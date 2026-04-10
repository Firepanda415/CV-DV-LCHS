#!/usr/bin/env python3
"""Build bound-related analysis assets for the Section 6 experiments.

This script is intentionally read-only with respect to repo-tracked TeX files.
It consumes existing sweep outputs and writes fresh analysis artifacts under
``results_clean_bounds/`` so the experiments section can be updated manually.

The current study has two distinct goals:

1. Provide a representative empirical check of the first-order heat-equation
   Trotter bound from Eq. ``(heat-trotter-error)`` by comparing observed
   truncated-reference error against a ``C/n`` guide line.
2. Summarize how the best oracle-baseline fidelity changes when the oscillator
   coefficient cutoff ``n_coeff`` is increased from 24 to 48 in the refined
   injection sweeps.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_TROTTER_SWEEP = ROOT / "results_clean_refine" / "sweep_all.csv"
DEFAULT_REFINED_SWEEPS = (
    ROOT / "results_clean_refined_dirichlet" / "sweep_all.csv",
    ROOT / "results_clean_refined_neumann" / "sweep_all.csv",
    ROOT / "results_clean_refined_periodic" / "sweep_all.csv",
)
DEFAULT_OUTPUT_DIR = ROOT / "results_clean_bounds"

BOUNDARY_LABELS = {
    "dirichlet": "Dirichlet",
    "neumann": "Neumann",
    "periodic": "Periodic",
}
BOUNDARY_ORDER = ("dirichlet", "neumann", "periodic")


def _coerce_value(raw: str) -> Any:
    """Convert CSV text into a typed Python value.

    Args:
        raw: Raw field text from ``csv.DictReader``.

    Returns:
        Parsed boolean, integer, float, or original string.
    """

    text = (raw or "").strip()
    if text == "":
        return ""
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if "." not in text and "e" not in text.lower():
            return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _read_valid_rows(path: Path) -> List[Dict[str, Any]]:
    """Load valid sweep rows from a CSV file.

    Args:
        path: Sweep CSV path.

    Returns:
        Parsed rows whose ``valid`` field is true.
    """

    with path.open() as handle:
        rows = [{key: _coerce_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    return [row for row in rows if bool(row.get("valid", False))]


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write a list of dictionaries as CSV."""

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def trotter_tightness_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize best truncated-reference errors by Trotter depth.

    The bound in Eq. ``(heat-trotter-error)`` is first order in the number of
    product-formula steps, so the natural empirical comparison is against a
    guide line of the form ``C / n``. We choose ``C`` so the guide matches the
    observed error at the smallest available ``n``.

    Args:
        rows: Valid sweep rows from the Dirichlet-focused Trotter sweep.

    Returns:
        Rows with empirical error and the normalized ``C / n`` guide.
    """

    grouped: Dict[int, List[float]] = {}
    for row in rows:
        n_trotter = int(row["n_trotter_steps"])
        grouped.setdefault(n_trotter, []).append(float(row["rel_error_vs_truncated"]))

    out: List[Dict[str, Any]] = []
    n_values = sorted(grouped)
    if not n_values:
        return out

    base_n = n_values[0]
    base_error = min(grouped[base_n])
    c_value = base_error * base_n
    for n_trotter in n_values:
        best_error = min(grouped[n_trotter])
        out.append(
            {
                "n_trotter_steps": n_trotter,
                "best_rel_error_vs_truncated": best_error,
                "c_over_n_guide": c_value / n_trotter,
            }
        )
    return out


def stateprep_consistency_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    """Summarize best exact fidelity versus ``n_coeff`` for refined sweeps.

    Args:
        paths: Sweep CSV paths for the refined Dirichlet, Neumann, and periodic
            oracle-baseline studies.

    Returns:
        Rows keyed by boundary condition and ``n_coeff``.
    """

    out: List[Dict[str, Any]] = []
    for path in paths:
        rows = _read_valid_rows(path)
        if not rows:
            continue
        boundary = str(rows[0]["boundary_condition"])
        grouped: Dict[int, List[float]] = {}
        for row in rows:
            grouped.setdefault(int(row["n_coeff"]), []).append(float(row["fidelity"]))
        for n_coeff in sorted(grouped):
            out.append(
                {
                    "boundary_condition": boundary,
                    "label": BOUNDARY_LABELS[boundary],
                    "n_coeff": n_coeff,
                    "best_exact_fidelity": max(grouped[n_coeff]),
                }
            )

    out.sort(key=lambda row: (BOUNDARY_ORDER.index(str(row["boundary_condition"])), int(row["n_coeff"])))
    return out


def _configure_matplotlib() -> None:
    """Configure a paper-friendly plotting style."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 7.5,
        }
    )


def make_trotter_tightness_figure(rows: Sequence[Mapping[str, Any]], *, out_path_base: Path) -> None:
    """Create a compact empirical-versus-bound scaling plot.

    Args:
        rows: Trotter summary rows created by :func:`trotter_tightness_rows`.
        out_path_base: Output path stem; ``.png`` and ``.pdf`` are written.
    """

    import matplotlib.pyplot as plt

    _configure_matplotlib()

    n_values = np.array([int(row["n_trotter_steps"]) for row in rows], dtype=float)
    observed = np.array([float(row["best_rel_error_vs_truncated"]) for row in rows], dtype=float)
    guide = np.array([float(row["c_over_n_guide"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    ax.loglog(
        n_values,
        observed,
        marker="o",
        color="#1f3b73",
        markeredgewidth=1.4,
        label=r"Observed best $\mathrm{rel\_error\_vs\_truncated}$",
    )
    ax.loglog(
        n_values,
        guide,
        linestyle="--",
        color="#b34700",
        label=r"$C/n$ guide (normalized at $n=8$)",
    )
    ax.set_xlabel(r"$n_{\mathrm{trotter}}$")
    ax.set_ylabel("Best relative error vs truncated reference")
    ax.legend(frameon=False, loc="upper right")

    fig.savefig(out_path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Build Section 6 bound-analysis assets from existing sweep data.")
    parser.add_argument(
        "--trotter-sweep",
        default=str(DEFAULT_TROTTER_SWEEP),
        help="CSV containing the existing n_trotter sweep used for the representative tightness check.",
    )
    parser.add_argument(
        "--refined-sweeps",
        default=",".join(str(path) for path in DEFAULT_REFINED_SWEEPS),
        help="Comma-separated refined oracle-baseline sweep CSVs for state-preparation consistency summaries.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where bound-analysis figures and CSV summaries will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the bound-analysis pipeline."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trotter_rows = trotter_tightness_rows(_read_valid_rows(Path(args.trotter_sweep)))
    stateprep_rows = stateprep_consistency_rows(Path(part.strip()) for part in args.refined_sweeps.split(",") if part.strip())

    _write_csv(trotter_rows, output_dir / "section6_trotter_tightness.csv")
    _write_csv(stateprep_rows, output_dir / "section6_stateprep_consistency.csv")
    make_trotter_tightness_figure(trotter_rows, out_path_base=output_dir / "section6_trotter_tightness")


if __name__ == "__main__":
    main()
