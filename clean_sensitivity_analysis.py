#!/usr/bin/env python3
"""Build a more interpretable Section 6 parameter-selection figure.

The checked-in manuscript figure projects the refined injection sweep onto
one-dimensional curves by fixing a single parameter and maximizing over the
rest. That view is compact, but it hides the geometry of the high-fidelity
region. This script creates a companion figure that exposes the raw
``(r_target, r_prime)`` response surface at the selected ``beta`` together
with a separate ``beta`` profile for ``n_coeff in {24, 48}``.

Outputs are written under ``results_clean_sensitivity/`` by default:

* ``section6_sensitivity_landscape.pdf`` and ``.png``:
  Three rows (Dirichlet, Neumann, Periodic). The left column shows exact
  infidelity ``1 - F`` over ``(r_target, r_prime)`` at the best ``beta`` for
  ``n_coeff = 48``. The right column shows the best achievable infidelity
  versus ``beta`` after optimizing over ``(r_target, r_prime)`` separately for
  ``n_coeff = 24`` and ``48``.
* ``section6_surface_rows.csv``:
  Plot-ready surface data used in the left column.
* ``section6_beta_profile_rows.csv``:
  Plot-ready beta-profile data used in the right column.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_REFINED_SWEEPS = (
    ROOT / "results_clean_refined_dirichlet" / "sweep_all.csv",
    ROOT / "results_clean_refined_neumann" / "sweep_all.csv",
    ROOT / "results_clean_refined_periodic" / "sweep_all.csv",
)
DEFAULT_OUTPUT_DIR = ROOT / "results_clean_sensitivity"

BOUNDARY_LABELS = {
    "dirichlet": "Dirichlet",
    "neumann": "Neumann",
    "periodic": "Periodic",
}
BOUNDARY_ORDER = ("dirichlet", "neumann", "periodic")


def _coerce_value(raw: str) -> Any:
    """Convert CSV text into a typed Python value."""

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
    """Load valid sweep rows from a CSV file."""

    with path.open() as handle:
        rows = [{key: _coerce_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    return [row for row in rows if bool(row.get("valid", False))]


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write a heterogeneous row sequence to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
            "grid.alpha": 0.16,
            "grid.linewidth": 0.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
        }
    )


def _bin_edges(values: Sequence[float]) -> np.ndarray:
    """Return pcolormesh edges for an irregular one-dimensional grid."""

    arr = np.array(sorted(float(value) for value in values), dtype=float)
    if arr.size == 1:
        step = 0.5
        return np.array([arr[0] - step, arr[0] + step], dtype=float)

    midpoints = 0.5 * (arr[:-1] + arr[1:])
    first = arr[0] - (midpoints[0] - arr[0])
    last = arr[-1] + (arr[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def _format_number(value: float) -> str:
    """Render compact numeric labels for plot annotations."""

    return f"{float(value):g}"


def _profile_summary_line(rows_for_n: Sequence[Mapping[str, Any]], *, n_coeff: int) -> str:
    """Summarize the fixed squeeze pair used in one beta-profile curve."""

    if not rows_for_n:
        return f"n{n_coeff}: no data"

    first = rows_for_n[0]
    return (
        f"n{n_coeff}: "
        f"{_format_number(float(first['anchor_r_target']))}/"
        f"{_format_number(float(first['anchor_r_prime']))}"
    )


def _best_beta_for_n_coeff(rows: Sequence[Mapping[str, Any]], *, n_coeff: int) -> float:
    """Select the beta whose best row gives the smallest exact infidelity."""

    candidates = [row for row in rows if int(row["n_coeff"]) == n_coeff]
    if not candidates:
        raise ValueError(f"No rows found for n_coeff={n_coeff}.")

    best_row = min(candidates, key=lambda row: 1.0 - float(row["fidelity"]))
    return float(best_row["beta"])


def surface_rows(path: Path, *, focus_n_coeff: int) -> List[Dict[str, Any]]:
    """Build surface rows for one boundary condition."""

    rows = _read_valid_rows(path)
    if not rows:
        return []

    boundary = str(rows[0]["boundary_condition"])
    beta_focus = _best_beta_for_n_coeff(rows, n_coeff=focus_n_coeff)
    panel = [
        row
        for row in rows
        if int(row["n_coeff"]) == focus_n_coeff and float(row["beta"]) == beta_focus
    ]
    if not panel:
        return []

    best = min(panel, key=lambda row: 1.0 - float(row["fidelity"]))
    out: List[Dict[str, Any]] = []
    for row in panel:
        infidelity = max(1e-12, 1.0 - float(row["fidelity"]))
        out.append(
            {
                "boundary_condition": boundary,
                "label": BOUNDARY_LABELS[boundary],
                "focus_n_coeff": focus_n_coeff,
                "beta_focus": beta_focus,
                "r_target": float(row["r_target"]),
                "r_prime": float(row["r_prime"]),
                "fidelity": float(row["fidelity"]),
                "infidelity": infidelity,
                "is_best_surface_point": (
                    float(row["r_target"]) == float(best["r_target"])
                    and float(row["r_prime"]) == float(best["r_prime"])
                ),
            }
        )
    out.sort(key=lambda row: (float(row["r_prime"]), float(row["r_target"])))
    return out


def beta_profile_rows(path: Path, *, n_coeff_values: Iterable[int]) -> List[Dict[str, Any]]:
    """Build beta-profile rows using one fixed squeeze pair per line."""

    rows = _read_valid_rows(path)
    if not rows:
        return []

    boundary = str(rows[0]["boundary_condition"])
    out: List[Dict[str, Any]] = []
    betas = sorted({float(row["beta"]) for row in rows})
    for n_coeff in n_coeff_values:
        candidates_for_n = [row for row in rows if int(row["n_coeff"]) == int(n_coeff)]
        if not candidates_for_n:
            continue
        anchor = min(candidates_for_n, key=lambda row: 1.0 - float(row["fidelity"]))
        anchor_r_target = float(anchor["r_target"])
        anchor_r_prime = float(anchor["r_prime"])
        for beta in betas:
            candidates = [
                row
                for row in rows
                if int(row["n_coeff"]) == int(n_coeff)
                and float(row["beta"]) == beta
                and float(row["r_target"]) == anchor_r_target
                and float(row["r_prime"]) == anchor_r_prime
            ]
            if not candidates:
                continue
            row = candidates[0]
            out.append(
                {
                    "boundary_condition": boundary,
                    "label": BOUNDARY_LABELS[boundary],
                    "n_coeff": int(n_coeff),
                    "beta": beta,
                    "anchor_r_target": anchor_r_target,
                    "anchor_r_prime": anchor_r_prime,
                    "fidelity": float(row["fidelity"]),
                    "infidelity": max(1e-12, 1.0 - float(row["fidelity"])),
                }
            )
    out.sort(key=lambda row: (int(row["n_coeff"]), float(row["beta"])))
    return out


def make_landscape_figure(
    surface_data: Sequence[Mapping[str, Any]],
    profile_data: Sequence[Mapping[str, Any]],
    *,
    out_path_base: Path,
) -> None:
    """Render the landscape-plus-profile figure."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError as exc:
        raise ImportError("matplotlib is required to generate sensitivity plots.") from exc

    if not surface_data or not profile_data:
        return

    _configure_matplotlib()

    grouped_surface: Dict[str, List[Mapping[str, Any]]] = {
        boundary: [row for row in surface_data if str(row["boundary_condition"]) == boundary]
        for boundary in BOUNDARY_ORDER
    }
    grouped_profile: Dict[str, List[Mapping[str, Any]]] = {
        boundary: [row for row in profile_data if str(row["boundary_condition"]) == boundary]
        for boundary in BOUNDARY_ORDER
    }

    positive_infidelities = [float(row["infidelity"]) for row in surface_data if float(row["infidelity"]) > 0.0]
    vmin = max(1e-6, min(positive_infidelities))
    vmax = max(positive_infidelities)

    fig, axes = plt.subplots(
        len(BOUNDARY_ORDER),
        2,
        figsize=(10.0, 10.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.05, 1.0)},
    )
    mesh = None
    line_colors = {24: "#b34700", 48: "#1f3b73"}
    legend_locs = {
        0: "upper right",
        1: "center right",
        2: "center right",
    }

    for row_index, boundary in enumerate(BOUNDARY_ORDER):
        surface_rows_for_boundary = grouped_surface[boundary]
        profile_rows_for_boundary = grouped_profile[boundary]
        if not surface_rows_for_boundary or not profile_rows_for_boundary:
            continue

        surface_ax = axes[row_index, 0]
        xs = sorted({float(row["r_target"]) for row in surface_rows_for_boundary})
        ys = sorted({float(row["r_prime"]) for row in surface_rows_for_boundary})
        z = np.full((len(ys), len(xs)), np.nan, dtype=float)
        for row in surface_rows_for_boundary:
            x_index = xs.index(float(row["r_target"]))
            y_index = ys.index(float(row["r_prime"]))
            z[y_index, x_index] = float(row["infidelity"])

        x_edges = _bin_edges(xs)
        y_edges = _bin_edges(ys)
        mesh = surface_ax.pcolormesh(
            x_edges,
            y_edges,
            z,
            shading="auto",
            cmap="magma_r",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        surface_ax.set_xticks(
            xs,
            labels=[_format_number(value) for value in xs],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        surface_ax.set_yticks(ys, labels=[_format_number(value) for value in ys])
        surface_ax.set_xlabel(r"$r_{\mathrm{target}}$")
        surface_ax.set_ylabel(r"$r^\prime$")
        surface_ax.set_title(
            f"{BOUNDARY_LABELS[boundary]}: "
            rf"$n_{{\mathrm{{coeff}}}}={int(surface_rows_for_boundary[0]['focus_n_coeff'])}$, "
            rf"$\beta={float(surface_rows_for_boundary[0]['beta_focus']):g}$"
        )

        profile_ax = axes[row_index, 1]
        for n_coeff in (24, 48):
            rows_for_n = [row for row in profile_rows_for_boundary if int(row["n_coeff"]) == n_coeff]
            betas = [float(row["beta"]) for row in rows_for_n]
            infidelities = [float(row["infidelity"]) for row in rows_for_n]
            anchor_r_target = float(rows_for_n[0]["anchor_r_target"])
            anchor_r_prime = float(rows_for_n[0]["anchor_r_prime"])
            profile_ax.plot(
                betas,
                infidelities,
                marker="o",
                color=line_colors[n_coeff],
                label=(
                    rf"$n_{{\mathrm{{coeff}}}}={n_coeff},\ "
                    rf"r_{{\mathrm{{target}}}}={_format_number(anchor_r_target)},\ "
                    rf"r^\prime={_format_number(anchor_r_prime)}$"
                ),
            )
        profile_ax.set_xlabel(r"$\beta$")
        profile_ax.set_ylabel(r"Infidelity at fixed $(r_{\mathrm{target}}, r^\prime)$")
        profile_ax.set_yscale("log")
        profile_ax.set_title(BOUNDARY_LABELS[boundary])
        profile_ax.legend(frameon=False, loc=legend_locs[row_index], fontsize=8.5)

    assert mesh is not None
    colorbar = fig.colorbar(mesh, ax=axes[:, 0], shrink=0.84, pad=0.03)
    colorbar.set_label(r"Infidelity $1 - F_{\mathrm{injection}}$")

    fig.savefig(out_path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Build a more interpretable Section 6 sensitivity figure.")
    parser.add_argument(
        "--refined-sweeps",
        default=",".join(str(path) for path in DEFAULT_REFINED_SWEEPS),
        help="Comma-separated refined injection sweep CSVs for Dirichlet, Neumann, and Periodic.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the figure and plot-ready CSVs will be written.",
    )
    parser.add_argument(
        "--focus-n-coeff",
        type=int,
        default=48,
        help="n_coeff value used for the response-surface panels.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the analysis."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    refined_sweeps = tuple(Path(item) for item in str(args.refined_sweeps).split(",") if item.strip())

    all_surface_rows: List[Dict[str, Any]] = []
    all_profile_rows: List[Dict[str, Any]] = []
    for path in refined_sweeps:
        all_surface_rows.extend(surface_rows(path, focus_n_coeff=args.focus_n_coeff))
        all_profile_rows.extend(beta_profile_rows(path, n_coeff_values=(24, args.focus_n_coeff)))

    _write_csv(all_surface_rows, output_dir / "section6_surface_rows.csv")
    _write_csv(all_profile_rows, output_dir / "section6_beta_profile_rows.csv")
    make_landscape_figure(
        all_surface_rows,
        all_profile_rows,
        out_path_base=output_dir / "section6_sensitivity_landscape",
    )


if __name__ == "__main__":
    main()
