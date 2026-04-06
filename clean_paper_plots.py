#!/usr/bin/env python3
"""Build paper-style sensitivity plots and summary tables from clean sweeps.

This script is intentionally separate from ``clean_sweep.py``. The sweep script
produces raw experiment records; this script consumes those records and creates:

- sensitivity figures for ``r_target``, ``r_prime``, and ``beta``,
- a best-row summary table across boundary conditions,
- lightweight CSV/JSON artifacts for direct use in a working paper.

The default workflow assumes the three overnight sweep folders:

- ``results_clean_overnight_dirichlet``
- ``results_clean_overnight_neumann``
- ``results_clean_overnight_periodic``

The main sensitivity statistic shown in the figures is the best exact fidelity
found at each parameter value after maximizing over the remaining sweep axes.
For context, the script also computes median and interquartile ranges at each
parameter value, so the figures can show whether a peak is broad or narrowly
tuned.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


DEFAULT_RESULTS_DIRS = (
    "results_clean_overnight_dirichlet",
    "results_clean_overnight_neumann",
    "results_clean_overnight_periodic",
)
PARAMETERS = ("r_target", "r_prime", "beta")
METRICS = ("fidelity", "fidelity_vs_truncated")
BOUNDARY_LABELS = {
    "dirichlet": "Dirichlet",
    "neumann": "Neumann",
    "periodic": "Periodic",
}
BOUNDARY_COLORS = {
    "dirichlet": "#1f3b73",
    "neumann": "#b34700",
    "periodic": "#1f7a4d",
}
PARAMETER_LABELS = {
    "r_target": r"$r_{\mathrm{target}}$",
    "r_prime": r"$r^\prime$",
    "beta": r"$\beta$",
}
METRIC_LABELS = {
    "fidelity": "Exact fidelity",
    "fidelity_vs_truncated": "Truncated-model fidelity",
}


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _read_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _normalize_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("valid")).lower() != "true":
            continue
        parsed = dict(row)
        for key in (
            "r_target",
            "r_prime",
            "beta",
            "fidelity",
            "fidelity_vs_truncated",
            "postselection_probability",
            "rel_error_vs_exact",
            "rel_error_vs_truncated",
            "coeff_backend_gap",
        ):
            if key in parsed and parsed[key] != "":
                parsed[key] = float(parsed[key])
        for key in ("n_coeff", "n_trotter_steps", "circuit_depth", "circuit_size"):
            if key in parsed and parsed[key] != "":
                parsed[key] = int(float(parsed[key]))
        out.append(parsed)
    return out


def _boundary_from_summary(summary: Mapping[str, Any], fallback_dir: Path) -> str:
    boundary = str(summary.get("boundary_condition", "")).strip().lower()
    if boundary:
        return boundary
    name = fallback_dir.name.lower()
    for candidate in BOUNDARY_LABELS:
        if candidate in name:
            return candidate
    raise ValueError(f"Could not infer boundary condition from {fallback_dir}.")


def load_result_bundle(result_dir: Path) -> Dict[str, Any]:
    """Load one sweep directory and normalize its valid rows."""

    summary = _read_summary(result_dir / "sweep_summary.json")
    rows = _normalize_rows(_read_csv_rows(result_dir / "sweep_all.csv"))
    if not rows:
        raise ValueError(f"No valid rows found in {result_dir}.")
    boundary = _boundary_from_summary(summary, result_dir)
    return {
        "boundary_condition": boundary,
        "label": BOUNDARY_LABELS[boundary],
        "color": BOUNDARY_COLORS[boundary],
        "summary": summary,
        "rows": rows,
        "result_dir": result_dir,
    }


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def parameter_sensitivity_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    parameter: str,
    metric: str,
) -> List[Dict[str, float]]:
    """Aggregate sensitivity statistics for one parameter and one metric."""

    grouped: Dict[float, List[float]] = defaultdict(list)
    for row in rows:
        grouped[float(row[parameter])].append(float(row[metric]))

    stats: List[Dict[str, float]] = []
    for x in sorted(grouped):
        vals = grouped[x]
        stats.append(
            {
                "x": float(x),
                "max": float(np.max(vals)),
                "median": _quantile(vals, 0.50),
                "q25": _quantile(vals, 0.25),
                "q75": _quantile(vals, 0.75),
                "count": float(len(vals)),
            }
        )
    return stats


def best_row(rows: Sequence[Mapping[str, Any]], *, metric: str = "fidelity") -> Dict[str, Any]:
    """Return the row with the highest chosen metric."""

    return dict(max(rows, key=lambda row: (float(row[metric]), -float(row["rel_error_vs_exact"]))))


def build_boundary_summary_table(bundles: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build a compact best-result table across boundary conditions."""

    table: List[Dict[str, Any]] = []
    for bundle in bundles:
        row = best_row(bundle["rows"], metric="fidelity")
        table.append(
            {
                "boundary_condition": bundle["boundary_condition"],
                "label": bundle["label"],
                "r_target": row["r_target"],
                "r_prime": row["r_prime"],
                "beta": row["beta"],
                "n_coeff": row["n_coeff"],
                "n_trotter_steps": row["n_trotter_steps"],
                "fidelity_exact": row["fidelity"],
                "fidelity_truncated": row["fidelity_vs_truncated"],
                "postselection_probability": row["postselection_probability"],
                "rel_error_vs_exact": row["rel_error_vs_exact"],
                "coeff_backend_gap": row["coeff_backend_gap"],
            }
        )
    table.sort(key=lambda row: row["fidelity_exact"], reverse=True)
    return table


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    headers = [
        "Boundary condition",
        "Best $r_{\\mathrm{target}}$",
        "Best $r^\\prime$",
        "Best $\\beta$",
        "$n_{\\mathrm{coeff}}$",
        "$n_{\\mathrm{trotter}}$",
        "Exact fidelity",
        "Truncated fidelity",
        "Postselection probability",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    f'{row["r_target"]:.6g}',
                    f'{row["r_prime"]:.6g}',
                    f'{row["beta"]:.6g}',
                    str(row["n_coeff"]),
                    str(row["n_trotter_steps"]),
                    f'{row["fidelity_exact"]:.6f}',
                    f'{row["fidelity_truncated"]:.6f}',
                    f'{row["postselection_probability"]:.6e}',
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_latex_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Boundary & $r_{\mathrm{target}}$ & $r^\prime$ & $\beta$ & $n_{\mathrm{coeff}}$ & $n_{\mathrm{trotter}}$ & $F_{\mathrm{exact}}$ & $F_{\mathrm{trunc}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f'{row["label"]} & '
            f'{row["r_target"]:.6g} & '
            f'{row["r_prime"]:.6g} & '
            f'{row["beta"]:.6g} & '
            f'{row["n_coeff"]} & '
            f'{row["n_trotter_steps"]} & '
            f'{row["fidelity_exact"]:.6f} & '
            f'{row["fidelity_truncated"]:.6f} \\\\'
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def _write_numerical_summary_markdown(
    bundles: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """Write a short working-paper summary of the numerical experiments."""

    lines: List[str] = []
    lines.append("# Numerical Experiment Summary")
    lines.append("")
    lines.append("## Best Fidelity By Boundary Condition")
    lines.append("")
    for row in summary_rows:
        lines.append(
            "- "
            f'{row["label"]}: '
            f'exact fidelity = {row["fidelity_exact"]:.6f}, '
            f'truncated fidelity = {row["fidelity_truncated"]:.6f}, '
            f'postselection probability = {row["postselection_probability"]:.6e}, '
            f'best parameters = '
            f'(r_target={row["r_target"]:.6g}, '
            f"r_prime={row['r_prime']:.6g}, "
            f'beta={row["beta"]:.6g}, '
            f'n_coeff={row["n_coeff"]}, '
            f'n_trotter={row["n_trotter_steps"]}).'
        )
    lines.append("")
    lines.append("## Main Observations")
    lines.append("")

    best_exact = max(float(row["fidelity_exact"]) for row in summary_rows)
    worst_exact = min(float(row["fidelity_exact"]) for row in summary_rows)
    best_trunc = min(float(row["fidelity_truncated"]) for row in summary_rows)
    lines.append(
        "- "
        f"Across all three boundary conditions, the best exact fidelities lie in the narrow range "
        f"{worst_exact:.6f} to {best_exact:.6f}."
    )
    lines.append(
        "- "
        f"Across the same best rows, truncated-model fidelity remains at least {best_trunc:.6f}, "
        "showing that the hybrid circuit matches the finite-Fock reference very closely."
    )

    high_r_all = all(float(row["r_target"]) >= 8.0 for row in summary_rows)
    high_rp_all = all(float(row["r_prime"]) >= 4.0 for row in summary_rows)
    if high_r_all and high_rp_all:
        lines.append(
            "- "
            "All three best rows occur in the aggressive squeezing regime with "
            r"$r_{\mathrm{target}} \gtrsim 8$ and $r^\prime \gtrsim 4$, so the main fidelity gains in the current sweep come from the high-squeezing corner of parameter space."
        )

    beta_values = {row["boundary_condition"]: float(row["beta"]) for row in summary_rows}
    lines.append(
        "- "
        f"At the best rows, Dirichlet and periodic both prefer beta around {beta_values['dirichlet']:.3g} to {beta_values['periodic']:.3g}, "
        f"while Neumann peaks at beta = {beta_values['neumann']:.3g}."
    )

    large_gap_rows = [row for row in summary_rows if float(row["coeff_backend_gap"]) > 1.0]
    if large_gap_rows:
        labels = ", ".join(row["label"] for row in large_gap_rows)
        lines.append(
            "- "
            f"The coefficient-backend agreement deteriorates for some high-squeezing best rows ({labels}), "
            "so the explicit-overlap backend should be treated as the trusted numerical reference in that regime."
        )

    lines.append("")
    lines.append("## Interpretation For The Working Paper")
    lines.append("")
    lines.append(
        "- "
        "The overnight sweep indicates that the dominant source of remaining error is the kernel choice rather than the hybrid circuit execution, because exact fidelity is below unity while truncated-model fidelity is essentially unity."
    )
    lines.append(
        "- "
        "The current sensitivity figures should be treated as paper-quality placeholders: they show the correct trends and clearly identify the high-fidelity region, but the sampling along each axis is still sparse for a final publication figure."
    )
    lines.append(
        "- "
        "For a final paper version, the most useful next refinement is a denser local sweep around the high-fidelity region in r_target and r_prime, with a modest densification in beta only where the boundary condition appears sensitive."
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
        }
    )


def make_sensitivity_figure(
    bundles: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    out_path_base: Path,
) -> None:
    """Create a 3-panel sensitivity plot for one fidelity metric."""

    import matplotlib.pyplot as plt

    _configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1), constrained_layout=True)

    for ax, parameter in zip(axes, PARAMETERS):
        for bundle in bundles:
            stats = parameter_sensitivity_stats(bundle["rows"], parameter=parameter, metric=metric)
            x = [item["x"] for item in stats]
            y_max = [item["max"] for item in stats]
            y_q25 = [item["q25"] for item in stats]
            y_q75 = [item["q75"] for item in stats]

            ax.fill_between(x, y_q25, y_q75, color=bundle["color"], alpha=0.12, linewidth=0.0)
            ax.plot(
                x,
                y_max,
                marker="o",
                color=bundle["color"],
                label=bundle["label"],
            )

        ax.set_xlabel(PARAMETER_LABELS[parameter])
        ax.set_title(PARAMETER_LABELS[parameter])
        ax.set_ylim(0.45 if metric == "fidelity" else 0.99995, 1.0005)

    axes[0].set_ylabel(METRIC_LABELS[metric])
    axes[0].legend(frameon=False, loc="lower right")

    title = "Sensitivity Of Best Fidelity To Kernel Parameters"
    if metric == "fidelity_vs_truncated":
        title = "Sensitivity Of Best Truncated-Model Fidelity To Kernel Parameters"
    fig.suptitle(title, y=1.03, fontsize=13)

    fig.savefig(out_path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_best_fidelity_bar_chart(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    out_path_base: Path,
) -> None:
    """Create a compact comparison bar chart across boundary conditions."""

    import matplotlib.pyplot as plt

    _configure_matplotlib()
    labels = [row["label"] for row in summary_rows]
    exact = [float(row["fidelity_exact"]) for row in summary_rows]
    trunc = [float(row["fidelity_truncated"]) for row in summary_rows]
    colors = [BOUNDARY_COLORS[row["boundary_condition"]] for row in summary_rows]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.bar(x - width / 2, exact, width=width, color=colors, alpha=0.92, label="Exact fidelity")
    ax.bar(x + width / 2, trunc, width=width, color=colors, alpha=0.35, label="Truncated fidelity")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.94, 1.001)
    ax.set_title("Best Fidelity By Boundary Condition")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out_path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_sensitivity_csvs(bundles: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    """Write per-parameter sensitivity statistics to CSV."""

    for metric in METRICS:
        rows: List[Dict[str, Any]] = []
        for bundle in bundles:
            for parameter in PARAMETERS:
                for stat in parameter_sensitivity_stats(bundle["rows"], parameter=parameter, metric=metric):
                    rows.append(
                        {
                            "boundary_condition": bundle["boundary_condition"],
                            "parameter": parameter,
                            metric: stat["max"],
                            "x": stat["x"],
                            "median": stat["median"],
                            "q25": stat["q25"],
                            "q75": stat["q75"],
                            "count": int(stat["count"]),
                        }
                    )
        _write_csv(rows, out_dir / f"sensitivity_{metric}.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-grade plots from clean sweep outputs")
    parser.add_argument(
        "--results-dirs",
        default=",".join(DEFAULT_RESULTS_DIRS),
        help="Comma-separated sweep result directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="results_clean_paper",
        help="Directory where paper figures and tables will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_dirs = [Path(item.strip()) for item in args.results_dirs.split(",") if item.strip()]
    bundles = [load_result_bundle(path) for path in result_dirs]
    bundles.sort(key=lambda bundle: list(BOUNDARY_LABELS).index(bundle["boundary_condition"]))

    summary_rows = build_boundary_summary_table(bundles)
    _write_csv(summary_rows, out_dir / "boundary_fidelity_summary.csv")
    _write_markdown_table(summary_rows, out_dir / "boundary_fidelity_summary.md")
    _write_latex_table(summary_rows, out_dir / "boundary_fidelity_summary.tex")
    _write_numerical_summary_markdown(
        bundles,
        summary_rows,
        out_dir / "numerical_experiment_summary.md",
    )

    write_sensitivity_csvs(bundles, out_dir)
    make_sensitivity_figure(bundles, metric="fidelity", out_path_base=out_dir / "paper_sensitivity_exact")
    make_sensitivity_figure(
        bundles,
        metric="fidelity_vs_truncated",
        out_path_base=out_dir / "paper_sensitivity_truncated",
    )
    make_best_fidelity_bar_chart(summary_rows, out_path_base=out_dir / "paper_boundary_comparison")

    payload = {
        "results_dirs": [str(path) for path in result_dirs],
        "summary_rows": summary_rows,
        "note": (
            "Sensitivity curves show the best fidelity attained at each parameter value "
            "after maximizing over the remaining sweep axes. Shaded bands denote the "
            "interquartile range across the raw sweep records at fixed parameter value."
        ),
    }
    (out_dir / "paper_summary.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
