"""Regenerate the parameter-sensitivity figure from the dense grid-search sweep.

Reads results_revision/viz_dense/landscape_dense.csv plus the r' extension
landscape_dense_ext.csv (coarse grid search; >= 8 values per axis; produced by
make_landscape_dense.py through the same exact-map code path as the selection
grid) and writes figures/sensitivity_landscape_v2.pdf.

Left column: fixed-scale map error eps_F over (r, r'), minimized over (beta, N).
Grey cells are invalid (r' >= r) or have numerically vanishing zeroth-moment
overlap. Right column: eps_F versus beta at the shared operating point's
(r, r') for N in {16, 24, 32}. No selection markers are drawn: the operating
point is chosen from this grid by weighing map error together with the
postselection probability, not by the eps_F argmin alone, and is stated in the
text.
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCES = [
    Path("results_revision/viz_dense/landscape_dense.csv"),
    Path("results_revision/viz_dense/landscape_dense_ext.csv"),
]
SELECTED = Path("results_revision/selected_params_v2.json")
OUTPUT = Path("figures/sensitivity_landscape_v2.pdf")
META = Path("results_revision/paper_hbar2/sensitivity_landscape_v2.meta.json")

BOUNDARIES = ["dirichlet", "neumann", "periodic"]
LABELS = {"dirichlet": "Dirichlet", "neumann": "Neumann", "periodic": "Periodic"}


def edges(values):
    values = np.asarray(values, dtype=float)
    mids = (values[1:] + values[:-1]) / 2
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def main():
    rows = []
    for source in SOURCES:
        with source.open() as handle:
            for row in csv.DictReader(handle):
                if row["rel_frobenius_error"] == "":
                    continue
                rows.append(
                    {
                        "boundary": row["boundary"],
                        "r": float(row["r_target"]),
                        "rp": float(row["r_prime"]),
                        "beta": float(row["beta"]),
                        "n": int(row["n_coeff"]),
                        "eps": float(row["rel_frobenius_error"]),
                    }
                )

    shared = json.loads(SELECTED.read_text())["boundaries"]["dirichlet"]
    sel_r, sel_rp = float(shared["r_target"]), float(shared["r_prime"])

    r_vals = sorted({row["r"] for row in rows})
    rp_vals = sorted({row["rp"] for row in rows})
    beta_vals = sorted({row["beta"] for row in rows})
    n_vals = sorted({row["n"] for row in rows})
    r_edges, rp_edges = edges(r_vals), edges(rp_vals)

    grids = {}
    for boundary in BOUNDARIES:
        sub = [row for row in rows if row["boundary"] == boundary]
        grid = np.full((len(rp_vals), len(r_vals)), np.nan)
        for yi, rp in enumerate(rp_vals):
            for xi, r in enumerate(r_vals):
                cell = [row["eps"] for row in sub if row["r"] == r and row["rp"] == rp]
                if cell:
                    grid[yi, xi] = min(cell)
        grids[boundary] = grid
    vmin = min(np.nanmin(grid) for grid in grids.values())
    vmax = max(np.nanmax(grid) for grid in grids.values())

    cmap = matplotlib.colormaps["viridis_r"].copy()
    cmap.set_bad("0.82")

    fig, axes = plt.subplots(3, 2, figsize=(9.8, 11.6))
    fig.subplots_adjust(hspace=0.46, wspace=0.30)

    for i, boundary in enumerate(BOUNDARIES):
        sub = [row for row in rows if row["boundary"] == boundary]

        ax = axes[i, 0]
        mesh = ax.pcolormesh(
            r_edges,
            rp_edges,
            grids[boundary],
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap,
            edgecolors="white",
            linewidth=0.4,
        )
        ax.set_xticks(r_vals, [f"{v:g}" for v in r_vals], fontsize=8)
        ax.set_yticks(rp_vals, [f"{v:g}" for v in rp_vals], fontsize=8)
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$r'$")
        ax.set_title(
            rf"{LABELS[boundary]}: $\min_{{\beta,\,N}}\ \varepsilon_F$ over $(r, r')$",
            fontsize=11,
        )
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[i, 1]
        markers = dict(zip(n_vals, ["o", "s", "^"]))
        for n in n_vals:
            curve = sorted(
                (row["beta"], row["eps"])
                for row in sub
                if row["r"] == sel_r and row["rp"] == sel_rp and row["n"] == n
            )
            ax.plot(
                [point[0] for point in curve],
                [point[1] for point in curve],
                marker=markers[n],
                markersize=4.5,
                label=rf"$N={n}$",
            )
        ax.set_yscale("log")
        ax.set_xticks(beta_vals)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\varepsilon_F$")
        ax.set_title(
            rf"{LABELS[boundary]}: $\varepsilon_F$ vs $\beta$ at $(r, r')=({sel_r:g}, {sel_rp:g})$",
            fontsize=11,
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")

    META.write_text(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sources": {
                    str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                    for source in SOURCES
                },
                "selected_params_artifact": str(SELECTED),
                "selected_params_sha256": hashlib.sha256(
                    SELECTED.read_bytes()
                ).hexdigest(),
                "metric": "rel_frobenius_error (fixed-alpha eps_F)",
                "left_panel": "min over (beta, n_coeff) per (r, r') cell; grey = invalid window or vanishing overlap; no selection markers by design",
                "right_panel": "eps_F vs beta at the shared operating point's (r, r') per n_coeff",
                "grid_points_per_axis": {
                    "r": len(r_vals),
                    "r_prime": len(rp_vals),
                    "beta": len(beta_vals),
                },
                "matplotlib": matplotlib.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
