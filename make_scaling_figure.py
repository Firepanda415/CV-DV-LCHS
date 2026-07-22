"""Regenerate the scaling-study figure from the dense visualization sweep.

Reads results_revision/viz_dense/scaling_dense.csv (produced by
make_scaling_dense.py at the Dirichlet-selected kernel point; 8 points per
axis) and writes figures/scaling_study.pdf.

Panel (a): fixed-scale map error eps_F versus T (exact-truncated-map rows).
Panel (b): measured product-formula error eps_t versus n_t at T=1 (circuit
rows) with a first-order 1/n_t guide.
Panel (c): postselection probability p of the benchmark input versus T.
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = Path("results_revision_v2/viz_dense/scaling_dense.csv")
OUTPUT = Path("figures/scaling_study.pdf")
META = Path("results_revision/paper_hbar2/scaling_study.meta.json")


def main():
    with SOURCE.open() as handle:
        rows = list(csv.DictReader(handle))
    model, circuit = [], []
    for row in rows:
        entry = {
            "n": int(row["n_coeff"]),
            "nf": int(row["n_fock"]),
            "t": float(row["total_time"]),
            "eps": float(row["rel_frobenius_error"]),
        }
        if row["case"] == "scaling_dense_model":
            entry["p"] = float(row["p_succ"])
            model.append(entry)
        else:
            entry["nt"] = int(row["n_trotter_steps"])
            entry["eps_t"] = float(row["eps_t"])
            circuit.append(entry)

    pairs = sorted({(row["n"], row["nf"]) for row in model})
    t_vals = sorted({row["t"] for row in model})
    nt_vals = sorted({row["nt"] for row in circuit})
    markers = dict(zip(pairs, ["o", "s", "^"]))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    fig.subplots_adjust(wspace=0.34)

    def pair_label(pair):
        return rf"$(N, N_{{\rm Fock}})=({pair[0]}, {pair[1]})$"

    ax = axes[0]
    for pair in pairs:
        sub = sorted(
            (row for row in model if (row["n"], row["nf"]) == pair),
            key=lambda row: row["t"],
        )
        ax.plot(
            [row["t"] for row in sub],
            [row["eps"] for row in sub],
            marker=markers[pair],
            markersize=4.5,
            label=pair_label(pair),
        )
    ax.set_yscale("log")
    ax.set_xticks(t_vals, [f"{v:g}" for v in t_vals], fontsize=8)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\varepsilon_F$")
    ax.set_title("(a) map error vs. evolution time", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for pair in pairs:
        sub = sorted(
            (row for row in circuit if (row["n"], row["nf"]) == pair),
            key=lambda row: row["nt"],
        )
        ax.plot(
            [row["nt"] for row in sub],
            [row["eps_t"] for row in sub],
            marker=markers[pair],
            markersize=4.5,
            label=pair_label(pair),
        )
    guide_anchor = max(row["eps_t"] for row in circuit if row["nt"] == nt_vals[0])
    ax.plot(
        nt_vals,
        [1.6 * guide_anchor * nt_vals[0] / nt for nt in nt_vals],
        linestyle="--",
        color="gray",
        label=r"$\propto 1/n_t$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(nt_vals, [str(v) for v in nt_vals], fontsize=8)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$n_t$")
    ax.set_ylabel(r"measured $\varepsilon_t$")
    ax.set_title(r"(b) product-formula error at $T=1$", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    for pair in pairs:
        sub = sorted(
            (row for row in model if (row["n"], row["nf"]) == pair),
            key=lambda row: row["t"],
        )
        ax.plot(
            [row["t"] for row in sub],
            [row["p"] for row in sub],
            marker=markers[pair],
            markersize=4.5,
            label=pair_label(pair),
        )
    ax.set_yscale("log")
    ax.set_xticks(t_vals, [f"{v:g}" for v in t_vals], fontsize=8)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$p$")
    ax.set_title("(c) postselection probability vs. time", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")

    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    META.write_text(
        json.dumps(
            {
                "output": str(OUTPUT),
                "source": str(SOURCE),
                "source_sha256": digest,
                "panel_a": "eps_F vs T, exact-truncated-map rows (8 T values)",
                "panel_b": "measured eps_t vs n_t at T=1, circuit rows (8 n_t values), 1/n_t guide",
                "panel_c": "benchmark-input p vs T (8 T values)",
                "matplotlib": matplotlib.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT}")
    print(f"source sha256 {digest}")


if __name__ == "__main__":
    main()
