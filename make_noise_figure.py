"""Render the photon-loss figure from results_revision_v2/noise_loss.csv.

Panel (a): benchmark-input conditional infidelity 1-F versus loss gamma
(log-log) with a linear-in-gamma guide. Panel (b): coherent-branch map
distortion versus gamma with the reference line (gamma/2) sqrt(<n^2>).
Panel (c): postselection probability versus gamma.
Writes figures/noise_loss.pdf.
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt

SOURCE = Path("results_revision_v2/noise_loss.csv")
OUTPUT = Path("figures/noise_loss.pdf")
META = Path("results_revision_v2/paper_hbar2_noise_loss.meta.json")

LABELS = {"dirichlet": "Dirichlet", "neumann": "Neumann", "periodic": "Periodic"}
MARKERS = {"dirichlet": "o", "neumann": "s", "periodic": "^"}


def main():
    with SOURCE.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in row:
            if key != "boundary":
                row[key] = float(row[key])

    gammas = sorted({row["gamma"] for row in rows})
    n2_sqrt = rows[0]["n2_sqrt"]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    for boundary in LABELS:
        sub = sorted(
            (row for row in rows if row["boundary"] == boundary),
            key=lambda row: row["gamma"],
        )
        ax.plot(
            [row["gamma"] for row in sub],
            [row["one_minus_F"] for row in sub],
            marker=MARKERS[boundary],
            markersize=4.5,
            label=LABELS[boundary],
        )
    ax.plot(
        gammas,
        [0.08 * g for g in gammas],
        linestyle="--",
        color="gray",
        label=r"$\propto(1-\eta)$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"loss $1-\eta$")
    ax.set_ylabel(r"$1-F$ (benchmark input)")
    ax.set_title("(a) conditional infidelity vs. loss", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for boundary in LABELS:
        sub = sorted(
            (row for row in rows if row["boundary"] == boundary),
            key=lambda row: row["gamma"],
        )
        ax.plot(
            [row["gamma"] for row in sub],
            [row["coherent_distortion"] for row in sub],
            marker=MARKERS[boundary],
            markersize=4.5,
            label=LABELS[boundary],
        )
    ax.plot(
        gammas,
        [0.5 * g * n2_sqrt for g in gammas],
        linestyle="--",
        color="crimson",
        label=r"$[(1-\eta)/2]\sqrt{\langle\hat n^2\rangle}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"loss $1-\eta$")
    ax.set_ylabel("coherent-branch map distortion")
    ax.set_title("(b) coherent-branch map distortion", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    for boundary in LABELS:
        sub = sorted(
            (row for row in rows if row["boundary"] == boundary),
            key=lambda row: row["gamma"],
        )
        ax.plot(
            [row["gamma"] for row in sub],
            [row["p_succ"] * 100 for row in sub],
            marker=MARKERS[boundary],
            markersize=4.5,
            label=LABELS[boundary],
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"loss $1-\eta$")
    ax.set_ylabel(r"$p$ [%]")
    ax.set_title("(c) postselection probability vs. loss", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    META.write_text(
        json.dumps(
            {
                "output": str(OUTPUT),
                "source": str(SOURCE),
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "matplotlib": matplotlib.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
