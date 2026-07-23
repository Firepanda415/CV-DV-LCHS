"""Two-order version of the joint cutoff-depth figure (Fig. 3 of the paper).

Renders the three-panel figure of `make_joint_tradeoff.py` with the
first-order proxy added to panels (b) and (c): for every error budget the
p = 2 curve of the archived study is drawn solid and the p = 1 counterpart
dashed, both with normalized prefactors (all commutator sums set to one) and
with their minimizing cutoffs marked.  For H = 0 the p = 1 factor of
Eq. (Lambda_p_definition) reduces to Lambda_1(NF) = 4 (NF - 1), so

    n_t^{(1)}(N; eps) = ceil( 4 (ceil(kappa N) - 1) / (eps - eps_tr(N)) ).

The truncation tail eps_tr(N) and the stretched-exponential fit of panel (a)
are read from the archived `results_joint_tradeoff/epsilon_tail.csv` and
refit with the same regression (log eps_tr against N^{1/4} over
40 <= N <= 200), so panel (a) is unchanged.

Validation gate (runs before anything is written): the p = 2 minimizers
recomputed from the archived tail must reproduce the per-(kappa, eps)
minimal-resource rows of `results_joint_tradeoff/joint_tradeoff.csv`
exactly.

Outputs:
    results_joint_tradeoff/joint_tradeoff_p1.csv   per-(kappa, eps, N) p=1 rows
    results_joint_tradeoff/joint_tradeoff_p1.meta.json
    figures/joint_cutoff_depth.pdf                 the paper figure (both orders)
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
TAIL_CSV = ROOT / "results_joint_tradeoff" / "epsilon_tail.csv"
ARCHIVE_CSV = ROOT / "results_joint_tradeoff" / "joint_tradeoff.csv"
OUTPUT_CSV = ROOT / "results_joint_tradeoff" / "joint_tradeoff_p1.csv"
META = ROOT / "results_joint_tradeoff" / "joint_tradeoff_p1.meta.json"
FIGURE_PATH = ROOT / "figures" / "joint_cutoff_depth.pdf"

EPSILONS = (1e-1, 5e-2, 3e-2, 1e-2, 5e-3)
KAPPAS = (("2", 2.0), ("4/3", 4.0 / 3.0))


def load_tail():
    n_vals, eps_vals = [], []
    with TAIL_CSV.open() as handle:
        for row in csv.DictReader(handle):
            n_vals.append(int(row["n_coeff"]))
            eps_vals.append(float(row["eps_tr"]))
    return np.array(n_vals), np.array(eps_vals)


N_GRID, EPS_TR = load_tail()


def lambda_normalized(n_fock, p):
    """Lambda_p with all commutator sums set to one, as in the p=2 study."""
    base = float(n_fock - 1)
    mixed = sum(2.0**a * base ** (a / 2.0) for a in range(1, p + 1))
    return mixed + 2.0 ** (p + 1) * base ** ((p + 1) / 2.0) + 1.0


def lambda_p1_pure_l(n_fock):
    """Eq. (Lambda_p_definition) at p = 1 for H = 0 with unit Gamma_1^(L)."""
    return 4.0 * (float(n_fock) - 1.0)


def proxy_rows(p, lam_func):
    rows = []
    optima = {}
    for kappa_label, kappa in KAPPAS:
        for eps in EPSILONS:
            best = None
            for idx, n_coeff in enumerate(N_GRID):
                margin = eps - EPS_TR[idx]
                if margin <= 0.0:
                    continue
                n_fock = int(np.ceil(kappa * n_coeff))
                n_t = int(np.ceil((lam_func(n_fock) / margin) ** (1.0 / p)))
                resource = int(n_coeff) * n_t
                rows.append({
                    "kappa": kappa_label, "epsilon": eps, "n_coeff": int(n_coeff),
                    "n_fock": n_fock, "eps_tr": EPS_TR[idx], "n_t": n_t,
                    "resource": resource,
                })
                if best is None or resource < best["resource"]:
                    best = rows[-1]
            optima[(kappa_label, eps)] = best
    return rows, optima


def validation_gate(p2_optima):
    archive = {}
    with ARCHIVE_CSV.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["kappa"], float(row["epsilon"]))
            r_val = int(row["resource"])
            if key not in archive or r_val < archive[key]["resource"]:
                archive[key] = {
                    "n_coeff": int(row["n_coeff"]), "n_fock": int(row["n_fock"]),
                    "n_t": int(row["n_t"]), "resource": r_val,
                }
    failures = []
    for key, ref in sorted(archive.items()):
        got = p2_optima[key]
        for field in ("n_coeff", "n_fock", "n_t", "resource"):
            if got[field] != ref[field]:
                failures.append(f"{key}: {field} {got[field]} != archived {ref[field]}")
    if failures:
        raise RuntimeError("VALIDATION GATE FAILED:\n" + "\n".join(failures))
    print(f"validation gate PASSED: {len(archive)} archived p=2 minimizers "
          "reproduced exactly", flush=True)


def make_figure(p2_rows, p2_optima, p1_rows, p1_optima, fit_params):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    ax.semilogy(N_GRID, EPS_TR, color="tab:blue", lw=1.4, label="numerical tail")
    fit_a, fit_c = fit_params
    ax.semilogy(
        N_GRID,
        np.exp(fit_a + fit_c * N_GRID**0.25),
        "--",
        color="gray",
        lw=1.0,
        label=rf"$\propto e^{{{fit_c:.2f}\,N^{{1/4}}}}$ fit",
    )
    power_law = EPS_TR[N_GRID == 16][0] * (N_GRID / 16.0) ** (-2.0)
    ax.semilogy(N_GRID, power_law, ":", color="gray", lw=1.0, label=r"$N^{-2}$")
    ax.set_xlabel(r"coefficient cutoff $N$", fontsize=9)
    ax.set_ylabel(r"$\epsilon_{\mathrm{tr}}(N)$", fontsize=9)
    ax.set_title("(a) coefficient-truncation tail", fontsize=10)
    ax.set_ylim(1e-3, 1.0)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(EPSILONS)))
    resources = [row["resource"] for row in p2_rows] + [row["resource"] for row in p1_rows]
    shared_ylim = (0.6 * min(resources), 1.8 * max(resources))
    for panel, (kappa_label, _kappa) in enumerate(KAPPAS, start=1):
        ax = axes[panel]
        for color, eps in zip(colors, EPSILONS):
            for rows, optima, style, marker in (
                (p2_rows, p2_optima, "-", "o"),
                (p1_rows, p1_optima, "--", "s"),
            ):
                curve = [
                    (row["n_coeff"], row["resource"])
                    for row in rows
                    if row["kappa"] == kappa_label and row["epsilon"] == eps
                ]
                n_vals, r_vals = zip(*curve)
                ax.semilogy(n_vals, r_vals, style, color=color, lw=1.3)
                best = optima[(kappa_label, eps)]
                ax.semilogy(
                    [best["n_coeff"]], [best["resource"]], marker=marker,
                    ms=5, mfc=color, mec="black", mew=0.7, ls="none",
                )
        ax.set_xlabel(r"coefficient cutoff $N$", fontsize=9)
        ax.set_ylabel(r"$\mathcal{R}(N;\epsilon)=N\,n_t(N;\epsilon)$", fontsize=9)
        ax.set_ylim(*shared_ylim)
        kappa_tex = "2" if kappa_label == "2" else "4/3"
        ax.set_title(
            rf"({chr(ord('a') + panel)}) joint proxy, $N_{{\rm Fock}}=\lceil {kappa_tex}\,N\rceil$",
            fontsize=10,
        )
        eps_handles = [
            mpatches.Patch(color=color, label=rf"$\epsilon={eps:g}$")
            for color, eps in zip(colors, EPSILONS)
        ]
        eps_legend = ax.legend(handles=eps_handles, loc="upper left", fontsize=7.5)
        ax.add_artist(eps_legend)
        order_handles = [
            mlines.Line2D([], [], color="dimgray", ls="-", lw=1.3,
                          marker="o", ms=4, mfc="dimgray", mec="black",
                          mew=0.6, label=r"$p=2$"),
            mlines.Line2D([], [], color="dimgray", ls="--", lw=1.3,
                          marker="s", ms=4, mfc="dimgray", mec="black",
                          mew=0.6, label=r"$p=1$"),
        ]
        ax.legend(handles=order_handles, loc="lower right", fontsize=7.5)
        ax.tick_params(labelsize=8)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)


def main():
    p2_rows, p2_optima = proxy_rows(2, lambda nf: lambda_normalized(nf, 2))
    validation_gate(p2_optima)
    p1_rows, p1_optima = proxy_rows(1, lambda_p1_pure_l)

    fit_range = (N_GRID >= 40) & (N_GRID <= 200)
    design = np.vstack(
        [np.ones(int(np.sum(fit_range))), N_GRID[fit_range] ** 0.25]
    ).T
    fit_coef, *_ = np.linalg.lstsq(design, np.log(EPS_TR[fit_range]), rcond=None)
    fit_params = (float(fit_coef[0]), float(fit_coef[1]))

    make_figure(p2_rows, p2_optima, p1_rows, p1_optima, fit_params)

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kappa", "epsilon", "n_coeff", "n_fock", "eps_tr", "n_t", "resource"],
        )
        writer.writeheader()
        for row in p1_rows:
            writer.writerow(row)

    p1_summary = {
        f"kappa={k} eps={e:g}": {
            "N_star": opt["n_coeff"], "n_fock": opt["n_fock"],
            "n_t": opt["n_t"], "resource": opt["resource"],
        }
        for (k, e), opt in sorted(p1_optima.items())
    }
    META.write_text(
        json.dumps(
            {
                "output": str(OUTPUT_CSV.relative_to(ROOT)),
                "figure": str(FIGURE_PATH.relative_to(ROOT)),
                "methodology": (
                    "p=1 counterpart of the archived p=2 proxy with normalized "
                    "prefactors: Lambda_1(NF) = 4(NF-1) for H=0 with unit "
                    "Gamma_1^(L), n_t = ceil(Lambda_1/(eps - eps_tr(N))), "
                    "eps_tr(N) read from the archived epsilon_tail.csv. "
                    "Validation gate: the recomputed p=2 minimizers reproduce "
                    "the archived joint_tradeoff.csv minimal-resource rows "
                    "exactly before anything is written. The figure draws p=2 "
                    "solid with circular N_star markers and p=1 dashed with "
                    "square markers, panel (a) unchanged."
                ),
                "stretched_exponential_fit": {
                    "intercept": fit_params[0], "coefficient": fit_params[1],
                },
                "p1_minimizers": p1_summary,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"fit coefficient (should be about -1.57): {fit_params[1]:.4f}")
    for key, val in p1_summary.items():
        print(f"  {key}: N_star={val['N_star']} n_t={val['n_t']} R={val['resource']}")
    print(f"wrote {OUTPUT_CSV.relative_to(ROOT)}, {META.relative_to(ROOT)}, "
          f"{FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
