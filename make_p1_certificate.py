"""First-order certificate connecting Theorem 3 to the benchmark operating point.

The Section 4.7 proxy of the manuscript is evaluated at p = 2 with normalized
prefactors, while every numerical experiment uses the first-order formula.
This script produces the p = 1 counterpart with the exact commutator structure
of the Dirichlet M = 4 heat benchmark, so the certified bound of Theorem 3 can
be compared with the measured product-formula error at the operating point
(N, N_Fock, n_t) = (32, 64, 100), where the measured error is
eps_t = 8.90e-4 at T = 1.

Computed quantities:
  1. eps_tr(N) at the operating cutoff (from results_joint_tradeoff/epsilon_tail.csv).
  2. Normalized p = 1 proxy minimizers N_star(eps), same philosophy as Fig. 3
     (unit commutator sums): Lambda_1(NF) = 4 (NF - 1) for H = 0.
  3. Exact-prefactor certificate for the benchmark inner splitting
     L = 2I - R_0 - R_1 with R_0 = X_0 and R_1 = (X_1 X_0 + Y_1 Y_0)/2, coupled
     through x = a + a^dagger.  First-order commutator bound (Childs et al.,
     PRX Quantum 2021, i < j sum, single nonzero pair):
       eps_t <= (t^2 / (2 n_t)) ||x_NF||^2 ||[R_0, R_1]||.
     Both the norm bound 2 sqrt(NF - 1) and the exact ||x_NF|| are used.
  4. A fine eps scan documenting that N = 32 is not a proxy minimizer at
     either order (a consequence of the plateau structure of eps_tr(N)).

Validation gate (runs before anything is written): the p = 2 minimizers
recomputed from epsilon_tail.csv with the script's own Lambda formula must
reproduce the archived per-(kappa, eps) minimal-R rows of
results_joint_tradeoff/joint_tradeoff.csv exactly.

Output: results_joint_tradeoff/p1_certificate.json.
"""

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
TAIL_CSV = REPO / "results_joint_tradeoff" / "epsilon_tail.csv"
ARCHIVE_CSV = REPO / "results_joint_tradeoff" / "joint_tradeoff.csv"
OUTPUT = REPO / "results_joint_tradeoff" / "p1_certificate.json"

T_TOT = 1.0
EPS_MEASURED = 8.90e-4
N_T_OP = 100
NF_OP = 64
EPS_LIST = (1e-1, 5e-2, 3e-2, 2e-2, 1e-2, 5e-3)
KAPPAS = (2.0, 4.0 / 3.0)


def load_tail():
    n_vals, eps_vals = [], []
    with TAIL_CSV.open() as handle:
        for row in csv.DictReader(handle):
            n_vals.append(int(row["n_coeff"]))
            eps_vals.append(float(row["eps_tr"]))
    return np.array(n_vals), np.array(eps_vals)


N_GRID, EPS_TR = load_tail()


def eps_tr(n):
    return float(EPS_TR[N_GRID == n][0])


def lambda_p2_normalized(nf):
    """Normalized p = 2 factor exactly as in make_joint_tradeoff.py."""
    base = float(nf - 1)
    mixed = sum(2.0**a * base ** (a / 2.0) for a in range(1, 3))
    return mixed + 2.0**3 * base ** (3.0 / 2.0) + 1.0


def lambda_p1_normalized(nf):
    """Eq. (Lambda_p_definition) at p = 1, H = 0, unit Gamma_1^(L)."""
    return 4.0 * (nf - 1.0)


def n_star(eps, kappa, p, lam_func, scale=1.0):
    """Minimize R = N * ceil((scale * lam / margin)^(1/p)) over admissible N."""
    best = None
    for idx, n in enumerate(N_GRID):
        margin = eps - EPS_TR[idx]
        if margin <= 0.0:
            continue
        nf = int(np.ceil(kappa * n))
        nt = int(np.ceil((scale * lam_func(nf) / margin) ** (1.0 / p)))
        r = int(n) * nt
        if best is None or r < best[3]:
            best = (int(n), nf, nt, r)
    return best


def validation_gate():
    """Reproduce the archived p = 2 minimal-R rows exactly."""
    archive = {}
    with ARCHIVE_CSV.open() as handle:
        for row in csv.DictReader(handle):
            kappa_txt = row["kappa"]
            kappa_val = (float(kappa_txt.split("/")[0]) / float(kappa_txt.split("/")[1])
                         if "/" in kappa_txt else float(kappa_txt))
            key = (kappa_val, float(row["epsilon"]))
            r_val = int(row["resource"])
            if key not in archive or r_val < archive[key][3]:
                archive[key] = (int(row["n_coeff"]), int(row["n_fock"]),
                                int(row["n_t"]), r_val)
    failures = []
    for (kappa, eps), ref in sorted(archive.items()):
        got = n_star(eps, kappa, 2, lambda_p2_normalized)
        if got != ref:
            failures.append(f"kappa={kappa} eps={eps}: {got} != archived {ref}")
    if failures:
        raise RuntimeError("VALIDATION GATE FAILED:\n" + "\n".join(failures))
    print(f"validation gate PASSED: {len(archive)} archived p=2 minimizers "
          "reproduced exactly", flush=True)


def main():
    validation_gate()

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    r0 = np.kron(np.eye(2), pauli_x)
    r1 = 0.5 * (np.kron(pauli_x, pauli_x) + np.kron(pauli_y, pauli_y))
    norm_comm = float(np.linalg.norm(r0 @ r1 - r1 @ r0, 2))

    offd = np.sqrt(np.arange(1, NF_OP))
    x_mat = np.diag(offd, 1) + np.diag(offd, -1)
    x_exact = float(np.max(np.abs(np.linalg.eigvalsh(x_mat))))
    x_bound = 2.0 * np.sqrt(NF_OP - 1.0)

    certificate = {}
    for label, xn in (("norm_bound", x_bound), ("exact_x_norm", x_exact)):
        eps_at_op = T_TOT**2 / (2.0 * N_T_OP) * xn**2 * norm_comm
        nt_cert = int(np.ceil(T_TOT**2 / 2.0 * xn**2 * norm_comm / EPS_MEASURED))
        certificate[label] = {
            "x_value": xn,
            "certified_eps_t_at_nt_100": eps_at_op,
            "certified_nt_for_measured_eps": nt_cert,
            "certified_over_measured_at_nt_100": eps_at_op / EPS_MEASURED,
        }

    normalized_p1 = {}
    for kappa in KAPPAS:
        rows = []
        for eps in EPS_LIST:
            b1 = n_star(eps, kappa, 1, lambda_p1_normalized)
            b2 = n_star(eps, kappa, 2, lambda_p2_normalized)
            rows.append({"eps": eps, "p1": b1, "p2": b2})
        normalized_p1[f"kappa_{kappa:.4f}"] = rows

    # Fine scan documenting that N = 32 never minimizes the proxy, at either
    # order, over eps in [0.0235, 0.12]: the eps_tr(N) plateau at N = 30, 31
    # makes the minimizer sequence jump over 32.
    minimizer_sets = {}
    for p_order, lam in ((1, lambda_p1_normalized), (2, lambda_p2_normalized)):
        seen = set()
        for eps in np.arange(0.0235, 0.12, 1e-4):
            b = n_star(float(eps), 2.0, p_order, lam)
            if b is not None:
                seen.add(b[0])
        minimizer_sets[f"p{p_order}"] = sorted(seen)

    payload = {
        "operating_point": {
            "N": 32, "N_Fock": NF_OP, "n_t": N_T_OP, "T": T_TOT,
            "eps_tr_at_N32": eps_tr(32),
            "measured_eps_t": EPS_MEASURED,
        },
        "benchmark_commutator": {
            "norm_comm_R0_R1": norm_comm,
            "x_norm_bound_2sqrt63": x_bound,
            "x_norm_exact_NF64": x_exact,
        },
        "p1_exact_prefactor_certificate": certificate,
        "p1_vs_p2_normalized_minimizers": normalized_p1,
        "minimizer_sets_kappa2_eps_0p0235_to_0p12": minimizer_sets,
        "n32_is_minimizer_p1": 32 in minimizer_sets["p1"],
        "n32_is_minimizer_p2": 32 in minimizer_sets["p2"],
        "conventions": (
            "First-order bound in the Childs et al. i<j form, "
            "eps_t <= (t^2/(2 n_t)) ||x_NF||^2 ||[R_0,R_1]||, for the "
            "Dirichlet M=4 inner splitting {2I, R_0, R_1} whose only "
            "noncommuting pair is (R_0, R_1) with [R_0,R_1] = i Y_1 Z_0. "
            "The Theorem-3 ordered-tuple form is an additional factor of two "
            "looser. Normalized minimizers use unit commutator sums as in "
            "Fig. 3."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"eps_tr(32) = {eps_tr(32):.6e}")
    print(f"||[R_0,R_1]|| = {norm_comm:.12f}")
    print(f"||x_64|| exact = {x_exact:.6f}, bound = {x_bound:.6f}")
    for label, cert in certificate.items():
        print(f"[{label}] certified eps_t at n_t=100: "
              f"{cert['certified_eps_t_at_nt_100']:.4f} "
              f"({cert['certified_over_measured_at_nt_100']:.0f}x measured), "
              f"certified n_t for 8.90e-4: {cert['certified_nt_for_measured_eps']}")
    print(f"N=32 in p1 minimizer set: {payload['n32_is_minimizer_p1']}, "
          f"in p2 set: {payload['n32_is_minimizer_p2']} "
          f"(sets: {minimizer_sets})")
    print(f"wrote {OUTPUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
