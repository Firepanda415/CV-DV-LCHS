"""Confront the postselection perturbation bound with the recorded benchmark data.

For each boundary condition of the M = 4 circuit-level benchmark at the shared
operating point, the finite perturbation theorem bounds the deviation between
the physical postselection probability and the finite-scale reference,

    |p_succ - p_ref| <= (2 ||e^{-AT} u_0||_2 eps_tot + eps_tot^2) / |alpha_{N,r}|^2,

where every right-hand quantity is recorded in finite_map_m4.csv
(||e^{-AT} u_0||_2 = sqrt(p_ref) |alpha_{N,r}|).  The script evaluates both
sides from the recorded columns, asserts that the bound holds for every
boundary, and writes the per-boundary values.

Output: results_revision_v2/postselection_bound_check.json (+ .meta.json).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "results_revision_v2" / "finite_map_m4.csv"
OUT = REPO / "results_revision_v2" / "postselection_bound_check.json"
META = REPO / "results_revision_v2" / "postselection_bound_check.meta.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    entries = {}
    with SRC.open() as handle:
        for row in csv.DictReader(handle):
            if not row["n_trotter_steps"] or int(row["n_trotter_steps"]) != 100:
                continue
            p_succ = float(row["p_succ"])
            p_ref = float(row["p_ref"])
            eps_tot = float(row["eps_tot"])
            alpha = abs(complex(float(row["alpha_target_re"]), float(row["alpha_target_im"])))
            decayed_norm = math.sqrt(p_ref) * alpha
            deviation = abs(p_succ - p_ref)
            bound = (2.0 * decayed_norm * eps_tot + eps_tot**2) / alpha**2
            assert deviation <= bound, f"perturbation bound violated for {row['boundary']}"
            entries[row["boundary"]] = {
                "p_succ": p_succ,
                "p_ref": p_ref,
                "eps_tot": eps_tot,
                "alpha_abs": alpha,
                "deviation": deviation,
                "bound": bound,
                "deviation_over_bound": deviation / bound,
            }
    assert set(entries) == {"dirichlet", "neumann", "periodic"}, "missing boundary rows"
    OUT.write_text(json.dumps(entries, indent=1) + "\n")

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    META.write_text(
        json.dumps(
            {
                "output": str(OUT.relative_to(REPO)),
                "source_sha256": sha256(Path(__file__)),
                "input_sha256": {str(SRC.relative_to(REPO)): sha256(SRC)},
                "git_head": git_head,
                "utc_timestamp": datetime.now(timezone.utc).isoformat(),
                "methodology": (
                    "|p_succ - p_ref| vs (2*sqrt(p_ref)*|alpha|*eps_tot + eps_tot^2)/|alpha|^2 "
                    "from the recorded circuit-level columns of finite_map_m4.csv at n_t = 100"
                ),
            },
            indent=1,
        )
        + "\n"
    )

    for boundary, vals in entries.items():
        print(
            f"{boundary:10s} |p-p_ref| = {vals['deviation']:.4e}  bound = {vals['bound']:.4e}"
            f"  ratio = {vals['deviation_over_bound']:.3f}"
        )
    print(f"wrote {OUT.name}, {META.name}")


if __name__ == "__main__":
    main()
