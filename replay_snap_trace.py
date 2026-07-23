#!/usr/bin/env python3
"""Replay one start of a completed SNAP+D artifact and record its objective trace.

L-BFGS-B is deterministic given the starting point and the objective, so
re-running one start with the same guess reproduces the original evaluation
sequence, and capping the replay at the recorded evaluation count reproduces
the original truncation point. The replayed final best must match the recorded
objective bit for bit, which validates the reconstruction before any
threshold-crossing time is read off the trace.

Run from a working directory holding ``results_revision/`` (the same pattern
as revision_eval.py). Applies to artifacts produced without a warm start and
with a single-level depth schedule.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_core import padded_seed_state
from clean_oracles import simulate_snap_d_state
import revision_eval as re_mod


class _Stop(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-coeff", type=int, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    doc = json.load(open(f"results_revision/snap_N{args.n_coeff}_seed11.json"))
    assert doc["depth_schedule"] == [doc["depth"]]
    assert not doc.get("sidecar", {}).get("parameters", {}).get("warm_start_from")
    rec = doc["start_records"][args.start_index]
    assert rec["start_index"] == args.start_index
    nfev_cap = int(rec["nfev"])

    kernel = re_mod._selected_kernel(
        "dirichlet", n_coeff=args.n_coeff, n_fock=None
    )
    coeffs, _ = re_mod._coefficients_for_finite(
        "dirichlet", kernel, argparse.Namespace()
    )
    # run_snap pads the coefficient state and optimize_snap_d pads its input
    # again. The second normalization shifts the state by one ulp, so the
    # replay must apply both to reproduce the production objective bitwise.
    target = padded_seed_state(
        padded_seed_state(coeffs, kernel.n_fock), kernel.n_fock
    )

    n_fock = kernel.n_fock
    depth = int(doc["depth"])
    n_snap = min(kernel.n_coeff, n_fock)
    params_per_layer = n_snap + 2
    n_params = depth * params_per_layer

    # Reproduce the guess sequence of optimize_snap_d: all random restarts
    # drawn upfront from default_rng(seed), no warm start.
    rng = np.random.default_rng(doc["seed"])
    guesses = []
    for _ in range(int(doc["restarts"])):
        guess = np.zeros(n_params, dtype=float)
        for layer in range(depth):
            offset = layer * params_per_layer
            guess[offset : offset + n_snap] = rng.uniform(
                -np.pi, np.pi, size=n_snap
            )
            guess[offset + n_snap] = rng.uniform(-0.5, 0.5)
            guess[offset + n_snap + 1] = rng.uniform(-0.5, 0.5)
        guesses.append(guess)
    guess = guesses[args.start_index]

    state = {"nfev": 0, "best": float("inf")}
    trace = []
    t0 = perf_counter()

    def cost(x: np.ndarray) -> float:
        state["nfev"] += 1
        prepared = simulate_snap_d_state(
            x, n_fock=n_fock, depth=depth, n_snap=n_snap
        )
        value = 1.0 - abs(np.vdot(target, prepared)) ** 2
        if value < state["best"]:
            state["best"] = value
            trace.append((state["nfev"], perf_counter() - t0, value))
        if state["nfev"] >= nfev_cap:
            raise _Stop
        return value

    try:
        minimize(
            cost,
            guess,
            method="L-BFGS-B",
            options={
                "maxiter": int(doc["maxiter"]),
                "maxfun": int(doc["maxfun"]),
                "ftol": 1e-15,
                "gtol": 1e-10,
            },
            callback=lambda _x: None,
        )
    except _Stop:
        pass

    args.out.write_text(
        json.dumps(
            {
                "n_coeff": args.n_coeff,
                "start_index": args.start_index,
                "nfev_cap": nfev_cap,
                "recorded_objective": rec["objective"],
                "recorded_wall_seconds": rec["wall_seconds"],
                "replayed_best": state["best"],
                "replay_wall_seconds": perf_counter() - t0,
                "bitwise_match": state["best"] == rec["objective"],
                "n_improvements": len(trace),
                "trace": trace,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"replay N{args.n_coeff} start {args.start_index}: "
        f"best {state['best']:.6e} vs recorded {rec['objective']:.6e} "
        f"(bitwise match: {state['best'] == rec['objective']})"
    )


main()
