"""Dense sensitivity-landscape data for the parameter figure (visualization sweep).

Evaluates the exact-truncated-map fixed-scale error over the dense grid in
results_revision/viz_dense/landscape_viz_grid.json via the same
evaluate_exact_map_row code path used for the pre-registered selection grid.
Points where the zeroth-moment overlap is numerically zero (fixed-scale error
undefined) are recorded with empty metric fields instead of aborting the sweep.

The raw maps behind every valid row (observed map columns, reference map,
alpha) are frozen to one companion npz per CSV, keyed by boundary and stacked
in grid_index order, so all reported metrics are recomputable without
re-running the sweep (raw_artifact column).

Descriptive only: parameter selection remains the pre-registered 81-point grid
(retuning_grid_registered.json, Gate B). Writes
results_revision/viz_dense/landscape_dense.csv.
"""

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from revision_eval import evaluate_exact_map_row

GRID = Path(
    sys.argv[1] if len(sys.argv) > 1
    else "results_revision/viz_dense/landscape_viz_grid.json"
)
OUTPUT = Path(
    sys.argv[2] if len(sys.argv) > 2
    else "results_revision/viz_dense/landscape_dense.csv"
)
META = OUTPUT.with_suffix(".meta.json")
RAW = OUTPUT.with_name(OUTPUT.stem + "_raw.npz")


def main():
    grid = json.loads(GRID.read_text())["boundaries"]
    rows = []
    raw_arrays = {}
    skipped = 0
    start = time.time()
    for boundary, points in grid.items():
        maps, alphas, indices = [], [], []
        reference_map = None
        for index, point in enumerate(points):
            try:
                row, raw = evaluate_exact_map_row(
                    point, boundary, 1.0, return_raw=True
                )
                row["raw_artifact"] = str(RAW)
                maps.append(raw["observed_vectors"])
                alphas.append(complex(raw["alpha_target"]))
                indices.append(index)
                reference_map = raw["reference_map"]
                raw_arrays.setdefault("input_vectors", raw["input_vectors"])
            except ValueError as error:
                row = {
                    "boundary": boundary,
                    **{key: point[key] for key in point},
                    "total_time": 1.0,
                    "evaluation": "exact_finite_map",
                    "rel_frobenius_error": "",
                    "invalid_reason": str(error),
                }
                skipped += 1
            row["grid_index"] = index
            row["is_best_fixed_alpha"] = False
            rows.append(row)
            if (index + 1) % 200 == 0:
                print(
                    f"[{boundary}] {index + 1}/{len(points)} "
                    f"({time.time() - start:.0f}s elapsed)",
                    flush=True,
                )
        raw_arrays[f"{boundary}_observed_vectors"] = np.asarray(maps)
        raw_arrays[f"{boundary}_alpha_target"] = np.asarray(alphas)
        raw_arrays[f"{boundary}_reference_map"] = reference_map
        raw_arrays[f"{boundary}_grid_index"] = np.asarray(indices, dtype=int)
    np.savez(RAW, **raw_arrays)
    for boundary in grid:
        valid = [
            row
            for row in rows
            if row["boundary"] == boundary and row["rel_frobenius_error"] != ""
        ]
        min(valid, key=lambda row: row["rel_frobenius_error"])[
            "is_best_fixed_alpha"
        ] = True

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    META.write_text(
        json.dumps(
            {
                "output": str(OUTPUT),
                "grid": str(GRID),
                "grid_sha256": hashlib.sha256(GRID.read_bytes()).hexdigest(),
                "raw_artifact": str(RAW),
                "raw_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
                "note": (
                    "post-hoc dense VISUALIZATION sweep; selection remains the "
                    "pre-registered grid (Gate B). Empty rel_frobenius_error "
                    "marks points with numerically zero zeroth-moment overlap. "
                    "Raw maps for every valid row are stacked per boundary in "
                    "the companion npz, aligned with the per-boundary "
                    "grid_index arrays."
                ),
                "rows": len(rows),
                "overlap_zero_points": skipped,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT} ({len(rows)} rows, {skipped} overlap-zero points)")


if __name__ == "__main__":
    main()
