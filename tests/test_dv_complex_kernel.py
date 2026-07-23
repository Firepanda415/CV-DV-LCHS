import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results_revision_v2"

CASES = {
    "heat_m4_dirichlet",
    "heat_m8_dirichlet",
    "heat_m16_dirichlet",
    "heat_m32_dirichlet",
}


def _rows(path):
    with path.open() as handle:
        return {row["case"]: row for row in csv.DictReader(handle)}


def test_complex_kernel_inventory_matches_phase_free_slices():
    baseline = _rows(V2 / "dv_circuit_extension.csv")
    baseline.update(_rows(V2 / "dv_circuit_extension_m32.csv"))
    inventory = _rows(V2 / "dv_circuit_extension_complex_kernel.csv")
    assert set(inventory) == CASES
    for case, row in inventory.items():
        ref = baseline[case]
        for key in ("lcu_qubits", "slice_1q", "slice_cx", "slice_crz"):
            assert row[key] == ref[key]
        # The phase diagonal on 9 ancilla qubits synthesizes to 2^9 - 1
        # rotations and 2^9 - 2 CX, once per circuit, independent of M.
        assert int(row["prep_unprep_1q"]) - int(ref["prep_unprep_1q"]) == 511
        assert int(row["prep_unprep_cx"]) - int(ref["prep_unprep_cx"]) == 510
        assert int(row["prep_unprep_crz"]) == int(ref["prep_unprep_crz"]) == 0
        assert (int(row["prep_unprep_1q"]), int(row["prep_unprep_cx"])) == (772, 606)
        for key in ("1q", "cx", "crz"):
            assert int(row[f"full100_{key}"]) == int(row[f"prep_unprep_{key}"]) + 100 * int(
                row[f"slice_{key}"]
            )


def test_cx_basis_inventory_is_exact_crz_conversion():
    three_type = _rows(V2 / "dv_circuit_extension_complex_kernel.csv")
    two_type = _rows(V2 / "dv_circuit_extension_complex_kernel_cx.csv")
    assert set(two_type) == CASES
    for case, row in two_type.items():
        ref = three_type[case]
        # CRZ(theta) = RZ(theta/2) CX RZ(-theta/2) CX, so each CRZ converts
        # to exactly two CX and two single-qubit rotations.
        for scope in ("slice", "prep_unprep", "full100"):
            crz = int(ref[f"{scope}_crz"])
            assert int(row[f"{scope}_1q"]) == int(ref[f"{scope}_1q"]) + 2 * crz
            assert int(row[f"{scope}_cx"]) == int(ref[f"{scope}_cx"]) + 2 * crz
        for key in ("1q", "cx"):
            assert int(row[f"full100_{key}"]) == int(row[f"prep_unprep_{key}"]) + 100 * int(
                row[f"slice_{key}"]
            )
        assert int(row["full100_depth_est"]) >= int(ref["full100_depth_est"])


def test_dyadic_kernel_fidelity_separates_complex_and_projected():
    rows = _rows(V2 / "dv_kernel_fidelity.csv")
    assert set(rows) == CASES
    for row in rows.values():
        assert int(row["branches"]) == 512
        assert float(row["beta"]) == 0.6
        inf_complex = float(row["one_minus_F_complex_kernel"])
        inf_projected = float(row["one_minus_F_positive_real"])
        assert 0.0 < inf_complex < 1.0e-6
        assert 5.0e-3 < inf_projected < 3.0e-2
        assert inf_complex < 1.0e-4 * inf_projected
