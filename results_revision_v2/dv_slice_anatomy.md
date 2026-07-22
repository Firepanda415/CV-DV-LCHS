# DV controlled-evolution slice anatomy (why the gate counts grow)

Question: why does the per-slice cost inflate 38 → 168 → 390 CX (M = 4, 8, 16)?
Hypothesis tested: "the uncontrolled Trotter block gets multiplied by
multi-controlling it." Verdict: **refined, not confirmed as stated** — the
control itself is cheap; the growth is in the uncontrolled ladder/carry
networks that are repeated once per control bit.

## Structure read off the builder output (m = log2 M system qubits, b LCU bits)

1. The upstream encodes the Laplacian with binary shift/ladder strings, not a
   Pauli decomposition: term count = 2m + 1 (5 / 7 / 9 / 11 at m = 2..5).
2. Exactly ONE controlled rotation per control bit: crz = b. With the retained
   trunc-multiplier 1.0 settings, b = 9 at every size (the archived
   dv_circuit_m{8,16}_summary.json files are regenerated at these settings).
   Control overhead is constant — phase-kickback style.
3. Everything else is uncontrolled machinery repeated per bit; verified
   component model, linear in b (checked at (m, b) = (2, 9), (3, 9), (3, 10),
   (4, 9), (4, 10); exact for these six components):
       crz = b,  rz = 4 b (m-1),  p = b + 2m,  h = 2m,
       ccx = b (m-2),  mcx_vchain(+dg) = 3 b (m-1)      [m >= 3]
   The residual plain-CX count has no single closed form across (m, b)
   (ladder-chain boundary terms; e.g. 15 at (3, 9) but 21 at (4, 9)); it is
   small and is absorbed in the decomposed quadratic fit below.
   The rz law re-verified at m = 5 with b = 9 (144).
4. Decomposed to the {1Q, CX, CRZ} reporting basis, per-slice CX fits
       c(m) = 46 m^2 - 100 m + 54   -> 38, 168, 390 (exact at m = 2, 3, 4).
   Falsification run at m = 5 (M = 32): measured 758 vs predicted 704 (+8%);
   the upstream switches its adder/carry compilation at m >= 5 (p-heavy,
   QFT-like increment), so the microscopic model is compilation-regime
   specific. The coarse law that survives: per-slice two-qubit cost grows
   ~ O(b log^2 M) — polylogarithmic in grid size, with b set by the quadrature
   precision (epsilon), not by M.

## Consequence for the CV-DV comparison wording

CV per-step cD/CX cost grows ~ O(M) at present (plain Pauli decomposition of
the tridiagonal L: cx/step = 4 / 20 / 68 at M = 4 / 8 / 16), while the DV slice
grows only polylogarithmically. This is the structural reason the block CX
ratio narrows (9.5x -> 8.4x -> 5.7x) and would eventually cross at larger M.
Claim wording must therefore not say the *gate-count* advantage grows with
size; what grows with size is the register-compactness advantage
(M_DV / N = 10.8 -> 19.8, and 9-10 ancilla qubits vs one mode). A ladder-string
compilation of the CV cD blocks is a natural future improvement and would
restore parity in the asymptotics.

Data: results_clean_dv_lchs_dirichlet/trotter100_summary.json (M=4),
results_revision_v2/dv_circuit_m{8,16}_summary.json, and the M=32 probe
(components printed in session log; not archived as a JSON).
