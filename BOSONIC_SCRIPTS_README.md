# Bosonic-Only PDE Scripts (New)

These are the new scripts. Historical files in `OLD_FILES_2ND/` are unchanged.

## Files
- `heat1d_bosonic_state_prep.py`
  - Modular state-prep interface.
  - Implemented now: `ClassicalInjectionStatePreparation`.
  - Placeholder: `GateBasedStatePreparation` (for later).
- `heat1d_bosonic_pde_solver.py`
  - Bosonic-qiskit PDE solver using injected CV state.
- `heat1d_bosonic_sensitivity.py`
  - Deterministic sensitivity/optimization using the new solver.
- `heat1d_bosonic_trajectory.py`
  - CHO-style trajectory validation (time sweep vs classical reference).

## Run
```bash
python heat1d_bosonic_pde_solver.py --state-prep classical-injection --calibrate-phase
```

```bash
python heat1d_bosonic_sensitivity.py \
  --state-prep classical-injection \
  --n-r-target 5 --n-r-prime 5 --n-beta 7
```

```bash
python heat1d_bosonic_trajectory.py \
  --state-prep classical-injection \
  --total-time 1.0 --n-steps 40
```
