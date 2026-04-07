# Method Comparison Summary

Givens is evaluated at the refined oracle-optimal kernel point and reproduces that injection baseline to numerical precision.
The `SNAP+D` rows below are compared against the `injection` baseline at the exact same SNAP+D kernel point, which isolates CV state-preparation loss from kernel-selection effects.

| Boundary | SNAP+D kernel point $(r_{\mathrm{target}}, r^\prime, \beta, n_{\mathrm{coeff}})$ | Givens exact fidelity | SNAP+D exact fidelity | SNAP+D oracle fidelity | Exact-fidelity gap | Postselection ratio | SNAP depth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dirichlet | (8, 4.19, 0.675, 48) | 0.998961 | 0.538375 | 0.632538 | 0.456492 | 0.613624 | 8 |
| Neumann | (8, 4.35, 0.3, 24) | 0.999716 | 0.976050 | 0.956378 | 0.019873 | 0.886613 | 6 |
| Periodic | (8.3, 4.19, 0.675, 48) | 0.999725 | 0.584585 | 0.659686 | 0.409699 | 1.083345 | 6 |
