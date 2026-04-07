# Numerical Experiment Summary

All results in this summary use `injection`, namely ideal direct loading of the CV ancilla state in the simulator.
Accordingly, these numbers should be interpreted as oracle-baseline end-to-end PDE fidelities, not as practical CV state-preparation benchmarks.

## Best Injection-Baseline Fidelity By Boundary Condition

- Periodic: exact fidelity = 0.999725, truncated fidelity = 1.000000, postselection probability = 4.842142e-02, best parameters = (r_target=8.1, r_prime=4.1, beta=0.3, n_coeff=48, n_trotter=100).
- Neumann: exact fidelity = 0.999716, truncated fidelity = 0.999984, postselection probability = 5.616264e-02, best parameters = (r_target=7.9, r_prime=4, beta=0.3, n_coeff=48, n_trotter=100).
- Dirichlet: exact fidelity = 0.998961, truncated fidelity = 0.999992, postselection probability = 6.278915e-02, best parameters = (r_target=7.9, r_prime=4.1, beta=0.5, n_coeff=48, n_trotter=100).

## Main Observations

- Across all three boundary conditions, the best exact fidelities lie in the narrow range 0.998961 to 0.999725.
- Across the same best rows, truncated-model fidelity remains at least 0.999984, showing that the hybrid circuit matches the finite-Fock reference very closely.
- At the best rows, Dirichlet and periodic both prefer beta around 0.5 to 0.3, while Neumann peaks at beta = 0.3.

## Interpretation For The Working Paper

- Because the CV ancilla is loaded by ideal injection in all of these runs, the table and plots isolate kernel quality and hybrid-evolution quality. They do not yet quantify the additional loss that will appear once SNAP+D is used as the actual state-preparation routine.
- The overnight sweep indicates that the dominant source of remaining error is the kernel choice rather than the hybrid circuit execution, because exact fidelity is below unity while truncated-model fidelity is essentially unity.
- The current sensitivity figures should be treated as paper-quality placeholders: they show the correct trends and clearly identify the high-fidelity region, but the sampling along each axis is still sparse for a final publication figure.
- The current sensitivity figures are envelope plots rather than fixed-slice plots. For each boundary condition and each displayed value of $r_{\mathrm{target}}$, $r^\prime$, or $\beta$, the plotted fidelity is the best value obtained after optimizing over the other swept parameters in the dataset. Consequently, the beta panel does not hold $r_{\mathrm{target}}$ and $r^\prime$ fixed at one optimum; those parameters may vary from point to point along the curve.
- For a final paper version, the most useful next refinement is a denser local sweep around the high-fidelity region in r_target and r_prime, with a modest densification in beta only where the boundary condition appears sensitive.

