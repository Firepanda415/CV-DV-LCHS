# Parameter Effects on CV Prep Fidelity and Final Fidelity

## 1) What is being optimized

With `use_fock_expansion=True`, the CV input is treated as an incoherent mixture:

\[
\rho_{\mathrm{in}} \approx \sum_n |C_n|^2 |n\rangle\langle n|.
\]

So the post-selected DV output is generally mixed:

\[
\rho_{\mathrm{post}} = \frac{1}{p_{\mathrm{post}}}\,\mathrm{Tr}_{\mathrm{CV}}\!\left[(|\phi\rangle\langle\phi|\otimes I)\,U\,(\rho_{\mathrm{in}}\otimes|b\rangle\langle b|)\,U^\dagger\right].
\]

Use these metrics:

- Final fidelity (mixed-state correct): \(F=\langle \hat u|\rho_{\mathrm{post}}|\hat u\rangle\), \(\hat u=u_{\mathrm{theory}}/\|u_{\mathrm{theory}}\|\).
- PDE-vector match (amplitude-sensitive): \(\|u_{\mathrm{theory}}-\sqrt{p_{\mathrm{post}}}\,\psi_{\mathrm{principal}}\|/\|u_{\mathrm{theory}}\|\).

## 2) How parameters enter theoretically

LCHS coefficient shape (from your quadrature implementation):

\/[
C_n \sim \int dk\; e^{-\gamma k^2}\,g_{\beta_k}(k)\,\frac{H_n(k/e^{r'})}{\sqrt{2^n n!}},
\quad \gamma=e^{-2r'}-e^{-2r},\; r' < r.
\\]

- `r_target = r`: sets post-selection squeeze and enters \(\gamma\).
- `r_prime = r'`: sets prep-basis squeeze and \(\gamma\); also controls Hermite argument scaling.
- `kernel_beta`: shapes kernel factor \(g_{\beta_k}(k)=\exp((1+ik)^{\beta_k})/(1-ik)\).
- `alpha_disp`: global scale of effective DV generator in evolution block.
- `energy_shift`: shifts identity term in generator (in this post-selected CV-DV map it changes overlap/success, not just a harmless phase).
- `beta` (in current run mode): mainly affects Gaussian-reference comparison (`|<psi_lchs|psi_gauss>|^2`), not the fock-mixture evolution itself.

## 3) Observed trends from your sweeps (coarse grid)

From `sensitivity_results/*.csv`:

- `kernel_beta`: increasing `0.4 -> 1.2` dropped final fidelity `~0.64 -> ~0.26`.
- `r_prime`: increasing `0.3 -> 1.5` dropped final fidelity `~0.55 -> ~0.21` and reduced post-probability strongly.
- `r_target`: weak/non-monotone; broad plateau in coarse scan.
- `alpha_disp`: increasing `0.6 -> 1.4` improved final fidelity `~0.34 -> ~0.53`.
- `energy_shift`: more negative improved fidelity (`-1.0` best in coarse scan).
- Coupled `(r_target,r_prime)`: best region is low `r_prime` with moderate `r_target`, respecting `r' < r`.

Your latest tuned run achieved:

- Final mixed-state fidelity: `0.9509`
- Purity: `0.9489` (still mixed)

Interpretation: direction match is high, but amplitude-level vector mismatch is still nontrivial.

## 4) What can still be improved (feasible in this project)

1. Optimize a multi-objective score, not fidelity alone:
   \[
   J = F - \lambda\,\frac{\|u_{\mathrm{theory}}-\sqrt{p_{\mathrm{post}}}\psi_{\mathrm{principal}}\|}{\|u_{\mathrm{theory}}\|} - \mu\,\max(0,p_{\min}-p_{\mathrm{post}}).
   \]
2. Do local joint scans around current optimum (`r_target`, `r_prime`, `alpha_disp`, `energy_shift`) instead of 1D-only sweeps.
3. Keep `r_prime` low and strictly below `r_target`; this is the most consistent high-fidelity region in your data.
4. Keep checking purity and post-selection probability with fidelity; mixed outputs can hide vector errors.
