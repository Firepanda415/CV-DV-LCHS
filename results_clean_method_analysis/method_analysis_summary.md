# Injection vs SNAP+D Analysis

This summary reports two complementary baselines.
Givens is evaluated at the refined oracle-optimal kernel point and reproduces that injection baseline to numerical precision.
SNAP+D is compared against the injection baseline at the same SNAP+D kernel point, so the reported gap isolates state-preparation loss.

## Main Findings

- The strongest `SNAP+D` result is Neumann, where exact PDE fidelity is 0.976050 against a matched injection baseline of 0.995923. The corresponding exact-fidelity gap is 0.019873.
- The weakest `SNAP+D` result is Dirichlet, where exact PDE fidelity drops to 0.538375 from a matched injection baseline of 0.994867.
- Dirichlet: `SNAP+D` oracle fidelity = 0.632538, exact PDE fidelity = 0.538375, matched injection exact fidelity = 0.994867, postselection ratio = 0.614, best snap depth = 8.
- Neumann: `SNAP+D` oracle fidelity = 0.956378, exact PDE fidelity = 0.976050, matched injection exact fidelity = 0.995923, postselection ratio = 0.887, best snap depth = 6.
- Periodic: `SNAP+D` oracle fidelity = 0.659686, exact PDE fidelity = 0.584585, matched injection exact fidelity = 0.994284, postselection ratio = 1.083, best snap depth = 6.

## Interpretation

- For Neumann, `SNAP+D` reproduces the oracle state well enough that the end-to-end PDE fidelity remains close to the injection benchmark. This indicates that the Neumann kernel state is substantially easier to synthesize with the current alternating SNAP+displacement ansatz.
- For Dirichlet and periodic, the observed PDE state still matches the exact truncated map induced by the prepared SNAP state almost perfectly, but the prepared SNAP state itself is too far from the ideal oracle. This means the dominant loss is state-preparation quality, not hybrid evolution quality.
- The low postselection ratios for Dirichlet and periodic do not explain the full gap by themselves, because the matched injection baselines at the same kernel points are already near unit PDE fidelity. The main bottleneck is the inability of the current low-depth SNAP+D ansatz to capture the target CV resource state in those two cases.
- The current Givens runs reproduce the matched injection baseline to numerical precision in all three cases, while reporting a uniform analytic resource count of 47 JC pulses and 47 qubit rotations. This is consistent with the present clean implementation, where Givens angles are synthesized analytically and then applied through simulator-side injection.
