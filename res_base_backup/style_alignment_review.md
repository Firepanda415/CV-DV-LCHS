# Style Alignment Review

Style baseline used for this review:
- [A Bayesian Approach for Characterizing and Mitigating Gate and Measurement Errors](https://www.osti.gov/pages/servlets/purl/1975101)
- [Unleashed from constrained optimization: quantum computing for quantum chemistry employing generator coordinate inspired method](https://www.nature.com/articles/s41534-024-00916-8)
- [Early Exploration of a Flexible Framework for Efficient Quantum Linear Solvers in Power Systems](https://ar5iv.org/html/2402.08136v3)

Target style for the working paper:
- direct, results-first, physics-journal prose
- fewer promotional adjectives
- shorter sentence units
- claims tied closely to the data or construction
- stable terminology instead of repeated rephrasing

The entries below cover only rewrite-worthy prose lines in `main.tex` and `experiments.tex`. All suggested replacements preserve the current technical claims, notation, and numerical values.

Each entry now includes:
- a rendered suggestion for quick reading
- a LaTeX block you can copy directly into the manuscript

## `main.tex`

### `main.tex:75`
Current issue: the abstract is technically strong, but it reads as over-compressed and uses inflated framing such as "advanced", "comprehensive", and "resource-efficient route".

Rendered suggestion:

> We introduce a hybrid oscillator-qubit formulation of linear combination of Hamiltonian simulation (LCHS) for linear ordinary and partial differential equations. The method replaces the discrete quadrature register in qubit-only LCHS with a continuous-variable ancillary mode, shifting the dominant approximation error from quadrature truncation to oscillator-state preparation and finite-Fock simulation. This removes the explicit $O(\log M)$ ancilla-qubit overhead of the discrete quadrature register. For a representative 100-step Dirichlet heat-equation benchmark, the hybrid implementation uses $400$ CNOTs in the Trotterized middle block, compared with nearly $4{,}800$ in a compressed discrete-variable realization. We derive error bounds for Fock-space truncation and product-formula simulation, and we benchmark the method on the one-dimensional heat equation with Dirichlet, Neumann, and periodic boundary conditions. Under ideal loading and Givens preparation, the exact-solution fidelity exceeds $0.9989$ in all three cases; optimized SNAP$+$D yields $0.996817$, $0.999678$, and $0.999531$. In the corresponding DV LCHS comparison, the optimal quadrature constructions require $M_{\mathrm{DV}}=320$, $192$, and $248$ terms, whereas the CV implementation uses $n_{\mathrm{coeff}}=48$. These results indicate that oscillator-state synthesis is the main practical bottleneck and that hybrid LCHS can reduce the representation cost of high-precision differential-equation simulation.

LaTeX copy-paste:

```latex
We introduce a hybrid oscillator-qubit formulation of linear combination of Hamiltonian simulation (LCHS) for linear ordinary and partial differential equations. The method replaces the discrete quadrature register in qubit-only LCHS with a continuous-variable ancillary mode, shifting the dominant approximation error from quadrature truncation to oscillator-state preparation and finite-Fock simulation. This removes the explicit $O(\log M)$ ancilla-qubit overhead of the discrete quadrature register. For a representative 100-step Dirichlet heat-equation benchmark, the hybrid implementation uses $400$ CNOTs in the Trotterized middle block, compared with nearly $4{,}800$ in a compressed discrete-variable realization. We derive error bounds for Fock-space truncation and product-formula simulation, and we benchmark the method on the one-dimensional heat equation with Dirichlet, Neumann, and periodic boundary conditions. Under ideal loading and Givens preparation, the exact-solution fidelity exceeds $0.9989$ in all three cases; optimized SNAP$+$D yields $0.996817$, $0.999678$, and $0.999531$. In the corresponding DV LCHS comparison, the optimal quadrature constructions require $M_{\mathrm{DV}}=320$, $192$, and $248$ terms, whereas the CV implementation uses $n_{\mathrm{coeff}}=48$. These results indicate that oscillator-state synthesis is the main practical bottleneck and that hybrid LCHS can reduce the representation cost of high-precision differential-equation simulation.
```

Why: your older papers usually start abstracts with the method and measured outcome, not with evaluative adjectives.

### `main.tex:92`
Current issue: the opening paragraph is generic and sounds like stock introduction language.

Rendered suggestion:

> Quantum simulation remains one of the main motivations for quantum computing, but scalable and precise implementations are still limited by both hardware and algorithm design. This challenge appears across condensed matter physics, quantum chemistry, high-energy physics, and materials science~\cite{Clinton_2024}.

LaTeX copy-paste:

```latex
Quantum simulation remains one of the main motivations for quantum computing, but scalable and precise implementations are still limited by both hardware and algorithm design. This challenge appears across condensed matter physics, quantum chemistry, high-energy physics, and materials science~\cite{Clinton_2024}.
```

Why: your prior papers are more effective when they open with the problem directly instead of with broad motivational scenery.

### `main.tex:94`
Current issue: the LCU/LCHS paragraph is accurate but over-explains standard material and delays the actual limitation.

Rendered suggestion:

> Hamiltonian simulation via linear combinations of unitaries (LCU) is a standard route to digital quantum simulation~\cite{wiebelcu}. In LCHS~\cite{LCHS1,LCHS2}, the target evolution is expressed as a weighted combination of efficiently implementable unitaries with controlled error. Existing realizations, however, are discrete-variable: continuous quantities must first be discretized, which introduces truncation error and can require a large ancilla register when high fidelity is needed.

LaTeX copy-paste:

```latex
Hamiltonian simulation via linear combinations of unitaries (LCU) is a standard route to digital quantum simulation~\cite{wiebelcu}. In LCHS~\cite{LCHS1,LCHS2}, the target evolution is expressed as a weighted combination of efficiently implementable unitaries with controlled error. Existing realizations, however, are discrete-variable: continuous quantities must first be discretized, which introduces truncation error and can require a large ancilla register when high fidelity is needed.
```

Why: this is closer to the compact setup style in your published papers, where background is introduced only to the extent needed for the paper's gap.

### `main.tex:96`
Current issue: the CV paragraph is solid, but the wording is still slightly promotional.

Rendered suggestion:

> Continuous-variable systems, such as harmonic-oscillator modes in photonic or cavity platforms, naturally represent bosonic degrees of freedom in large Hilbert spaces. Qubit platforms, by contrast, provide the control nonlinearity needed for conditional operations. A hybrid oscillator-qubit architecture can therefore separate continuous weighting from qubit-space system evolution more naturally than a qubit-only implementation.

LaTeX copy-paste:

```latex
Continuous-variable systems, such as harmonic-oscillator modes in photonic or cavity platforms, naturally represent bosonic degrees of freedom in large Hilbert spaces. Qubit platforms, by contrast, provide the control nonlinearity needed for conditional operations. A hybrid oscillator-qubit architecture can therefore separate continuous weighting from qubit-space system evolution more naturally than a qubit-only implementation.
```

Why: this keeps the comparison concrete and avoids phrases like "expanded computational expressiveness".

### `main.tex:98`
Current issue: the paragraph repeats the pitch from the abstract instead of stating the paper's contribution plainly.

Rendered suggestion:

> Here we formulate LCHS in a hybrid CV-DV architecture. The continuous-variable mode replaces the discrete quadrature register, while the qubit register still implements the Hamiltonian evolution. The point is not to remove approximation error altogether, but to move it away from quadrature discretization and toward oscillator-state preparation and finite-Fock simulation, where the tradeoffs are different and potentially more favorable.

LaTeX copy-paste:

```latex
Here we formulate LCHS in a hybrid CV-DV architecture. The continuous-variable mode replaces the discrete quadrature register, while the qubit register still implements the Hamiltonian evolution. The point is not to remove approximation error altogether, but to move it away from quadrature discretization and toward oscillator-state preparation and finite-Fock simulation, where the tradeoffs are different and potentially more favorable.
```

Why: your prior papers typically state the contribution and the tradeoff explicitly, without repeating high-level sales language.

### `main.tex:120-124`
Current issue: the setup sentence is fine mathematically, but the phrase "provides an optimal state preparation cost and a near-optimal scaling in matrix queries on all parameters" is clunky.

Rendered suggestion:

> LCHS gives optimal state-preparation cost and near-optimal matrix-query complexity for linear ordinary differential equations (ODEs) of the form

LaTeX copy-paste:

```latex
LCHS gives optimal state-preparation cost and near-optimal matrix-query complexity for linear ordinary differential equations (ODEs) of the form
```

Why: this is shorter and reads more like your technical exposition elsewhere.

### `main.tex:128-134`
Current issue: this block packs too many definitions and implications into one sentence chain.

Rendered suggestion:

> Write $A(t)=L(t)+iH(t)$ using the Cartesian decomposition
> \[
> L(t)=\frac{A(t)+A^\dagger(t)}{2}, \qquad H(t)=\frac{A(t)-A^\dagger(t)}{2i}.
> \]
> We assume $L(t)\succeq 0$. If this does not hold, one may shift the spectrum by substituting $u(t)=e^{ct}v(t)$ so that $L(t)+cI\succeq 0$, where $-c$ is the minimum over the smallest eigenvalues of $L(t)$. Under this decomposition, LCHS gives ... Here $g(k):=\frac{f(k)}{1-ik}$ for a kernel function $f(k)$, and the near-optimal choice is $f(k)=\frac{e^{2^\beta}}{2\pi e^{(1+ik)^\beta}}$ with $\beta\in(0,1)$~\cite{LCHS2}. The bracketed terms in Eqs.~\eqref{eq:LCHS-sol-homo} and~\eqref{eq:LCHS-sol-inhomo} are unitary, so LCHS approximates the integrals by a truncated discrete sum and then uses LCU circuitry to prepare the solution state~\cite{LCHS1,LCHS2}.

LaTeX copy-paste:

```latex
Write $A(t)=L(t)+iH(t)$ using the Cartesian decomposition
\[
L(t)=\frac{A(t)+A^\dagger(t)}{2}, \qquad H(t)=\frac{A(t)-A^\dagger(t)}{2i}.
\]
We assume $L(t)\succeq 0$. If this does not hold, one may shift the spectrum by substituting $u(t)=e^{ct}v(t)$ so that $L(t)+cI\succeq 0$, where $-c$ is the minimum over the smallest eigenvalues of $L(t)$. Under this decomposition, LCHS gives
...
Here $g(k):=\frac{f(k)}{1-ik}$ for a kernel function $f(k)$, and the near-optimal choice is $f(k)=\frac{e^{2^\beta}}{2\pi e^{(1+ik)^\beta}}$ with $\beta\in(0,1)$~\cite{LCHS2}. The bracketed terms in Eqs.~\eqref{eq:LCHS-sol-homo} and~\eqref{eq:LCHS-sol-inhomo} are unitary, so LCHS approximates the integrals by a truncated discrete sum and then uses LCU circuitry to prepare the solution state~\cite{LCHS1,LCHS2}.
```

Why: the older papers tend to separate definition, assumption, and consequence rather than stacking them in one long progression.

### `main.tex:139-148`
Current issue: the motivation is correct, but "brings a huge cost", "promising direction", and the last sentence all sound less authored and more generated.

Rendered suggestion:

> Combined with finite-difference discretization and linearization methods such as Carleman linearization~\cite{liu2021carleman}, LCHS can also be applied to nonlinear PDEs after reduction to a linear ODE system. The main circuit-level cost comes from the number of terms in the discretized LCHS integral, denoted by $M$. For the complementary solution,
> \[
> M \in \Ocal\left( T \max_t \|L(t)\|\left(\log \frac{1}{\epsilon}\right)^{1+1/\beta} \right),
> \]
> where $\epsilon$ is the total LCHS error~\cite{LCHS2}. The corresponding LCU circuit uses $\lceil\log_2(M)\rceil$ ancilla qubits and, more importantly, multi-controlled Hamiltonian-simulation blocks of the same width. For a time-independent ODE with $\epsilon\approx 10^{-5}$, $\beta=0.8$, $T=1000$, and $\|L\|=1$, $M$ can already reach the $10^6$ scale. More precise scalings are given in~\cite{pocrnic2025constant}. Motivated by continuous LCU~\cite{bell2025co}, we therefore consider a hybrid CV-DV version of LCHS. In this setting, $\Ocal(1)$ ancillary oscillators together with CV state preparation and CV measurement replace the $\lceil\log_2(M)\rceil$-qubit quadrature register, while the Hamiltonian evolution remains in qubit space. This does not remove all truncation error, but it removes the quadrature truncation itself and shifts the remaining approximation to CV state preparation and hybrid Hamiltonian simulation.

LaTeX copy-paste:

```latex
Combined with finite-difference discretization and linearization methods such as Carleman linearization~\cite{liu2021carleman}, LCHS can also be applied to nonlinear PDEs after reduction to a linear ODE system. The main circuit-level cost comes from the number of terms in the discretized LCHS integral, denoted by $M$. For the complementary solution,
\[
M \in \Ocal\left( T \max_t \|L(t)\|\left(\log \frac{1}{\epsilon}\right)^{1+1/\beta} \right),
\]
where $\epsilon$ is the total LCHS error~\cite{LCHS2}. The corresponding LCU circuit uses $\lceil\log_2(M)\rceil$ ancilla qubits and, more importantly, multi-controlled Hamiltonian-simulation blocks of the same width. For a time-independent ODE with $\epsilon\approx 10^{-5}$, $\beta=0.8$, $T=1000$, and $\|L\|=1$, $M$ can already reach the $10^6$ scale. More precise scalings are given in~\cite{pocrnic2025constant}. Motivated by continuous LCU~\cite{bell2025co}, we therefore consider a hybrid CV-DV version of LCHS. In this setting, $\Ocal(1)$ ancillary oscillators together with CV state preparation and CV measurement replace the $\lceil\log_2(M)\rceil$-qubit quadrature register, while the Hamiltonian evolution remains in qubit space. This does not remove all truncation error, but it removes the quadrature truncation itself and shifts the remaining approximation to CV state preparation and hybrid Hamiltonian simulation.
```

Why: this version keeps the numerical motivation but removes the rhetorical overreach and weak phrasing.

### `main.tex:178-197`
Current issue: this methodology block is conceptually clear, but the exposition is more verbose than it needs to be.

Rendered suggestion:

> We first consider the time-independent homogeneous case $A(t)=A$ and $b(t)=0$. Equations~\eqref{eq:exp-sol} and~\eqref{eq:LCHS-sol-homo} then reduce to
> \[
> u(t)=\left[\int_{\mathbb{R}} g(k)e^{-it(kL+H)}\,dk\right]u_0.
> \]
> In continuous LCU, a single ancillary oscillator couples to the qubit register and implements the joint evolution
> \[
> e^{-it(L\otimes \hat{k}+H\otimes I)},
> \]
> where $\hat{k}$ denotes one of the canonical quadrature operators and $H$ acts trivially on the oscillator. To realize the required kernel $g(k)$, we encode it through an initial oscillator state and a postselected oscillator measurement. Let $\ket{\Psi}=\ket{\psi}_{\rm osc}\ket{u_0}_q$ and $\bra{\Phi}=\bra{\phi}_{\rm osc}\otimes \mathbf{I}_q$. Choosing $k=\hat{x}$ and following Ref.~\cite{bell2025co}, we obtain
> \[
> \bra{\Phi}e^{-it(\hat{x}\otimes L+\mathbf{I}_{\rm osc}\otimes H)}\ket{\Psi}
> =
> \left[\int_{\mathbb{R}}\phi^*(x)\psi(x)e^{-it(xL+H)}\,dx\right]\ket{u_0}_q.
> \]
> Comparing with Eq.~\eqref{eq:sol-homo-lchs} shows that $g(x)=\phi^*(x)\psi(x)$, so the task is to choose the preparation and postselection states so that their overlap reproduces the LCHS kernel.

LaTeX copy-paste:

```latex
We first consider the time-independent homogeneous case $A(t)=A$ and $b(t)=0$. Equations~\eqref{eq:exp-sol} and~\eqref{eq:LCHS-sol-homo} then reduce to
\[
u(t)=\left[\int_{\mathbb{R}} g(k)e^{-it(kL+H)}\,dk\right]u_0.
\]
In continuous LCU, a single ancillary oscillator couples to the qubit register and implements the joint evolution
\[
e^{-it(L\otimes \hat{k}+H\otimes I)},
\]
where $\hat{k}$ denotes one of the canonical quadrature operators and $H$ acts trivially on the oscillator. To realize the required kernel $g(k)$, we encode it through an initial oscillator state and a postselected oscillator measurement. Let $\ket{\Psi}=\ket{\psi}_{\rm osc}\ket{u_0}_q$ and $\bra{\Phi}=\bra{\phi}_{\rm osc}\otimes \mathbf{I}_q$. Choosing $k=\hat{x}$ and following Ref.~\cite{bell2025co}, we obtain
\[
\bra{\Phi}e^{-it(\hat{x}\otimes L+\mathbf{I}_{\rm osc}\otimes H)}\ket{\Psi}
=
\left[\int_{\mathbb{R}}\phi^*(x)\psi(x)e^{-it(xL+H)}\,dx\right]\ket{u_0}_q.
\]
Comparing with Eq.~\eqref{eq:sol-homo-lchs} shows that $g(x)=\phi^*(x)\psi(x)$, so the task is to choose the preparation and postselection states so that their overlap reproduces the LCHS kernel.
```

Why: the revised version keeps the same derivation but reads more like a technical setup section than a guided lecture.

### `main.tex:199-214`
Current issue: the squeezed-state setup and proposition lead-in are serviceable, but they can be tightened and the grammar around the proposition can be improved.

Rendered suggestion:

> We take the postselection state $\ket{\phi}$ to be a squeezed vacuum with position-space wavefunction
> \[
> \phi_r(x):=S(r,0)\phi_0(x)=\left(\frac{2}{e^{2r}\pi}\right)^{1/4}e^{-x^2/e^{2r}}
> =\left(\frac{2}{\sigma^2\pi}\right)^{1/4}e^{-x^2/\sigma^2},
> \]
> where $\sigma=e^r$. The corresponding initial state is then
> \[
> \psi_r(x):=\left(\frac{\sigma^2\pi}{2}\right)^{1/4}g(x)e^{x^2/\sigma^2}
> =\mathcal{N}g(x)e^{x^2/\sigma^2}.
> \]
> The limits $r=0$ and $r\to\infty$ are discussed in Section~\ref{ss:state-prep}. This construction leads to the following CV-DV form of LCHS:
>
> Let $A\in\mathbb{C}^{n\times n}$ satisfy $A=L+iH$ with
> \[
> L=\frac{A+A^\dagger}{2}\succeq 0,\qquad H=\frac{A-A^\dagger}{2i}.
> \]
> Then $\ket{u(t)}=e^{-At}\ket{u(0)}$ can be represented using a CV auxiliary mode. The joint evolution acts on the ancillary oscillator and qubit register as
> \[
> \ket{u(t)}
> =
> (\bra{\phi}_{\rm osc}\otimes\mathbf{I}_q)e^{-it(\hat{x}\otimes L+\mathbf{I}_{\rm osc}\otimes H)}(\ket{\psi}_{\rm osc}\otimes\ket{u_0}_q).
> \]

LaTeX copy-paste:

```latex
We take the postselection state $\ket{\phi}$ to be a squeezed vacuum with position-space wavefunction
\[
\phi_r(x):=S(r,0)\phi_0(x)=\left(\frac{2}{e^{2r}\pi}\right)^{1/4}e^{-x^2/e^{2r}}
=\left(\frac{2}{\sigma^2\pi}\right)^{1/4}e^{-x^2/\sigma^2},
\]
where $\sigma=e^r$. The corresponding initial state is then
\[
\psi_r(x):=\left(\frac{\sigma^2\pi}{2}\right)^{1/4}g(x)e^{x^2/\sigma^2}
=\mathcal{N}g(x)e^{x^2/\sigma^2}.
\]
The limits $r=0$ and $r\to\infty$ are discussed in Section~\ref{ss:state-prep}. This construction leads to the following CV-DV form of LCHS:

Let $A\in\mathbb{C}^{n\times n}$ satisfy $A=L+iH$ with
\[
L=\frac{A+A^\dagger}{2}\succeq 0,\qquad H=\frac{A-A^\dagger}{2i}.
\]
Then $\ket{u(t)}=e^{-At}\ket{u(0)}$ can be represented using a CV auxiliary mode. The joint evolution acts on the ancillary oscillator and qubit register as
\[
\ket{u(t)}
=
(\bra{\phi}_{\rm osc}\otimes\mathbf{I}_q)e^{-it(\hat{x}\otimes L+\mathbf{I}_{\rm osc}\otimes H)}(\ket{\psi}_{\rm osc}\otimes\ket{u_0}_q).
\]
```

Why: your published style tends to make proposition statements read like mathematics first, prose second.

### `main.tex:1258`
Current issue: the first conclusion paragraph repeats too much of the introduction and uses "central idea" language that reads slightly generic.

Rendered suggestion:

> We introduced a hybrid oscillator-qubit form of LCHS for linear differential equations and used the heat equation as the main benchmark. The key step is to replace the discrete quadrature register of standard DV LCHS with a continuous-variable ancillary mode, so that the LCHS integral is realized through a qumode sandwich acting on a joint oscillator-qubit evolution. Within this formulation, we derived the corresponding state-preparation conditions, obtained truncation and product-formula error bounds, and gave a gate-level compilation in terms of Gaussian CV operations, controlled displacements, and standard qubit Pauli compilation.

LaTeX copy-paste:

```latex
We introduced a hybrid oscillator-qubit form of LCHS for linear differential equations and used the heat equation as the main benchmark. The key step is to replace the discrete quadrature register of standard DV LCHS with a continuous-variable ancillary mode, so that the LCHS integral is realized through a qumode sandwich acting on a joint oscillator-qubit evolution. Within this formulation, we derived the corresponding state-preparation conditions, obtained truncation and product-formula error bounds, and gave a gate-level compilation in terms of Gaussian CV operations, controlled displacements, and standard qubit Pauli compilation.
```

Why: the conclusion should restate the paper cleanly rather than re-pitch it.

### `main.tex:1260-1262`
Current issue: the result summary is good, but it is still more expansive than your usual results-first style.

Rendered suggestion:

> For the one-dimensional heat equation with Dirichlet, Neumann, and periodic boundary conditions, the ideal-loading CV-DV construction reaches exact-solution fidelities above $0.9989$ in all three cases. The numerical trends match the theoretical analysis: increasing the retained oscillator truncation improves accuracy, and the hybrid product formula shows the expected first-order behavior. At the circuit level, the dominant practical error comes from CV state preparation rather than from the subsequent CV-DV evolution. This bottleneck can nevertheless be reduced substantially in practice. With optimized SNAP$+$D synthesis, the end-to-end PDE fidelity reaches $0.996817$ for Dirichlet, $0.999678$ for Neumann, and $0.999531$ for periodic boundary conditions, while the corresponding oscillator-state fidelities are $0.994380$, $0.995048$, and $0.996856$. The analytic Givens construction matches the ideal-loading baseline to numerical precision and provides a hardware-oriented reference in JC-pulse and qubit-rotation counts.

LaTeX copy-paste:

```latex
For the one-dimensional heat equation with Dirichlet, Neumann, and periodic boundary conditions, the ideal-loading CV-DV construction reaches exact-solution fidelities above $0.9989$ in all three cases. The numerical trends match the theoretical analysis: increasing the retained oscillator truncation improves accuracy, and the hybrid product formula shows the expected first-order behavior. At the circuit level, the dominant practical error comes from CV state preparation rather than from the subsequent CV-DV evolution. This bottleneck can nevertheless be reduced substantially in practice. With optimized SNAP$+$D synthesis, the end-to-end PDE fidelity reaches $0.996817$ for Dirichlet, $0.999678$ for Neumann, and $0.999531$ for periodic boundary conditions, while the corresponding oscillator-state fidelities are $0.994380$, $0.995048$, and $0.996856$. The analytic Givens construction matches the ideal-loading baseline to numerical precision and provides a hardware-oriented reference in JC-pulse and qubit-rotation counts.
```

Why: your older papers usually consolidate the evidence and conclusion in one compact results paragraph.

### `main.tex:1264-1266`
Current issue: this comparison section is informative, but the phrase "algorithmic ceiling" is repeated and the interpretation is slightly overstated.

Rendered suggestion:

> The DV comparison clarifies both the attainable fidelity and the representation cost. In the classical DV quadrature construction with $\epsilon=0.1$ and $\eta=1$, the best fidelities in the scanned range occur at $\beta=0.60$ for Dirichlet, $\beta=0.90$ for Neumann, and $\beta=0.80$ for periodic, giving $F_{\mathrm{DV}}=0.997666$, $0.997740$, and $0.999722$. The CV-DV Givens baseline remains higher by $0.001295$ for Dirichlet and $0.001976$ for Neumann, while the periodic gap is only $0.000003$. At the same time, the corresponding DV quadrature constructions require $M_{\mathrm{DV}}=320$, $192$, and $248$ terms, whereas the present CV study uses $n_{\mathrm{coeff}}=48$. On the circuit side, MPS-compressed DV ancilla preparation can be efficient, but for the representative Dirichlet benchmark considered here the repeated 100-slice DV circuit still remains deeper than the corresponding CV-DV realization.

LaTeX copy-paste:

```latex
The DV comparison clarifies both the attainable fidelity and the representation cost. In the classical DV quadrature construction with $\epsilon=0.1$ and $\eta=1$, the best fidelities in the scanned range occur at $\beta=0.60$ for Dirichlet, $\beta=0.90$ for Neumann, and $\beta=0.80$ for periodic, giving $F_{\mathrm{DV}}=0.997666$, $0.997740$, and $0.999722$. The CV-DV Givens baseline remains higher by $0.001295$ for Dirichlet and $0.001976$ for Neumann, while the periodic gap is only $0.000003$. At the same time, the corresponding DV quadrature constructions require $M_{\mathrm{DV}}=320$, $192$, and $248$ terms, whereas the present CV study uses $n_{\mathrm{coeff}}=48$. On the circuit side, MPS-compressed DV ancilla preparation can be efficient, but for the representative Dirichlet benchmark considered here the repeated 100-slice DV circuit still remains deeper than the corresponding CV-DV realization.
```

Why: this keeps the comparison grounded in the actual measurements instead of in repeated conceptual labels.

### `main.tex:1268-1269`
Current issue: the closing claim is reasonable, but "viable route" and "robust practical resource advantage" sound slightly generic.

Rendered suggestion:

> Overall, these results support CV-DV LCHS as a practical alternative when the quadrature cost of DV LCHS is the dominant limitation. They also identify oscillator-state synthesis as the main remaining obstacle between the observed representation advantage and a broader circuit-level advantage.

LaTeX copy-paste:

```latex
Overall, these results support CV-DV LCHS as a practical alternative when the quadrature cost of DV LCHS is the dominant limitation. They also identify oscillator-state synthesis as the main remaining obstacle between the observed representation advantage and a broader circuit-level advantage.
```

Why: your prior conclusions usually end on the limiting factor and next technical step, not on a generalized impact phrase.

## `experiments.tex`

### `experiments.tex:1-3`
Current issue: the opening is clear, but it can be made more compact and less repetitive.

Rendered suggestion:

> We study the one-dimensional heat equation with Dirichlet, Neumann, and periodic boundary conditions on a two-qubit register. Unless stated otherwise, the target is the exact discretized solution $u(T)=e^{-AT}u(0)$ at $T=1$, with oscillator truncation $n_{\mathrm{fock}}=64$ and first-order Trotterization with $n_{\mathrm{trotter}}=100$ steps. The numerical study uses a four-dimensional semidiscrete system with $M=4$, $\alpha=1$, $h=1$, $T=1$, and initial state $u(0)=\ket{01}$. Since $\alpha/h^2=1$ in all three cases, the semidiscrete generators are

LaTeX copy-paste:

```latex
We study the one-dimensional heat equation with Dirichlet, Neumann, and periodic boundary conditions on a two-qubit register. Unless stated otherwise, the target is the exact discretized solution $u(T)=e^{-AT}u(0)$ at $T=1$, with oscillator truncation $n_{\mathrm{fock}}=64$ and first-order Trotterization with $n_{\mathrm{trotter}}=100$ steps. The numerical study uses a four-dimensional semidiscrete system with $M=4$, $\alpha=1$, $h=1$, $T=1$, and initial state $u(0)=\ket{01}$. Since $\alpha/h^2=1$ in all three cases, the semidiscrete generators are
```

Why: the revision removes duplicated setup language and gets to the benchmark definition faster.

### `experiments.tex:49-57`
Current issue: the metric definitions are technically fine, but the paragraph is denser than necessary.

Rendered suggestion:

> Here $u_{\mathrm{trunc}}$ denotes the exact finite-Fock reference for the same kernel parameters, and $\psi_{\mathrm{target}}$ denotes the ideal truncated oscillator state. We use $F_{\mathrm{snap}}$, $F_{\mathrm{givens}}$, and $F_{\mathrm{DV}}$ for the end-to-end PDE fidelities obtained with SNAP$+$D, Givens preparation, and the classical DV LCHS quadrature, respectively. The CV postselection probability is
> \[
> p_{\mathrm{post}}
> :=
> \left\|
> (\langle 0|\,S^\dagger(r_{\mathrm{target}})\otimes I)\,\Psi_{\mathrm{out}}
> \right\|^2 .
> \]
> The quantity $n_{\mathrm{coeff}}$ denotes the number of retained oscillator Fock amplitudes in the truncated CV oracle state.

LaTeX copy-paste:

```latex
Here $u_{\mathrm{trunc}}$ denotes the exact finite-Fock reference for the same kernel parameters, and $\psi_{\mathrm{target}}$ denotes the ideal truncated oscillator state. We use $F_{\mathrm{snap}}$, $F_{\mathrm{givens}}$, and $F_{\mathrm{DV}}$ for the end-to-end PDE fidelities obtained with SNAP$+$D, Givens preparation, and the classical DV LCHS quadrature, respectively. The CV postselection probability is
\[
p_{\mathrm{post}}
:=
\left\|
(\langle 0|\,S^\dagger(r_{\mathrm{target}})\otimes I)\,\Psi_{\mathrm{out}}
\right\|^2 .
\]
The quantity $n_{\mathrm{coeff}}$ denotes the number of retained oscillator Fock amplitudes in the truncated CV oracle state.
```

Why: your experimental sections are stronger when definitions are separated cleanly instead of chained in one long explanatory sentence.

### `experiments.tex:64-72`
Current issue: the parameter-selection subsection is strong, but the current version sounds more polished than authored.

Rendered suggestion:

> We select the kernel parameters $r_{\mathrm{target}}$, $r^\prime$, and $\beta$ through injection-based sweeps. Because the CV resource state is loaded ideally in this study, the sweep isolates kernel quality rather than gate-level preparation error. The full search domain is
> \[
> r_{\mathrm{target}}\in[1.92,8.4],\qquad
> r^\prime\in[0.975,4.5],\qquad
> \beta\in[0.3,0.9],
> \]
> with $n_{\mathrm{coeff}}\in\{24,48\}$ and $n_{\mathrm{trotter}}=100$. We then resample the highest-fidelity region more densely on $r_{\mathrm{target}}\in[7.8,8.4]$ and $r^\prime\in[4.0,4.5]$, which is the range shown in Fig.~\ref{fig:clean_sensitivity_exact}. For each displayed parameter value, the plotted point is the best exact fidelity obtained while the remaining sweep parameters vary over the tested grid. Across all three boundary conditions, the best oracle-baseline points lie in a narrow high-squeezing region. The optima are $(7.9,4.1,0.5,48)$ for Dirichlet, $(7.9,4.0,0.3,48)$ for Neumann, and $(8.1,4.1,0.3,48)$ for periodic, with exact fidelities above $0.9989$ in all three cases.

LaTeX copy-paste:

```latex
We select the kernel parameters $r_{\mathrm{target}}$, $r^\prime$, and $\beta$ through injection-based sweeps. Because the CV resource state is loaded ideally in this study, the sweep isolates kernel quality rather than gate-level preparation error. The full search domain is
\[
r_{\mathrm{target}}\in[1.92,8.4],\qquad
r^\prime\in[0.975,4.5],\qquad
\beta\in[0.3,0.9],
\]
with $n_{\mathrm{coeff}}\in\{24,48\}$ and $n_{\mathrm{trotter}}=100$. We then resample the highest-fidelity region more densely on $r_{\mathrm{target}}\in[7.8,8.4]$ and $r^\prime\in[4.0,4.5]$, which is the range shown in Fig.~\ref{fig:clean_sensitivity_exact}. For each displayed parameter value, the plotted point is the best exact fidelity obtained while the remaining sweep parameters vary over the tested grid. Across all three boundary conditions, the best oracle-baseline points lie in a narrow high-squeezing region. The optima are $(7.9,4.1,0.5,48)$ for Dirichlet, $(7.9,4.0,0.3,48)$ for Neumann, and $(8.1,4.1,0.3,48)$ for periodic, with exact fidelities above $0.9989$ in all three cases.
```

Why: this shifts the paragraph toward procedure and observed outcome, which is closer to your established style.

### `experiments.tex:98-100`
Current issue: the theorem-comparison paragraph is informative but overloaded.

Rendered suggestion:

> Theorem~\ref{thm: state-prep} is asymptotic and contains unspecified constants $\mathcal K$ and $\mathcal M$, so the experiments test qualitative agreement rather than calibrated numerical tightness. In the ideal-loading sweep, increasing the oscillator cutoff from $n_{\mathrm{coeff}}=24$ to $48$ improves the best exact fidelity in all three cases: from $0.9956816$ to $0.9989611$ for Dirichlet, from $0.9995599$ to $0.9997155$ for Neumann, and from $0.9992957$ to $0.9997249$ for periodic. This is consistent with the theorem's prediction that larger truncated oscillator resources improve the approximation. The same sweep also places the optimum in a large-squeezing regime, with both $r_{\mathrm{target}}$ and $r^\prime$ between roughly $4$ and $8$. For the hybrid evolution, Eq.~\eqref{eq:heat-trotter-error} predicts first-order scaling in the number of product-formula steps. Using the Dirichlet data and comparing with the truncated CV reference, the best observed relative errors are $3.0835\times 10^{-3}$ at $n_{\mathrm{trotter}}=8$, $1.5428\times 10^{-3}$ at $n_{\mathrm{trotter}}=16$, and $7.7171\times 10^{-4}$ at $n_{\mathrm{trotter}}=32$. These values track a normalized $C/n$ guide line closely, which gives a representative empirical check of the expected first-order behavior.

LaTeX copy-paste:

```latex
Theorem~\ref{thm: state-prep} is asymptotic and contains unspecified constants $\mathcal K$ and $\mathcal M$, so the experiments test qualitative agreement rather than calibrated numerical tightness. In the ideal-loading sweep, increasing the oscillator cutoff from $n_{\mathrm{coeff}}=24$ to $48$ improves the best exact fidelity in all three cases: from $0.9956816$ to $0.9989611$ for Dirichlet, from $0.9995599$ to $0.9997155$ for Neumann, and from $0.9992957$ to $0.9997249$ for periodic. This is consistent with the theorem's prediction that larger truncated oscillator resources improve the approximation. The same sweep also places the optimum in a large-squeezing regime, with both $r_{\mathrm{target}}$ and $r^\prime$ between roughly $4$ and $8$. For the hybrid evolution, Eq.~\eqref{eq:heat-trotter-error} predicts first-order scaling in the number of product-formula steps. Using the Dirichlet data and comparing with the truncated CV reference, the best observed relative errors are $3.0835\times 10^{-3}$ at $n_{\mathrm{trotter}}=8$, $1.5428\times 10^{-3}$ at $n_{\mathrm{trotter}}=16$, and $7.7171\times 10^{-4}$ at $n_{\mathrm{trotter}}=32$. These values track a normalized $C/n$ guide line closely, which gives a representative empirical check of the expected first-order behavior.
```

Why: the revision keeps the same evidence but makes the logic easier to follow from statement to data to conclusion.

### `experiments.tex:105-107`
Current issue: the table lead-in is good, but the interpretation can be stated more directly.

Rendered suggestion:

> Table~\ref{tab:clean_snap_comparison} compares Givens synthesis and SNAP$+$D preparation at the parameter values in Table~\ref{tab:clean_oracle_baseline}. In each case, the SNAP$+$D row uses the best depth identified at that kernel point. The optimized SNAP$+$D circuits remain close to the ideal baseline in all three boundary conditions. The Neumann and periodic cases are closest to the Givens reference, with differences of $3.78\times 10^{-5}$ and $1.94\times 10^{-4}$, respectively. Dirichlet remains slightly farther away, with $F_{\mathrm{snap}}=0.996817$ and a gap of $2.14\times 10^{-3}$.

LaTeX copy-paste:

```latex
Table~\ref{tab:clean_snap_comparison} compares Givens synthesis and SNAP$+$D preparation at the parameter values in Table~\ref{tab:clean_oracle_baseline}. In each case, the SNAP$+$D row uses the best depth identified at that kernel point. The optimized SNAP$+$D circuits remain close to the ideal baseline in all three boundary conditions. The Neumann and periodic cases are closest to the Givens reference, with differences of $3.78\times 10^{-5}$ and $1.94\times 10^{-4}$, respectively. Dirichlet remains slightly farther away, with $F_{\mathrm{snap}}=0.996817$ and a gap of $2.14\times 10^{-3}$.
```

Why: this reads more like a results paragraph and less like commentary on a table.

### `experiments.tex:126-132`
Current issue: this is one of the clearest parts scientifically, but it is too long and interpretive for one paragraph.

Rendered suggestion:

> The oracle-baseline optima occur at large values of $r_{\mathrm{target}}$ and $r^\prime$. In the CV-DV LCHS construction, the ancillary oscillator enters through the qumode sandwich $\langle \phi | e^{-it(\hat{x}\otimes L + I\otimes H)} | \psi \rangle$, so these large squeezing values indicate that the CV resource state must realize a sufficiently broad quadrature-space kernel over the relevant spectral window of $A$. The gain in exact PDE fidelity is accompanied by a lower postselection probability, while $F_{\mathrm{trunc}}$ remains essentially unity.
>
> The three fidelities play different roles. $F_{\mathrm{oracle}}$ measures how accurately SNAP$+$D reproduces the target oscillator seed, $F_{\mathrm{snap}}$ measures the final PDE solution fidelity, and $F_{\mathrm{trunc}}$ measures agreement with the exact finite-Fock evolution generated by the prepared seed. The observation $F_{\mathrm{trunc}}\approx 1$ together with smaller $F_{\mathrm{oracle}}$ shows that, once the oscillator seed is fixed, the hybrid evolution itself is accurate and the remaining discrepancy comes mainly from state preparation. For Dirichlet, Neumann, and periodic boundary conditions, the optimized SNAP$+$D circuits reach $F_{\mathrm{oracle}}=0.994380$, $0.995048$, and $0.996856$, with corresponding end-to-end fidelities $F_{\mathrm{snap}}=0.996817$, $0.999678$, and $0.999531$.
>
> Givens synthesis provides an analytic exact-synthesis baseline together with JC-pulse and qubit-rotation counts. In all three boundary conditions, Givens matches the injection fidelity to numerical precision and reports $47$ JC pulses and $47$ qubit rotations.
>
> For the hybrid evolution block, the gate structure is simpler than the full circuit depth may suggest. In the present circuits, the 100-step Trotterized CV-DV evolution uses qubit basis-change gates, CNOT ladders, unconditional displacements $D(\alpha)$, and qubit-controlled displacements $CD$. The two outer squeezing gates contribute a fixed Gaussian overhead, while SNAP$+$D contributes the non-Gaussian oracle-preparation cost. Here one $\mathrm{SNAP}_{48}+D$ layer means a SNAP gate with $48$ active number-state phases followed by one displacement. Direct state loading is used only for the ideal oracle baseline and is not counted as a synthesis primitive.

LaTeX copy-paste:

```latex
The oracle-baseline optima occur at large values of $r_{\mathrm{target}}$ and $r^\prime$. In the CV-DV LCHS construction, the ancillary oscillator enters through the qumode sandwich $\langle \phi | e^{-it(\hat{x}\otimes L + I\otimes H)} | \psi \rangle$, so these large squeezing values indicate that the CV resource state must realize a sufficiently broad quadrature-space kernel over the relevant spectral window of $A$. The gain in exact PDE fidelity is accompanied by a lower postselection probability, while $F_{\mathrm{trunc}}$ remains essentially unity.

The three fidelities play different roles. $F_{\mathrm{oracle}}$ measures how accurately SNAP$+$D reproduces the target oscillator seed, $F_{\mathrm{snap}}$ measures the final PDE solution fidelity, and $F_{\mathrm{trunc}}$ measures agreement with the exact finite-Fock evolution generated by the prepared seed. The observation $F_{\mathrm{trunc}}\approx 1$ together with smaller $F_{\mathrm{oracle}}$ shows that, once the oscillator seed is fixed, the hybrid evolution itself is accurate and the remaining discrepancy comes mainly from state preparation. For Dirichlet, Neumann, and periodic boundary conditions, the optimized SNAP$+$D circuits reach $F_{\mathrm{oracle}}=0.994380$, $0.995048$, and $0.996856$, with corresponding end-to-end fidelities $F_{\mathrm{snap}}=0.996817$, $0.999678$, and $0.999531$.

Givens synthesis provides an analytic exact-synthesis baseline together with JC-pulse and qubit-rotation counts. In all three boundary conditions, Givens matches the injection fidelity to numerical precision and reports $47$ JC pulses and $47$ qubit rotations.

For the hybrid evolution block, the gate structure is simpler than the full circuit depth may suggest. In the present circuits, the 100-step Trotterized CV-DV evolution uses qubit basis-change gates, CNOT ladders, unconditional displacements $D(\alpha)$, and qubit-controlled displacements $CD$. The two outer squeezing gates contribute a fixed Gaussian overhead, while SNAP$+$D contributes the non-Gaussian oracle-preparation cost. Here one $\mathrm{SNAP}_{48}+D$ layer means a SNAP gate with $48$ active number-state phases followed by one displacement. Direct state loading is used only for the ideal oracle baseline and is not counted as a synthesis primitive.
```

Why: your prior papers usually break interpretation, metric meaning, and resource accounting into separate compact units instead of one continuous paragraph.

### `experiments.tex:154-179`
Current issue: the DV-baseline setup is mathematically correct, but the prose around it is more verbose than your usual comparison sections.

Rendered suggestion:

> For the DV LCHS baseline, the first question is the size of the quadrature discretization. In the homogeneous heat-equation benchmark considered here, $H=0$ and $L=A^{(\mathrm{bc})}$, so the standard DV LCHS formulas of Refs.~\cite{LCHS1,LCHS2} give ...
>
> Here $h_1$ is the quadrature step size, $K$ is the truncation half-width of the $k$ interval, $Q$ is the Gauss--Legendre order on each subinterval, $M_{\mathrm{DV}}$ is the number of resulting LCU terms, and $m_c$ is the control-register width. In Table~\ref{tab:dv_resource_compare}, we fix $\eta=1$ and $\epsilon=0.1$. For each boundary condition, we then choose the value of $\beta$ from the scan $\beta\in[0.60,1)$ that maximizes the classical DV fidelity
> \[
> F_{\mathrm{DV}} = F(u_{\mathrm{DV}},u_{\mathrm{exact}}),
> \]
> where $u_{\mathrm{DV}}$ is the classical DV LCHS output for the same initial state.

LaTeX copy-paste:

```latex
For the DV LCHS baseline, the first question is the size of the quadrature discretization. In the homogeneous heat-equation benchmark considered here, $H=0$ and $L=A^{(\mathrm{bc})}$, so the standard DV LCHS formulas of Refs.~\cite{LCHS1,LCHS2} give
...

Here $h_1$ is the quadrature step size, $K$ is the truncation half-width of the $k$ interval, $Q$ is the Gauss--Legendre order on each subinterval, $M_{\mathrm{DV}}$ is the number of resulting LCU terms, and $m_c$ is the control-register width. In Table~\ref{tab:dv_resource_compare}, we fix $\eta=1$ and $\epsilon=0.1$. For each boundary condition, we then choose the value of $\beta$ from the scan $\beta\in[0.60,1)$ that maximizes the classical DV fidelity
\[
F_{\mathrm{DV}} = F(u_{\mathrm{DV}},u_{\mathrm{exact}}),
\]
where $u_{\mathrm{DV}}$ is the classical DV LCHS output for the same initial state.
```

Why: this version introduces the comparison with less rhetorical staging and clearer experimental control.

### `experiments.tex:197-209`
Current issue: the DV comparison is good, but the prose can be made more results-led and less discursive.

Rendered suggestion:

> The DV comparison separates representation cost from circuit cost. In representation size, the optimal DV quadrature uses $320$, $192$, and $248$ terms for Dirichlet, Neumann, and periodic boundary conditions, compared with $n_{\mathrm{coeff}}=48$ on the CV side. The corresponding factors are approximately $6.67$, $4.00$, and $5.17$. In fidelity, the best classical DV solutions remain slightly below the CV-DV Givens baseline by $0.001295$ for Dirichlet and $0.001976$ for Neumann, while the periodic difference is only $0.000003$.
>
> The circuit comparison is more nuanced. On the DV side, the ancilla amplitude vector can be compressed as a matrix product state (MPS), which reduces the cost of ancilla preparation relative to generic amplitude loading. For the representative Dirichlet instance summarized here, the practical circuit is a nine-control-qubit implementation built at $\beta=0.8$, with ancilla preparation and unpreparation applied once and only the controlled LCHS evolution block repeated. The ancilla preparation block contains $130$ aggregated one-qubit gates and $48$ CNOT gates. One repeated controlled-evolution slice contains $53$ aggregated one-qubit gates and $47$ CNOT-type controls. The full 100-step DV circuit therefore has depth $9227$, with $5561$ aggregated one-qubit gates and $4796$ CNOT-type controls. By comparison, the Dirichlet CV-DV implementation uses $1400$ single-qubit gates, $400$ CNOTs, $100$ unconditional displacements, and $300$ controlled displacements in the Trotterized middle block, together with a $30$-layer SNAP$+$D preparation routine. Thus, even with MPS compression in the ancilla stage, the repeated DV qubit circuit remains deeper for this benchmark.

LaTeX copy-paste:

```latex
The DV comparison separates representation cost from circuit cost. In representation size, the optimal DV quadrature uses $320$, $192$, and $248$ terms for Dirichlet, Neumann, and periodic boundary conditions, compared with $n_{\mathrm{coeff}}=48$ on the CV side. The corresponding factors are approximately $6.67$, $4.00$, and $5.17$. In fidelity, the best classical DV solutions remain slightly below the CV-DV Givens baseline by $0.001295$ for Dirichlet and $0.001976$ for Neumann, while the periodic difference is only $0.000003$.

The circuit comparison is more nuanced. On the DV side, the ancilla amplitude vector can be compressed as a matrix product state (MPS), which reduces the cost of ancilla preparation relative to generic amplitude loading. For the representative Dirichlet instance summarized here, the practical circuit is a nine-control-qubit implementation built at $\beta=0.8$, with ancilla preparation and unpreparation applied once and only the controlled LCHS evolution block repeated. The ancilla preparation block contains $130$ aggregated one-qubit gates and $48$ CNOT gates. One repeated controlled-evolution slice contains $53$ aggregated one-qubit gates and $47$ CNOT-type controls. The full 100-step DV circuit therefore has depth $9227$, with $5561$ aggregated one-qubit gates and $4796$ CNOT-type controls. By comparison, the Dirichlet CV-DV implementation uses $1400$ single-qubit gates, $400$ CNOTs, $100$ unconditional displacements, and $300$ controlled displacements in the Trotterized middle block, together with a $30$-layer SNAP$+$D preparation routine. Thus, even with MPS compression in the ancilla stage, the repeated DV qubit circuit remains deeper for this benchmark.
```

Why: this pulls the actual comparison to the front and makes the interpretation read closer to your published result sections.

## Global notes

These patterns recur across the manuscript and are worth applying consistently beyond the flagged lines:

1. Replace evaluative adjectives with concrete evidence. "Advanced", "comprehensive", "powerful", and "promising" are usually unnecessary once the construction and numbers are on the page.
2. Prefer one claim per sentence in the introduction and conclusion. The manuscript is strongest when it moves in short problem -> method -> implication steps.
3. In the experiments section, let the metric or comparison lead the paragraph. Several paragraphs become more natural when the first sentence states what is being compared, and the next sentence states what the data show.
4. Use "indicate", "show", "give", and "remain" more often than "support", "enable", or "identify" when the evidence is empirical rather than categorical.
