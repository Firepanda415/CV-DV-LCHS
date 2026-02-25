# Actionable Implementation Plan LaTeX File for the 1D Heat Equation Using Hybrid CV-DV LCHS

## Repository errata and applied corrections

The executable implementation and tests in this repository use the following
checked conventions. Treat these as authoritative over any conflicting wording
in this long-form report:

- Gate-level convention is validated by `tests/test_gate_math.py`.
- For HybridLane `Displacement(a, phi)` with `hbar=2`:
  - `Delta x = 2 a cos(phi)`
  - `Delta p = 2 a sin(phi)`
  - therefore `phi=0` is x-branch and `phi=-pi/2` is p-branch.
- `ConditionalDisplacement` decomposition is checked as
  `CP^\dagger D(i alpha) CP`.
- With \(\hat{x}=(a+a^\dagger)/2\), the sign identity is
  `CD(+i*lambda/2) = exp(+i lambda sigma_z x_hat)`.
  To realize `exp(-i lambda sigma_z x_hat)`, use `alpha=-i*lambda/2`
  (or equivalently flip the sign of \(\lambda\)).
- Current scripts live at repository root:
  - `heat1d_lchs_cv_dv.py`
  - `heat1d_lchs_sensitivity.py`

These points are now reflected in runtime diagnostics and should be used for
all reproducibility runs.

## Scope, deliverables, and baseline choices

This response provides a single self-contained LaTeX document (ready to save as a `.tex` file) that specifies an implementable plan to reproduce the **1D heat equation** example with a **hybrid CV-DV Linear Combination of Hamiltonian Simulation (LCHS)** circuit, using **hybridlane** as the frontend and **Bosonic Qiskit** as the simulation backend (via the `bosonicqiskit.hybrid` PennyLane device). The plan prioritizes: (i) correctness of operator definitions, (ii) clear separation between classical preprocessing and quantum circuit steps, (iii) a concrete parameter and error budget, and (iv) unit-test guidance.

Key baseline choices (explicitly editable in the LaTeX file):
- **PDE**: 1D heat equation on a rod with **Dirichlet boundary conditions** as the default (can switch to periodic/Neumann with a clear change list).
- **Spatial discretization**: finite differences on \(N\) points with **\(N=2^m\)** to align with amplitude encoding into \(m\) qubits (smallest demo: \(m=2\Rightarrow N=4\)).
- **Dynamics**: homogeneous case \(b(t)=0\) first, because it isolates the core LCHS machinery.
- **Hamiltonian evolution**: **first-order Lie–Trotter** and commuting-group Trotterization for simplicity, consistent with your requirement.

The core LCHS identity expresses stable non-unitary evolution as a continuous weighted superposition of Hamiltonian simulations (unitaries). The LCHS formulation and its improved-kernel variants (with hyperparameter \(\beta\)) are described in the LCHS literature and related presentations.

## Mathematical specification needed for DV-only readers

The LaTeX file includes (at minimum) two “theoretical solutions” as you requested:

- **Closed-form PDE solution** (continuous): for Dirichlet BCs, separation of variables yields a sine-series solution with exponentially decaying Fourier modes.  
- **CV-DV LCHS solution expression**: a qumode “sandwich” integral that reproduces \(e^{-At}\) as an integral over unitary evolutions \(e^{-i t(kL+H)}\) weighted by a kernel. The continuous-kernel LCHS identity is standard, and the improved kernel family depends on \(\beta\in(0,1)\).

The plan also defines *every variable before use* in formulas, and makes explicit what is classical vs quantum.

## Error sources, tuning knobs, and how to choose parameters

The LaTeX file decomposes end-to-end error into distinct contributions that map to implementation knobs:

- **Spatial discretization error** (classical, PDE-to-ODE): controlled by grid spacing \(h\) (and thus \(N\)); for standard second-derivative finite differences, spatial error is typically \(O(h^2)\) under regularity assumptions.  
- **Kernel / LCHS truncation-discretization error**: in pure LCHS implementations one truncates the integral \(\int_{\mathbb R}\) to \([-K,K]\) and then quadratures it; improved kernels parameterized by \(\beta\) are designed to achieve near-optimal complexity and lead to characteristic \(K\)-vs-\(\epsilon\) scaling (often described via \((\log(1/\epsilon))^{1/\beta}\)-type behavior).  
- **Hybridlane/Bosonic-Qiskit operator-definition mismatch risk**: explicitly addressed (see next section).
- **Trotterization error** (quantum compilation): controlled by Trotter steps \(n_{\text{Trot}}\), commutator structure, and commuting-group decomposition.  
- **Finite CV cutoff and finite squeezing effects**: controlled by `max_fock_level` (simulation cutoff) and the two squeeze parameters (one for projection, one for preparation basis).

The file contains practical guidance for setting these parameters in a staged way (start with tiny \(N\), high cutoff, conservative squeezing; then tighten).

## Operator-to-software mapping and sanity checks

The LaTeX file includes an “operator ledger” that maps each mathematical operator to:
1) its hybridlane operation class,
2) the corresponding Bosonic Qiskit gate family, and
3) the intended mathematical definition.

This is critical because identical names can mask different conventions.

A few verified anchors (the file points readers to these as “must-pass sanity checks”):

- **Displacement** in hybridlane is \(D(\alpha)=\exp(\alpha a^\dagger-\alpha^* a)\), with \(\alpha=a e^{i\phi}\).  
- **Squeezing** in hybridlane is \(S(\zeta)=\exp\left[\frac12(\zeta^* a^2-\zeta(a^\dagger)^2)\right]\).  
- **Bosonic Qiskit gate definitions (paper)** include single-mode squeezing as \(e^{\frac12(\theta^*aa-\theta a^\dagger a^\dagger)}\) and controlled displacement as \(e^{\sigma_z\otimes(\theta a^\dagger-\theta^* a)}\).  
- **Hybridlane ConditionalDisplacement** is explicitly documented as \(CD(\alpha)=\exp[\sigma_z(\alpha a^\dagger-\alpha^* a)]\), matching the Bosonic Qiskit controlled displacement form.  

Important “mismatch risk” found by inspection:
- The Bosonic Qiskit `operators.py` implementation of single-mode squeezing appears to form the exponential with a creation-operator-squared term where annihilation-operator-squared is expected (relative to the paper and hybridlane definition).  
The LaTeX file makes this **actionable**: it requires a unit test that numerically compares the matrix generated by the backend gate against the analytic formula for small cutoffs, and it documents the required parameter/sign transform if the implementation is found to follow the “swapped” convention.

Finally, the plan explicitly leverages the fact that the **hybrid interaction** needed for \(e^{-i\lambda \hat{x}\otimes P}\) can be synthesized from **controlled displacement** by choosing purely imaginary displacement amplitude (because \(a=x+ip\) under the quantum-optics convention). This keeps the implementation within the standard ISA-like gate set exposed by hybridlane/Bosonic Qiskit.

## Unit-test strategy that validates the implementation

The LaTeX file contains a test suite blueprint (pytest-style) with “smallest nontrivial” tests before scaling:

- **Gate-definition tests**: Displacement and squeezing matrix equivalence at small cutoff; controlled displacement generates the intended \(\hat{x}\)-coupling when the parameter is purely imaginary; and the squeezing-convention mismatch test described above.  
- **LCHS identity sanity test**: choose \(L=\lambda I\) so the target non-unitary becomes scalar \(e^{-\lambda t}\); verify the sandwich circuit reproduces it within tolerance. This isolates kernel/state-prep without Pauli decomposition confounders.  
- **End-to-end PDE test**: for \(N=4\) (2 qubits), compare quantum-produced vector (postselected output) to classical reference \(e^{-At}u_0\), including success probability vs \(\|u(t)\|^2\).  
- **Trotter scaling test**: fix everything else and vary \(n_{\text{Trot}}\); verify error decreases with expected scaling for first-order formulas.  

## LaTeX file content

```latex
\documentclass[11pt]{article}

\usepackage{amsmath,amssymb,amsfonts}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{enumitem}

\geometry{margin=1in}

% --------------------
% Minimal macros
% --------------------
\newcommand{\dd}{\mathrm{d}}
\newcommand{\ii}{\mathrm{i}}
\newcommand{\ee}{\mathrm{e}}
\newcommand{\RR}{\mathbb{R}}
\newcommand{\CC}{\mathbb{C}}

\title{Actionable Plan: 1D Heat Equation Demo via Hybrid CV--DV LCHS\\
in \texttt{hybridlane} with Bosonic Qiskit Backend}
\author{}
\date{\today}

\begin{document}
\maketitle

\section{Goal, deliverables, and success criteria}

\subsection*{Goal}
Implement the 1D heat equation example using the hybrid CV--DV LCHS approach in \texttt{hybridlane}, simulated with the Bosonic Qiskit backend (\texttt{bosonicqiskit.hybrid} PennyLane device).

\subsection*{Primary deliverable}
A runnable example script (suggested path in the \texttt{hybridlane} repo):
\[
\texttt{examples/heat\_equation\_1d\_cvdv\_lchs.py}
\]
that:
\begin{enumerate}[itemsep=2pt,topsep=2pt]
\item builds a discretized heat-equation generator $A$ for $N=2^m$ grid points,
\item constructs the CV--DV LCHS circuit (state prep $\rightarrow$ hybrid evolution $\rightarrow$ postselection/projection),
\item produces (postselected) output vector $\tilde{\mathbf u}(t)$,
\item compares to the classical reference $\mathbf u(t)=\exp(-At)\mathbf u(0)$,
\item prints an error report and unit-test-friendly diagnostics.
\end{enumerate}

\subsection*{Definition of success}
Choose a small, deterministic reference case (default below: Dirichlet BCs and $f(x)=\sin(\pi x/L)$). For $m=2$ ($N=4$) or $m=3$ ($N=8$), demonstrate:
\[
\frac{\|\tilde{\mathbf u}(t)-\mathbf u(t)\|_2}{\|\mathbf u(t)\|_2} \le \epsilon_{\mathrm{target}}
\]
with a documented error budget. A good initial target is $\epsilon_{\mathrm{target}}=10^{-2}$ and then tighten.

\section{Mathematical specification}

\subsection{Continuous PDE model and analytic solution}

\paragraph{PDE.}
Let $u(x,t)$ denote temperature on a rod of length $L>0$. Let $\alpha>0$ be thermal diffusivity.
The 1D heat equation is
\begin{equation}
\frac{\partial u(x,t)}{\partial t} = \alpha \frac{\partial^2 u(x,t)}{\partial x^2}, \quad 0<x<L,\; t\ge 0.
\end{equation}

\paragraph{Default boundary condition (Dirichlet).}
\begin{equation}
u(0,t)=0,\qquad u(L,t)=0,\qquad t\ge 0.
\end{equation}

\paragraph{Initial condition.}
\begin{equation}
u(x,0)=f(x),\qquad 0<x<L.
\end{equation}

\paragraph{Closed-form solution (Dirichlet).}
Define Fourier sine coefficients
\begin{equation}
b_n := \frac{2}{L}\int_0^L f(x)\sin\!\left(\frac{n\pi x}{L}\right)\dd x,\qquad n=1,2,\dots
\end{equation}
Then the analytic solution is
\begin{equation}
u(x,t) = \sum_{n=1}^{\infty} b_n \sin\!\left(\frac{n\pi x}{L}\right)\exp\!\left(-\alpha \left(\frac{n\pi}{L}\right)^2 t\right).
\end{equation}

\paragraph{Default reference case (recommended for unit tests).}
Set $f(x)=\sin(\pi x/L)$, which yields $b_1=1$ and $b_{n\ne 1}=0$, hence:
\begin{equation}
u(x,t)=\sin\!\left(\frac{\pi x}{L}\right)\exp\!\left(-\alpha \left(\frac{\pi}{L}\right)^2 t\right).
\end{equation}
This gives a clean, single-mode reference.

\subsection{Spatial discretization to a linear ODE}

\paragraph{Grid.}
Choose $N$ interior points with $N=2^m$ to amplitude-encode into $m$ qubits.
For Dirichlet BCs, set grid spacing
\begin{equation}
h := \frac{L}{N+1},\qquad x_j := jh,\quad j=1,\dots,N.
\end{equation}

\paragraph{State vector.}
Define $\mathbf u(t)\in\RR^N$ by $\mathbf u_j(t)\approx u(x_j,t)$.

\paragraph{Discrete Laplacian (Dirichlet).}
Define $T\in\RR^{N\times N}$ as the tridiagonal matrix:
\begin{equation}
T := \begin{bmatrix}
2 & -1 \\
-1 & 2 & -1 \\
& \ddots & \ddots & \ddots \\
&& -1 & 2 & -1 \\
&&& -1 & 2
\end{bmatrix}.
\end{equation}
Then define the generator
\begin{equation}
A := \frac{\alpha}{h^2} T.
\end{equation}

\paragraph{Resulting ODE.}
\begin{equation}
\frac{\dd}{\dd t}\mathbf u(t) = -A\mathbf u(t),\qquad \mathbf u(0)=\mathbf u_0.
\end{equation}

\paragraph{Theoretical discrete solution.}
\begin{equation}
\mathbf u(t)=\exp(-At)\,\mathbf u_0.
\end{equation}

\subsection{CV--DV LCHS target expression}

\paragraph{Cartesian decomposition (general form).}
Given $A=L+\ii H$, define
\begin{equation}
L:=\frac{A+A^\dagger}{2},\qquad H:=\frac{A-A^\dagger}{2\ii}.
\end{equation}
For diffusion-like dynamics we assume $L\succeq 0$ (stable).

\paragraph{LCHS continuous representation.}
We choose a kernel weight $g(k)$ such that
\begin{equation}
\exp\!\left(-\int_0^t A(s)\dd s\right)
=\int_{\RR} g(k)\left[\mathcal{T}\exp\!\left(-\ii\int_0^t (kL(s)+H(s))\dd s\right)\right]\dd k.
\end{equation}
For time-independent $A$, this simplifies to
\begin{equation}
\exp(-At)=\int_{\RR} g(k)\,\exp\!\left(-\ii t(kL+H)\right)\dd k.
\end{equation}

\paragraph{Improved kernel family (used in your working paper).}
Pick $\beta\in(0,1)$ and define
\begin{equation}
f(k)=\frac{\ee^{2\beta}}{2\pi \ee\, (1+\ii k)^\beta},\qquad
g(k)=\frac{f(k)}{1-\ii k}.
\end{equation}
Near-Cauchy reference (formal limit as \(\beta\to 1^{-}\)):
\[
\beta\to 1^{-} \Rightarrow f(k)\to\frac{1}{\pi(1+\ii k)},\; g(k)\to\frac{1}{\pi(1+k^2)}.
\]

\paragraph{CV--DV ``sandwich'' theorem target.}
Let $\hat{x}$ be the qumode position quadrature operator (quantum optics convention).
We want a circuit implementing the hybrid unitary
\begin{equation}
U(t):=\exp\!\left(-\ii t(\hat{x}\otimes L + I_{\mathrm{osc}}\otimes H)\right).
\end{equation}
Choose two normalized qumode states $|\psi\rangle_{\mathrm{osc}}$ (prepared) and $|\phi\rangle_{\mathrm{osc}}$ (postselected)
with position wavefunctions $\psi(x)$ and $\phi(x)$ such that
\begin{equation}
\phi^*(x)\psi(x)=g(x).
\end{equation}
Then the induced (unnormalized) DV state is
\begin{equation}
|\mathbf u(t)\rangle_{\mathrm{DV}}
=
\left(\langle \phi|_{\mathrm{osc}}\otimes I\right)\,U(t)\,\left(|\psi\rangle_{\mathrm{osc}}\otimes|\mathbf u_0\rangle_{\mathrm{DV}}\right)
=
\left[\int_{\RR} g(x)\,\exp\!\left(-\ii t(xL+H)\right)\dd x\right]|\mathbf u_0\rangle_{\mathrm{DV}}.
\end{equation}
In the ideal setting, this equals $\exp(-At)|\mathbf u_0\rangle$.

\paragraph{Heat equation specialization.}
For the heat equation, $A$ is real symmetric so $H=0$ and $L=A$.
Thus the target reduces to
\begin{equation}
|\mathbf u(t)\rangle
=
\left[\int_{\RR} g(x)\,\exp(-\ii t x A)\dd x\right]|\mathbf u_0\rangle.
\end{equation}

\section{Implementation architecture: classical vs quantum responsibilities}

\subsection{Classical preprocessing responsibilities}
Given $(\alpha,L,t,N=2^m)$ and initial function $f(x)$:
\begin{enumerate}[itemsep=3pt,topsep=3pt]
\item Build $A=(\alpha/h^2)T\in\RR^{N\times N}$.
\item Build a reference solution $\mathbf u_{\mathrm{ref}}(t)=\exp(-At)\mathbf u_0$ (SciPy).
\item Obtain a Pauli decomposition on $m$ qubits:
\[
A = \sum_{j=1}^{J} c_j P_j,
\]
where each $P_j$ is a Pauli string on $m$ qubits and $c_j\in\RR$.
For small $m$, use automated Pauli decomposition (e.g., PennyLane utilities). For larger $m$, exploit structure.
\item Choose LCHS kernel hyperparameter $\beta\in(0,1)$ and define $g(\cdot)$.
\item Choose two squeeze parameters:
\begin{itemize}[itemsep=2pt,topsep=2pt]
\item $r_{\mathrm{proj}}$: squeezing for the postselection state $|\phi\rangle$.
\item $r_{\mathrm{basis}}$: squeezing used as the basis for approximating the prepared state $|\psi\rangle$ in a truncated Fock space.
\end{itemize}
\item Compute prepared-state coefficients $\{C_n\}_{n=0}^{N_{\max}}$ so that
\begin{equation}
|\psi\rangle \approx \sum_{n=0}^{N_{\max}} C_n\,S(r_{\mathrm{basis}},0)|n\rangle.
\end{equation}
Normalize $\sum_{n=0}^{N_{\max}}|C_n|^2=1$.
\end{enumerate}

\subsection{Quantum circuit responsibilities}
With one qumode wire (call it \texttt{"m"}) and $m$ qubits (wires \texttt{0..m-1}):
\begin{enumerate}[itemsep=3pt,topsep=3pt]
\item Prepare the DV initial state $|\mathbf u_0\rangle$ (for this repository, a computational-basis state from \texttt{init\_q0, init\_q1}).
\item Prepare the qumode state $|\psi\rangle_{\mathrm{osc}}$ (details in Sec.~\ref{sec:stateprep}).
\item Apply the hybrid evolution $U(t)=\exp(-\ii t\,\hat{x}\otimes A)$ by Trotterization over Pauli terms (Sec.~\ref{sec:trotter}).
\item Postselect/projection: project the qumode onto $|\phi\rangle_{\mathrm{osc}}$ (Sec.~\ref{sec:projection}).
\item Return the (postselected) DV state vector and the success probability.
\end{enumerate}

\section{Operator ledger: math definitions and software mapping}

\subsection{Primitive CV operators and conventions}
We assume the quantum-optics convention:
\[
\hat{x}=\frac{a+a^\dagger}{2},\quad \hat{p}=\frac{a-a^\dagger}{2\ii},\quad a=\hat{x}+\ii\hat{p}.
\]
This matches the Gaussian wavefunction convention used in the CV LCHS derivations.

\subsection{Hybridlane operator definitions (must match the math)}
Use these \texttt{hybridlane} operators:
\begin{itemize}[itemsep=2pt,topsep=2pt]
\item \texttt{hqml.Displacement(a, phi, wires="m")} implements
$\exp(\alpha a^\dagger-\alpha^* a)$ with $\alpha=ae^{\ii\phi}$.
\item \texttt{hqml.Squeezing(r, phi, wires="m")} implements
$S(\zeta)=\exp\left[\frac12(\zeta^*a^2-\zeta(a^\dagger)^2)\right]$ with $\zeta=re^{\ii\phi}$.
\item \texttt{hqml.ConditionalDisplacement(a, phi, wires=[q,\,"m"])} implements
$CD(\alpha)=\exp\left[\sigma_z(\alpha a^\dagger-\alpha^* a)\right]$ with $\alpha=ae^{\ii\phi}$.
\end{itemize}

\subsection{Bosonic Qiskit gate definitions and the squeezing-convention check}
Bosonic Qiskit defines:
\[
\text{single-mode squeezing: }\exp\left[\frac12(\theta^*aa-\theta a^\dagger a^\dagger)\right],
\quad
\text{controlled displacement: }\exp\left[\sigma_z\otimes(\theta a^\dagger-\theta^*a)\right].
\]
\textbf{Action item:} verify that the backend's matrix-level implementation of single-mode squeezing
matches the above definition (and thus the \texttt{hybridlane} definition) for a small cutoff, e.g. \texttt{max\_fock\_level=8}.
If a mismatch is detected, introduce a parameter transform (conjugation/sign) \emph{in the translation layer} and lock it in with a unit test.

\section{State preparation and projection}
\label{sec:stateprep}

\subsection{Postselection state $|\phi\rangle$}
Use a squeezed vacuum for projection:
\[
|\phi\rangle := S(r_{\mathrm{proj}},0)|0\rangle.
\]
In simulation, projection can be performed as:
\[
\langle \phi|\psi_{\mathrm{final}}\rangle
=
\langle 0|S^\dagger(r_{\mathrm{proj}},0)|\psi_{\mathrm{final}}\rangle,
\]
so it suffices to (classically) apply $S^\dagger$ on the qumode and then take overlap with vacuum.

\subsection{Prepared state $|\psi\rangle$ as a truncated squeezed-Fock expansion}
We need $\phi^*(x)\psi(x)=g(x)$, hence
\[
\psi(x)=\frac{g(x)}{\phi^*(x)}.
\]
Given $\phi$ is a squeezed vacuum, this implies $\psi(x)$ is generally non-Gaussian (especially for the near-optimal kernel with $\beta\in(0,1)$).

We approximate $|\psi\rangle$ by:
\[
|\psi\rangle\approx\sum_{n=0}^{N_{\max}} C_n\,S(r_{\mathrm{basis}},0)|n\rangle.
\]

\subsection{Two practical implementation modes for $|\psi\rangle$}
\paragraph{Mode A (used in this repository): incoherent Fock-component aggregation.}
Because backend support for direct arbitrary qumode-state injection is limited in this project stack,
we use a finite Fock expansion with incoherent weights $|C_n|^2$ and aggregate per-component circuit outputs.
This is computationally practical but introduces a model mismatch versus coherent superposition preparation.

\paragraph{Mode B (hardware-aligned target): coherent gate-based preparation.}
Use extra ancilla qubits plus number-selective operations (e.g. SNAP / selective rotations) to synthesize the
superposition $\sum C_n|n\rangle$ and then apply a squeezing gate. This requires additional compilation work.

\section{Hybrid Hamiltonian evolution by Trotterization}
\label{sec:trotter}

\subsection{Target unitary}
For heat equation, target:
\[
U(t)=\exp(-\ii t\,\hat{x}\otimes A),\qquad A=\sum_{j=1}^{J} c_j P_j.
\]
Thus:
\[
U(t)=\exp\left(-\ii t\sum_{j=1}^{J} c_j(\hat{x}\otimes P_j)\right).
\]

\subsection{First-order Trotter step}
Let $n_{\mathrm{Trot}}\in\mathbb{N}$ and $\delta t:=t/n_{\mathrm{Trot}}$.
Use:
\[
U(t)\approx \left(\prod_{j=1}^{J}\exp(-\ii \delta t\, c_j\,\hat{x}\otimes P_j)\right)^{n_{\mathrm{Trot}}}.
\]

\subsection{Implementing $\exp(-\ii \lambda\,\hat{x}\otimes P)$ using controlled displacement}
Key synthesis idea:
\[
CD(\alpha)=\exp\left[\sigma_z(\alpha a^\dagger-\alpha^*a)\right]
=\exp\left(+\ii(2\,\mathrm{Im}\,\alpha)\,\sigma_z\hat{x}\right)\cdot\exp\left(-\ii(2\,\mathrm{Re}\,\alpha)\,\sigma_z\hat{p}\right)
\times e^{i\varphi_{\mathrm{BCH}}},
\]
where \(e^{i\varphi_{\mathrm{BCH}}}\) is a global phase from non-commutation of \(\hat{x},\hat{p}\).
Choosing $\alpha = -\ii\lambda/2$ yields:
\[
CD(-\ii\lambda/2)=\exp(-\ii\lambda\,\sigma_z\hat{x}).
\]

To implement $\exp(-\ii \lambda\,\hat{x}\otimes P)$ for a multi-qubit Pauli string $P$:
\begin{enumerate}[itemsep=2pt,topsep=2pt]
\item Conjugate $P$ into a single-qubit $Z$ via Clifford basis changes and a parity-compute circuit.
\item Apply \texttt{hqml.ConditionalDisplacement(a, phi)} with $\alpha=-\ii\lambda/2$ on the parity qubit and qumode.
\item Uncompute the parity and undo basis changes.
\end{enumerate}

\section{Projection, success probability, and classical readout}
\label{sec:projection}

\subsection{Postselection}
In the current implementation we use projector-conditioned Pauli expectations, not direct full-statevector readout.
Let $\Pi_\phi$ be the CV projector used for postselection (implemented as inverse preparation + vacuum projector).
Measure:
\[
p_{\mathrm{succ}} = \ev{\Pi_\phi},
\qquad
m_{ab}=\ev{\Pi_\phi\,(P_a\otimes P_b)}.
\]
Reconstruct the conditional DV density matrix:
\[
\rho_{\mathrm{post}}=\frac{1}{4\,p_{\mathrm{succ}}}\sum_{a,b\in\{I,X,Y,Z\}} m_{ab}\,(P_a\otimes P_b).
\]
For vector-style PDE comparison, use the principal-eigenvector proxy of \(\rho_{\mathrm{post}}\).

\subsection{Convert back to a PDE comparison vector}
In this repository the DV register is compared to the finite-difference target via
\(\rho_{\mathrm{post}}\) and its principal eigenvector proxy; the PDE-priority metric is reported on
\(\sqrt{p_{\mathrm{succ}}}\,\psi_{\mathrm{principal}}\).

\section{Error budget and parameter selection guide}

Define a target end-to-end error $\epsilon_{\mathrm{target}}$ and allocate:
\[
\epsilon_{\mathrm{target}}
\ge
\epsilon_{\mathrm{space}}
+\epsilon_{\mathrm{kernel}}
+\epsilon_{\mathrm{prep}}
+\epsilon_{\mathrm{proj}}
+\epsilon_{\mathrm{fock}}
+\epsilon_{\mathrm{trot}}.
\]

\subsection{Recommended staged procedure}
\begin{enumerate}[itemsep=3pt,topsep=3pt]
\item \textbf{Stage 0 (gate sanity):} validate Displacement, ConditionalDisplacement, and Squeezing definitions by matrix tests at small cutoff.
\item \textbf{Stage 1 (LCHS identity scalar test):} set $A=\lambda I$ on 1 qubit, verify the sandwich yields $e^{-\lambda t}$.
\item \textbf{Stage 2 (PDE small-N):} run $m=2$ ($N=4$), compare to $\exp(-At)\mathbf u_0$.
\item \textbf{Stage 3 (near-optimal kernel):} switch to $\beta\in(0,1)$ and prepared non-Gaussian $|\psi\rangle$.
\end{enumerate}

\subsection{How to pick $\beta$, $r_{\mathrm{proj}}$, $r_{\mathrm{basis}}$, and cutoff}
\begin{itemize}[itemsep=3pt,topsep=3pt]
\item Start near the Cauchy baseline with $\beta\approx 0.95$ (the current scripts enforce $\beta\in(0,1)$).
\item Move to $\beta\in[0.8,0.95]$ for near-optimal behavior, and record how $N_{\max}$ and cutoff must increase to maintain fidelity.
\item Increase $r_{\mathrm{proj}}$ to make the projection state more sharply peaked in $x$ (but track qumode energy and cutoff needs).
\item Tune $r_{\mathrm{basis}}<r_{\mathrm{proj}}$ to reduce coefficient spread in the truncated expansion of $|\psi\rangle$.
\item Increase \texttt{max\_fock\_level} until projection/cutoff errors stop dominating (diagnose via tail probability beyond cutoff).
\end{itemize}

\section{Unit test blueprint}

Create tests under:
\[
\texttt{tests/test\_gate\_math.py}
\]

\subsection*{Test list}
\begin{enumerate}[itemsep=3pt,topsep=3pt]
\item \textbf{Gate-definition tests}
\begin{enumerate}[itemsep=2pt,topsep=2pt]
\item Displacement matrix equality between hybridlane and analytic $D(\alpha)$.
\item ConditionalDisplacement generates $\exp(-\ii\lambda \sigma_z \hat{x})$ for purely imaginary $\alpha=-\ii\lambda/2$.
\item Squeezing-definition check: compare backend squeezing matrix against analytic $S(\zeta)$; if mismatch, pin down the parameter transform and assert it.
\end{enumerate}
\item \textbf{Scalar LCHS test} ($A=\lambda I$): output scalar matches $e^{-\lambda t}$ within tolerance.
\item \textbf{End-to-end PDE test} ($m=2$): compare postselected output to $\exp(-At)\mathbf u_0$.
\item \textbf{Trotter convergence test}: error decreases as $n_{\mathrm{Trot}}$ increases (log-log fit optional).
\item \textbf{Success probability regression}: compare measured $p_{\mathrm{succ}}$ against classical $\|\mathbf u(t)\|_2^2$ in the high-accuracy limit.
\end{enumerate}

\section{Concluding clarifications needed from you}

To finalize the example spec without ambiguity, please answer:

\begin{enumerate}[itemsep=3pt,topsep=3pt]
\item Which boundary condition should the official demo use: Dirichlet (default here), Neumann, or periodic?
\item Should the first implementation target the homogeneous heat equation ($b(t)=0$) only, or do you want the inhomogeneous term included in v1?
\item Do you want v1 to use a near-Cauchy baseline ($\beta\approx0.95$) for fastest validation, or jump directly to broader near-optimal $\beta\in(0,1)$ sweeps?
\end{enumerate}

\end{document}
```
