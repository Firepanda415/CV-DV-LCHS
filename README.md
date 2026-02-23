Use Hybridlane, see documentation https://pnnl.github.io/hybridlane/getting-started.html




Sensitivity improvements


**Settings:**

**  **total_time: 1

**  **n_steps: 100

**  **dt: 0.01

**  **init_qubits: (0, 1)

**  **dv_qubits: 2

**  **n_dim: 32

**  **r_target: 1.2

**  **r_prime: 0.3

**  **beta: 0.7

**  **kernel_beta: 0.4

**  **alpha_disp: 1.4

**  **energy_shift: -1.0

**  **use_gaussian_prep: False

**  **use_displacement: True

**  **use_fock_expansion: True

**  **fock_expansion_cutoff: 1e-08

**  **max_fock_level: 32

**  **hbar: 2.0

**Preparing LCHS states (N=32, r_target=1.2)...**

**States ready.**

**CV state fidelity |<psi_lchs|psi_gauss>|^2 = 0.9043636375546708**

**Circuit stats:**

**  **wires: [0, 'm0', 1]

**  **qubits: 2 qumodes: 1

**  **operations: 2205

**  **measurements: 17

**  **op counts: {'FockLadder': 1, 'PauliX': 2, 'Squeezing': 2, 'Displacement': 100, 'Hadamard': 1000, 'ConditionalDisplacement': 300, 'CNOT': 400, 'RZ': 400}

**  **measurement counts: {'ExpectationMP': 17}

**Note: fock expansion uses incoherent weights |Cn|^2 (mixture approximation).**

**Post-selection probability: 0.004257139352953996**

**Density diagnostics:**

**  **Tr(rho_post) before sanitize: 1.0

**  **min eig before sanitize: 0.005293910084476784

**  **clipped negative eig weight: 0.0

**  **Tr(rho_post) after sanitize: 0.9999999999999996

**  **purity Tr(rho_post^2): 0.9488566675922195

**  **Note: rho_post is mixed; vector-only comparisons use a principal-eigenvector proxy.

**||u_theory||: 0.08793250337087748**

**Fidelity F=<u_hat|rho_post|u_hat> (mixed-state correct): 0.9509343183558827**

**Norm diff (normalized vectors): 0.15460513178640503**

**Norm diff (best-fit scaled): 0.013554136226750054**

**Norm diff (unnormalized u_theory vs sqrt(p_post)*psi_post): 0.02552999324063933**
