# Computational Approach: AI-REML for Social Genetic Effects

## 1. Model

We estimate variance components for a **Social Genetic Effects (SGE)** model applied to school peer effects. For observed individual $i$ in school class (group) $g_i$:

$$y_i = \mathbf{x}_i'\boldsymbol{\beta} + a_{d,i} + \sum_{j \in g_i,\, j \neq i} a_{s,j} + \sum_{h=1}^{H} c_{h(i)} + e_i$$

where:
- $\boldsymbol{\beta}$ are fixed effects (intercept, covariates),
- $a_{d,i}$ is the **direct** genetic effect of individual $i$ on their own outcome,
- $a_{s,j}$ is the **indirect (social/peer)** genetic effect of classmate $j$ on $i$'s outcome,
- $c_{h(i)}$ is the **shared environment effect** of group $h(i)$ for environment factor $h$ (e.g., household), with $c_{h,l} \sim N(0, \sigma^2_{c_h})$,
- $e_i \sim N(0, \sigma^2_e)$ is the residual.

In matrix notation:

$$\mathbf{y} = X\boldsymbol{\beta} + Z_d \mathbf{a}_d + Z_s \mathbf{a}_s + \sum_{h=1}^{H} Z_h \mathbf{c}_h + \mathbf{e}$$

where $Z_h$ is the $n \times n_h$ indicator matrix mapping observations to environment-group levels. The $H$ environment factors are optional: setting $H=0$ gives the base SGE model. The total number of variance components is $n_p = 4 + H$.

$$\begin{pmatrix} \mathbf{a}_d \\ \mathbf{a}_s \end{pmatrix} \sim N\!\left(\mathbf{0},\; G \otimes A\right), \quad G = \begin{pmatrix} \sigma^2_d & \sigma_{ds} \\ \sigma_{ds} & \sigma^2_s \end{pmatrix}$$

where $Z_d$ maps observations to pedigree IDs (identity-like, $n \times N$), $Z_s$ maps each observation to the sum of classmates' social effects (peer design matrix, $n \times N$), and each $Z_h$ maps observations to environment-group indicators. The genetic effects follow the distribution above, where $A$ is the $N \times N$ additive genetic relationship matrix derived from the pedigree, and $\otimes$ denotes the Kronecker product. The variance components to estimate are $\boldsymbol{\theta} = (\sigma^2_d, \sigma^2_s, \sigma_{ds}, \sigma^2_e, \sigma^2_{c_1}, \ldots, \sigma^2_{c_H})$, giving $n_p = 4 + H$ free parameters.

The solution vector in the MME has dimension $\dim = n_f + 2N + \sum_h n_h$, where $n_h$ is the number of distinct levels of environment factor $h$.

**Key dimensions for the real data:**
- $N \approx 7{,}000{,}000$ (pedigree size)
- $n \approx 1{,}000{,}000$ (observed individuals)
- $\dim(\text{MME}) = n_f + 2N + \sum_h n_h \approx 14{,}000{,}000 + \sum_h n_h$ (the env term adds one dimension per distinct level per factor)

### 1.1 Missing outcome data (observed-data formulation)

The implementation uses an **observed-data REML formulation**:
- $n$ is the number of individuals with observed outcome values used in $\mathbf{y}$ and $X$.
- $N$ is the full pedigree size used for the random-effect vectors $(\mathbf{a}_d, \mathbf{a}_s)$ and for $A^{-1}$.

Individuals with missing outcomes are therefore not included as rows in $\mathbf{y}$ (no residual term for that individual), but they can still be part of the pedigree graph through $A^{-1}$. This means their direct and social genetic effects remain in the model state and are estimated as BLUPs, borrowing information from related observed individuals.

Operationally, the contribution of an unobserved pedigree ID depends on available design information:
- If an ID is present only in the pedigree, it contributes through genetic covariance structure and is shrunk toward the pedigree prior.
- If an ID is also represented in the group/design mapping used to build the social operator, it can additionally contribute through peer-structure equations even without its own outcome row.

In the current `run_emai.jl` workflow, group/design mappings are built from the observed table (`yX.csv`). To activate the second case for missing-outcome individuals, the input pipeline must provide group membership independently of observed outcomes.

This separation is exactly why we distinguish $n$ (observed outcomes) from $N$ (all pedigree IDs): missing outcomes are integrated out at the observation level while still allowing genetic information propagation to unobserved individuals.

### 1.2 Required input data and effect scope

The model requires two linked data layers: an observed-data layer (length $n$) and a pedigree layer (size $N$).

Required inputs in the current workflow are:

| Input | Dimension | Role |
|---|---|---|
| Outcome vector ($\mathbf{y}$) | $n$ | Observed response values |
| Fixed-effect design ($X$) | $n \times n_f$ | Covariates and intercept |
| Observation-to-pedigree IDs (`id_ped`) | $n$ | Maps each observed row to a pedigree ID in $1..N$ |
| Peer-group IDs (`group`) | $n$ | Defines social exposure structure among observed rows |
| Sparse $A^{-1}$ triplets | on $N \times N$ | Pedigree precision structure |
| Sparse Cholesky of $A^{-1}$ | on $N \times N$ | Enables implicit application of $A$ |
| Optional environment IDs (`env_ids[h]`) | $n$ each | Observation-level grouping for each shared environment factor |

This implies different scope for different random effects:

1. **Direct and indirect genetic effects** $(\mathbf{a}_d, \mathbf{a}_s)$ are defined on the full pedigree size $N$.
2. **Shared environment effects** $\mathbf{c}_h$ are defined through `env_ids[h]` of length $n$ (observed rows), with one effect per observed environment level $1..n_h$.

So yes: in the current formulation, genetic effects are pedigree-wide, while environment effects are observation/group-level by construction.

Practically, this means:

1. Unobserved pedigree members can still receive BLUPs for $a_d$ and $a_s$ through pedigree covariance.
2. Environment effects are estimated for levels represented in the supplied `env_ids` mapping.
3. Expanding environment effects beyond observed rows would require an expanded input mapping design, not only a larger pedigree.

## 2. Mixed Model Equations (MME)

Henderson's MME for this model are:

$$\underbrace{\begin{pmatrix} X'X & X'Z_d & X'Z_s \\ Z_d'X & Z_d'Z_d + \lambda_{dd}A^{-1} & Z_d'Z_s + \lambda_{ds}A^{-1} \\ Z_s'X & Z_s'Z_d + \lambda_{ds}A^{-1} & Z_s'Z_s + \lambda_{ss}A^{-1} \end{pmatrix}}_{\text{LHS (coefficient matrix } C\text{)}} \begin{pmatrix} \hat{\boldsymbol{\beta}} \\ \hat{\mathbf{a}}_d \\ \hat{\mathbf{a}}_s \end{pmatrix} = \begin{pmatrix} X'\mathbf{y} \\ Z_d'\mathbf{y} \\ Z_s'\mathbf{y} \end{pmatrix}$$

where $\Lambda = G^{-1}\sigma^2_e$ provides the ratios $\lambda_{dd}, \lambda_{ds}, \lambda_{ss}$.

The coefficient matrix $C$ is of dimension $(n_f + 2N + \sum_h n_h) \times (n_f + 2N + \sum_h n_h)$ — roughly $\gtrsim 14M$ with env effects included — and is never formed explicitly. Instead, we implement it as a **matrix-free operator** that computes $C \cdot v$ for any vector $v$ via component-wise operations.

## 3. Why Not Direct Likelihood Optimisation?

A natural starting point for estimating $\boldsymbol{\theta}$ is to maximise the **marginal (restricted) log-likelihood** directly:

$$\ell_R(\boldsymbol{\theta}) = -\frac{1}{2}\left[\log|V| + \log|X'V^{-1}X| + \mathbf{y}'P\mathbf{y}\right]$$

where $V = Z_d A Z_d' \sigma^2_d + Z_s A Z_s' \sigma^2_s + \cdots + I\sigma^2_e$ is the $n \times n$ marginal covariance matrix. A derivative-free optimiser (Nelder–Mead, Brent, etc.) could then call this function repeatedly until convergence.

The fundamental obstacle is the **log-determinant** $\log|V|$. For $n \approx 400{,}000$, forming $V$ explicitly already requires $\sim 1.2$ TB of memory. Even if $V$ were sparse, computing its log-determinant exactly requires a Cholesky factorisation, whose fill-in for a pedigree-structured matrix is enormous. The second term $\log|X'V^{-1}X|$ requires computing $V^{-1}X$, which is equally infeasible.

**AI-REML avoids all log-determinants.** The score vector and Average Information matrix (Sections 3.1–3.3) are expressed entirely in terms of $P\mathbf{y}$ and $P\mathbf{z}_j$ — vectors obtained by solving the MME. The log-likelihood value itself is never evaluated. Instead, the algorithm operates as a **gradient-based quasi-Newton method**: the gradient (score vector $\mathbf{s}$) is computed without evaluating the objective, and parameter updates use the Average Information matrix as an approximate Hessian. Convergence is monitored through the score norm $\|\mathbf{s}\|$ and the relative parameter change $\|\Delta\boldsymbol{\theta}\|/\|\boldsymbol{\theta}\|$.

This means:
- **No determinants** are computed at any point in the algorithm.
- **No dense $n \times n$ matrices** are ever formed or stored.
- **The objective is never evaluated**—only its gradient, which is much cheaper and numerically more stable.
- The cost per iteration is dominated by $k+5$ CG solves of the $(n_f + 2N + \sum_h n_h)$-dimensional MME, each of which is feasible via the matrix-free PCG approach (Section 5).

The trade-off relative to derivative-free optimisation is that AI-REML requires analytically correct score and curvature expressions, and is a Newton-type method that can diverge if the initial variance components are far from the optimum. The step-halving safeguard (Section 3.4) mitigates the latter.

## 4. Estimation: Average Information REML (AI-REML)

We use AI-REML (Gilmour, Thompson & Cullis, 1995) to estimate $\boldsymbol{\theta}$. The update at iteration $t$ is:

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} + \alpha \cdot \text{AI}^{-1} \cdot \mathbf{s}$$

where $\mathbf{s}$ is the REML score vector and AI is the Average Information matrix ($4 \times 4$).

### 3.1 Score vector

**Intuition:** The score vector $\mathbf{s}$ is the gradient of the restricted log-likelihood with respect to the variance parameters. Each component $s_i$ tells you the direction and magnitude to move parameter $\theta_i$ to improve the likelihood. When all $s_i \approx 0$, the algorithm has converged.

**Definition:** For parameter $\theta_i$ with associated variance structure $V_i = \partial V / \partial \theta_i$:

$$s_i = \frac{1}{2}\left(\mathbf{y}'P V_i P\mathbf{y} - \text{tr}(P V_i)\right)$$

where $P = V^{-1} - V^{-1}X(X'V^{-1}X)^{-1}X'V^{-1}$ is the projection matrix and $V$ is the $n \times n$ marginal covariance matrix. In practice, $P\mathbf{y}$ is obtained from the MME solution (see Section 4).

**Variance derivatives:** The table below lists $V_i = \partial V / \partial \theta_i$ for each variance parameter. These are the partial derivatives of the marginal covariance matrix with respect to each variance component. They appear in both the score computation (above) and the Hutchinson trace estimation (Section 3.2) and Average Information matrix (Section 3.3).

| Parameter | $\partial V / \partial \theta_i$ |
|---|---|
| $\sigma^2_d$ | $Z_d A Z_d'$ |
| $\sigma^2_s$ | $Z_s A Z_s'$ |
| $\sigma_{ds}$ | $Z_d A Z_s' + Z_s A Z_d'$ |
| $\sigma^2_e$ | $I_n$ |
| $\sigma^2_{c_h}$ | $Z_h Z_h'$ |

The $Z_h Z_h'$ application is a **scatter-gather** operation: for output $i$, $(Z_h Z_h' u)_i = \sum_{j:\, h(j)=h(i)} u_j$. This is $O(n)$ and requires no matrix storage.

### 3.2 Stochastic trace estimation (Hutchinson)

The trace $\text{tr}(PV_i)$ is intractable for large $n$. We use the **Hutchinson estimator** with $k$ random probe vectors $\mathbf{z}_j \sim N(0, I)$:

$$\text{tr}(PV_i) \approx \frac{1}{k} \sum_{j=1}^{k} \mathbf{z}_j' V_i P\mathbf{z}_j$$

Each $P\mathbf{z}_j$ requires solving the MME with $\mathbf{z}_j$ as the right-hand side (same structure as for $P\mathbf{y}$). The same $k$ probe vectors are reused across all REML iterations for comparability.

### 3.3 Average Information matrix

$$\text{AI}_{ij} = \frac{1}{2} \mathbf{y}' P V_i P V_j P \mathbf{y}$$

Computing $P V_j P\mathbf{y}$ requires solving the MME with $V_j P\mathbf{y}$ as the "data" vector — another CG solve. Since $j \in \{1, 2, 3, 4\}$, only **4 CG solves** are needed (not 10), and the 10 elements of the symmetric AI matrix are filled by cheap matrix-vector products $V_i \cdot (\cdot)$ plus dot products.

### 3.4 Step-size control

A step-halving safeguard ensures:
- All variance components remain positive ($\sigma^2_d, \sigma^2_s, \sigma^2_e > 0$)
- $G$ remains positive definite ($\sigma^2_d \sigma^2_s > \sigma_{ds}^2$)

The step size $\alpha$ starts at the requested damping value and is halved up to 20 times until both conditions are satisfied.

## 5. Solving the MME: Conjugate Gradient with Matrix-Free Operator

The core computational task is solving systems of the form $C \mathbf{x} = \mathbf{b}$ where $C$ is the $(n_f + 2N + \sum_h n_h) \times (n_f + 2N + \sum_h n_h)$ MME coefficient matrix. We use the **Preconditioned Conjugate Gradient (PCG)** algorithm.

### 4.1 Matrix-free operator

The operator $C \cdot v$ is computed without forming $C$ explicitly. For $v = (v_f, v_d, v_s)$:

1. **Data contribution:** Compute $\hat{y} = Xv_f + Z_dv_d + Z_sv_s$, then:
   - $\text{res}_f = X'\hat{y}$
   - $\text{res}_d = Z_d'\hat{y}$
   - $\text{res}_s = Z_s'(Xv_f + Z_dv_d) + Z_s'Z_sv_s$
2. **Genetic prior:** Add $(G^{-1}\sigma^2_e) \otimes A^{-1}$ contribution to the genetic blocks.

Key optimizations:
- $Z_s \cdot v_s$ is computed via group sums: for each observation, sum $v_s$ over classmates.
- $Z_s'Z_s \cdot v_s$ exploits the identity: for member $i$ in group of size $n_g$, $(Z_s'Z_sv)_i = (n_g - 2)\sum_j v_j + v_i$ (no explicit matrix needed).
- $A^{-1}$ is stored as a sparse matrix (Henderson's rules); $A^{-1}v$ is a standard sparse matrix-vector product.

### 4.2 Applying $A = A^{-1\, -1}$

Several operations require $A \cdot x$ (not $A^{-1} \cdot x$). Since $A$ is dense and enormous, we compute $A \cdot x$ via the Cholesky factorization of $A^{-1}$:

$$A^{-1} = LL', \quad A \cdot x = (LL')^{-1}x = L'^{-1}(L^{-1}x)$$

The Cholesky factor $L$ is precomputed from the pedigree in $R$ (using Henderson's rules block-by-block) and passed to Julia as a sparse lower-triangular matrix. Each application of $A$ requires two sparse triangular solves.

### 4.3 Preconditioners

Two preconditioner options are implemented:

**Block Preconditioner** (default, recommended):

$$M^{-1} = \begin{pmatrix} \text{diag}(X'X)^{-1} & 0 \\ 0 & (G \otimes A) / \sigma^2_e \end{pmatrix}$$

The values of $G$ and $\sigma^2_e$ used in the preconditioner are the **current estimates at each AI-REML iteration**. The preconditioner is recomputed before each set of CG solves, so it adapts as variance estimates improve. This typically speeds convergence compared to using fixed initial values.

This captures the full off-diagonal structure of $A^{-1}$ in the genetic block. Each application requires 4 sparse triangular solves (2 for $A \cdot r_d$ and 2 for $A \cdot r_s$), plus a $2 \times 2$ $G$-matrix blend.

**Environmental effect blocks** (when present): For each environment factor $h$, a diagonal preconditioner is used at each level $j$:

$$M_h^{-1}(j) = \frac{1}{n_{h,j} + \frac{\sigma^2_e}{\sigma^2_{c_h}}}$$

where $n_{h,j}$ is the number of observations at level $j$ of factor $h$, $\sigma^2_e$ is the current residual variance, and $\sigma^2_{c_h}$ is the current environmental variance component for factor $h$. Like the genetic block, both $\sigma^2_e$ and $\sigma^2_{c_h}$ use current iteration estimates.

**Diagonal Preconditioner** (lighter alternative):

$$M^{-1} = \text{diag}(C)^{-1}$$

Uses only the diagonal of the MME coefficient matrix. Cheaper per CG iteration (no triangular solves) but typically requires many more iterations to converge.

### 4.4 Warm starting

CG solutions are warm-started across REML iterations: the solution from the previous iteration serves as the initial guess for the next. This is particularly effective because parameter changes between successive REML iterations are small, so the MME solution changes slowly.

## 6. Parallelization Strategy

The main parallelizable components are:

| Task | Count per iteration | Parallel strategy |
|---|---|---|
| $z$-solves ($P\mathbf{z}_j$) | $k$ (e.g., 50) | `@threads :static` over $j$ |
| Trace estimation ($V_i P\mathbf{z}_j$) | $4k$ (e.g., 200) | `@threads :static` flat loop |
| AI CG solves ($PV_jP\mathbf{y}$) | 4 | `@threads :static` over $j$ |
| AI dot products ($V_i \cdot (\cdot)$) | 10 | `@threads :static` over pairs |

### Thread safety

The matrix-free MME operator and preconditioner use mutable workspace buffers for intermediate results. To avoid data races, each thread receives its own **`ThreadWorkspace`** containing independent copies of all mutable buffers:

- RHS buffer, MME operator workspaces ($w_n$, $w_{n2}$, $w_{\dim}$, $A_d$, $A_s$, $G^{-1}$)
- $A$-application temporaries ($\text{tmp}_{N1}$, $\text{tmp}_{N2}$, $\text{tmp}_{N3}$)
- Preconditioner temporaries ($\text{tmp}_1$, $\text{tmp}_2$)
- Output buffer ($v_i\_\text{out}$)
- Environment temporaries ($\text{env\_tmp}[h]$ for each factor $h$)

Threading uses `:static` scheduling to guarantee a stable `threadid()` → workspace mapping throughout each parallel loop.

**Implementation note:** In Julia 1.12, multiple consecutive `Threads.@threads :static` blocks within a single function corrupt each other's closure environments, producing non-deterministic wrong results. Each `@threads` block must therefore be wrapped in its own dedicated function. The five helpers are: `_threaded_vi_py!`, `_threaded_trace!`, `_threaded_ai_cg!`, `_threaded_ai_matvec!`, `_threaded_z_solves!`.

Read-only data (model matrices $A^{-1}$, $L$, $X$, group structure, environment ID arrays) is shared safely across all threads.

## 7. Memory Layout

For $N = 7{,}000{,}000$ and $n = 1{,}000{,}000$:

| Component | Approx. size |
|---|---|
| $A^{-1}$ (sparse, $\sim 3$ nonzeros/row avg) | $\sim 500$ MB |
| $L_{A^{-1}}$ (sparse lower triangular) | $\sim 250$ MB |
| Solution vectors ($\text{sol}_y$ + $k$ $\text{sol}_z$) | $\sim 5.6$ GB ($k=50$) |
| Probe vectors ($k$ vectors of length $n$) | $\sim 400$ MB |
| Per-thread workspace ($\times n_t$ threads) | $\sim 1.9$ MB/thread |
| $X$ (dense, $n \times n_f$) | $\sim 16$ MB ($n_f = 2$) |
| **Total estimate** | **$\sim 10\text{--}15$ GB** |

## 8. Convergence Monitoring

Each REML iteration reports:
- **Wall-clock timing** per step (CG(y), CG(z), AI) and cumulative time
- **ETA** based on average iteration time
- **Parameter estimates** with standard errors (from $\text{AI}^{-1}$)
- **Score vector** and Newton step ($\delta$)
- **$G$-matrix health:** genetic correlation $\rho_{ds}$, condition number, eigenvalues
- **Score norm** $\|\mathbf{s}\|$ (should decrease toward zero)
- **Relative parameter change** $\|\Delta\boldsymbol{\theta}\| / \|\boldsymbol{\theta}\|$ (convergence criterion: $< 10^{-5}$)

All diagnostics are also written to a CSV log file for external monitoring during long runs.

## 9. Post-Estimation

After convergence, a final MME solve with the converged variance components yields:
- **BLUE** $\hat{\boldsymbol{\beta}}$ (fixed effects estimates)
- **BLUP** $\hat{\mathbf{a}}_d$ and $\hat{\mathbf{a}}_s$ (predicted direct and social genetic effects)

Results are stored in an `AiRemlResult` struct and serialized to disk via JLD2.

Standard errors for the variance components are obtained from the inverse of the (regularized) AI matrix. Significance testing for individual parameters (e.g., $H_0: \sigma^2_s = 0$) can be performed via a Wald-type $z$-test: $z = \hat{\theta}_i / \text{SE}_i$.

## 10. Software

- **Julia 1.12** with `IterativeSolvers.jl` (PCG solver), `SparseArrays`, `JLD2`
- **R** for pedigree processing: computing $A^{-1}$ (Henderson's rules) and its Cholesky factor $L$, exported as sparse triplet CSV files
- Data exchange via CSV (triplet format for sparse matrices, dense format for $y$, $X$, group assignments)

## References

- Gilmour, A. R., Thompson, R., & Cullis, B. R. (1995). Average information REML: An efficient algorithm for variance parameter estimation in linear mixed models. *Biometrics*, 51(4), 1440–1450.
- Henderson, C. R. (1976). A simple method for computing the inverse of a numerator relationship matrix used in prediction of breeding values. *Biometrics*, 32(1), 69–83.
- Hutchinson, M. F. (1990). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics — Simulation and Computation*, 19(2), 433–450.
- Bijma, P. (2014). The quantitative genetics of indirect genetic effects: a selective review of modelling issues. *Heredity*, 112(1), 61–69.
