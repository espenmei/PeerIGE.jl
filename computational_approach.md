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

### 1.1 Why this becomes a computational problem

For a behavior-genetic audience, the main point is not that the statistical model is unusually exotic. The main point is that a scientifically natural model leads to a computational object that is too large for naive mixed-model software.

The modeling ingredients are familiar:
- fixed effects for measured covariates,
- one direct genetic effect per pedigree individual,
- one social genetic effect per pedigree individual,
- optional shared environment effects,
- pedigree-based covariance through $A$.

What makes the problem hard is the scale of the implied linear system. Once the model is written as Henderson's mixed model equations, the unknown vector has dimension

$$\dim = n_f + 2N + \sum_h n_h.$$

With pedigree-scale data, this is not a bookkeeping detail. It is the central computational bottleneck. Even before considering repeated REML iterations, the naive coefficient matrix would have dimension roughly $14$ million by $14$ million on the real data scale.

This is why the computational contribution matters scientifically: the goal is not to fit a different simplified model, but to fit the same direct-plus-social genetic model without ever constructing the impossible matrix that the textbook formulation suggests.

### 1.2 Communicating the scale to non-specialists

An effective way to explain the difficulty is to contrast the scientific model with the naive computation it implies.

| Scientific question | Naive computation | Why it fails |
|---|---|---|
| Estimate direct and social genetic effects jointly | Form the full MME coefficient matrix | The matrix dimension is on the order of $n_f + 2N + \sum_h n_h$, which is far too large to store densely |
| Use REML for variance components | Repeatedly factorize or invert very large matrices | AI-REML requires many large linear solves across iterations |
| Use pedigree relationships | Form $A$ explicitly | $A$ is dense and enormous even when $A^{-1}$ is sparse |

The practical message is simple: the obstacle is not just that the model is "a bit slower" than a standard animal model. The obstacle is that the obvious dense computation is not feasible in memory or time.

If one were to store a dense matrix of dimension $d$, the storage alone would be approximately

$$8d^2 \text{ bytes}.$$

So even a problem with $d = 200{,}000$ would already require about $320$ GB just to store one dense double-precision matrix. At the real pedigree scale considered here, direct dense storage is completely out of reach, which is why all core computations must be sparse and matrix-free.

### 1.3 Visual intuition: sparsity does not survive the covariance construction

The figure below is useful for communicating why the covariance-side formulation becomes difficult even when some of the building blocks look sparse or structured.

![Covariance illustration comparing the pedigree relationship matrix $A$, the social-design cross-product $Z_s Z_s'$, and the induced social-genetic covariance term $Z_s A Z_s'$. The main point is that the propagated covariance can be substantially denser than the structured ingredients used to define it.](covariance_illustration.png)

The key message is not that every intermediate matrix is dense. It is that the covariance contribution induced by social structure and pedigree propagation can become much denser than the original ingredients. In particular:
- the pedigree precision machinery is exploitable because $A^{-1}$ and its sparse Cholesky factor are sparse,
- the social-design cross-product $Z_s Z_s'$ is still highly structured,
- but the propagated covariance term $Z_s A Z_s'$ is much denser, which makes direct covariance construction, storage, and factorization unattractive.

For a methodological behavior-genetics audience, this picture helps separate two ideas that are easy to conflate: pedigree computations can exploit sparse precision structure, while the marginal covariance seen by the phenotype can still become computationally unwieldy.

### 1.4 Missing outcome data (observed-data formulation)

The implementation uses an **observed-data REML formulation**:
- $n$ is the number of individuals with observed outcome values used in $\mathbf{y}$ and $X$.
- $N$ is the full pedigree size used for the random-effect vectors $(\mathbf{a}_d, \mathbf{a}_s)$ and for $A^{-1}$.

Individuals with missing outcomes are therefore not included as rows in $\mathbf{y}$ (no residual term for that individual), but they can still be part of the pedigree graph through $A^{-1}$. This means their direct and social genetic effects remain in the model state and are estimated as BLUPs, borrowing information from related observed individuals.

Operationally, the contribution of an unobserved pedigree ID depends on available design information:
- If an ID is present only in the pedigree, it contributes through genetic covariance structure and is shrunk toward the pedigree prior.
- If an ID is also represented in the group/design mapping used to build the social operator, it can additionally contribute through peer-structure equations even without its own outcome row.

In the current workflow, observation rows for $\mathbf{y}$ and $X$ come from `yX_obs.csv`, while social group membership is supplied separately through the full `roster.csv` file. This preserves social-group membership for missing-outcome individuals and keeps effective peer counts (group size minus self) correct for observed rows.

This separation is exactly why we distinguish $n$ (observed outcomes) from $N$ (all pedigree IDs): missing outcomes are integrated out at the observation level while still allowing genetic information propagation to unobserved individuals.

### 1.5 Required input data and effect scope

The model requires two linked data layers: an observed-data layer (length $n$) and a pedigree layer (size $N$).

Required inputs in the current workflow are:

| Input | Dimension | Role |
|---|---|---|
| Outcome vector ($\mathbf{y}$) | $n$ | Observed response values |
| Fixed-effect design ($X$) | $n \times n_f$ | Covariates and intercept |
| Observation-to-pedigree IDs (`id_ped`) | $n$ | Maps each observed row in `yX_obs.csv` to a pedigree ID in $1..N$ |
| Peer-group roster (`roster.csv`) | roster rows | Defines social exposure structure from full group membership; observed rows are mapped into this structure |
| Sparse $A^{-1}$ triplets | on $N \times N$ | Pedigree precision structure |
| Sparse Cholesky of $A^{-1}$ | on $N \times N$ | Enables implicit application of $A$ |
| Optional environment IDs (`env_ids[h]`) | $n$ each | Observation-level grouping for each shared environment factor |

In the current implementation, `yX_obs.csv` must contain one row per observed outcome, including `y`, `id_ped`, and any covariates included in $X$. Optional environment columns such as `household`, `classroom`, and `school` also live in `yX_obs.csv`. The separate `roster.csv` file contains `id_ped` and `group` for all pedigree IDs that participate in the social grouping structure, including individuals without an observed outcome.

This implies different scope for different random effects:

1. **Direct and indirect genetic effects** $(\mathbf{a}_d, \mathbf{a}_s)$ are defined on the full pedigree size $N$.
2. **Shared environment effects** $\mathbf{c}_h$ are defined through `env_ids[h]` of length $n$ (observed rows), with one effect per observed environment level $1..n_h$.

So yes: in the current formulation, genetic effects are pedigree-wide, while environment effects are observation/group-level by construction.

Practically, this means:

1. Unobserved pedigree members can still receive BLUPs for $a_d$ and $a_s$ through pedigree covariance.
2. Environment effects are estimated for levels represented in the supplied `env_ids` mapping.
3. Expanding environment effects beyond observed rows would require an expanded input mapping design, not only a larger pedigree.

### 1.6 Data organization and cache efficiency

**Row ordering in `yX_obs.csv` affects computational performance.** The critical inner loop (`apply_Zs_transpose!` and related group-sum operations) repeatedly accesses the group-sum buffer using observation indices. If observations are **ordered by social group**, these accesses exhibit strong spatial locality, keeping the group-sum array warm in L1/L2 cache throughout iteration. Random ordering causes frequent cache misses.

**Recommendation:** Sort `yX_obs.csv` rows by the `group` column (primary key), then by `id_ped` (secondary key). This clustering can yield **5–15% wall-clock speedup** in the hot loop, depending on group size and system cache size, without algorithmic cost.

`roster.csv` and the sparse matrix files do not benefit materially from sorting; they are processed differently or only once during setup.

## 2. Mixed Model Equations (MME)

Henderson's MME for this model are:

$$\underbrace{\begin{pmatrix} X'X & X'Z_d & X'Z_s \\ Z_d'X & Z_d'Z_d + \lambda_{dd}A^{-1} & Z_d'Z_s + \lambda_{ds}A^{-1} \\ Z_s'X & Z_s'Z_d + \lambda_{ds}A^{-1} & Z_s'Z_s + \lambda_{ss}A^{-1} \end{pmatrix}}_{\text{LHS (coefficient matrix } C\text{)}} \begin{pmatrix} \hat{\boldsymbol{\beta}} \\ \hat{\mathbf{a}}_d \\ \hat{\mathbf{a}}_s \end{pmatrix} = \begin{pmatrix} X'\mathbf{y} \\ Z_d'\mathbf{y} \\ Z_s'\mathbf{y} \end{pmatrix}$$

where $\Lambda = G^{-1}\sigma^2_e$ provides the ratios $\lambda_{dd}, \lambda_{ds}, \lambda_{ss}$.

The coefficient matrix $C$ is of dimension $(n_f + 2N + \sum_h n_h) \times (n_f + 2N + \sum_h n_h)$ — roughly $\gtrsim 14M$ with env effects included — and is never formed explicitly. Instead, we implement it as a **matrix-free operator** that computes $C \cdot v$ for any vector $v$ via component-wise operations.

For presentation purposes, this is the key computational message: the inferential target is standard REML for a pedigree mixed model, but the algorithm succeeds only because it replaces explicit matrix construction with repeated operator applications and iterative solves.

### 2.1 Why not use a sparse direct solver on the MME?

This argument is also intended to rule out an obvious alternative: assembling the sparse MME and solving it by sparse Cholesky or sparse LDL$'$ factorization at each REML iteration.

In principle, the MME coefficient matrix is structured and many of its ingredients are sparse or implicitly sparse. In practice, however, sparse direct methods are still unattractive at the target scale for three related reasons:

1. The system dimension is enormous, so even storing the assembled MME is already a major burden.
2. Sparse factorization introduces substantial fill-in, especially once the direct-effect, social-effect, and pedigree-coupling blocks interact.
3. AI-REML requires repeated solves as the variance components change, so a sparse direct factorization would need to be rebuilt or heavily updated many times across iterations.

So the issue is not only whether one sparse factorization is theoretically possible. The issue is whether repeated factorization of a changing, extremely large system is computationally realistic. The matrix-free PCG approach is chosen because it avoids assembling $C$, avoids factorization fill-in, and reduces the repeated work to operator applications plus preconditioned iterative solves.

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

This is the point where randomized methods enter for a fundamental reason, not as a convenience add-on. The difficult term is the trace itself. Computing $\text{tr}(PV_i)$ exactly would require access to diagonal or factorization information about an operator built from very large solves, which would largely undo the computational savings of the matrix-free formulation. Hutchinson's estimator replaces that exact trace with an unbiased Monte Carlo estimate that only requires matrix-vector style operations and the same MME solves already needed elsewhere in AI-REML.

So the logic is:
- iterative methods are used because repeated direct factorization of the MME is not scalable,
- randomized trace estimation is used because exact trace evaluation is not scalable,
- both choices are driven by the same goal: keep every expensive step reducible to repeated applications of structured operators and large linear solves.

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
- Data exchange via CSV (triplet format for sparse matrices; dense observed-data table in `yX_obs.csv`; full social roster in `roster.csv`)

## References

- Gilmour, A. R., Thompson, R., & Cullis, B. R. (1995). Average information REML: An efficient algorithm for variance parameter estimation in linear mixed models. *Biometrics*, 51(4), 1440–1450.
- Henderson, C. R. (1976). A simple method for computing the inverse of a numerator relationship matrix used in prediction of breeding values. *Biometrics*, 32(1), 69–83.
- Hutchinson, M. F. (1990). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics — Simulation and Computation*, 19(2), 433–450.
- Bijma, P. (2014). The quantitative genetics of indirect genetic effects: a selective review of modelling issues. *Heredity*, 112(1), 61–69.
