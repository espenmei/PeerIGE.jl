# Computational Approach: AI-REML for Social Genetic Effects

## 1. Model

We estimate variance components for a **Social Genetic Effects (SGE)** model applied to school peer effects. For observed individual $i$ in school class (group) $g_i$:

$$y_i = \mathbf{x}_i'\boldsymbol{\beta} + a_{d,i} + \sum_{j \in g_i,\, j \neq i} a_{s,j} + e_i$$

where:
- $\boldsymbol{\beta}$ are fixed effects (intercept, covariates),
- $a_{d,i}$ is the **direct** genetic effect of individual $i$ on their own outcome,
- $a_{s,j}$ is the **indirect (social/peer)** genetic effect of classmate $j$ on $i$'s outcome,
- $e_i \sim N(0, \sigma^2_e)$ is the residual.

In matrix notation:

$$\mathbf{y} = X\boldsymbol{\beta} + Z_d \mathbf{a}_d + Z_s \mathbf{a}_s + \mathbf{e}$$

where $Z_d$ maps observations to pedigree IDs (identity-like, $n \times N$), and $Z_s$ maps each observation to the sum of classmates' social effects (peer design matrix, $n \times N$). The genetic effects follow:

$$\begin{pmatrix} \mathbf{a}_d \\ \mathbf{a}_s \end{pmatrix} \sim N\!\left(\mathbf{0},\; G \otimes A\right), \quad G = \begin{pmatrix} \sigma^2_d & \sigma_{ds} \\ \sigma_{ds} & \sigma^2_s \end{pmatrix}$$

where $A$ is the $N \times N$ additive genetic relationship matrix derived from the pedigree, and $\otimes$ denotes the Kronecker product. The four variance components to estimate are $\boldsymbol{\theta} = (\sigma^2_d, \sigma^2_s, \sigma_{ds}, \sigma^2_e)$.

**Key dimensions for the real data:**
- $N \approx 7{,}000{,}000$ (pedigree size)
- $n \approx 1{,}000{,}000$ (observed individuals)
- $\dim(\text{MME}) = n_f + 2N \approx 14{,}000{,}002$

## 2. Mixed Model Equations (MME)

Henderson's MME for this model are:

$$\underbrace{\begin{pmatrix} X'X & X'Z_d & X'Z_s \\ Z_d'X & Z_d'Z_d + \lambda_{dd}A^{-1} & Z_d'Z_s + \lambda_{ds}A^{-1} \\ Z_s'X & Z_s'Z_d + \lambda_{ds}A^{-1} & Z_s'Z_s + \lambda_{ss}A^{-1} \end{pmatrix}}_{\text{LHS (coefficient matrix } C\text{)}} \begin{pmatrix} \hat{\boldsymbol{\beta}} \\ \hat{\mathbf{a}}_d \\ \hat{\mathbf{a}}_s \end{pmatrix} = \begin{pmatrix} X'\mathbf{y} \\ Z_d'\mathbf{y} \\ Z_s'\mathbf{y} \end{pmatrix}$$

where $\Lambda = G^{-1}\sigma^2_e$ provides the ratios $\lambda_{dd}, \lambda_{ds}, \lambda_{ss}$.

The coefficient matrix $C$ is of dimension $\sim 14M \times 14M$ and is never formed explicitly. Instead, we implement it as a **matrix-free operator** that computes $C \cdot v$ for any vector $v$ via component-wise operations.

## 3. Estimation: Average Information REML (AI-REML)

We use AI-REML (Gilmour, Thompson & Cullis, 1995) to estimate $\boldsymbol{\theta}$. The update at iteration $t$ is:

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} + \alpha \cdot \text{AI}^{-1} \cdot \mathbf{s}$$

where $\mathbf{s}$ is the REML score vector and AI is the Average Information matrix ($4 \times 4$).

### 3.1 Score vector

For parameter $\theta_i$ with associated variance structure $V_i = \partial V / \partial \theta_i$:

$$s_i = \frac{1}{2}\left(\mathbf{y}'P V_i P\mathbf{y} - \text{tr}(P V_i)\right)$$

where $P = V^{-1} - V^{-1}X(X'V^{-1}X)^{-1}X'V^{-1}$ is the projection matrix. In practice, $P\mathbf{y}$ is obtained from the MME solution (see Section 4).

The variance derivatives are:
| Parameter | $V_i$ |
|---|---|
| $\sigma^2_d$ | $Z_d A Z_d'$ |
| $\sigma^2_s$ | $Z_s A Z_s'$ |
| $\sigma_{ds}$ | $Z_d A Z_s' + Z_s A Z_d'$ |
| $\sigma^2_e$ | $I_n$ |

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

## 4. Solving the MME: Conjugate Gradient with Matrix-Free Operator

The core computational task is solving systems of the form $C \mathbf{x} = \mathbf{b}$ where $C$ is the $\sim 14M \times 14M$ MME coefficient matrix. We use the **Preconditioned Conjugate Gradient (PCG)** algorithm.

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

This captures the full off-diagonal structure of $A^{-1}$ in the genetic block. Each application requires 4 sparse triangular solves (2 for $A \cdot r_d$ and 2 for $A \cdot r_s$), plus a $2 \times 2$ $G$-matrix blend.

**Diagonal Preconditioner** (lighter alternative):

$$M^{-1} = \text{diag}(C)^{-1}$$

Uses only the diagonal of the MME coefficient matrix. Cheaper per CG iteration (no triangular solves) but typically requires many more iterations to converge.

### 4.4 Warm starting

CG solutions are warm-started across REML iterations: the solution from the previous iteration serves as the initial guess for the next. This is particularly effective because parameter changes between successive REML iterations are small, so the MME solution changes slowly.

## 5. Parallelization Strategy

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

Threading uses `:static` scheduling to guarantee a stable `threadid()` → workspace mapping throughout each parallel loop.

Read-only data (model matrices $A^{-1}$, $L$, $X$, group structure) is shared safely across all threads.

## 6. Memory Layout

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

## 7. Convergence Monitoring

Each REML iteration reports:
- **Wall-clock timing** per step (CG(y), CG(z), AI) and cumulative time
- **ETA** based on average iteration time
- **Parameter estimates** with standard errors (from $\text{AI}^{-1}$)
- **Score vector** and Newton step ($\delta$)
- **$G$-matrix health:** genetic correlation $\rho_{ds}$, condition number, eigenvalues
- **Score norm** $\|\mathbf{s}\|$ (should decrease toward zero)
- **Relative parameter change** $\|\Delta\boldsymbol{\theta}\| / \|\boldsymbol{\theta}\|$ (convergence criterion: $< 10^{-5}$)

All diagnostics are also written to a CSV log file for external monitoring during long runs.

## 8. Post-Estimation

After convergence, a final MME solve with the converged variance components yields:
- **BLUE** $\hat{\boldsymbol{\beta}}$ (fixed effects estimates)
- **BLUP** $\hat{\mathbf{a}}_d$ and $\hat{\mathbf{a}}_s$ (predicted direct and social genetic effects)

Results are stored in an `AiRemlResult` struct and serialized to disk via JLD2.

Standard errors for the variance components are obtained from the inverse of the (regularized) AI matrix. Significance testing for individual parameters (e.g., $H_0: \sigma^2_s = 0$) can be performed via a Wald-type $z$-test: $z = \hat{\theta}_i / \text{SE}_i$.

## 9. Software

- **Julia 1.12** with `IterativeSolvers.jl` (PCG solver), `SparseArrays`, `JLD2`
- **R** for pedigree processing: computing $A^{-1}$ (Henderson's rules) and its Cholesky factor $L$, exported as sparse triplet CSV files
- Data exchange via CSV (triplet format for sparse matrices, dense format for $y$, $X$, group assignments)

## References

- Gilmour, A. R., Thompson, R., & Cullis, B. R. (1995). Average information REML: An efficient algorithm for variance parameter estimation in linear mixed models. *Biometrics*, 51(4), 1440–1450.
- Henderson, C. R. (1976). A simple method for computing the inverse of a numerator relationship matrix used in prediction of breeding values. *Biometrics*, 32(1), 69–83.
- Hutchinson, M. F. (1990). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics — Simulation and Computation*, 19(2), 433–450.
- Bijma, P. (2014). The quantitative genetics of indirect genetic effects: a selective review of modelling issues. *Heredity*, 112(1), 61–69.
