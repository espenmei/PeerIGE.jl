# A latent Mundlak adjustment for school-level sorting

## Main idea

A Mundlak-type adjustment could, in principle, model school-level dependence
between the environment and latent direct and indirect genetic effects.
However, this would be a **joint latent correlated-effects model**, not an
ordinary regression adjustment using observed school means. Its identification
and ability to recover genetic variance components would need to be established.

The starting point is the direct-plus-social model described in
[the computational approach](../computational_approach.md).

## 1. The school-sorting problem

Let $s(i)$ denote individual $i$'s school and $\mathcal C(i)$ their classroom
roster. Consider

$$
y_i
= \mathbf x_i^\top\boldsymbol\beta
+ a_{d,i}
+ \sum_{\substack{j\in\mathcal C(i)\\j\ne i}} a_{s,j}
+ c_{s(i)}
+ e_i.
$$

Here:

- $a_{d,i}$ is the individual's direct genetic effect;
- $a_{s,j}$ is classmate $j$'s indirect, or social, genetic effect;
- $c_{s(i)}$ is a shared school environmental effect;
- $e_i$ is the individual residual.

The baseline pedigree model assumes

$$
\begin{pmatrix}
\mathbf a_d\\
\mathbf a_s
\end{pmatrix}
\sim
N\!\left(
\mathbf 0,\,
G\otimes A
\right),
\qquad
G=
\begin{pmatrix}
\sigma_d^2 & \sigma_{ds}\\
\sigma_{ds} & \sigma_s^2
\end{pmatrix}.
$$

A conventional independent school random intercept additionally assumes that
the school environment is independent of the genetic effects. School sorting
can make that assumption implausible: pupils' genetic propensities may be
associated with the environments of the schools they attend.

## 2. A latent Mundlak-type specification

For a defined school roster $\mathcal S_s$ of size $n_s$, define

$$
\bar a_{d,s}
= \frac{1}{n_s}\sum_{j\in\mathcal S_s}a_{d,j},
\qquad
\bar a_{s,s}
= \frac{1}{n_s}\sum_{j\in\mathcal S_s}a_{s,j}.
$$

The proposed extension is

$$
c_s
= \gamma_d\bar a_{d,s}
+ \gamma_s\bar a_{s,s}
+ u_s,
$$

where the remaining school effect $\mathbf u$ is assumed independent of the
genetic effects, conditional on the included covariates and the specified
membership structure. Equivalently, the conditional school-environment mean is

$$
E\!\left[
c_s\mid \mathbf a_d,\mathbf a_s
\right]
= \gamma_d\bar a_{d,s}
+ \gamma_s\bar a_{s,s}.
$$

Substitution gives

$$
y_i
= \mathbf x_i^\top\boldsymbol\beta
+ a_{d,i}
+ \sum_{\substack{j\in\mathcal C(i)\\j\ne i}}a_{s,j}
+ \gamma_d\bar a_{d,s(i)}
+ \gamma_s\bar a_{s,s(i)}
+ u_{s(i)}
+ e_i.
$$

Thus:

- $\gamma_d$ describes the association of school environment with
  school-average direct genetic propensity;
- $\gamma_s$ describes its association with school-average social genetic
  propensity.

These are conditional associations, not automatically causal effects of
changing a school's genetic composition.

**School averages and classroom exposure serve different roles.** School means
represent the proposed sorting adjustment; the classroom sum remains the
actual social-exposure term. School-level means are therefore appropriate to
investigate when the hypothesized selection operates between schools, even
though social effects operate within classrooms.

The school roster must be defined substantively, for example by school and
cohort when appropriate. It should not silently become the subset with
observed outcomes. The choice of a school mean versus a school total also
matters when school sizes differ; the equations above specifically assume a
mean-based relationship.

## 3. Why this must be estimated jointly

In an ordinary Mundlak adjustment, covariates and their group averages are
observed. Here, both school averages contain unknown genetic effects.

It is therefore **not equivalent** to:

1. fit the original independent-effects model;
2. compute school averages of its genetic BLUPs;
3. add those averages as error-free fixed covariates.

Those BLUPs depend on the same outcomes and on the independence assumptions
being questioned. Their shrinkage and prediction uncertainty cannot simply be
ignored. A two-stage plug-in regression is not the joint latent model above.

### Matrix representation

Let $M$ map pedigree effects to school averages and let $H$ map school effects
to observed individuals. Then

$$
\mathbf y
= X\boldsymbol\beta
+ (Z_d+\gamma_d HM)\mathbf a_d
+ (Z_s+\gamma_s HM)\mathbf a_s
+ H\mathbf u
+ \mathbf e.
$$

For fixed $\gamma_d$ and $\gamma_s$, the predictor remains linear in the latent
effects. The two coefficients are additional parameters of the genetic design
and hence of the marginal covariance, not ordinary fixed-effect coefficients
on known columns of $X$.

School averages and their transposes can be applied with sums and broadcasts.
This formulation does not require a dense relationship matrix or an explicitly
formed full mixed-model coefficient matrix. However, fitting it would require
extending the likelihood optimization and its derivatives; it is not just an
extra pair of input covariates.

## 4. Scope and identification limitations

The proposal represents one particular form of dependence. It does not
automatically correct every consequence of nonrandom school assignment.

- **Other sorting mechanisms:** parental characteristics, socioeconomic
  circumstances, and other latent traits may require additional structure.
- **Within-school sorting:** nonrandom classroom assignment remains a separate
  issue.
- **Genetic distribution assumptions:** the specification models dependence of
  school environment on genetic composition. It does not by itself establish
  that the assumed genetic distribution conditional on the observed school
  assignment is correct.
- **Identification:** school-composition contributions may be difficult to
  distinguish from existing genetic contributions and school environmental
  variation. Pedigree links across schools, multiple classrooms per school,
  and variation in composition can help, but do not guarantee identification.
- **Exposure scale:** the classroom term here is a sum. A diluted or
  mean-exposure model must retain its intended scaling separately from the
  school-composition adjustment.

Consequently, recovery of $\sigma_d^2$, $\sigma_s^2$, and $\sigma_{ds}$ must be
demonstrated rather than inferred from a better likelihood alone.

## 5. School fixed effects as a comparison

A simpler robustness analysis replaces the school random intercept with
unrestricted school fixed effects:

$$
y_i
= \mathbf x_i^\top\boldsymbol\beta
+ \alpha_{s(i)}
+ a_{d,i}
+ \sum_{\substack{j\in\mathcal C(i)\\j\ne i}}a_{s,j}
+ e_i.
$$

For an additive school-level environmental intercept, this avoids requiring
independence between that intercept and school composition. School-constant
genetic-composition contributions are absorbed as well.

Identification then relies on the remaining within-school contrasts and
pedigree covariance structure. School fixed effects do not resolve
within-school classroom sorting, guarantee identification of social genetic
variance, or repair every possible misspecification of the genetic covariance.

## 6. Concrete verification strategy

Before using the latent extension for substantive inference:

1. Simulate pedigree-linked direct and social genetic effects with known
   $\sigma_d^2$, $\sigma_s^2$, and $\sigma_{ds}$.
2. Generate school environmental effects using known $\gamma_d$ and
   $\gamma_s$, with classrooms nested within schools.
3. Compare an independent-school random-effects model, a school fixed-effects
   model, and the joint latent extension.
4. Across repeated simulations, assess bias, uncertainty calibration,
   convergence, and identifiability of the genetic parameters and the two
   $\gamma$ coefficients.
5. Include zero-sorting cases, varying school/class sizes, pedigree links
   across schools, and misspecified sorting mechanisms.

**Bottom line:** school-average latent direct and social genetic effects are
a coherent basis for a proposed correlated-school-effects model. They are not
observed Mundlak controls, and the extension needs joint estimation and
simulation-based identification checks before being treated as a correction
for school sorting.

---

## 7. Implementing school fixed effects: design analysis

*Analysis only; nothing below is implemented.* References are to the current
code: the matrix-free operator in [emai_mme.jl](../src/emai_mme.jl), the
preconditioners in [emai_precond.jl](../src/emai_precond.jl), and the score/AI
assembly in [emai_aireml.jl](../src/emai_aireml.jl).

### 7.1 Relation to the latent Mundlak terms

With unrestricted school intercepts $\alpha_s$, any school-constant term is
absorbed, including $\gamma_d\bar a_{d,s}+\gamma_s\bar a_{s,s}$ from §2. So,
for an additive school intercept, school fixed effects nest the latent Mundlak
specification and require no extra variance-component parameters. The cost is
that between-school genetic variation no longer contributes phenotypic
information; it is informed only through pedigree links via $A^{-1}$.

### 7.2 Notation and scale

Let $S$ be the number of schools (thousands), $H_S\in\{0,1\}^{n\times S}$ the
observation-to-school indicator, and $D_S=H_S^\top H_S=\operatorname{diag}(n_1,\dots,n_S)$
the observed counts. Write the remaining design as

$$
W=\begin{bmatrix}X & Z_d & Z_s & Z_{1} & \cdots & Z_{H}\end{bmatrix},
$$

with prior precision block $\Lambda$ (zero for $X$, $G^{-1}\otimes A^{-1}$
scaled by $\sigma_e^2$ for the genetic block, $\sigma_e^2/\sigma_{c_h}^2$ for
environment blocks). The full model is

$$
\mathbf y = H_S\boldsymbol\alpha + W\boldsymbol\theta + \mathbf e .
$$

$H_S$ has exactly one nonzero per row, so it is naturally stored as the integer
vector `school_id[i]` already used for `env_ids[h]`; no matrix needs to be
formed. $S\approx 10^3$–$10^4$ is negligible against $\dim(\text{MME})\approx 1.4\times10^7$.

### 7.3 Three representation options

**Option A — dummy columns in `X`.** `SGEModel.X` is a dense `Matrix{Float64}`.
With $n=4\times10^5$ and $S=5000$, dense dummies need about $16$ GB and turn
every `mul!(·, X, ·)` and `X'·` into $O(nS)$ work. A sparse `X` would fix the
storage, but the fixed-effect covariance routine would still perform one CG
solve per column, i.e. thousands of solves. **Not recommended.**

**Option B — an explicit school block with zero prior (reuse the env-block
machinery).** Environment factors already enter the operator as an indicator
block with ridge $\lambda_h=\sigma_e^2/\sigma_{c_h}^2$. A school fixed effect is
the same block with $\lambda=0$ (formally $\sigma_c^2\to\infty$), excluded from
the variance-component list. The changes would be: one extra layout range of
length $S$, $H_S^\top\hat{\mathbf y}$ and $H_S\boldsymbol\alpha$ loops in the
operator, `build_rhs!`, and `compute_Px_from_sol!`, and a diagonal
preconditioner $1/n_s$. The cost is $O(n)$ per matvec.

Concerns:

- The CG system now contains a block with no prior shrinkage. It is positive
  definite only if $[X\;H_S]$ has full column rank. The intercept and every
  school-constant covariate in `X` must be dropped or a reference coding imposed.
- The diagonal preconditioner ignores the strong coupling between
  $\boldsymbol\alpha$ and the intercept-like directions of the genetic and
  classroom blocks. This is the same weakness that makes diagonal
  preconditioning of unshrunk group effects slow. Expect more CG iterations,
  which is costly given that `cg_maxiter=2000` is already reached on real data.

**Option C — exact absorption (within-school projection).** Eliminate
$\boldsymbol\alpha$ analytically. Define the within-school projector

$$
M_S = I_n - H_S D_S^{-1} H_S^\top ,
\qquad
(M_S\mathbf v)_i = v_i - \bar v_{s(i)} .
$$

The Schur complement of the $\boldsymbol\alpha$ block gives the reduced MME.
§7.4 shows that this unweighted form is exact because
$R=\sigma_e^2 I$, and lists when it is not.

$$
\left(W^\top M_S W + \Lambda\right)\hat{\boldsymbol\theta}
= W^\top M_S\mathbf y,
\qquad
\hat{\boldsymbol\alpha}=D_S^{-1}H_S^\top\!\left(\mathbf y - W\hat{\boldsymbol\theta}\right).
$$

Matrix-free, this adds one projection per matvec. However, the current operator
does **not** form the full $\hat{\mathbf y}=W\mathbf v$ before applying
transposes: it computes $Z_s^\top$ on the partial $\hat{\mathbf y}$ excluding
$Z_s\mathbf w$, and adds $Z_s^\top Z_s\mathbf w$ through the specialized
`apply_ZsZs_accumulate!` kernel. The projection cannot simply be inserted after
step 3. Two restructurings are possible:

1. Reorder the steps: assemble the full $\hat{\mathbf y}$ (including
   $Z_s\mathbf w$), demean it in place, then apply all transposes, including
   `apply_Zs_transpose!` on the complete projected vector. The specialized
   $Z_s^\top Z_s$ kernel is then unnecessary in this path.
2. Keep the current order and subtract a rank-$S$ correction,
   $W^\top M_S W\mathbf v = W^\top W\mathbf v - W^\top H_S\mathbf m$, with
   school means $\mathbf m = D_S^{-1}H_S^\top W\mathbf v$. This requires the
   full $\hat{\mathbf y}$ for $\mathbf m$ and another transpose pass, so it is
   less attractive.

Option 1 is simpler; the cost of the change in operation order needs checking
against the existing kernel. Because $M_S$ is a symmetric idempotent projector,
$W^\top M_S W = (M_S W)^\top(M_S W)$, so the operator remains symmetric.
**This is the recommended option.**

Properties:

- The right-hand side becomes $W^\top M_S\mathbf v$: demean the source vector
  ($\mathbf y$, a probe $\mathbf z$, or $V_j P\mathbf y$) by school before the
  existing `build_rhs!` transposes.
- Cost: $O(n+S)$ per matvec, plus a length-$S$ buffer for school sums. That
  buffer must be per-thread (in `ThreadWorkspace`), like the existing
  `group_sum`; it must not be shared across threads.
- The solution vector and layout are unchanged. No new block, no unshrunk
  unknowns in CG, and the coupling with $\boldsymbol\alpha$ is removed exactly
  rather than approximated by a preconditioner.
- The same rank requirement applies: school-constant columns of `X`,
  including the intercept, become zero after $M_S$ and make the reduced fixed
  block singular. Detect them by within-school sum of squares, e.g.
  $\lVert M_S\mathbf x_j\rVert^2 < \varepsilon\lVert\mathbf x_j-\bar x_j\rVert^2$,
  and drop them with a clear message.
- Schools with a single observed student contribute nothing after demeaning:
  that student's row becomes zero. This is the correct fixed-effects behavior,
  but should be reported.

### 7.4 When is within-school demeaning exact? REML, score, and AI

Simple within-school demeaning is **not** a generic substitute for fixed
effects in a mixed model. In general, eliminating fixed effects is a GLS
operation, and its projection depends on the covariance. The claim here is
narrower. Under the conditions below, unweighted demeaning is exact **for this
model**, provided it is applied to the data **and every design matrix**, not
just to $\mathbf y$ or to the final residuals. The derivation follows.

#### 7.4.1 Assumptions

The current model satisfies the following assumptions:

- **(A1) Independent, homoscedastic residuals:** $R=\operatorname{Var}(\mathbf e)=\sigma_e^2 I_n$.
  Correlation comes only from random effects:
  $V = W_r\Gamma W_r^\top + \sigma_e^2 I_n$, where
  $W_r=[Z_d\;Z_s\;Z_1\cdots Z_H\;(Z_{pe})]$ and $\Gamma$ contains
  $G\otimes A$ and the environment variances.
- **(A2) Unrestricted school effects:** the school effects $\boldsymbol\alpha$
  are fixed, with no prior term in the MME.
- **(A3) Full-rank remaining fixed effects:** after school-constant columns
  are removed, $[X\;H_S]$ has full column rank.

#### 7.4.2 Derivation 1: elimination in Henderson's MME

The MME for $(\boldsymbol\alpha,\boldsymbol\beta,\mathbf u)$ are

$$
\begin{bmatrix}
H_S^\top R^{-1}H_S & H_S^\top R^{-1}W\\
W^\top R^{-1}H_S & W^\top R^{-1}W+\Lambda
\end{bmatrix}
\begin{bmatrix}\boldsymbol\alpha\\ \boldsymbol\theta\end{bmatrix}
=
\begin{bmatrix}H_S^\top R^{-1}\mathbf y\\ W^\top R^{-1}\mathbf y\end{bmatrix},
\qquad
\Lambda=\begin{bmatrix}0&0\\0&\Gamma^{-1}\end{bmatrix}.
$$

Here $\boldsymbol\theta=(\boldsymbol\beta,\mathbf u)$, and $\Lambda$ is the
unscaled precision. The residual-scaled form in §7.2 is $\sigma_e^2$ times it,
and multiplying the system by $\sigma_e^2$ leaves the solution unchanged.

The random-effect covariance enters only through $\Lambda$, in the
$\boldsymbol\theta$ block. School effects couple to the rest of the system
only through $R^{-1}$. Taking the Schur complement of the $\boldsymbol\alpha$
block is exact linear algebra, not an approximation. It gives

$$
\left(W^\top Q_R W + \Lambda\right)\hat{\boldsymbol\theta}=W^\top Q_R\mathbf y,
\qquad
Q_R = R^{-1}-R^{-1}H_S\left(H_S^\top R^{-1}H_S\right)^{-1}H_S^\top R^{-1}.
$$

$Q_R$ is the GLS projection, which generally depends on the covariance through
$R$. **Only** under (A1),

$$
Q_R=\frac{1}{\sigma_e^2}\left(I-H_SD_S^{-1}H_S^\top\right)=\frac{M_S}{\sigma_e^2},
$$

so the projection is ordinary within-school demeaning. It is independent of
all variance parameters, including $\sigma_e^2$ after scaling. $V$ is not
diagonal, but that does not matter here: random-effect correlation is handled
by $\Lambda$, not by the projection.

The derivation uses $W^\top M_S W$, i.e. all of $X$, $Z_d$, $Z_s$, $Z_h$, and
$Z_{pe}$ demeaned. The following are **not equivalent**:

- demeaning $\mathbf y$ and fitting the original operator $W^\top W+\Lambda$;
- demeaning only the residual returned by an unabsorbed solve.

Either shortcut changes the estimator. This is why §7.3 requires projection of
the full $\hat{\mathbf y}=W\mathbf v$ inside the operator.

#### 7.4.3 Derivation 2: REML error contrasts

Let $K\in\mathbb R^{n\times(n-S)}$ be an orthonormal basis of
$\operatorname{null}(H_S^\top)$. Then $KK^\top=M_S$, $K^\top K=I$, and
$K^\top H_S=0$. The transformed data satisfy

$$
K^\top\mathbf y\sim N\!\left(
K^\top X\boldsymbol\beta,\;
(K^\top W_r)\,\Gamma\,(K^\top W_r)^\top+\sigma_e^2 I_{n-S}
\right).
$$

Because of (A1) and $K^\top K=I$, this is again a mixed model with
**independent residuals**, projected random-effect designs $K^\top W_r$, and
the **same** parameters $(G,\sigma_{c_h}^2,\sigma_e^2)$. If $R$ were not
proportional to $I$, $K^\top RK$ would not be diagonal, and the residual
structure of the transformed model would change.

REML can be written using any full-rank set of error contrasts
$L^\top\mathbf y$ satisfying $L^\top[X\;H_S]=0$. Different such sets give
log-likelihoods that differ only by a constant independent of
$(G,\sigma_c^2,\sigma_e^2)$ (Harville, D. A., 1974, *Bayesian inference for
variance components using only error contrasts*, Biometrika 61:383–385). Applying REML for the remaining
$K^\top X$ in the transformed model is one such choice. It therefore has the
same score, AI matrix, and REML estimates as explicit school dummies. The
explicit and absorbed constructions should also produce the same

$$
P=V^{-1}-V^{-1}\tilde X(\tilde X^\top V^{-1}\tilde X)^{-1}\tilde X^\top V^{-1}
=\tilde K(\tilde K^\top V\tilde K)^{-1}\tilde K^\top,
\qquad \tilde X=[X\;H_S],
$$

where $\tilde K$ is any full-rank basis of $\operatorname{null}(\tilde X^\top)$.
In particular, $PH_S=0$ and $P=M_SP=PM_S$.

#### 7.4.4 Consequences for the score, traces, and AI in this code

The existing score, trace, and AI code can be unchanged **only because of
how it is currently structured**. `compute_score_and_ai` works entirely in
observation space:

- $\mathbf y^\top PV_iP\mathbf y$ uses $P\mathbf y$;
- the Hutchinson estimate of $\operatorname{tr}(PV_i)$ is
  $k^{-1}\sum_j\mathbf z_j^\top V_iP\mathbf z_j$;
- the AI uses $P V_j P\mathbf y$.

Every $P\mathbf v$ is produced by one MME solve followed by
`compute_Px_from_sol!`. The $V_i$ operators are the **unprojected**
$\partial V/\partial\theta_i$, which do not involve fixed effects. There is no
hard-coded $n-\operatorname{rank}(X)$, closed-form $\sigma_e^2$ update, or
log-determinant term. If every $P\mathbf v$ is exactly the $P$ above, the
score and AI formulas apply unchanged, including for $\sigma_e^2$, where $V_{\sigma_e^2}=I$ and the Hutchinson
estimate targets $\operatorname{tr}(P)$ under the full fixed-effect space.

Every call that produces $P\mathbf v$ for $\mathbf y$, a probe $\mathbf z$, or
$V_j P\mathbf y$ must satisfy all three conditions:

1. **Absorbed right-hand side:** $W^\top M_S\mathbf v$.
2. **Absorbed operator:** $W^\top M_SW+\Lambda$.
3. **Demeaned residual:**

$$
P\mathbf v=\frac{1}{\sigma_e^2}M_S\left(\mathbf v-W\hat{\boldsymbol\theta}_v\right).
$$

This follows by substituting the eliminated $\hat{\boldsymbol\alpha}_v$ into
the usual residual. Omitting any condition in any call gives a different
matrix from $P$ and biases the score.

Probes $\mathbf z$ need not be projected because $P$ already annihilates
school directions. The vectors $V_iP\mathbf z$ and $V_jP\mathbf y$ are generally
**not** school-centered. That is harmless in the trace term, which only takes
the dot product $\mathbf z_j^\top V_iP\mathbf z_j$. In the AI step,
$V_jP\mathbf y$ is passed to another solve, so condition 1 is what projects it.
The $V_i$ operators themselves must stay unprojected.

These conditions define when demeaning is exact:

| Change | Status |
|---|---|
| Residual heteroscedasticity: known weights, $R=\sigma_e^2\operatorname{diag}(w)^{-1}$ | Exact only with **weighted** demeaning, $Q_R\propto W_e-W_eH_S(H_S^\top W_eH_S)^{-1}H_S^\top W_e$, where $W_e=\operatorname{diag}(w)$ |
| Residual heteroscedasticity: weights depending on variance parameters | Exact projection depends on those parameters and must be recomputed when they change |
| Correlated residuals, such as an R-side school or class covariance | Simple demeaning is **not** exact; $Q_R$ is a dense-within-school GLS projection, and the error-contrast covariance becomes $K^\top RK$ |
| Random school effect retained alongside fixed school effects | Not a demeaning error, but $M_SZ_{\text{school}}=0$, so its variance is unidentified (§7.5) |

**Approximation note:** with inexact CG, the operator, right-hand side, and
residual must use the same projection. Otherwise the identity
$H_S^\top P\mathbf v=0$ fails beyond CG tolerance, which is the check in
invariant 2 below.

REML degrees of freedom are $n-\operatorname{rank}(\tilde X)$: about $S$
fewer, and fewer still if any schools contain only one observation. No current
code path uses this quantity. It matters only if a log-likelihood or
closed-form residual-variance update is added later.

Invariants to state and test before implementing:

1. The reduced operator is symmetric positive definite whenever the reduced
   `X` has full column rank, because $\Lambda$'s genetic and environment blocks
   are positive definite and $W^\top M_S W$ is positive semidefinite.
2. Every $P\mathbf v$ satisfies $H_S^\top P\mathbf v = \mathbf 0$ (residuals sum
   to zero within each school), up to CG tolerance.
3. The threaded and non-threaded paths give identical results within solver
   tolerance.

### 7.5 Interaction with existing model features

- **School random environment factor:** must be removed when school fixed
  effects are used; otherwise its $Z_h$ is exactly in the column space of
  $H_S$, and $\sigma_{c_{\text{school}}}^2$ is unidentified, drifting toward
  the variance floor.
- **Classroom and household random effects:** classrooms nested in schools
  remain identified from within-school between-class variation. Households
  that span schools are only partially demeaned; this is fine.
- **Social design $Z_s$ and dilution:** with summed peer exposure, school
  demeaning removes school-mean exposure but retains variation from class size
  and classroom composition. With mean exposure, there is less within-school
  variation in exposure magnitude. Identification of $\sigma_s^2$ may weaken
  in either case and must be checked in simulation.
- **Single-class schools (sharp case):** if school and class coincide, and
  exposure is a sum, then
  $(Z_s\mathbf a_s)_i = T_g - a_{s,i}$, where $T_g$ is the class total. The
  total $T_g$ is school-constant and absorbed, leaving
  $M_S(\mathbf a_d - \mathbf a_s)$. Such schools inform only the combination
  $\operatorname{Var}(a_d-a_s)=\sigma_d^2+\sigma_s^2-2\sigma_{ds}$ through
  phenotypes. Any separation then relies on pedigree links to other schools.
  Separate identification of $\sigma_s^2$ therefore comes mainly from schools
  with **several classes or cohorts**, as in peer-effect designs using school
  fixed effects and within-school random class assignment. Before choosing
  between school-level and school×cohort fixed effects, report the share of
  observations in single-class schools.
- **Peer environment effect:** enters through $Z_s$ and is projected in the same
  way; no separate treatment is needed.
- **Missing outcomes:** $H_S$ and $D_S$ use observed rows only, consistent
  with the observed-data REML formulation. Unobserved roster members still
  contribute genetically through $Z_s$ and the pedigree.

### 7.6 Preconditioning and CG convergence

The block preconditioner approximates the genetic block by
$(G\otimes A)/\sigma_e^2$, i.e. it is prior dominated and ignores the data term.
Its genetic part is therefore unchanged by absorption. Its fixed-effect diagonal
should be computed from the projected covariates,
$\operatorname{diag}(X^\top M_SX)$, rather than $\operatorname{diag}(X^\top X)$.
Within-school variances can be much smaller than raw sums of squares, so this
is a correction, not cosmetic. For the diagonal preconditioner,
$\operatorname{diag}(Z_d^\top M_S Z_d)_i = 1-1/n_{s(i)}$ for observed $i$, a
small correction that can be ignored or inserted cheaply. The environment
diagonal changes from `counts[j]` to
$n_j-\sum_s n_{js}^2/n_s$, where $n_{js}$ is the number of observations in
environment level $j$ and school $s$. For nested classrooms, this is
$n_j(1-n_j/n_{s(j)})$. Both corrections, and the $Z_s$ diagonal, cost one
$O(n)$ pass.

The main convergence concern is the *spectrum*, not the preconditioner. After
demeaning, school-mean genetic directions lose data information and are
controlled mainly by the prior. That often improves rather than worsens
conditioning, because those directions were previously strongly coupled to
the intercept and school environment. This is a hypothesis to verify, not a
guarantee: compare CG iteration counts at identical variance components with
and without absorption.

### 7.7 Recovering school effects and fixed-effect inference

- $\hat{\boldsymbol\alpha}=D_S^{-1}H_S^\top(\mathbf y - W\hat{\boldsymbol\theta})$
  is one $O(n)$ pass after the final solve. Store it separately from the MME
  solution, because it is not part of the CG unknowns.
- The current `fixed_effect_vcov!` would still apply to the remaining
  within-school-identified columns of `X`, using one CG solve per column
  through the reduced operator.
- The full covariance of $\hat{\boldsymbol\alpha}$ would require thousands of
  solves and is probably unnecessary. If needed, compute specific contrasts or
  Hutchinson-type diagonal estimates.
- The intercept is no longer separately estimated; any overall mean is part of
  $\hat{\boldsymbol\alpha}$.

### 7.8 Suggested validation sequence

These are bounded checks, each covering one property:

1. **Operator identity (small problem):** form the explicit dense Schur
   complement $W^\top M_S W+\Lambda$ and compare its matvec with the
   matrix-free absorbed operator for random vectors. Also check
   $\mathbf u^\top C\mathbf v=\mathbf v^\top C\mathbf u$.
2. **Absorption versus explicit representation:** on a small simulation,
   fit Option B (explicit zero-prior school block) and Option C. REML
   estimates, $P\mathbf y$, and the remaining $\hat{\boldsymbol\beta}$ should
   agree within CG and Hutchinson tolerance, ideally with fixed probe seeds.
   As a negative control, a fit that demeans only $\mathbf y$ but keeps the
   unprojected operator should **not** agree. This confirms that the test can
   detect an incomplete projection.
3. **Projection invariant:** check
   $\max_s\lvert(H_S^\top P\mathbf y)_s\rvert\approx 0$.
4. **Recovery under sorting:** extend the §6 simulation with sorting into
   schools by genetic composition; compare independent school random effects,
   school fixed effects, and the latent Mundlak extension on recovery of
   $\sigma_d^2,\sigma_s^2,\sigma_{ds}$.
5. **Threaded equivalence and CG cost:** compare threaded and non-threaded
   runs, and inspect `aireml_log.csv` for iteration counts and CG convergence
   relative to the current school-random-effect model.

### 7.9 Summary recommendation

Implement school fixed effects by **absorption**: within-school demeaning of
the full $\hat{\mathbf y}$ in a reordered operator and of residuals in
$P\mathbf v$, demeaning in `build_rhs!`, a
per-thread length-$S$ buffer, removal of school-constant columns of `X`, and
post hoc recovery of $\hat{\boldsymbol\alpha}$. This adds negligible memory
and $O(n)$ work per matvec and keeps the MME dimension unchanged. Under
independent homoscedastic residuals ($R=\sigma_e^2I$), it reproduces exactly,
up to CG and Hutchinson error, the REML of the explicit fixed-effects model.
That holds only if the operator, every right-hand side, and every $P\mathbf v$
residual use the same projection (§7.4). Weighted residuals would require
weighted demeaning; correlated residuals would require a GLS projection.
Use the explicit zero-prior block as a validation reference.

---

To view the equations in VS Code, open this Markdown file and run
**Markdown: Open Preview to the Side**. The equations use `$...$` and
`$$...$$` math delimiters supported by VS Code's built-in Markdown preview.
