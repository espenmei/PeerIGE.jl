# Assortative mating in the pedigree SGE model

**Status:** Research note. Nothing here is implemented in `src/`. The note
collects the theory, our empirical diagnostics, and candidate model extensions.
Each candidate must be validated by simulation before it is used for inference.

The baseline model is the direct-plus-social model in
[the computational approach](../computational_approach.md).

## 1. Why this matters for education outcomes

Pedigree REML on Norwegian education outcomes gives a direct heritability of
roughly 0.8–0.9. This is well above twin and family estimates for comparable
traits. Spousal correlations for educational attainment are among the highest
for any human trait. Assortative mating (AM) is therefore a leading candidate
for part of the excess, together with geography, population structure, and
family-level environment.

The standard numerator relationship matrix $A$ assumes that parents are
genetically uncorrelated unless they share ancestors in the pedigree. Under AM,
that assumption fails for every couple. The failure compounds over generations.
It also affects distant relatives more than close ones. A model using the full
pedigree, including many distant pairs, is therefore exposed more than a
sibling or twin design.

## 2. What the correlation diagnostic shows

These values come from `application/diagnose_am.R` (a local script, not under
version control), run on `edu10_MAT_checked`. Rerun the script to refresh them.
$r$ is the residual phenotypic correlation and $a$ is the expected additive
relationship.

| Relationship | $r/a$ |
| --- | --- |
| Full siblings | 0.72 |
| Full first cousins | 1.19 |
| Full second cousins | 2.04 |
| Avuncular | ≈ 0.65 |
| Maternal / paternal half-siblings | 0.96 / 0.73 |
| Cousin ratio, mothers' sisters / fathers' brothers (MM/FF) | 1.24 |

Weighted regression of $r$ on $a$, with weights $1/\mathrm{SE}(r)^2$, under a
simple $G+E$ model:

| Fit | All households | Different households |
| --- | --- | --- |
| Through the origin, $h^2$ | 0.75 | 0.87 |
| With intercept, $h^2$ ($c$) | 0.61 (0.057) | 0.59 (0.058) |

Interpretation:

- Under $A$, $r/a$ should be roughly constant across degrees of relatedness.
  It instead **rises** with distance. Phenotypic correlation falls by about
  0.64 per meiosis rather than 0.5.
- A simple AM path model fits these correlations reasonably well, with spousal
  genetic correlation $r_A \approx 0.29$ and $h^2 \approx 0.56$.
- The intercept fit gives a lower $h^2$ than the fit through the origin. This
  is consistent with distant relatives pulling the slope up. The intercept is
  an extrapolation, not an identified shared-environment component.
- **Not explained by AM:** the maternal-side excess (half-siblings and MM/FF
  cousins). AM is symmetric between the sexes. Maternal effects or maternal-line
  environmental sharing are more plausible explanations.
- Pair counts are not REML weights. Check `am_h2_weights` before concluding
  which relationship classes drive the pedigree estimate.

## 3. Theory: copaths, Pearson–Aitken and the Schur complement

### 3.1 Selection on a multivariate normal

Partition the latent variables $L$ (genetic and environmental values) and the
mate-choice variables $P$ (phenotypes):

$$
\Sigma =
\begin{pmatrix}
\Sigma_{LL} & \Sigma_{LP}\\
\Sigma_{PL} & \Sigma_{PP}
\end{pmatrix}.
$$

Mate choice changes $\Sigma_{PP}$ to $\Sigma_{PP}^*$, adding a correlation
$\mu$ between the spouses. The regression of $L$ on $P$ and the conditional
covariance (the Schur complement)
$\Sigma_{LL} - \Sigma_{LP}\Sigma_{PP}^{-1}\Sigma_{PL}$ are assumed unchanged.
This gives the Pearson–Aitken selection formulas:

$$
\Sigma_{LP}^* = \Sigma_{LP}\Sigma_{PP}^{-1}\Sigma_{PP}^*,
\qquad
\Sigma_{LL}^* = \Sigma_{LL}
- \Sigma_{LP}\Sigma_{PP}^{-1}\,(\Sigma_{PP} - \Sigma_{PP}^*)\,
  \Sigma_{PP}^{-1}\Sigma_{PL}.
$$

Cloninger's copath rules for path diagrams are a graphical shorthand for this
calculation. They have two rules:

- A chain crosses at most one copath.
- After crossing a copath, the chain must follow the arrows forward.

### 3.2 Two spouses

Let $P_i = aA_i + eE_i$ with standardized variables, and let
$\operatorname{Cor}(P_1, P_2) = \mu$ after mating. Then

$$
\operatorname{Cov}(A_1, A_2) = a^2\mu,\qquad
\operatorname{Cov}(A_1, E_2) = ae\mu,\qquad
\operatorname{Cov}(E_1, E_2) = e^2\mu.
$$

Variances within each spouse are unchanged, and
$\operatorname{Cov}(A_1, E_1) = 0$. The cross term $ae\mu$ matters for us. It
means a child's genes become correlated with the other parent's environment,
which may be transmitted as family environment. AM therefore creates
gene–environment correlation in the offspring generation. A model without
family-environment transmission will attribute this to $A$.

### 3.3 The next generation and equilibrium

Write $r_A = \operatorname{Cor}(A_m, A_f)$ for the spousal genetic correlation.
Under direct phenotypic AM, $r_A = h^2\mu$. With $a_k = \tfrac12(a_m + a_f)
+ \varepsilon_k$, the offspring variance is

$$
\operatorname{Var}(a_k) = \tfrac14 V_m + \tfrac14 V_f
+ \tfrac12 \operatorname{Cov}(a_m, a_f) + \operatorname{Var}(\varepsilon_k).
$$

In the infinitesimal (Bulmer) model, the Mendelian sampling variance is
$\operatorname{Var}(\varepsilon_k) = \tfrac12\sigma_0^2(1 - \bar F)$, where
$\sigma_0^2$ is the random-mating genic variance. AM does not change this. All
of the extra variance comes from the covariance between parents. At equilibrium:

$$
V = \frac{\sigma_0^2}{1 - r_A},\qquad
\operatorname{Cor}_A(\text{full sibs}) = \operatorname{Cor}_A(\text{parent, offspring})
= \tfrac12(1 + r_A).
$$

Heuristically, every parent–offspring link carries a factor
$\tfrac12(1 + r_A)$ instead of $\tfrac12$, so the excess grows geometrically
with the number of meioses. A decay of 0.64 per meiosis corresponds to
$r_A \approx 0.28$. This agrees with the path-model fit in Section 2, but it
applies the genetic decay to phenotypic correlations and ignores environmental
sharing.

**Units matter.** If $\sigma_a^2$ is parameterized as the current
(equilibrium) genetic variance $V$, then the Mendelian sampling variance in
units of $V$ is $\tfrac12(1 - r_A)(1 - \bar F)$, not $\tfrac12(1 - \bar F)$.
"AM leaves $d_i$ unchanged" holds only in units of $\sigma_0^2$.

## 4. Consequences for pedigree REML

AM causes two separate problems:

1. **Real extra genetic variance.** The AM-induced variance is genuine genetic
   variance in the current generation. Including it in $h^2$ is not a bias as
   such, but it is not transferable to a random-mating population.
2. **Misspecified covariance pattern.** $A$ under-predicts the covariance of
   distant relatives relative to close relatives. REML then compromises across
   relationship classes. The net effect on $\hat\sigma_d^2$, $\hat\sigma_s^2$,
   $\hat\sigma_{ds}$ and the environmental components depends on which classes
   carry the information. The direction and size of the bias must be
   established by simulation, not assumed.

**Cross-trait AM.** Parents mostly choose partners on their own education,
income, and social background, not on their children's school outcomes. The
relevant quantity is the spousal correlation of the genetic values *for the
trait being analysed*. This is a cross-trait quantity. The equilibrium
constraint $r_A = h^2\mu$ does not apply. We should estimate or fix $r_A$
directly, possibly per generation, rather than trying to separate $h^2$ and
$\mu$.

**Social genetic effects.** AM also acts on $a_s$ through cross-trait
correlation. The simplest extension is
$\operatorname{Var}\begin{pmatrix}a_d\\ a_s\end{pmatrix} = G \otimes A^*(r_A)$,
with the same $r_A$ for both effects. This assumption is not motivated by
biology. It only keeps the model tractable.

## 5. Building an AM-aware relationship structure

Constraints from the project: never form $A$, $A^*$ or the MME coefficient
matrix explicitly. Hutchinson trace probes and the preconditioner rely on a
sparse factor of the precision matrix, so any change to the precision matrix
must keep its factor sparse.

### 5.1 What does not work: adjusting $d_i$ only

The Henderson/Quaas decomposition
$A^{-1} = (I - T)^\top D^{-1}(I - T)$ encodes a directed acyclic graph in
which parents are correlated only through shared ancestors. Changing the
Mendelian sampling variances $d_i$ changes the offspring variances but never
creates a covariance between spouses. It therefore cannot produce the in-law
paths or the inflated correlations between distant relatives. Changing $d_i$
is needed only for the units issue in Section 3.3.

### 5.2 Founder couples: a latent couple node (exact, sparse)

For a couple whose members are both founders, add a latent node $w_c$:

$$
a_m = \sqrt{r_A}\,w_c + \sqrt{1 - r_A}\,\varepsilon_m,\qquad
a_f = \sqrt{r_A}\,w_c + \sqrt{1 - r_A}\,\varepsilon_f.
$$

This is still a directed acyclic graph. The augmented precision matrix stays
sparse, with one extra node per couple. Variances are preserved and the
spousal correlation is exactly $r_A$. Descendants follow the usual recursion.

Limitations:

- It requires $r_A \ge 0$.
- A founder with several partners needs a different construction, such as one
  node per founder or correlated couple nodes.
- It only covers founders. In a deep pedigree, most couples are non-founders.

### 5.3 Non-founder couples: AM as conditioning ($Q_\tau$)

For spouses whose values are already determined by their own parents,
correlation can only be induced by conditioning. A sparse Gaussian version
treats each couple $c = (m, f)$ as a pseudo-observation
$0 = a_m - a_f + \nu_c$, with $\nu_c \sim N(0, \tau\sigma_a^2)$:

$$
Q_\tau = A^{-1} + \frac1\tau \sum_{c}(e_m - e_f)(e_m - e_f)^\top,
\qquad \operatorname{Var}(a) = \sigma_a^2\,Q_\tau^{-1}.
$$

Properties:

- $Q_\tau$ adds one off-diagonal entry per couple and stays sparse.
- For two unrelated founder spouses with unit variance, the result is
  $\operatorname{Cor}(a_m, a_f) = r = 1/(1 + \tau)$, and each variance is
  shrunk to $1/(1 + r)$.
- **This is not Pearson–Aitken.** Conditioning shrinks spousal variances
  instead of preserving them, and it also induces correlation among the
  spouses' ancestors. The parameter $\tau$ therefore needs calibration
  against $r_A$ by simulation.
- Marginal variances could be restored with a diagonal rescaling
  $D^{-1/2} Q_\tau D^{-1/2}$, which preserves sparsity. This requires
  $\operatorname{diag}(Q_\tau^{-1})$, which can be obtained by selected
  inversion or stochastic estimation. That cost is significant but feasible.
- **A sparse $Q_\tau$ does not mean a sparse factor.** Spousal edges create new
  cycles in the pedigree graph and cause fill-in in the Cholesky factor. This
  must be measured on the real 7M-node pedigree before any implementation.
- $\tau$ (or $r_A$) enters the precision matrix nonlinearly. It is not a simple
  scale factor like the other variance components, so the AI-REML derivative
  becomes $\partial V/\partial\tau = -\sigma_a^2 Q_\tau^{-1}
  (\partial Q_\tau/\partial\tau) Q_\tau^{-1}$. A profile over a grid of $\tau$
  values is the simpler first approach.

## 6. Competing explanations

These produce similar "distant relatives are too alike" patterns and must be
ruled out or modelled alongside AM:

- **Geography and population structure.** Shared region, municipality, or
  school catchment inflates correlations at every distance, and the inflation
  is relatively largest for distant pairs. This is the first thing to control
  for.
- **Social homogamy and family transmission.** Partners selected on social
  background, or an environmental effect transmitted down family lines. The
  family-transmission model in the [roadmap](../ROADMAP.md) with $\rho/2 \le
  0.5$ cannot by itself produce a decay above 0.5 per meiosis. With correlated
  spousal environments it could.
- **Maternal effects or maternal-line environment.** These are needed to
  explain the maternal-side excess, which AM cannot produce.
- **Household and peer environment.** These already enter the model as
  `env_cols` and peer environment, but they only cover close relatives.

## 7. Identification and data

- **Affinal pairs with $a = 0$.** The most direct AM test among phenotyped
  students is a pair whose only connection goes through a marriage. For
  example, $i$ and $j$ where $j$'s father is married to the sister of $i$'s
  mother, and $j$ is his child with another partner ("step-cousins"). Under
  $A$ their expected genetic correlation is 0; under AM it is about $r_A/8$.
  Shared environment through family contact is a confounder, but plausibly
  weaker than for blood cousins. Phenotypic AM and social homogamy both
  predict a non-zero correlation, so these pairs test whether marriage induces
  resemblance at all, not which mechanism causes it.
- **Parental education from the registers.** The observed spousal correlation
  of parents' education gives $\mu$ directly. Combined with an estimate of the
  genetic correlation between parental education and the children's outcome,
  it bounds $r_A$ independently of the pedigree fit.
- **Pedigree depth.** Refit on the same phenotyped individuals with the
  pedigree truncated to 1, 2 and 3 generations of ancestors. Under AM the
  estimate should fall as distant connections are removed.

## 8. Plan

1. Add geography/school residualisation to `diagnose_am.R`, plus affinal
   ($a = 0$) pair classes.
2. Run the pedigree-depth sensitivity analysis with AI-REML.
3. Write a forward simulation with known $r_A$ (and optionally
   social homogamy and maternal effects). Measure the bias of the current SGE
   REML on the real pedigree structure.
4. Only then implement and test an AM-aware precision matrix:
   - Measure the Cholesky fill-in of $Q_\tau$ on the real pedigree.
   - Profile over $\tau$.
   - Calibrate $\tau$ against $r_A$ by simulation.
   - Check that the threaded and non-threaded paths agree.

## 9. References

Standard references:

- Fisher, R. A. (1918). The correlation between relatives on the supposition of
  Mendelian inheritance. *Trans. R. Soc. Edinburgh*, 52, 399–433.
- Wright, S. (1921). Systems of mating. III. Assortative mating based on
  somatic resemblance. *Genetics*, 6, 144–161.
- Bulmer, M. G. (1971). The effect of selection on genetic variability.
  *American Naturalist*, 105, 201–211.
- Henderson, C. R. (1976). A simple method for computing the inverse of a
  numerator relationship matrix used in prediction of breeding values.
  *Biometrics*, 32, 69–83.
- Quaas, R. L. (1976). Computing the diagonal elements and inverse of a large
  numerator relationship matrix. *Biometrics*, 32, 949–953.
- Cloninger, C. R. (1980). Interpretation of intrinsic and extrinsic structural
  relations by path analysis: theory and applications to assortative mating.
  *Genetical Research*, 36, 133–145.
- Keller, M. C., et al. (2009). Modeling extended twin family data I:
  description of the Cascade model. *Twin Research and Human Genetics*, 12,
  8–18.
- Yengo, L., et al. (2018). Imprint of assortative mating on the human genome.
  *Nature Human Behaviour*, 2, 948–954.
- Border, R., et al. (2022). Assortative mating biases marker-based
  heritability estimators. *Nature Communications*, 13, 660.
- Border, R., et al. (2022). Cross-trait assortative mating is widespread and
  inflates genetic correlation estimates. *Science*, 378, 754–761.

Not yet verified (suggested by an AI assistant). Check before citing:

- Tallis & Leppard (1988), *Theor. Appl. Genet.* 75, on selection and AM for
  multiple polygenic traits.
- Gianola (1982), *Theor. Appl. Genet.* 62, on AM and the genetic correlation.
- Kennedy & Sorensen (1988) and Sorensen & Kennedy (1984), *J. Anim. Sci.*, on
  mixed-model methods under AM and selection.
- Keller, Medland & Duncan (2010), *Behavior Genetics* 40, 377–393, comparing
  extended twin family designs.
- Heath & Eaves (1985), *Behavior Genetics* 15, on separating phenotypic AM
  from social homogamy.
- A 2025 *Nature* paper on indirect assortative mating and its
  intergenerational consequences.

Cited incorrectly in the AI text and not about AM: Lo, Fernando & Grossman
(1993) is about covariance between relatives in multibreed populations.
