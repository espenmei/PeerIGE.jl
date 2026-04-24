# Peer Effects Through an Indirect Genetic Effects (IGE) Framework

## Purpose Of This Document

This document is a working outline for describing the IGE approach to peer effects.

Goals for this draft:
1. Outline the statistical and methodological approach.
2. Contrast the approach with the linear-in-means model.
3. Explain what is solved computationally and why.

This is a base structure and is intentionally incomplete. Additional sections can be added as the project evolves.

---

## 0. Background And Motivation

### 0.1 Why Peer Effects In Education Matter

Education is paramount both for individuals and society. Educational achievement is a strong predictor of health, income, and social outcomes across the life course, and it has been estimated to account for a substantial share of cross-national variation in economic growth. Understanding the determinants of educational achievement therefore has both scientific and policy importance.

Peer effects — the influence that individuals have on others in their social environment — are central to most theories of human development. Most people intuitively recognize that the peers they grew up with mattered for who they became. Yet, despite decades of empirical research, fundamental questions about peer effects in education remain unresolved: How large are they? Through what mechanisms do they operate? Who is most sensitive to peer influences?

### 0.2 Four Core Challenges In Peer Effects Research

Four interrelated methodological challenges have prevented clear answers:

1. **Measurement.** An often overlooked but fundamental challenge is how to operationalize peer effects. The dominant approach in economics and sociology identifies specific observable peer characteristics — gender composition, mean prior test score, share with behavioral problems — and regresses outcomes on these. This *fixed characteristics* approach implicitly assumes that the most influential peer characteristics have been identified and measured. In practice, the features through which peers influence each other are numerous, subtle, and often unobserved. Limiting analysis to one or two observable peer variables captures only a narrow slice of the true peer influence.

2. **Confounding and stratification.** Demonstrating peer effects requires ruling out other explanations for outcome similarity among peers. Students are not assigned to classrooms at random; families sort across schools and neighborhoods based on socioeconomic position, ethnicity, and preferences. This creates correlation between peer composition and unobserved background factors, making naive comparisons misleading. Random assignment to classrooms, natural experiments, and genetic instruments have each been used as identification strategies.

3. **The reflection problem.** When peer influence is modeled through peers' outcomes — as in the linear-in-means model — the simultaneity of influence creates an identification problem known as the reflection problem (Manski, 1993): it is difficult to distinguish own influence on peers from peer influence on oneself because both processes generate the same correlational patterns.

4. **Sensitivity and heterogeneity.** The workhorse linear-in-means model assumes that every student responds identically to a given unit of peer input. Theoretical models from psychology and genetics suggest that individuals differ substantially in environmental sensitivity, but this heterogeneity has received little attention in the education literature.

### 0.3 A New Framework: Peer Effects As Individual Properties And Complex Traits

The IGE approach to peer effects offers a fundamentally different conceptual starting point, grounded in two central propositions:

**Peer effects are general qualities of the individual.** A peer effect, properly construed, refers to the causal influence that a focal individual *exerts on others*, not to any one specific behavior or attribute of theirs that has been measured. In the traditional fixed characteristics approach, the effect of peers is represented by the group mean of a particular variable (e.g., mean test score, share male). The proposed framework instead represents each individual's total peer-effect potential as a single latent quantity — a comprehensive summary of all the ways the person influences those around them. This sidesteps the measurement problem entirely.

**Peer effects are complex traits.** Social interactions have played a vital role in human evolution and development. The capacity to influence others — through behavior, motivation, norm-setting, cooperation — is almost certainly shaped by many genetic variants and environmental factors. This observation motivates treating peer effects as *complex traits* in the quantitative genetics sense: latent properties of individuals with both heritable and environmental components, studied at the population level through variance decomposition rather than through measurement of specific mechanisms.

Together these two propositions suggest connecting the education literature on peer effects to the quantitative genetics framework for indirect (social) genetic effects, which was developed to model exactly this situation: heritable latent effects of one individual on the phenotype of others.

---

## 1. Statistical And Methodological Approach

### 1.1 Core Idea

The IGE framework decomposes outcome variation into:
1. Direct genetic effects: the effect of an individual's own genotype on their own outcome.we should add to our outline 
2. Indirect (social) genetic effects: the effect of peers' genotypes on the individual's outcome.
3. Shared environmental effects: group-level context components (for example household, classroom, school).
4. Residual variation.

A compact representation is:

$$
\mathbf{y} = X\beta + Z_d a_d + Z_s a_s + \sum_{h=1}^H Z_h c_h + e
$$

with

$$
\begin{pmatrix}a_d \\ a_s\end{pmatrix} \sim N\left(0,\, G \otimes A\right), \quad
G = \begin{pmatrix}\sigma_d^2 & \sigma_{ds} \\ \sigma_{ds} & \sigma_s^2\end{pmatrix}
$$

Key parameters:
1. $\sigma_d^2$: direct genetic variance.
2. $\sigma_s^2$: social genetic variance.
3. $\sigma_{ds}$: direct-social covariance.
4. $\sigma_e^2$: residual variance.
5. $\sigma_{c_h}^2$: shared environment variances.

### 1.2 Peer Effects As A Latent Variable

An important conceptual point is that this approach treats peer effects as a **latent construct** rather than as a single directly observed peer behavior. This connects to the two central propositions laid out in Section 0.3: peer effects are general individual-level properties, and they are complex traits subject to genetic and environmental decomposition.

The closest antecedent in the psychology literature is the *social relations model* (Kenny & La Voie, 1984; Back & Kenny, 2010), which partitions outcome variation in round-robin social interaction designs into actor effects, partner effects, and relationship-specific residuals. The partner effect in the social relations model — the consistent contribution a given person makes to the outcomes of all those who interact with them — is precisely what the IGE framework calls a peer effect. The present framework extends this logic from small round-robin laboratory studies to population-scale register data by replacing round-robin assignment with pedigree-based covariance modeling.

Under this view, the peer environment is not reduced to one measured peer characteristic or one peer mean. Instead, it is conceptualized as the combined influence of many peer attributes and behaviors that are at least partly heritable and that jointly shape the focal individual's outcome.

This gives a different interpretation of the social genetic component:
1. It captures the aggregate influence of many heritable peer behaviors.
2. It is not limited to one observed mechanism.
3. It represents a broad latent peer-effect signal transmitted through the peer group.

In that sense, the indirect genetic component can be interpreted as a type of **peer value-added measure**. It summarizes how much peers contribute to outcomes through the bundle of traits and behaviors they bring into the social environment, above and beyond the individual's own direct effects and the shared environmental terms included in the model.

This framing is useful because it shifts attention away from asking whether one specific peer variable explains outcomes, and toward estimating the total latent peer contribution embedded in the social setting.

A key practical advantage of this approach is that it requires **no knowledge or measurement of which peer characteristics actually matter**. In real social settings the features through which peers influence each other can be numerous, subtle, and intertwined — motivation, cooperation, disruptive behavior, academic norms, and so on. Identifying and measuring the relevant ones ex ante would be difficult or impossible. The IGE framework sidesteps this entirely: because the social genetic component is estimated from pedigree-based covariance structure rather than from observed peer traits, it captures the aggregate heritable contribution of whatever peer features matter, without requiring the researcher to enumerate or measure them. The complexity of the underlying mechanisms is absorbed into $\sigma_s^2$ and $\sigma_{ds}$.

### 1.3 Sum Versus Mean Specification

One point that needs to be made explicit is that this framework currently uses the **raw leave-one-out sum** of peer contributions, not the leave-one-out mean.

That is, the social component is conceptualized as:

$$
\sum_{j \in g_i,\, j \neq i} a_{s,j}
$$

rather than:

$$
\frac{1}{n_{g_i}-1}\sum_{j \in g_i,\, j \neq i} a_{s,j}
$$

as in the normalized linear-in-means specification.

This matters mainly for interpretation:
1. If group size is constant, the difference is mostly a scale normalization.
2. If group size varies, the two specifications imply different estimands.

The raw-sum specification treats peer influence as **total peer input**. Larger groups can generate a larger total social contribution because more peers contribute to the latent peer environment.

The normalized specification treats peer influence as **average per-peer input**. Group size does not mechanically scale the social term because the sum is divided by the number of peers.

For the present framework, the raw-sum specification fits the conceptualization of peer effects as a latent aggregate of many heritable peer behaviors. It is therefore consistent with the interpretation of the indirect genetic component as a latent peer value-added measure.

At the same time, this is a substantive modeling choice rather than a harmless normalization whenever group size varies. The final write-up should therefore make clear that the baseline model targets **total peer influence**, not average peer influence.

### 1.4 Note On Bijma's Dilution Parameter

It may also be useful to relate this discussion to **Bijma's dilution parameter** from the indirect genetic effects literature.

The basic idea is to allow peer influence to scale with group size through a parameterized dilution rule, for example:

$$
\frac{1}{(n_{g_i}-1)^d}\sum_{j \in g_i,\, j \neq i} a_{s,j}
$$

where $d$ governs how strongly peer contributions are diluted as group size increases.

Useful benchmark cases are:
1. $d = 0$: no dilution, corresponding to the raw-sum specification.
2. $d = 1$: full dilution, corresponding to the leave-one-out mean.
3. $0 < d < 1$: partial dilution, where larger groups matter more than in the mean specification but less than in the raw-sum specification.

This is useful conceptually because it shows that the current model and the linear-in-means normalization can be viewed as two endpoints of a broader class of peer-effect specifications.

This may be valuable later as:
1. A robustness analysis.
2. A way to connect the IGE framework more directly to existing peer-effects specifications.
3. A way to test whether peer influence behaves more like total exposure or average exposure.

### 1.5 What Genetic Structure Adds Over Latent Peer Effects

A natural question is what is gained by modeling peer effects through genetic structure, relative to treating peer effects as a purely latent environmental construct — for example a random pen effect.

A random group effect absorbs shared environmental variation within pens, including any social influence that is common to pen members. This already captures *something* about peer context, but it does not distinguish between:

1. The influence peers have *on each other* through their heritable traits and behaviors.
2. Shared environmental conditions that affect all pen members in common.

The genetic structure in the IGE model — specifically the contrast between $\mathbf{ZZ}'$ (pen membership) and $\mathbf{ZAZ}'$ (peer genetic relatedness) — provides the lever to separate these two sources. Because $\mathbf{A}$ encodes pedigree relatedness, animals in different pens can share social genetic signal through ancestry, even when they share no pen environment. This cross-pen genetic covariance, which $\mathbf{ZAZ}'$ introduces but a latent pen effect cannot, is the basis for identification.

More concretely, the genetic IGE framework makes the following estimable that a latent-peer-effect model does not:

1. **Separation of direct and social genetic effects.** Because the direct effect travels through $\mathbf{A}$ while the social effect travels through $\mathbf{ZAZ}'$, the model can attribute variance to each component separately.

2. **The direct–social genetic covariance $\sigma_{ds}$.** This covariance describes whether animals with high direct genetic merit tend to also have social genotypes that help or hinder their pen mates. A positive $\sigma_{ds}$ means the same genetic endowment that raises your own outcome also tends to raise others'; a negative $\sigma_{ds}$ suggests a competitive architecture. This parameter is only estimable because direct and social components have distinct covariance signatures in the data.

3. **Total heritable variance including social feedback.** The total heritable variance in a socially structured population is not $\sigma_d^2$ alone but depends on $\sigma_d^2$, $\sigma_s^2$, $\sigma_{ds}$, and group size — a quantity sometimes called the total breeding value variance. Recovering this requires both components.

In short: a latent pen effect models *what pens share in common*. The IGE model additionally models *how heritable peer characteristics transmit across and within pens through ancestry*, which is what makes the direct–social decomposition and their covariance identified.

### 1.6 Identification Logic

Identification relies on non-collinear variation across:
1. Pedigree relatedness structure.
2. Peer-group composition.
3. Shared environment groupings.

In practice, this means we need enough independent variation to separate:
1. Social genetic signal from shared environmental clustering.
2. Direct and social genetic components.

#### 1.6.1 Variation in group size as an additional lever

Unequal group sizes provide a further identification argument, because the social genetic and the shared environment components imply different variance structures as group size changes.

Under the IGE model, the phenotypic variance of individual $i$ in a group of size $n_g$ contains a term proportional to $(n_g - 1)\sigma_s^2 + 2(n_g-1)\sigma_{ds}$: the social genetic contribution grows with the number of peers. A shared pen effect $c_h \sim N(0, \sigma_c^2)$, by contrast, adds $\sigma_c^2$ to every pen member regardless of pen size.

This means that if pen sizes vary, the two components imply *different patterns of residual heteroskedasticity across groups*:

1. Under a pure shared environment model, residual variance is homogeneous across group sizes (conditional on fixed effects).
2. Under a social genetic model, residual variance increases with group size because larger pens accumulate more peer genetic signal.

Variation in group size therefore provides an additional empirical lever to separate peer effects from common pen effects, above and beyond the cross-pen pedigree signal. Groups of different sizes generate different predicted variances under the social genetic model, and that pattern is not explained by a common environment term alone.

This is worth keeping in mind for both the identification argument and the simulation design: including variation in pen size strengthens the case for identifiability.

### 1.7 Estimation Strategy

Variance components are estimated with AI-REML, using:
1. Matrix-free mixed model operators.
2. Preconditioned conjugate gradient linear solves.
3. Hutchinson trace estimation for intractable traces.

This makes the approach scalable to very large pedigrees and large observed samples.

---

## 2. Contrast With The Linear-In-Means Model

### 2.1 Typical Linear-In-Means Setup

A canonical reduced-form peer model is often written as:

$$
y_i = \alpha + \rho \bar y_{g,-i} + \gamma x_i + \delta \bar x_{g,-i} + u_i
$$

where $\bar y_{g,-i}$ is peers' mean outcome and $\bar x_{g,-i}$ peers' mean covariate.

### 2.2 Core Differences

1. Source of peer variation:
- Linear-in-means uses peers' observed outcomes/covariates.
- IGE uses pedigree-based random effect decomposition to model latent social-genetic transmission.

2. Reflection problem:
- Linear-in-means must address simultaneity and reflection explicitly.
- IGE shifts focus to covariance decomposition using relatedness and structured random effects.

4. Conceptualization of peer effects:
- Linear-in-means typically operationalizes peer effects through observed peer averages.
- IGE can be framed as a latent peer-effects model, where peer influence reflects the combined contribution of many heritable behaviors rather than one observed peer regressor.

5. Causal interpretation target:
- Linear-in-means often targets average peer outcome effects.
- IGE targets variance components and covariance structure of direct vs indirect genetic architecture.

6. Treatment of unobserved heterogeneity:
- Linear-in-means depends heavily on controls/instruments/fixed effects.
- IGE explicitly models genetic covariance through $A$ and shared environment through $Z_h c_h$ terms.

### 2.3 What Is Comparable Across The Two

1. Both aim to describe social interaction effects.
2. Both require strong assumptions for causal interpretation.
3. Both can be biased by correlated environments and sorting if not modeled well.

### 2.4 Suggested Framing In The Final Write-Up

1. Linear-in-means as a mean-effect regression framework.
2. IGE as a variance-decomposition framework for latent social-genetic transmission.
3. Emphasize that the IGE object can be viewed as a peer value-added measure at the latent-variable level.
4. Clarify that the estimands are different, so one is not a direct drop-in replacement for the other.

---

## 3. What We Solve (Computationally And Statistically)

### 3.1 Statistical Problem

Given observed outcomes, covariates, pedigree structure, and peer grouping, estimate:
1. Variance components $\theta = (\sigma_d^2, \sigma_s^2, \sigma_{ds}, \sigma_e^2, \sigma_{c_1}^2, \ldots, \sigma_{c_H}^2)$.
2. Fixed effects $\beta$.
3. BLUPs for direct and social genetic effects (and environment effects when included).

### 3.2 Computational Problem

At each AI-REML iteration, solve large linear systems from Henderson mixed model equations without forming the dense full system explicitly.

Core operations:
1. Repeated MME solves for $P y$, $P z_j$, and AI terms.
2. Matrix-vector products with implicit operators.
3. Approximate trace terms via Monte Carlo probes.

### 3.3 Why This Is Hard

1. Problem size: pedigree dimension can be in the millions.
2. Dense relationship matrix $A$ is not explicitly formable.
3. AI-REML requires repeated solves and stable numerical updates.

### 3.4 What The Current Implementation Delivers

1. Sparse and matrix-free representation using $A^{-1}$ and sparse Cholesky factors.
2. Threaded implementation for key iteration steps.
3. Numerical safeguards for positive-definite variance updates.

---

## 4. Optional Sections To Add Later

1. Assumptions and threat-to-identification checklist.
2. Missing outcomes and partial-observation handling (see note below).
3. Interpretation guide for $\sigma_{ds}$ and sign constraints.
4. Simulation design and parameter-recovery diagnostics.
5. Real-data workflow and reporting template.
6. Sensitivity analyses and robustness checks.

### 4.1 Note On Missing Outcomes And Group Size

**Priority: implement before any real-data application.**

Handling missing outcomes in the IGE model requires care because the observation space and the group membership space are not the same thing.

In a standard mixed model, dropping animals with missing outcomes from the data is straightforward: absent animals simply do not contribute to the likelihood. In the IGE model, however, every pen member contributes a social genetic effect $a_{s,j}$ to the outcome of each pen-mate, regardless of whether that animal is itself observed. Dropping an unobserved animal from the group structure therefore does two things incorrectly:

1. It reduces the effective pen size, biasing the social genetic variance components that depend on group size.
2. It omits the unobserved animal's social genetic value from the model, so the social contribution to observed pen-mates is understated.

The correct treatment separates three roles:

| Role | Who qualifies |
|---|---|
| In pedigree — contributes to **A** | All animals, observed or not |
| In group — social effect acts on pen-mates | All pen members, observed or not |
| Has phenotype — row in **y** and **X** | Observed animals only |

**Workaround until fixed:** drop rows with missing `y` from `yX` before running the model. This conflates the second and third roles and underestimates effective group size, but is otherwise safe for datasets with negligible missingness.

#### Minimal-change design for the fix

`GroupStructure` already separates the two concerns internally:
- `members_id` / `id_to_group` — pedigree-level pen membership (used in `apply_ZsZs_accumulate!` and the social sum loop).
- `obs_to_id` — maps observation rows to pedigree IDs (used in `apply_Zs_transpose!` and the RHS assembly).

The fix requires no changes to the MME operator or AI-REML code. Only `GroupStructure` and its constructor need to change, by accepting an optional separate full-roster input:

```julia
GroupStructure(obs_group_ids, obs_animal_ids, N_pedigree;
               full_group_ids = nothing, full_animal_ids = nothing)
```

If `full_*` is supplied, `members_id` and `id_to_group` are built from the complete pen roster; `obs_to_id` is still built from the observed subset only. If `full_*` is omitted, behavior is identical to today (backward-compatible). The call site in `run_emai.jl` passes an optional pen-roster CSV alongside `yX`.

---

## 5. Minimal Next Draft Tasks

1. Decide target audience: methods appendix vs main text methods section.
2. Add one worked toy example with direct and social variance components.
3. Add a concise figure showing direct path, peer path, and shared environment path.
4. Add a short subsection on what can and cannot be claimed causally.
