# PeerIGE.jl

Julia code for AI-REML estimation of direct and social genetic effects in large pedigree-based mixed models.

## What This Repository Contains

- `src/emai.jl`: thin façade that loads the model. The implementation is split across `src/emai_core.jl` (shared types and observed-data / group construction), `src/emai_precond.jl` (diagonal and block preconditioners), `src/emai_mme.jl` (matrix-free mixed model equations, CG solves, `V_i` operators), `src/emai_aireml.jl` (Hutchinson trace probes, score/AI assembly, and the `fit_aireml!` driver), `src/emai_show.jl` (model report with iteration diagnostics, printed after every AI-REML iteration and by `display(model)`), and `src/emai_post.jl` (post-fit `stochastic_loglik`, `loglik_difference` and `psi_score` for model comparison). Together they fit the full social genetic effects model with variance components (`σ2_d`, `σ2_s`, `σ_ds`, `σ2_e`) plus optional shared environment effects
- `run_emai.jl`: runnable entrypoint; fits both the base SGE model and the SGE+household model, prints a comparison table
- `run_emai_hpc.jl`: single-input HPC runner for `obs_table_<INPUT>.csv` / `roster_<INPUT>.csv` files and trimmed pedigree triplets
- `computational_approach.md`: math and algorithm overview
- `sim/`: simulation scripts in R (`sim_peer.R`, `sim_peer_evn.R`; `sim_peer_evn.R` adds household/classroom/school shared environment effects); each writes the two-table format (`yX_obs.csv`, `roster.csv`)
- `yX_obs.csv`: observed-only rows — one row per observed outcome, with columns `y`, `id_ped`, covariates, and optional environment IDs (e.g. `household`)
- `roster.csv`: full social group membership for all relevant pedigree IDs, with columns `id_ped` and `group` (includes unobserved animals)
- `Ainv_triplet.csv`, `LAinv_triplet.csv`: sparse pedigree matrix inputs in triplet format
- `bench_threads.jl`: threading benchmark (times a threaded run and reports per-iteration timing and final estimates)
- `tests`: test entrypoints and smoke checks for missing-response handling and H=0/H=1 models

## Folder Structure

- `src`: model implementation (`emai.jl` façade plus `emai_core.jl`, `emai_precond.jl`, `emai_mme.jl`, `emai_aireml.jl`, `emai_show.jl`, `emai_post.jl`)
- `sim`: R simulation scripts that generate the input CSVs
- `outputs/logs`: AI-REML and benchmark log CSVs
- `outputs/results`: JLD2 model result files
- `outputs/benchmarks`: benchmark CSV logs
- `outputs/smoke`: smoke-test logs
- `tests`: test scripts and the main test entrypoint (`runtests.jl`)
- `diagnostics`: one-off diagnostic scripts (`diag_*.jl`)

## Environment

This workspace uses a local Julia project:

```powershell
julia --project=.
```

Dependencies are tracked in `Project.toml` and `Manifest.toml`.

## Running The Main Model

Run the full model with:

```powershell
julia --project=. run_emai.jl
```

Run the test entrypoint after code changes with:

```powershell
julia --project=. tests/runtests.jl
```

The runner currently:

1. Loads observed phenotype/design data from `yX_obs.csv` and social group membership from `roster.csv`
2. Validates required columns and builds `GroupStructure` from the two-table input
3. Loads `Ainv` and sparse Cholesky inputs from triplet CSV files
4. Fits **Model 1**: base SGE (`σ2_d`, `σ2_s`, `σ_ds`, `σ2_e`)
5. Fits **Model 2**: SGE + household shared environment, and **Model 3**: SGE + household + classroom + school
6. Prints a side-by-side parameter comparison table
7. All runs are threaded — start with `--threads 8`

Saving examples in the runner are commented out for the offline HPC environment.

### Interactive R data preparation

`setup_edu_data.R` is an editable, cell-based copy of the education setup.
Configure its input paths and run the cells in order. It retains the screenshot
version's class-size filter, earliest education-year rule, whole-class year
exclusion, twin exclusion from outcomes, and fixed effects (no term-1 covariate).
By default it writes a separate `data/edu10_MAT_checked` bundle.

The exported observations include `household_imputed`, its complement
`household_observed`, the target and actual household years, and
`household_year_shifted`. A register-observed household from the 2015 fallback
is **not** necessarily a household observed at the outcome date. Household IDs
are still pooled across years as in the original; verify that register codes
are longitudinally stable before interpreting that random effect.

The script stops on conflicting grades in the earliest recorded year rather
than choosing one by file order. It verifies source-person identities, parent
edges, unchanged observation/design ordering, and the sparse factor identity
after pedigree trimming and reversal. Pedigree reversal changes the pedigree
row order and observation-to-pedigree IDs, **not** the observation row order.

`diagnostics/diagnose_h2_household.R` checks repeated IDs, household sizes, and
fixed-effect-adjusted sibling correlations. Between-household comparisons
require an observed/imputed flag; household-stratified comparisons exclude
explicitly shifted household years. These are descriptive correlations, not
REML heritability estimates or independent-pair significance tests.

Validate both R scripts on synthetic data with
`Rscript tests/test_edu_data_setup.R`.

### Single-input HPC runner

`run_emai_hpc.jl` fits one input dataset and saves the complete fitted model.
Edit `DATA_DIR`, `RUN_NAME`, the model block, and fitting options in the script;
it takes no command-line model selector. Run it from the Julia project:

```bash
julia --threads "$SLURM_CPUS_PER_TASK" --project=. run_emai_hpc.jl
```

It reads `yX_obs.csv`, `X_obs.csv`, `roster.csv`, `Ainv_triplet.csv`
(upper triangle of `Ainv`), and `LAinv_triplet.csv` (lower factor,
`Ainv = L_Ainv * L_Ainv'`) from `DATA_DIR`. The CSV log, fitted model, and a
copy of the run script are written under `DATA_DIR/fits`. Input checks are done
by `load_observed_data`, `build_group_structure`, and `fit_aireml!`.

When transferring an updated implementation, replace the Julia files under
`src/` on the HPC system and start a new Julia process. This does not require
overwriting an existing manually configured `run_emai_hpc.jl`.

## Fitted Models And Inference Snapshots

The fitted model is the primary object. Supply design-column names when building
it (`load_observed_data` also returns `fixef_names`):

```julia
model = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0;
    fixef_names=["(Intercept)", "x"])
# Choose one fit call. ψ is fixed by default:
fit_aireml!(model; max_iter=25, k=20)

# Or estimate ψ jointly with the variance components. This requires variable
# roster group sizes and centers scaling at their median:
fit_aireml!(model; max_iter=50, k=64, estimate_ψ=true,
    ψ_bounds=(0.0, 1.0), ψ_max_step=0.25)

model.σ2_d                 # fitted variance component
fixef(model)               # view of beta in the authoritative solution vector
ranef(model).a_d           # view of pedigree-wide direct BLUPs
model.fit.se               # variance-component SEs
model.fit.converged
model.fit.step_crit

save_model(model, "fitted_model.jld2")
model = load_model("fitted_model.jld2")

result = snapshot(model)   # independent, no y/X/pedigree matrices or scratch
save_results(result, "inference_snapshot.jld2")
result = load_results("inference_snapshot.jld2")
```

Every iteration header shows the current ψ and whether it is fixed or estimated.
When `log_file` is set, the iteration CSV also includes a `psi` column in both
cases; `se_psi` and `psi_score_mcse` are included only when ψ is estimated.
The reported value is after that iteration's parameter update.

`model.fit` stores AI, regularized variance-component covariance/SEs, score,
initial parameters, convergence diagnostics, and fit settings. Variance
components live only on the model and BLUE/BLUPs only in `model.sol_y`; the
accessors return views, not additional copies. Hutchinson solution buffers
and thread workspaces are local to fitting and are not retained or saved.
Refitting requires explicit `fit_aireml!(model; refit=true, ...)` and starts
from its current estimates. A failed refit clears the previous fit metadata.
Do not manually change model inputs or parameters after fitting and then use
cached inference; build/refit the model instead.

A snapshot contains variance components, beta, all BLUPs, fixed-effect and
environment-factor names, parameter names, observation/pedigree and environment
mappings, dilution/peer settings, AI/covariance/SEs, and convergence/settings
metadata. Its arrays are independent of the model.

By default, `snapshot(model)` also includes `beta_vcov` and `beta_se`.
These require one extra matrix-free CG solve per fixed-effect column, on the
first request only, and are cached on the model. Call
`fixed_effect_vcov!(model)` directly to populate this cache without a snapshot,
or use `snapshot(model; include_fixef_se=false)` to explicitly omit these
quantities (`nothing`). This optional inference work is not part of fitting.
Like `fit_aireml!`, `fixed_effect_vcov!` returns the same model, not the matrix:

```julia
fixed_effect_vcov!(model)
beta_vcov = model.fit.beta_vcov
```

BLUP prediction-error variances/reliabilities are **not** computed. A snapshot
supports inference on stored coefficients and variance components, not arbitrary
new computations requiring the original design or pedigree; save the full model
for those.

As before, AI/variance-component SEs are evaluated **before the last parameter
update**, while the final BLUE/BLUP solve and fixed-effect covariance use the
returned parameters. `ai_params` records the AI evaluation point explicitly,
which matters particularly for nonconverged fits. `converged=false` must not be
ignored when interpreting a snapshot.

`run_aireml(model; ...)` remains compatible: it mutates the model and returns
an `AiRemlResult` with fixed-effect SEs omitted, preserving its previous cost.
It allows refitting by default, as before. New code should use `fit_aireml!`.
The serialized struct layout has changed; existing JLD2 files should be
re-exported using their original source version rather than assuming cross-version
struct compatibility.

## Key Outputs

- `outputs/logs/*.csv`: per-iteration timing, parameter estimates, scores, convergence diagnostics
- `outputs/results/*.jld2`: optional complete `SGEModel` files or independent `AiRemlResult` snapshots

## Post-Fit Inspection

Set the single `data_dir` and editable `runs` list at the top of
`post_fit.jl`. The defaults compare:

- Mean: `ds_meandilution_class_household`
- Sum: `ds_sum_dilution_class_household`
- Estimated ψ: `ds_optdilution_class_household`

Use the **original** `roster.csv`, `yX_obs.csv` and `fits/RUN_NAME.jld2`
files in that directory. All fits must use identical observations, fixed
effects, pedigree, social roster and random-effect designs apart from dilution.
Full saved models are loaded one at a time and reduced to snapshots; their
data/design fingerprints are checked. Snapshots can also be compared, but
cannot establish equality of the original y, X or pedigree, so emit a warning.

Open `post_fit.jl` in VS Code and start a Julia REPL with `--project=.`. Edit
`data_dir` and `runs`, then execute the **Setup** and **Load and check the fits**
cells. Run the other `##` sections individually to inspect the tables and plots
you want. They leave `report` and the generated `figures` available in the
REPL; creating one plot does not generate the others. Re-running Load clears
the plot collection, preventing figures from different comparisons being mixed.
The calculations live in `src/emai_post_compare.jl`, which must accompany the
script when copying it to another machine.

Run the **Save tables and the figures created so far** cell when you want to
write output. It saves seven CSV tables and only the figures you generated to
`data_dir/fits/comparison`, overwriting files with matching names. Other figures
from an earlier save remain unchanged. The tables cover fit status and ψ/SE, raw and common-reference components,
derived quantities, fixed effects, group-size curves, and pairwise BLUP
correlations/RMSE/top-10% overlap. Plots include all component trajectories
(including classroom/household), ψ, step length, convergence criterion, estimate
correlation matrices and BLUP scatterplots. BLUP statistics use **all observed
individuals**; only scatterplots use a reproducible sample of at most 20,000.
Top-tail overlap is the fraction of equally sized top-10% sets shared by two
fits (ties follow the common observation order). No BLUP prediction-error SEs
are claimed; fixed-effect SEs appear only when already cached in the saved fit.

Raw social variances/covariances are **not directly comparable**: estimated ψ
uses centered scaling, while fixed ψ uses raw `(m-1)^(-ψ)` scaling. Common-scale
tables use per-peer variance `c(m₀)^2 σ²_s`, covariance `c(m₀) σ_ds` and,
if fitted, `c(m₀)^2 σ²_pe`, at the rounded median non-singleton roster size m₀.
BLUP plots use `a_d`, `c(m₀) a_s`, and `a_d + (m₀-1)c(m₀)a_s`.
Derived quantities and delta-method SEs include ψ and its cross-covariances when
estimated. The reference phenotype variance/social variance received assume
unrelated group members and omit focal-peer genetic covariance: these are
reference-model summaries, **not** the empirical variance of y. `T²` can exceed 1.
Intervals are approximate 95% Wald intervals, unreliable at boundaries or under
weak identification. AI/SEs belong to the pre-last-update point; inspect
`max_last_update` and convergence before interpreting them. Non-converged fits
are visibly flagged and are not ranked by these descriptive plots.

Expensive likelihood and ψ-score calculations are **never run automatically**.
With full, converged models, explicitly run in a suitably sized batch job:

```julia
likelihood = compare_likelihood(report; K=64)
scores = compare_psi_scores(report; K=64)  # fixed-ψ fits only
```

These also load one full model at a time. Likelihood differences are relative to
the first configured fit and use paired probes with a shared seed; output
includes Monte Carlo SEs, maximum Lanczos steps, the configured limit/tolerance,
and counts of probes reaching that limit. MCSE does **not** include
Lanczos truncation error: repeat with tighter tolerances/more steps before
ranking close fits, and increase K if Monte Carlo uncertainty is too large.
Score diagnostics require the other parameters to be at an optimum; boundaries
or weak identification invalidate a naive normal interpretation. See
`computational_approach.md`, Section 9.1. Matrix-free post-fit work can require
substantial **per-thread** memory; start with few threads. Plots.jl is required
in the global environment, deliberately not as a project dependency. On a
headless node, set `ENV["GKSwstype"]="100"` before `using Plots`.

## Input File Contract

The runner expects two CSV files in the working directory:

| File | Required columns | Notes |
|---|---|---|
| `yX_obs.csv` | `y`, `id_ped`, covariates | One row per **observed** outcome; no missing `y` |
| `roster.csv` | `id_ped`, `group` | One row per pedigree individual in any social group (may include unobserved animals) |

Additional columns (e.g. `household`, `classroom`, `school`) in `yX_obs.csv` are used for shared environment models. Covariates included in `X` are configured via `X_COLS` at the top of `run_emai.jl`.

## Optional Social Peer Environmental Effect

`SGEModel(...; peer_env=true, σ2_pe=0.05)` adds a non-genetic peer effect $p_j \sim N(0, \sigma^2_{pe})$ for every roster member. It enters peers' outcomes through the same diluted, self-excluding peer sum as `a_s`, with no covariance with the residual. It is typically fitted together with a classroom effect on the same groups:

```julia
model = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.05, 0.0, 2.0;
    env_ids=[classroom_ids], σ2_env=[0.3], peer_env=true, σ2_pe=0.05)
```

`σ²_pe` is the last parameter in the score/AI/SE vectors and appears as `s2pe`/`se_pe` in the iteration log. `AiRemlResult` adds `σ2_pe`, `p_env` (BLUPs) and `pe_ids` (pedigree ID per BLUP). Against a classroom effect it is identified only through class-size variation, and against `σ2_s` only through relatedness. See `computational_approach.md` §1.7.

## Direct-Only Model

`SGEModel(...; social=false)` fits $y = X\beta + Z_d a_d + \sum_h Z_h c_h + e$ without the social genetic block. Use it, for example, for a direct + household + residual baseline:

```julia
model = SGEModel(y, X, Ainv, L_Ainv, gs, 0.7, 0.0, 0.0, 0.2; social=false,
    env_ids=[household_ids], σ2_env=[0.02], env_names=["household"])
fit_aireml!(model; max_iter=50, k=32)
```

- The positional `σ2_s` and `σ_ds` values are ignored and stored as 0. `gs` only maps observations to pedigree IDs; singleton groups are enough.
- The parameter, score, AI and SE vectors are `[σ²_d, σ²_e, σ²_env...]`. The log columns are `s2d,s2e,s2env…`, with `rho_ds`/`cond_G` logged as `NaN`.
- `peer_env`, nonzero `ψ`, `estimate_ψ` and `psi_score` are rejected. `compare_fits` only compares social fits.
- To compare against a social fit, use `stochastic_loglik` on each fitted model and `loglik_difference`. The SE is unpaired because the MME dimensions differ.
- `has_social(model)` reports which model type is in use.

## Current Working Assumptions

- Julia 1.12
- Large-scale pedigree setting where the MME coefficient matrix is never formed explicitly
- `A` is applied implicitly through the Cholesky factorization of `A^{-1}`
- Numerical safeguards are intentional and should not be removed casually
- Threading uses `:static` scheduling; each `@threads` block must live in its own function (see DEVELOPMENT.md)
- `GroupStructure` precomputes `obs_to_group` and `ThreadWorkspace`/`MMEOperator` carry a `group_sum` buffer to avoid redundant per-observation group summation

## Where To Start

- Read `computational_approach.md` for the model and algorithm
- Read `DEVELOPMENT.md` for the execution and validation workflow
- Use `ROADMAP.md` to pick the next bounded development task