# PeerIGE.jl

Julia code for AI-REML estimation of direct and social genetic effects in large pedigree-based mixed models.

## What This Repository Contains

- `src/emai.jl`: thin façade that loads the model. The implementation is split across `src/emai_core.jl` (shared types and observed-data / group construction), `src/emai_precond.jl` (diagonal and block preconditioners), `src/emai_mme.jl` (matrix-free mixed model equations, CG solves, `V_i` operators), and `src/emai_aireml.jl` (Hutchinson trace probes, score/AI assembly, and the `fit_aireml!` driver). Together they fit the full social genetic effects model with variance components (`σ2_d`, `σ2_s`, `σ_ds`, `σ2_e`) plus optional shared environment effects
- `run_emai.jl`: runnable entrypoint; fits both the base SGE model and the SGE+household model, prints a comparison table
- `computational_approach.md`: math and algorithm overview
- `sim/`: simulation scripts in R (`sim_peer.R`, `sim_peer_evn.R`; `sim_peer_evn.R` adds household/classroom/school shared environment effects); each writes the two-table format (`yX_obs.csv`, `roster.csv`)
- `yX_obs.csv`: observed-only rows — one row per observed outcome, with columns `y`, `id_ped`, covariates, and optional environment IDs (e.g. `household`)
- `roster.csv`: full social group membership for all relevant pedigree IDs, with columns `id_ped` and `group` (includes unobserved animals)
- `Ainv_triplet.csv`, `LAinv_triplet.csv`: sparse pedigree matrix inputs in triplet format
- `bench_threads.jl`: threading benchmark (times a threaded run and reports per-iteration timing and final estimates)
- `tests`: test entrypoints and smoke checks for missing-response handling and H=0/H=1 models

## Folder Structure

- `src`: model implementation (`emai.jl` façade plus `emai_core.jl`, `emai_precond.jl`, `emai_mme.jl`, `emai_aireml.jl`)
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

## Fitted Models And Inference Snapshots

The fitted model is the primary object. Supply design-column names when building
it (`load_observed_data` also returns `fixef_names`):

```julia
model = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0;
    fixef_names=["(Intercept)", "x"])
fit_aireml!(model; max_iter=25, k=20)  # returns the same model

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