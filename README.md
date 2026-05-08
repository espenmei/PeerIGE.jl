# PeerIGE.jl

Julia code for AI-REML estimation of direct and social genetic effects in large pedigree-based mixed models.

## What This Repository Contains

- `emai.jl`: full social genetic effects model with variance components (`σ2_d`, `σ2_s`, `σ_ds`, `σ2_e`) plus optional shared environment effects (`σ2_env` for 0–2 grouping factors)
- `run_emai.jl`: runnable entrypoint; fits both the base SGE model and the SGE+household model, prints a comparison table
- `emai_direct.jl`: simpler direct-effects-only baseline
- `run_emai_direct.jl`: runnable entrypoint for the direct-effects baseline
- `computational_approach.md`: math and algorithm overview
- `sim_peer.R`, `sim_peer_evn.R`, `sim_direct.R`: simulation scripts in R (`sim_peer_evn.R` adds household/classroom/school shared environment effects); each writes the two-table format (`yX_obs.csv`, `roster.csv`)
- `yX_obs.csv`: observed-only rows — one row per observed outcome, with columns `y`, `id_ped`, covariates, and optional environment IDs (e.g. `household`)
- `roster.csv`: full social group membership for all relevant pedigree IDs, with columns `id_ped` and `group` (includes unobserved animals)
- `Ainv_triplet.csv`, `LAinv_triplet.csv`: sparse pedigree matrix inputs in triplet format
- `bench_threads.jl`: threading benchmark (compares sequential vs threaded timing and correctness)
- `tests`: test entrypoints and smoke checks for missing-response handling and H=0/H=1 models

## Folder Structure

- `outputs/logs`: AI-REML and benchmark log CSVs
- `outputs/results`: JLD2 model result files
- `outputs/benchmarks`: benchmark CSV logs
- `outputs/smoke`: smoke-test logs
- `tests`: test scripts and the main test entrypoint (`runtests.jl`)
- `diagnostics`: one-off diagnostic scripts (`diag_*.jl`)
- `docs`: rendered HTML documentation artifacts (`computational_approach.html` and assets)

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

Run the direct-effects baseline with:

```powershell
julia --project=. run_emai_direct.jl
```

Run the test entrypoint after code changes with:

```powershell
julia --project=. tests/runtests.jl
```

The runner currently:

1. Loads observed phenotype/design data from `yX_obs.csv` and social group membership from `roster.csv`
2. Validates required columns and builds `GroupStructure` from the two-table input
3. Loads `Ainv` and sparse Cholesky inputs from triplet CSV files
4. Fits **Model 1**: base SGE (`σ2_d`, `σ2_s`, `σ_ds`, `σ2_e`), saves to `outputs/results/aireml_result_base.jld2`
5. Fits **Model 2**: SGE + household shared environment, saves to `outputs/results/aireml_result_env.jld2`
6. Prints a side-by-side parameter comparison table
7. All runs use `threaded=true` by default — start with `--threads 8`

## Key Outputs

- `outputs/logs/*.csv`: per-iteration timing, parameter estimates, scores, convergence diagnostics
- `outputs/results/*.jld2`: serialized `AiRemlResult`/`DirectRemlResult` with estimates, BLUE/BLUP outputs, AI matrix, and standard errors

## Input File Contract

The runner expects two CSV files in the working directory:

| File | Required columns | Notes |
|---|---|---|
| `yX_obs.csv` | `y`, `id_ped`, covariates | One row per **observed** outcome; no missing `y` |
| `roster.csv` | `id_ped`, `group` | One row per pedigree individual in any social group (may include unobserved animals) |

Additional columns (e.g. `household`, `classroom`, `school`) in `yX_obs.csv` are used for shared environment models. Covariates included in `X` are configured via `X_COLS` at the top of `run_emai.jl`.

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