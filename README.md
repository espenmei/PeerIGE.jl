# PeerIGE.jl

Julia code for AI-REML estimation of direct and social genetic effects in large pedigree-based mixed models.

## What This Repository Contains

- `emai.jl`: full social genetic effects model with four variance components (`σ2_d`, `σ2_s`, `σ_ds`, `σ2_e`)
- `run_emai.jl`: runnable entrypoint for the full model using default file names and parameters
- `emai_direct.jl`: simpler direct-effects-only baseline
- `run_emai_direct.jl`: runnable entrypoint for the direct-effects baseline
- `computational_approach.md`: math and algorithm overview
- `sim_peer.R`, `sim_direct.R`: simulation scripts in R
- `yX.csv`, `X.csv`: dense input data
- `Ainv_triplet.csv`, `LAinv_triplet.csv`: sparse pedigree matrix inputs in triplet format

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

Run a short smoke test after code changes with:

```powershell
julia --project=. test_smoke_emai.jl
```

The runner currently:

1. Loads phenotype and design data from `yX.csv` and `X.csv`
2. Loads `Ainv` and sparse Cholesky inputs from triplet CSV files
3. Builds the `SGEModel`
4. Runs AI-REML
5. Writes progress to `aireml_log.csv`
6. Saves final results to `aireml_result.jld2`

## Key Outputs

- `aireml_log.csv`: per-iteration timing, parameter estimates, scores, convergence diagnostics
- `aireml_result.jld2`: serialized `AiRemlResult` with variance estimates, BLUE/BLUP outputs, AI matrix, and standard errors

## Current Working Assumptions

- Julia 1.12
- Large-scale pedigree setting where the MME coefficient matrix is never formed explicitly
- `A` is applied implicitly through the Cholesky factorization of `A^{-1}`
- Numerical safeguards are intentional and should not be removed casually

## Where To Start

- Read `computational_approach.md` for the model and algorithm
- Read `DEVELOPMENT.md` for the execution and validation workflow
- Use `ROADMAP.md` to pick the next bounded development task