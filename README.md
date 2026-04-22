# PeerIGE.jl

Julia code for AI-REML estimation of direct and social genetic effects in large pedigree-based mixed models.

## What This Repository Contains

- `emai.jl`: full social genetic effects model with variance components (`σ2_d`, `σ2_s`, `σ_ds`, `σ2_e`) plus optional shared environment effects (`σ2_env` for 0–2 grouping factors)
- `run_emai.jl`: runnable entrypoint; fits both the base SGE model and the SGE+household model, prints a comparison table
- `emai_direct.jl`: simpler direct-effects-only baseline
- `run_emai_direct.jl`: runnable entrypoint for the direct-effects baseline
- `computational_approach.md`: math and algorithm overview
- `sim_peer.R`, `sim_peer_evn.R`, `sim_direct.R`: simulation scripts in R (`sim_peer_evn.R` adds a household shared environment effect)
- `yX.csv`, `X.csv`: dense input data
- `Ainv_triplet.csv`, `LAinv_triplet.csv`: sparse pedigree matrix inputs in triplet format
- `bench_threads.jl`: threading benchmark (compares sequential vs threaded timing and correctness)
- `test_smoke_emai.jl`: quick smoke test for H=0 and H=1 models

## Folder Structure

- `outputs/logs`: AI-REML and benchmark log CSVs
- `outputs/results`: JLD2 model result files
- `outputs/benchmarks`: benchmark CSV logs
- `outputs/smoke`: smoke-test logs
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

Run a short smoke test after code changes with:

```powershell
julia --project=. test_smoke_emai.jl
```

The runner currently:

1. Loads phenotype and design data from `yX.csv` and `X.csv`
2. Loads `Ainv` and sparse Cholesky inputs from triplet CSV files
3. Fits **Model 1**: base SGE (`σ2_d`, `σ2_s`, `σ_ds`, `σ2_e`), saves to `outputs/results/aireml_result_base.jld2`
4. Fits **Model 2**: SGE + household shared environment, saves to `outputs/results/aireml_result_env.jld2`
5. Prints a side-by-side parameter comparison table
6. Both runs use `threaded=true` by default — start with `--threads 8`

## Key Outputs

- `outputs/logs/*.csv`: per-iteration timing, parameter estimates, scores, convergence diagnostics
- `outputs/results/*.jld2`: serialized `AiRemlResult`/`DirectRemlResult` with estimates, BLUE/BLUP outputs, AI matrix, and standard errors

## Current Working Assumptions

- Julia 1.12
- Large-scale pedigree setting where the MME coefficient matrix is never formed explicitly
- `A` is applied implicitly through the Cholesky factorization of `A^{-1}`
- Numerical safeguards are intentional and should not be removed casually
- Threading uses `:static` scheduling; each `@threads` block must live in its own function (see DEVELOPMENT.md)

## Where To Start

- Read `computational_approach.md` for the model and algorithm
- Read `DEVELOPMENT.md` for the execution and validation workflow
- Use `ROADMAP.md` to pick the next bounded development task