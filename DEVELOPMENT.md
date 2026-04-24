# Development Guide

## Default Workflow

Use short, bounded development cycles:

1. Identify one target change
2. Inspect the affected functions before editing
3. State the invariants that must remain true
4. Implement one focused change
5. Run a representative verification step immediately

This project is numerically sensitive. Avoid broad cleanup passes across `emai.jl` unless the goal is explicitly modularization.

## Standard Run Command

Always run Julia with the workspace project activated:

```powershell
julia --project=. run_emai.jl
```

For REPL work:

```julia
using Pkg
Pkg.activate(".")
```

## Important Runtime Parameters

The main execution entry points are:

- `run_emai.jl`: runnable script with explicit data loading, model setup, and solver arguments
- `run_aireml(model; ...)` in `emai.jl`: core outer optimization loop once a model is already constructed
- `run_emai_direct.jl`: runnable script for the direct-effects baseline
- `run_emai_direct(; ...)` in `emai_direct.jl`: reusable helper for the direct model workflow

Parameters worth tuning carefully:

- `max_iter`: outer AI-REML iteration cap
- `k`: number of Hutchinson probe vectors
- `damping`: initial step length for parameter updates
- `cg_maxiter`: PCG iteration cap for MME solves
- `cg_reltol`: PCG convergence tolerance
- `ai_maxiter`, `ai_reltol`: tolerances for AI-matrix-related solves
- `use_block_precond`: selects block versus diagonal preconditioner
- `threaded`: enables threaded z-solves, trace estimation, and AI work
- `seed`: controls reproducible Hutchinson probes

## Invariants To Preserve

- `G = [σ2_d σ_ds; σ_ds σ2_s]` must remain positive definite
- `σ2_d`, `σ2_s`, and `σ2_e` must stay positive
- `σ2_env[h]` must stay positive for each env factor `h`
- Threaded paths must use thread-local mutable buffers only
- `A` must never be formed explicitly; apply it through sparse triangular solves
- Numerical guards like `max(..., 1e-6)` and step-halving are deliberate stability controls
- **Each `Threads.@threads :static` block must live in its own dedicated helper function.** Multiple consecutive `@threads` blocks within a single function corrupt each other's closure environments in Julia 1.12 (non-deterministic results). The helpers are: `_threaded_vi_py!`, `_threaded_trace!`, `_threaded_ai_cg!`, `_threaded_ai_matvec!`, `_threaded_z_solves!`.

## High-Risk Areas

- `BlockPreconditioner` and `DiagPreconditioner`
- `MMEOperator` and its `mul!` overloads
- `apply_A!`, `get_Vi_u!`, `get_Vi_env_u!`, and the `_threaded_*` helper functions
- `compute_score_and_ai`
- parameter updates in `run_aireml`
- `env_ids`, `env_n_levels`, `env_counts`, `env_tmp` fields in `SGEModel` and `ThreadWorkspace`

For changes in these sections, define the expected mathematical identity before editing.

## Validation Checklist

After meaningful code changes, verify at least the following:

1. Julia imports succeed under `--project=.`
2. Run `julia --threads 8 --project=. tests/runtests.jl` (tests missing-response group mapping plus H=0 and H=1 sequentially)
3. `aireml_log_base.csv` and `aireml_log_env.csv` are produced with sensible timing and parameters
4. Parameter updates remain finite and the `G` matrix stays valid
5. Threaded changes do not introduce shared mutable state
6. For any change touching a `_threaded_*` helper or `compute_score_and_ai`, run `julia --threads 8 --project=. bench_threads.jl` and confirm `Max |Δ| < 1e-4` and speedup > 1.5x

## Suggested Task Shapes For AI Collaboration

Good tasks:

- extract one data-loading section into a function
- add input validation for matrix and vector dimensions
- add a small test for one operator identity
- explain or document one numerical block
- compare threaded and non-threaded behavior on one run

Avoid open-ended tasks like “clean up the whole file” or “optimize everything”.

## Documentation Priority

When adding new logic, also document:

- why the change is needed
- which invariant it preserves
- how to validate it against the current implementation