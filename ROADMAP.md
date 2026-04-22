# Roadmap

## Completed (2026-04-14)

- **Shared environment effects**: `SGEModel` accepts `env_ids` and `σ2_env` keyword arguments for 0–2 grouping factors (e.g., household). The MME operator, RHS builders, block preconditioner, BLUP extraction, and `AiRemlResult` all handle an arbitrary number of env factors. The scatter-gather operation for $V_{env} = Z_h Z_h'$ is O(n).
- **Simulation update**: `sim_peer_evn.R` generates data with a household shared environment (sibling-pair households, `s2_c=0.5`).
- **Dual model runner**: `run_emai.jl` fits both Model 1 (base SGE) and Model 2 (SGE + household) and prints a side-by-side comparison.
- **Threading fix**: Diagnosed and fixed a Julia 1.12 bug where multiple consecutive `Threads.@threads :static` blocks within the same function corrupt each other's closure environments. Each `@threads` block is now wrapped in its own helper function (`_threaded_vi_py!`, `_threaded_trace!`, `_threaded_ai_cg!`, `_threaded_ai_matvec!`, `_threaded_z_solves!`).
- **Threading benchmark**: `bench_threads.jl` confirms ~2x speedup on 8 threads (H=0: 2.16x, H=1: 1.98x) with parameter match to machine precision (Max |Δ| < 3e-8).

## Immediate Tracks

### 1. Modularization

- extract data loading and model construction from the script tail into named functions
- isolate preconditioner construction behind a smaller public API
- separate threaded and non-threaded solve setup more clearly

### 2. Correctness And Validation

- add dimension and input checks before model construction
- add focused tests for `apply_A!`, preconditioners, `get_Vi_u!`, and `get_Vi_env_u!`
- compare key outputs between `emai_direct.jl` and simplified settings in `emai.jl`

### 3. Experiment Workflow

- make run parameters configurable without editing the script body each time
- standardize output naming for logs and result files
- document a baseline run configuration for regression checking

### 4. Documentation

- keep `computational_approach.md` aligned with implementation details
- document Norwegian domain terms that appear in comments and variable names
- add short rationale notes where numerical safeguards are easy to misread as arbitrary

## Recommended Order

1. Extract script-tail setup into functions
2. Add dimension and file validation
3. Add a baseline verification run workflow
4. Add small tests for the safest operator blocks
5. Only then consider larger structural refactors

## Session Rule

Each work session should end with one of these outcomes:

- one verified code change
- one verified explanation of a mathematical block
- one new validation artifact
- one documentation improvement that reduces future ambiguity