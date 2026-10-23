# Development Guide

## Default Workflow

Use short, bounded development cycles:

1. Identify one target change
2. Inspect the affected functions before editing
3. State the invariants that must remain true
4. Implement one focused change
5. Run a representative verification step immediately

This project is numerically sensitive. Avoid broad cleanup passes across `src/emai*.jl` unless the goal is explicitly modularization.

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

For offline deployment, rebuild the source archive after changing the source.
It is a generated artifact, not a versioned copy of the implementation:

```powershell
Compress-Archive -Path .\src\*.jl -DestinationPath .\src.zip -Force
```

## Important Runtime Parameters

The main execution entry points are:

- `run_emai.jl`: runnable script with explicit data loading, model setup, and solver arguments
- `fit_aireml!(model; ...)` in `src/emai_aireml.jl`: core outer optimization loop; returns the mutated model
- `snapshot(model)`: optional independent inference export; its first default call adds fixed-effect covariance solves
- `run_aireml(model; ...)`: compatibility wrapper returning a snapshot without fixed-effect covariance solves

Parameters worth tuning carefully:

- `max_iter`: outer AI-REML iteration cap
- `k`: number of Hutchinson probe vectors
- `damping`: initial step length for parameter updates
- `cg_maxiter`: PCG iteration cap for MME solves
- `cg_reltol`: PCG convergence tolerance
- `ai_maxiter`, `ai_reltol`: tolerances for AI-matrix-related solves
- `use_block_precond`: selects block versus diagonal preconditioner
- `seed`: controls reproducible Hutchinson probes

Note: z-solves, trace estimation, and AI work are always threaded via the
`_threaded_*` helpers; workspaces are allocated per thread. Launch Julia with
`--threads N` to control parallelism (a single thread simply runs the
`@threads` loops serially).

## Invariants To Preserve

- `G = [σ2_d σ_ds; σ_ds σ2_s]` must remain positive definite
- `σ2_d`, `σ2_s`, and `σ2_e` must stay positive
- `σ2_env[h]` must stay positive for each env factor `h`
- `σ2_pe` must stay positive when `peer_env=true`; with `peer_env=false` the model, layout, and results must be bit-identical to the model without the feature
- Threaded paths must use thread-local mutable buffers only
- `A` must never be formed explicitly; apply it through sparse triangular solves
- Numerical guards like `max(..., 1e-6)` and step-halving are deliberate stability controls. Do not replace step-halving in `run_aireml` by per-component flooring in `update_params!`: an overshooting AI step then sends several variances to their floors in one iteration, after which AI-REML only climbs back by roughly a factor `(1+α)` per iteration (scores in the millions, estimates near zero for many iterations).
- AI-REML starts with the requested `damping` and adapts the next iteration's step based on progress of the undamped correction in AI-SE units, up to `α=1`. Feasibility step-halving remains active. Convergence uses `max|Δ|/SE < 1e-2`, not only the realized damped change; `adaptive_damping=false` disables the progress-based adjustment.
- **Each `Threads.@threads :static` block must live in its own dedicated helper function.** Multiple consecutive `@threads` blocks within a single function corrupt each other's closure environments in Julia 1.12 (non-deterministic results). The helpers are: `_threaded_vi_py!`, `_threaded_trace!`, `_threaded_ai_cg!`, `_threaded_ai_matvec!`, `_threaded_z_solves!`.

## High-Risk Areas

### Fitted state and performance

- Model mutators (`fit_aireml!`, `fixed_effect_vcov!`, `update_params!`,
  model-owned `solve_mme_y!`/`solve_mme_z!`) return the same model.
  Buffer mutators return the destination, not the model whose inputs they read.
  `apply_GinvAinv!` returns both destination blocks as `(out_d, out_s)`;
  threaded helpers return their first destination collection. Scratch buffers
  are not part of these return contracts. Cached and uncached paths must agree.
- Variances remain on `SGEModel`; the final BLUE/BLUP solution remains in
  `sol_y`. `fixef`/`ranef` expose views. Do not add pedigree-sized copies to
  `model.fit`, or retain probe solution buffers and thread workspaces.
- `model.fit` is absent until fitting completes. Refitting must be explicit
  (`refit=true`); it clears old metadata before changing the estimates.
- Snapshot arrays must not alias model arrays. Save/load tests cover the full
  model and independent snapshot.
- Fixed-effect covariance is optional post-fit work. Since
  `MMEOperator = sigma2_e * C`, solving against a fixed-effect unit RHS and
  multiplying the fixed block by `sigma2_e` gives `(X'V^-1 X)^-1`.
  Never form the full coefficient matrix. Nonconvergence of these inference
  solves is an error, not an apparently valid SE.
- The optimizer's last AI belongs to the pre-update parameters: retain
  `ai_params` rather than labeling it as an exact final-point Hessian.
- Before structural fit changes, capture warmed repeated timings on identical
  inputs, settings, probes and thread count. Compare estimates, full solutions,
  AI, SEs, iteration counts and convergence, not only wall time.
- Validate with `tests/test_fitted_model.jl`, the complete suite on one and
  multiple threads, and `bench_threads.jl` when threaded workspace plumbing
  changes. Keep optional snapshot inference time separate from fit time.

The fitted-state refactor was compared against commit `c492e5f` in one process
with alternating before/after run order (Julia 1.12.5, eight threads,
N=20,000, n=18,000, five iterations, k=10, seed=42). Nine base-model pairs
and five one-environment pairs gave median paired runtime ratios of 0.998
and 0.949 respectively, both below the 1.05 regression threshold. Allocated
bytes decreased slightly. Parameters, the complete MME solution, AI, SEs,
iteration counts and convergence flags matched bit-for-bit in every pair.
These timings exclude optional post-fit fixed-effect inference; they are a
regression check on simulation data, not a real-pedigree-scale speed guarantee.

### Diagnosing collapsed variance estimates

- `run_aireml` first checks `Ainv*x ≈ L_Ainv*(L_Ainv'*x)` with two
  deterministic random probes (relative tolerance `1e-6`). This uses existing
  workspace buffers and sparse products, not a reconstructed precision or
  relationship matrix. A mismatch stops the fit: the MME uses `Ainv`, whereas
  genetic score derivatives and the block preconditioner use `L_Ainv`.
- After trimming, both inputs must use the same new pedigree order. A permuted
  Cholesky factor must not be supplied as though it were an unpermuted factor.
  Observed and roster IDs must also be remapped consistently; the factor check
  cannot verify these external ID mappings.
- Every unconverged MME solve now warns, even with `verbose_cg=false`, including
  probe (`z`) and AI (`direct`) solves. Inspect these warnings before interpreting
  the scores or changing damping. `ai_maxiter` is separate from `cg_maxiter`.
- `var(y)` is not the sum of variance parameters: fixed effects, pedigree
  relatedness, the direct/social covariance, peer counts, and dilution affect
  the marginal variance. Small social parameters can produce large outcome
  variance through peer sums.
- Validate these checks with `julia --threads 8 --project=. tests/test_solver_diagnostics.jl`.

### Numerically sensitive components

- `BlockPreconditioner` and `DiagPreconditioner`
- `MMEOperator` and its `mul!` overloads
- `apply_A!`, `get_Vi_u!`, `get_Vi_env_u!`, and the `_threaded_*` helper functions
- `compute_score_and_ai`
- parameter updates in `run_aireml`
- `env_ids`, `env_n_levels`, `env_counts`, `env_tmp` fields in `SGEModel` and `ThreadWorkspace`
- `peer_env`, `pe_ids`, `pe_data_diag` in `SGEModel` and the `pe` block of `SolutionLayout` (the operator reuses `op.Ad` as scratch for `a_s + scatter(p)` before `apply_GinvAinv!` overwrites it)

For changes in these sections, define the expected mathematical identity before editing.

## Validation Checklist

After meaningful code changes, verify at least the following:

1. Julia imports succeed under `--project=.`
2. Run `julia --threads 8 --project=. tests/runtests.jl` (tests missing-response group mapping plus H=0 and H=1 sequentially)
3. `aireml_log_base.csv` and `aireml_log_env.csv` are produced with sensible timing and parameters
4. Parameter updates remain finite and the `G` matrix stays valid
5. Threaded changes do not introduce shared mutable state
6. For any change touching a `_threaded_*` helper or `compute_score_and_ai`, run `julia --threads 8 --project=. bench_threads.jl` and confirm it converges to sensible estimates with reasonable per-iteration timing

## Suggested Task Shapes For AI Collaboration

Good tasks:

- extract one data-loading section into a function
- add input validation for matrix and vector dimensions
- add a small test for one operator identity
- explain or document one numerical block

Avoid open-ended tasks like “clean up the whole file” or “optimize everything”.

## Documentation Priority

When adding new logic, also document:

- why the change is needed
- which invariant it preserves
- how to validate it against the current implementation