---
description: "Use when modifying PeerIGE.jl model code, AI-REML updates, social design logic, missing-response handling, shared environment effects, or tests for emai.jl and related Julia files. Summarizes non-negotiable mathematical invariants from computational_approach.md."
name: "PeerIGE Model Invariants"
applyTo: "emai*.jl, run_emai*.jl, tests/**/*.jl"
---

# PeerIGE Model Invariants

- Preserve the observed-data formulation: `y` and `X` are built only from observed outcomes, while direct and social genetic effects live on the full pedigree size `N`.
- Missing-outcome individuals may be absent from `y` but can still contribute through pedigree structure and, when group membership is supplied, through the social design. Do not collapse these roles by silently dropping them from group membership.
- The social term is the sum of peers' indirect effects excluding self, not a class mean. Do not normalize by group size unless the estimand is intentionally being changed.
- Shared environment effects are observation-level grouping terms `Z_h c_h`; they are not pedigree-wide random effects.
- Keep `G = [σ2_d σ_ds; σ_ds σ2_s]` positive definite and keep variance components positive. Any parameter update logic must preserve `σ2_d * σ2_s > σ_ds^2` and `σ2_e > 0`.
- Do not replace AI-REML with direct dense likelihood optimization. The implementation depends on matrix-free solves for `P*y`, `P*z_j`, the score, and the Average Information matrix.
- Never form dense `A`, dense `V`, or the full MME coefficient matrix explicitly. Preserve the sparse `Ainv` plus sparse-triangular-solve approach for applying `A`.
- Preserve the current effect scope: genetic effects are pedigree-wide, while environment effects are defined by observed-row mappings in `env_ids`.
- When changing math-heavy code, explain the change in terms of the model equation or operator identity, not only the local syntax.
- Preferred validation after model changes: run `julia --project=. tests/runtests.jl`; if threading, preconditioners, or AI workspaces change, also run `julia --project=. bench_threads.jl`.