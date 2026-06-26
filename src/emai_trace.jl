# =====================================================================
# STOCHASTIC TRACE PROBES
# Randomisation used by the Hutchinson trace estimator. The same probe
# vectors must be reused across every AI-REML iteration so that the trace
# estimate tr(P*V_i) is comparable between iterations. Included via emai.jl.
# =====================================================================

# Draw k fixed Rademacher/Gaussian probe vectors z_j ∈ ℝⁿ for the Hutchinson
# estimator tr(P*V_i) ≈ (1/k) Σ_j z_j' (V_i (P z_j)). Seeding here keeps the
# probes deterministic and identical across iterations of run_aireml.
function make_trace_probes(n::Int, k::Int; seed::Int=42)
    Random.seed!(seed)
    return [randn(n) for _ in 1:k]
end
