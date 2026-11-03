# Text report for fitted models. fit_aireml! prints the same report after every
# iteration (with a temporary fit object), so all diagnostics live here.
# `fit` may be an AiRemlFit or an AiRemlResult: both store AI, vcov, se, score,
# initial_params, ai_params, converged, n_iter, step_crit, alpha and settings.

# Also used by fit_aireml! for the CSV log.
function _g_health(σ2_d, σ2_s, σ_ds)
    rho_ds = σ_ds / sqrt(σ2_d * σ2_s)
    eig_G = eigvals([σ2_d σ_ds; σ_ds σ2_s])
    cond_G = maximum(eig_G) / max(minimum(eig_G), 1e-12)
    return rho_ds, cond_G, eig_G
end

function _show_aireml_report(io::IO, n_obs, n_pedigree, fixef_names, beta, beta_se,
    param_names, params, fit; env_names=String[], env_levels=Int[], peer_env=false,
    ψ=0.0, in_progress=false)
    println(io, in_progress ? "AI-REML iteration $(fit.n_iter)" : "AI-REML fit")
    status = fit === nothing ? "not fitted" :
        fit.converged ? "converged after $(fit.n_iter) iteration(s)" :
        in_progress ? "in progress ($(fit.n_iter) of max $(fit.settings.max_iter) iterations)" :
        "not converged after $(fit.n_iter) iteration(s)"
    println(io, "  Status: ", status)
    println(io, "  Data:   ", n_obs, " observations; ", n_pedigree, " pedigree members")
    social = "σ²_s" in param_names
    println(io, "  Design: ", length(fixef_names), " fixed effects; ",
        length(env_names), " shared environment factor(s)",
        peer_env ? "; peer environment enabled" : "",
        social ? "" : "; direct-only (no social genetic effect)")
    if !isempty(env_names)
        println(io, "  Environment levels: ",
            join(["$(env_names[i])=$(env_levels[i])" for i in eachindex(env_names)], ", "))
    end
    fit === nothing && return nothing

    s = fit.settings
    social && println(io, "  ψ:      ", ψ, " (", "ψ" in param_names ? "estimated" : "fixed", ")")
    @printf(io, "  Settings: k=%d, damping=%.3g%s, precond=%s, CG reltol=%.0e (iter ≤ %d) then %.0e, cg_maxiter=%d, seed=%d\n",
        s.k, s.damping, s.adaptive_damping ? " (adaptive)" : "",
        s.use_block_precond ? "Block" : "Diag", s.cg_reltol_early, s.cg_early_iters,
        s.cg_reltol, s.cg_maxiter, s.seed)

    # Δ_AI is the undamped AI step and |Δ|/SE the convergence criterion per parameter.
    se, score = fit.se, fit.score
    delta, boundary_active = _boundary_constrained_delta(fit.AI + 1e-6 * I,
        score, fit.ai_params; variance_count=length(param_names) - Int("ψ" in param_names),
        social)
    z = abs.(delta) ./ max.(se, eps())
    println(io)
    println(io, "Variance components")
    @printf(io, "  %-10s %12s %12s %12s %12s %10s %12s\n",
        "Component", "Estimate", "SE", "Score", "Δ_AI", "|Δ|/SE", "Start")
    for i in eachindex(param_names)
        @printf(io, "  %-10s %12.6g %12.6g %+12.4g %+12.4g %10.3g %12.6g%s\n",
            param_names[i], params[i], se[i], score[i], delta[i], z[i],
            fit.initial_params[i], z[i] == maximum(z) && !fit.converged ? "  ◀" : "")
    end
    if !isempty(boundary_active)
        println(io, "  Held near lower bound: ", join(param_names[boundary_active], ", "),
            "; Δ_AI uses the reduced AI system.")
    end

    if !isempty(fixef_names)
        println(io)
        println(io, "Fixed effects", in_progress ? " (GLS at start-of-iteration variances)" : "")
        println(io, "  ", rpad("Name", 24), lpad("Estimate", 14), lpad("SE", 14))
        max_rows = max(1, min(12, displaysize(io)[1] - 18))
        n_show = min(length(fixef_names), max_rows)
        for i in 1:n_show
            print(io, "  ", rpad(fixef_names[i], 24), lpad(@sprintf("%.6g", beta[i]), 14))
            println(io, lpad(beta_se === nothing ? "not computed" : @sprintf("%.6g", beta_se[i]), 14))
        end
        n_show < length(fixef_names) &&
            println(io, "  … ", length(fixef_names) - n_show, " more fixed effects")
        beta_se === nothing && println(io, "  Fixed-effect SEs have not been computed.")
    end

    println(io)
    println(io, "Iteration diagnostics")
    @printf(io, "  Criterion:  max|Δ_AI|/SE=%.3e (%s); converged when < 0.01\n",
        fit.step_crit, param_names[argmax(z)])
    @printf(io, "  Steps:      last α=%.3e; last relative change=%.3e; relative change from start=%.3e\n",
        fit.alpha, norm(params .- fit.ai_params) / norm(fit.ai_params),
        norm(params .- fit.initial_params) / norm(fit.initial_params))
    @printf(io, "  |score|:    %.4e\n", norm(score))
    if social
        rho_ds, cond_G, eig_G = _g_health(params[1], params[2], params[3])
        @printf(io, "  G:          ρ_ds=%.4f; cond(G)=%.1f; eig=[%.4g, %.4g]",
            rho_ds, cond_G, eig_G[1], eig_G[2])
        abs(rho_ds) > 0.95 && print(io, "  ⚠ ρ nær grense!")
        cond_G > 1e4 && print(io, "  ⚠ dårlig kondisjonert!")
        println(io)
    else
        println(io, "  G:          direct-only model (no social genetic effect)")
    end
    eig_AI = eigvals(Symmetric(fit.AI))
    @printf(io, "  AI:         eig=[%.4g, %.4g]; cond(AI)=%.3g", eig_AI[1], eig_AI[end],
        eig_AI[end] / max(eig_AI[1], 1e-300))
    eig_AI[1] <= 0 && print(io, "  ⚠ AI ikke positiv definitt!")
    println(io)

    # Sampling correlations from AI⁻¹; |r| > 0.9 means the components are hard to separate.
    d = sqrt.(abs.(diag(fit.vcov)))
    R = fit.vcov ./ (d * d')
    short = [first(name, 8) for name in param_names]
    println(io, "  Sampling correlations of estimates (AI⁻¹; ! marks |r| > 0.9):")
    println(io, "    ", rpad("", 9), join(lpad.(short[1:end-1], 9)))
    for i in 2:length(param_names)
        print(io, "    ", rpad(short[i], 9))
        for j in 1:i-1
            print(io, lpad(@sprintf("%+.2f%s", R[i, j], abs(R[i, j]) > 0.9 ? "!" : " "), 9))
        end
        println(io)
    end
    println(io, "  Score, SE, Δ_AI and correlations are evaluated before the last update.")
    return nothing
end

# Report for a model with a given fit object; fit_aireml! passes the current iteration's fit.
function _show_model(io::IO, model::SGEModel, fit; in_progress=false)
    beta_se = fit === nothing || fit.beta_vcov === nothing ? nothing : sqrt.(diag(fit.beta_vcov))
    estimate_ψ = _fit_estimates_psi(model, fit)
    return _show_aireml_report(io, length(model.y), size(model.Ainv, 1),
        model.fixef_names, view(model.sol_y, model.layout.fixed), beta_se,
        get_param_names(model; include_ψ=estimate_ψ), get_params(model; include_ψ=estimate_ψ), fit;
        env_names=model.env_names, env_levels=model.env_n_levels,
        peer_env=model.peer_env, ψ=model.ψ, in_progress)
end

Base.show(io::IO, ::MIME"text/plain", model::SGEModel) = _show_model(io, model, model.fit)

function Base.show(io::IO, ::MIME"text/plain", result::AiRemlResult)
    env_levels = [length(ids) == 0 ? 0 : maximum(ids) for ids in result.env_ids]
    base = "σ²_s" in result.param_names ?
        [result.σ2_d, result.σ2_s, result.σ_ds, result.σ2_e] : [result.σ2_d, result.σ2_e]
    return _show_aireml_report(io, result.n_obs, result.N_pedigree,
        result.fixef_names, result.beta, result.beta_se,
        result.param_names, vcat(base,
            result.σ2_env, result.peer_env ? [result.σ2_pe] : Float64[],
            "ψ" in result.param_names ? [result.ψ] : Float64[]), result;
        env_names=result.env_names, env_levels=env_levels,
        peer_env=result.peer_env, ψ=result.ψ)
end

function Base.show(io::IO, model::SGEModel)
    print(io, "SGEModel(", model.fit === nothing ? "not fitted" : "fitted",
        ", n=", length(model.y), ", N=", size(model.Ainv, 1), ")")
end

function Base.show(io::IO, result::AiRemlResult)
    print(io, "AiRemlResult(", result.converged ? "converged" : "not converged",
        ", n_iter=", result.n_iter, ", n=", result.n_obs,
        ", N=", result.N_pedigree, ")")
end
