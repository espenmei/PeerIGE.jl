# =====================================================================
# AI-REML DRIVER
# Score and Average Information assembly, parameter updates, the result
# struct, and the fit_aireml! iteration loop. Included via emai.jl.
# =====================================================================
# Returnerer varianstype-representasjon for gjeldende modell (d, s, ds, e, env_h)
function get_variance_components(model::SGEModel)
    comps = VarianceComponent[
        VarianceComponent(VC_D, 0),
        VarianceComponent(VC_S, 0),
        VarianceComponent(VC_DS, 0),
        VarianceComponent(VC_E, 0)
    ]
    for h in 1:length(model.env_ids)
        push!(comps, VarianceComponent(VC_ENV, h))
    end
    if model.peer_env
        push!(comps, VarianceComponent(VC_PE, 0))
    end
    return comps
end

function get_param_names(model::SGEModel; include_ψ::Bool=_fit_estimates_psi(model))
    names = ["σ²_d", "σ²_s", "σ_ds", "σ²_e"]
    for h in 1:length(model.env_ids)
        push!(names, "σ²_env$h")
    end
    model.peer_env && push!(names, "σ²_pe")
    include_ψ && push!(names, "ψ")
    return names
end

function compute_score_and_ai(model::SGEModel, Py, z_list, Pz_list, workspaces, preconditioners;
    ai_reltol=1e-5, ai_maxiter=500, verbose=true, estimate_ψ::Bool=false,
    return_psi_mcse::Bool=false)
    comps = get_variance_components(model)
    n_p = length(comps)
    score = zeros(n_p)
    AI = zeros(n_p, n_p)
    k = length(Pz_list)
    n = length(Py)
    estimate_ψ && k < 2 &&
        throw(ArgumentError("Estimating ψ requires at least two trace probes"))

    # 1. Forhåndsberegn alle Vi * Py
    verbose && println("    -> Beregner Vi * Py...")
    Vi_Py_list = [Vector{Float64}(undef, n) for _ in 1:n_p]
    _threaded_vi_py!(Vi_Py_list, Py, comps, model, workspaces)

    # 2. SCORE: s_i = 0.5 * (y'P * Vi * Py - tr(P * Vi))
    #    Trace-estimering: parallelliser k get_Vi_u-kall per parameter
    verbose && println("    -> Beregner trace (k=$k)...")
    n_trace = n_p * k
    trace_results = zeros(n_trace)
    _threaded_trace!(trace_results, n_trace, k, Pz_list, z_list, comps, model, workspaces)
    for i in 1:n_p
        tr_val = sum(@view trace_results[(i-1)*k+1 : i*k])
        score[i] = 0.5 * (Py' * Vi_Py_list[i] - tr_val)
    end

    # 3. AI-MATRISE (Average Information)
    #    Trenger bare n_p CG-solves (én per V_j).
    #    AI[i,j] = 0.5 * Py' * Vi * P * Vj * Py
    verbose && println("    -> Beregner AI-matrise ($(n_p + (estimate_ψ ? 1 : 0)) CG-solves)...")

    # Steg 1: Beregn P * V_j * Py
    P_Vi_Py_list = [Vector{Float64}(undef, n) for _ in 1:n_p]
    _threaded_ai_cg!(P_Vi_Py_list, Vi_Py_list, model, workspaces, preconditioners, ai_reltol, ai_maxiter)

    # Steg 2: Beregn AI[i,j] = 0.5 * Py' * Vi * P * Vj * Py  (billige mat-vec)
    pairs = [(i, j) for i in 1:n_p for j in 1:i]
    ai_vals = zeros(length(pairs))
    _threaded_ai_matvec!(ai_vals, pairs, P_Vi_Py_list, Py, comps, model, workspaces)
    for (idx, (i_idx, j_idx)) in enumerate(pairs)
        AI[i_idx, j_idx] = ai_vals[idx]
        AI[j_idx, i_idx] = ai_vals[idx]
    end

    psi_mcse = NaN
    if estimate_ψ
        ψ_idx = n_p + 1
        λ = _psi_log_weights(model; centered=true)
        QPy = zeros(n)
        _apply_Q!(QPy, Py, model, workspaces[1])
        VdotPy = zeros(n)
        _apply_Qt!(VdotPy, λ .* Py, model, workspaces[1])
        @. VdotPy += λ * QPy

        trace_terms = zeros(k)
        Qz_list = [zeros(n) for _ in 1:k]
        Threads.@threads :static for j in 1:k
            tid = Threads.threadid()
            ws = workspaces[tid]
            Qz = Qz_list[j]
            _apply_Q!(Qz, z_list[j], model, ws)
            trace_terms[j] = 2 * dot(Pz_list[j], λ .* Qz)
        end
        score = vcat(score, 0.5 * (dot(Py, VdotPy) - mean(trace_terms)))
        psi_mcse = 0.5 * std(trace_terms) / sqrt(k)

        PVdotPy = [zeros(n)]
        _threaded_ai_cg!(PVdotPy, [VdotPy], model, workspaces, preconditioners,
            ai_reltol, ai_maxiter)
        AI_aug = zeros(n_p + 1, n_p + 1)
        AI_aug[1:n_p, 1:n_p] .= AI
        for i in 1:n_p
            AI_aug[i, ψ_idx] = AI_aug[ψ_idx, i] = 0.5 * dot(Vi_Py_list[i], PVdotPy[1])
        end
        AI_aug[ψ_idx, ψ_idx] = 0.5 * dot(VdotPy, PVdotPy[1])
        AI = AI_aug
    end
    return return_psi_mcse ? (score, AI, psi_mcse) : (score, AI)
end

# --- Helper functions: each wraps a single @threads block to isolate closures ---

function _threaded_vi_py!(Vi_Py_list, Py, comps, model, workspaces)
    n_p = length(comps)
    Threads.@threads :static for i in 1:n_p
        tid = Threads.threadid()
        get_Vi_u!(Vi_Py_list[i], Py, comps[i], model, workspaces[tid])
    end
    return Vi_Py_list
end

function _threaded_trace!(trace_results, n_trace, k, Pz_list, z_list, comps, model, workspaces)
    Threads.@threads :static for flat_idx in 1:n_trace
        tid = Threads.threadid()
        ws = workspaces[tid]
        i = (flat_idx - 1) ÷ k + 1
        j = (flat_idx - 1) % k + 1
        get_Vi_u!(ws.vi_out, Pz_list[j], comps[i], model, ws)
        trace_results[flat_idx] = dot(z_list[j], ws.vi_out) / k
    end
    return trace_results
end

function _threaded_ai_cg!(P_Vi_Py_list, Vi_Py_list, model, workspaces, preconditioners, ai_reltol, ai_maxiter)
    n_p = length(Vi_Py_list)
    tmp_sols = [copy(model.sol_y) for _ in 1:n_p]
    Threads.@threads :static for j in 1:n_p
        tid = Threads.threadid()
        ws = workspaces[tid]
        solve_mme_direct!(tmp_sols[j], model, Vi_Py_list[j], ws, preconditioners[tid],
            reltol=ai_reltol, maxiter=ai_maxiter)
        compute_Px_from_sol!(P_Vi_Py_list[j], model, Vi_Py_list[j], tmp_sols[j], ws)
    end
    return P_Vi_Py_list
end

function _threaded_ai_matvec!(ai_vals, pairs, P_Vi_Py_list, Py, comps, model, workspaces)
    Threads.@threads :static for idx in eachindex(pairs)
        tid = Threads.threadid()
        ws = workspaces[tid]
        i_idx, j_idx = pairs[idx]
        get_Vi_u!(ws.vi_out, P_Vi_Py_list[j_idx], comps[i_idx], model, ws)
        ai_vals[idx] = 0.5 * dot(Py, ws.vi_out)
    end
    return ai_vals
end

function _threaded_z_solves!(sol_z, model, z_fixed_list, Pz_list, workspaces, preconditioners,
    k, cg_reltol, cg_maxiter)
    Threads.@threads :static for i in 1:k
        tid = Threads.threadid()
        ws = workspaces[tid]
        solve_mme_z!(sol_z[i], model, z_fixed_list[i], ws, preconditioners[tid],
            reltol=cg_reltol, maxiter=cg_maxiter)
        compute_Px_from_sol!(Pz_list[i], model, z_fixed_list[i], sol_z[i], ws)
    end
    return sol_z
end

# =====================================================================
# RESULTAT-STRUCT OG LAGRING (JLD2)
# =====================================================================

struct AiRemlResult
    # Varianskomponenter
    σ2_d::Float64
    σ2_s::Float64
    σ_ds::Float64
    σ2_e::Float64
    σ2_env::Vector{Float64}        # H env variances (empty if H=0)
    # BLUE og BLUP (fra sol_y)
    beta::Vector{Float64}      # Faste effekter (nf)
    a_d::Vector{Float64}       # Direkte genetiske effekter (N)
    a_s::Vector{Float64}       # Sosiale genetiske effekter (N)
    c_env::Vector{Vector{Float64}} # Env BLUPs (H vectors, each n_h)
    # Inferens
    AI::Matrix{Float64}        # Average Information matrise (n_p × n_p)
    se::Vector{Float64}        # Standardfeil (n_p)
    # Metadata
    converged::Bool
    n_iter::Int
    n_obs::Int                 # n (antall observasjoner)
    N_pedigree::Int            # N (antall i stamtavlen)
    k::Int                     # Antall Hutchinson-prober
    # Mapping: obs → stamtavle-ID (for å koble BLUP til individer)
    obs_to_id::Vector{Int}
    ψ::Float64                 # Dilution exponent used for this fit
    # Social peer environmental effect (σ²_pe = 0 and empty vectors when peer_env=false)
    peer_env::Bool
    σ2_pe::Float64
    p_env::Vector{Float64}     # Peer env BLUPs, one per roster member
    pe_ids::Vector{Int}        # Pedigree ID for each entry of p_env
    fixef_names::Vector{String}
    env_names::Vector{String}
    param_names::Vector{String}
    env_ids::Vector{Vector{Int}}
    beta_vcov::Union{Nothing,Matrix{Float64}}
    beta_se::Union{Nothing,Vector{Float64}}
    vcov::Matrix{Float64}
    score::Vector{Float64}
    initial_params::Vector{Float64}
    ai_params::Vector{Float64}
    step_crit::Float64
    alpha::Float64
    settings::AiRemlSettings
end

function fitted_state(model::SGEModel)
    fit = model.fit
    fit === nothing && throw(ArgumentError("Model has not been fitted; call fit_aireml! first"))
    return fit
end

"""Return a view of the fitted fixed effects (changes to the view change the model)."""
fixef(model::SGEModel) = (fitted_state(model); @view model.sol_y[model.layout.fixed])

"""Return views of the fitted direct, social, shared and peer environmental effects."""
function ranef(model::SGEModel)
    fitted_state(model)
    lay = model.layout
    return (a_d=view(model.sol_y, lay.direct), a_s=view(model.sol_y, lay.social),
        c_env=[view(model.sol_y, r) for r in lay.env], p_env=view(model.sol_y, lay.pe))
end

"""
    fixed_effect_vcov!(model)

Compute and cache `(X'V^-1 X)^-1` using one matrix-free MME solve per fixed
effect. Return the same model, including when the covariance is already cached.
Read the covariance from `model.fit.beta_vcov`. This optional post-fit operation
does not change the estimates.
"""
function fixed_effect_vcov!(model::SGEModel)
    fit = fitted_state(model)
    cached = fit.beta_vcov
    cached !== nothing && return model
    settings = fit.settings
    ws = model_workspace(model)
    precond = settings.use_block_precond ? get_preconditioner(model, ws) : get_diag_preconditioner(model)
    op = make_mme_operator(model, ws)
    dest = zeros(get_dim(model))
    covariance = zeros(model.layout.nf, model.layout.nf)
    for j in axes(covariance, 2)
        fill!(ws.rhs_buffer, 0.0)
        ws.rhs_buffer[j] = 1.0
        fill!(dest, 0.0)
        _, history = cg!(dest, op, ws.rhs_buffer; Pl=precond,
            reltol=settings.cg_reltol, maxiter=settings.cg_maxiter, log=true)
        history.isconverged || error(
            "Fixed-effect covariance CG did not converge for column $j after $(history.iters) iterations")
        # The operator is residual-scaled: C_scaled = sigma2_e * C.
        covariance[:, j] .= model.σ2_e .* view(dest, model.layout.fixed)
    end
    fit.beta_vcov = Matrix(Symmetric((covariance + covariance') / 2))
    return model
end

"""
    snapshot(model; include_fixef_se=true)

Make an independent inference result, without data, pedigree matrices or fitting
workspaces. Fixed-effect covariance/SEs require extra solves on the first call;
set `include_fixef_se=false` to omit them explicitly. BLUP prediction-error
variances are not computed.
"""
function snapshot(model::SGEModel; include_fixef_se::Bool=true)
    fit = fitted_state(model)
    if include_fixef_se
        fixed_effect_vcov!(model)
    end
    beta_vcov = include_fixef_se ? copy(fit.beta_vcov) : nothing
    effects = ranef(model)
    return AiRemlResult(
        model.σ2_d, model.σ2_s, model.σ_ds, model.σ2_e, copy(model.σ2_env),
        copy(fixef(model)), copy(effects.a_d), copy(effects.a_s), [copy(c) for c in effects.c_env],
        copy(fit.AI), copy(fit.se), fit.converged, fit.n_iter,
        length(model.y), size(model.Ainv, 1), fit.settings.k,
        copy(model.groups.obs_to_id), model.ψ, model.peer_env,
        model.peer_env ? model.σ2_pe : 0.0, copy(effects.p_env), copy(model.pe_ids),
        copy(model.fixef_names), copy(model.env_names), get_param_names(model),
        copy.(model.env_ids), beta_vcov,
        beta_vcov === nothing ? nothing : sqrt.(diag(beta_vcov)),
        copy(fit.vcov), copy(fit.score), copy(fit.initial_params), copy(fit.ai_params),
        fit.step_crit, fit.alpha, fit.settings)
end

function save_results(result::AiRemlResult, filename::String)
    jldsave(filename; result=result)
    sz_mb = filesize(filename) / 1024^2
    @printf("Resultat lagret til %s (%.1f MB)\n", filename, sz_mb)
end

function load_results(filename::String)
    return load(filename, "result")
end

"""Save the complete fitted model, including data and pedigree inputs."""
function save_model(model::SGEModel, filename::String)
    fitted_state(model)
    jldsave(filename; model)
    return filename
end

load_model(filename::String) = load(filename, "model")

function update_params!(model::SGEModel, delta, damping)
    # Foreslåtte endringer
    new_d2 = model.σ2_d + damping * delta[1]
    new_s2 = model.σ2_s + damping * delta[2]
    new_ds = model.σ_ds + damping * delta[3]
    new_e2 = model.σ2_e + damping * delta[4]
    # Sikkerhetsmekanismer (Bending)
    model.σ2_d = max(1e-4, new_d2)
    model.σ2_s = max(1e-6, new_s2)
    model.σ2_e = max(1e-2, new_e2)
    # Hold korrelasjonen i sjakk: |rho| < 0.99
    max_cov = 0.99 * sqrt(model.σ2_d * model.σ2_s)
    model.σ_ds = clamp(new_ds, -max_cov, max_cov)
    # Env variances
    for h in 1:length(model.σ2_env)
        model.σ2_env[h] = max(1e-6, model.σ2_env[h] + damping * delta[4+h])
    end
    if model.peer_env
        model.σ2_pe = max(1e-6, model.σ2_pe + damping * delta[pe_param_index(model)])
    end
    return model
end

const BOUNDARY_TOL = 1e-5

# Maximize score'Δ - Δ'AI_reg*Δ/2 with Δ_i >= 0 for variances near their floor.
# Release active constraints using the reduced quadratic's gradient, not the full
# unconstrained step: correlations can otherwise pin a component with a positive score.
function _boundary_constrained_delta(AI_reg, score, params;
    variance_count::Int=length(params))
    candidates = [i for i in vcat(2, collect(5:variance_count))
        if params[i] <= BOUNDARY_TOL]
    delta = AI_reg \ score
    all(i -> delta[i] >= 0, candidates) && return delta, Int[]

    active = copy(candidates)
    fill!(delta, 0.0)
    for _ in 1:max(100, 10 * length(score)^2)
        free = setdiff(eachindex(score), active)
        trial = zeros(length(score))
        trial[free] = AI_reg[free, free] \ score[free]
        blocking = [i for i in candidates if !(i in active) && trial[i] < 0]
        if !isempty(blocking)
            fractions = [delta[i] / (delta[i] - trial[i]) for i in blocking]
            fraction, pos = findmin(fractions)
            delta .+= fraction .* (trial .- delta)
            i = blocking[pos]
            delta[i] = 0.0
            push!(active, i)
            continue
        end

        delta = trial
        gradient = score - AI_reg * delta
        tol = 100 * eps(Float64) *
            max(1.0, norm(score, Inf), norm(AI_reg, Inf) * norm(delta, 1))
        isempty(active) && return delta, active
        inward, pos = findmax(gradient[active])
        inward <= tol && return delta, sort(active)
        deleteat!(active, pos)
    end
    error("Boundary-constrained AI step did not satisfy its optimality conditions")
end

function _boundary_constrained_delta(model::SGEModel, AI_reg, score)
    return _boundary_constrained_delta(AI_reg, score, get_params(model);
        variance_count=4 + length(model.σ2_env) + Int(model.peer_env))
end

function _psi_reference_size(model::SGEModel)
    sizes = [length(members) for members in model.groups.members_id if length(members) > 1]
    isempty(sizes) && throw(ArgumentError("Cannot estimate ψ: no peer groups with at least two members"))
    length(unique(sizes)) > 1 ||
        throw(ArgumentError("Cannot estimate ψ: peer-group roster sizes do not vary"))
    return median(sizes)
end

function _set_group_scale!(model::SGEModel; centered::Bool)
    ref_peers = centered ? _psi_reference_size(model) - 1 : 1.0
    model.group_scale .= [
        length(members) <= 1 ? 0.0 :
        ((length(members) - 1.0) / ref_peers)^(-model.ψ)
        for members in model.groups.members_id
    ]
    if model.peer_env
        model.pe_ids, model.pe_data_diag = build_peer_env_levels(model.groups, model.group_scale)
    end
    return model
end

function _convert_psi_scale!(model::SGEModel; to_centered::Bool)
    ref_peers = _psi_reference_size(model) - 1
    factor = ref_peers^model.ψ
    if to_centered
        model.σ2_s /= factor^2
        model.σ_ds /= factor
        model.peer_env && (model.σ2_pe /= factor^2)
        model.sol_y[model.layout.social] ./= factor
        model.peer_env && (model.sol_y[model.layout.pe] ./= factor)
    else
        model.σ2_s *= factor^2
        model.σ_ds *= factor
        model.peer_env && (model.σ2_pe *= factor^2)
        model.sol_y[model.layout.social] .*= factor
        model.peer_env && (model.sol_y[model.layout.pe] .*= factor)
    end
    _set_group_scale!(model; centered=to_centered)
    return model
end

function adapt_damping(α::Float64, step_crit::Float64, previous_step_crit::Float64;
    enabled::Bool=true)
    if !enabled || !isfinite(previous_step_crit)
        return α
    elseif step_crit < 0.9 * previous_step_crit
        return min(1.0, 1.5 * α)
    elseif step_crit > 1.1 * previous_step_crit
        return max(1e-4, 0.5 * α)
    end
    return α
end

# Draw k fixed Gaussian probe vectors z_j ∈ ℝⁿ for the Hutchinson trace
# estimator tr(P*V_i) ≈ (1/k) Σ_j z_j' (V_i (P z_j)). Seeding here keeps the
# probes deterministic and identical across every iteration of run_aireml,
# which is required for the trace estimate to be comparable between iterations.
function make_trace_probes(n::Int, k::Int; seed::Int=42)
    Random.seed!(seed)
    return [randn(n) for _ in 1:k]
end

"""
    fit_aireml!(model; refit=false, kwargs...)

Fit in place and return `model`. Variances and `sol_y` are authoritative;
`model.fit` holds inference and convergence metadata, not duplicate BLUPs.
Use `refit=true` to continue from a previously fitted model. Fitting scratch
is local to this call. An interrupted/failed refit clears the old fit state.
"""
function fit_aireml!(model::SGEModel; max_iter=15, k=5, damping=0.2,
    cg_maxiter=2000, cg_reltol=1e-7,
    cg_reltol_early=1e-5, cg_early_iters=3,
    ai_maxiter=500, ai_reltol=1e-5,
    use_block_precond=true, verbose_cg=true,
    adaptive_damping::Bool=true,
    log_file::String="",
    seed::Int=42, refit::Bool=false,
    estimate_ψ::Bool=false, ψ_bounds::Tuple{Float64,Float64}=(0.0, 1.0),
    ψ_max_step::Float64=0.25)
    isfinite(damping) && damping > 0 ||
        throw(ArgumentError("damping must be finite and positive"))
    max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
    k > 0 || throw(ArgumentError("k must be positive"))
    estimate_ψ && k < 2 &&
        throw(ArgumentError("Estimating ψ requires k ≥ 2 to estimate its Monte Carlo SE"))
    cg_maxiter > 0 && ai_maxiter > 0 ||
        throw(ArgumentError("CG iteration limits must be positive"))
    all(t -> isfinite(t) && t > 0, (cg_reltol, cg_reltol_early, ai_reltol)) ||
        throw(ArgumentError("CG tolerances must be finite and positive"))
    cg_early_iters >= 0 || throw(ArgumentError("cg_early_iters must be nonnegative"))
    isfinite(ψ_max_step) && ψ_max_step > 0 ||
        throw(ArgumentError("ψ_max_step must be finite and positive"))
    all(isfinite, ψ_bounds) && ψ_bounds[1] < ψ_bounds[2] ||
        throw(ArgumentError("ψ_bounds must be finite and strictly increasing"))
    estimate_ψ && !(ψ_bounds[1] <= model.ψ <= ψ_bounds[2]) &&
        throw(ArgumentError("Starting ψ=$(model.ψ) is outside ψ_bounds=$ψ_bounds"))
    model.fit === nothing || refit ||
        throw(ArgumentError("Model is already fitted; pass refit=true to fit again"))
    previously_estimated_ψ = _fit_estimates_psi(model)
    settings = AiRemlSettings(; max_iter, k, damping, cg_maxiter, cg_reltol,
        cg_reltol_early, cg_early_iters, ai_maxiter, ai_reltol,
        use_block_precond, verbose_cg, adaptive_damping, log_file, seed)
    model.fit = nothing
    n = length(model.y)
    N = size(model.Ainv, 1)
    dim = length(model.sol_y)
    H = length(model.env_ids)
    param_names = get_param_names(model; include_ψ=estimate_ψ)
    n_p = length(param_names)

    nt = Threads.nthreads()
    max_tid = Threads.maxthreadid()   # Julia 1.12: threadid() kan returnere > nthreads()

    sol_z = [zeros(dim) for _ in 1:k]

    # Vi må bruke de SAMME z-vektorene i hver iterasjon
    # for at Hutchinson-estimatet skal være sammenlignbart (se make_trace_probes)
    z_fixed_list = make_trace_probes(n, k; seed=seed)

    # Preallokkerte buffere for Px-beregninger (gjenbrukes per iterasjon)
    Py = Vector{Float64}(undef, n)
    Pz_list = [Vector{Float64}(undef, n) for _ in 1:k]

    # Workspaces for CG-solves og trace-estimering: ett per tråd (indeksert med threadid())
    workspaces = create_thread_workspaces(model, max_tid)

    factor_error = validate_pedigree_factor(model, workspaces[1])
    @printf("Pedigree factor check: max relative residual=%.2e\n", factor_error)

    if estimate_ψ != previously_estimated_ψ
        _convert_psi_scale!(model; to_centered=estimate_ψ)
    end
    initial_params = get_params(model; include_ψ=estimate_ψ)

    precond_name = use_block_precond ? "Block" : "Diag"
    println("Starter AI-REML: k=$k, damping=$damping, precond=$precond_name, cg_maxiter=$cg_maxiter, threads=$nt")
    println("CG reltol schedule: early=$cg_reltol_early (iter <= $cg_early_iters), late=$cg_reltol")
    println("N=$( size(model.Ainv,1)), n=$n, dim=$dim")
    println("-"^70)
    flush(stdout)

    # CSV-logg for ekstern overvåking (tail -f log.csv)
    use_log = !isempty(log_file)
    if use_log
        open(log_file, "w") do io
            pe_cols = model.peer_env ? ",s2pe" : ""
            pe_se_cols = model.peer_env ? ",se_pe" : ""
            psi_se_cols = estimate_ψ ? ",se_psi" : ""
            param_cols = join(["s2d","s2s","sds","s2e", ["s2env$h" for h in 1:H]...], ",") * pe_cols * ",psi"
            se_cols = join(["se_d","se_s","se_ds","se_e", ["se_env$h" for h in 1:H]...], ",") * pe_se_cols * psi_se_cols
            println(io, "iter,time_y,time_z,time_ai,time_total,cum_time,eta_min,",
                param_cols, ",", se_cols, ",",
                "score_norm,step_crit,alpha,rho_ds,cond_G,converged",
                estimate_ψ ? ",psi_score_mcse" : "")
        end
        println("Iterasjonslogg skrives til: $log_file")
    end

    # Variabler som trengs etter loopen (for AiRemlResult)
    AI = zeros(n_p, n_p)
    step_crit = Inf
    previous_step_crit = Inf
    alpha_next = min(Float64(damping), 1.0)
    last_iter = 0
    last_alpha = NaN
    score = zeros(n_p)
    ai_params = copy(initial_params)
    iter_times = Float64[]   # for ETA-beregning
    t_start_total = time()

    for iter in 1:max_iter
        last_iter = iter
        t_iter_start = time()
        iter_cg_reltol = iter <= cg_early_iters ? cg_reltol_early : cg_reltol

        # A. Oppdater preconditionere basert på nye varianser
        if use_block_precond
            preconditioners = [get_preconditioner(model, ws) for ws in workspaces]
        else
            preconditioners = fill(get_diag_preconditioner(model), length(workspaces))
        end

        # 1. Oppdater modellens egne løsninger (Warm Start) — y-solve er sekvensiell
        t_y = @elapsed begin
            solve_mme_y!(model, workspaces[1], preconditioners[1], reltol=iter_cg_reltol, maxiter=cg_maxiter, verbose=verbose_cg)
            compute_Px_from_sol!(Py, model, model.y, model.sol_y, workspaces[1])
        end

        # 2. Parallelliser k z-solves
        if verbose_cg
            @printf("    Z-solves: k=%d (threaded)...\n", k)
            flush(stdout)
        end
        t_z = @elapsed begin
            _threaded_z_solves!(sol_z, model, z_fixed_list, Pz_list, workspaces, preconditioners,
                k, iter_cg_reltol, cg_maxiter)
        end
        if verbose_cg
            @printf("    Z-solves ferdig: %.1fs\n", t_z)
            flush(stdout)
        end

        # 3. Beregn Score og AI
        if verbose_cg
            println("    Score + AI...")
            flush(stdout)
        end
        t_ai = @elapsed begin
            if estimate_ψ
                score, AI_new, psi_mcse = compute_score_and_ai(model, Py, z_fixed_list, Pz_list,
                    workspaces, preconditioners,
                    ai_reltol=ai_reltol, ai_maxiter=ai_maxiter, verbose=verbose_cg,
                    estimate_ψ=true, return_psi_mcse=true)
            else
                score, AI_new = compute_score_and_ai(model, Py, z_fixed_list, Pz_list,
                    workspaces, preconditioners,
                    ai_reltol=ai_reltol, ai_maxiter=ai_maxiter, verbose=verbose_cg)
            end
            copyto!(AI, AI_new)
        end
        if estimate_ψ
            @printf("    ψ score Monte Carlo SE=%.3e (k=%d)\n", psi_mcse, k)
        end
        if verbose_cg
            @printf("    Score + AI ferdig: %.1fs\n", t_ai)
            flush(stdout)
        end

        # 4. Løs systemet og beregn Standardfeil (SE)
        AI_reg = AI + 1e-6 * I
        delta, boundary_active = _boundary_constrained_delta(model, AI_reg, score)
        if !isempty(boundary_active)
            names = get_param_names(model; include_ψ=estimate_ψ)
            println("    Fixed at lower bound this iteration: ",
                join(names[boundary_active], ", "))
        end
        AI_inv = inv(AI_reg)
        se = sqrt.(abs.(diag(AI_inv)))
        applied_delta = copy(delta)
        ψ_idx = n_p
        if estimate_ψ
            applied_delta[ψ_idx] = clamp(delta[ψ_idx], -ψ_max_step, ψ_max_step)
            applied_delta[ψ_idx] =
                clamp(model.ψ + applied_delta[ψ_idx], ψ_bounds...) - model.ψ
        end

        # 5. Lagre AI-evalueringspunktet (brukes av show for Δparam)
        copyto!(ai_params, get_params(model; include_ψ=estimate_ψ))

        # 6. Step-halving: finn steglengde som holder alle varianser > 0 og G positiv definitt.
        # Komponenter allerede på gulvet med utovervendt steg er holdt fast over (aktiv mengde).
        # Må ikke erstattes av ren komponentvis projeksjon i update_params!: et overskytende
        # AI-steg sender da flere varianser til gulvet i én iterasjon, og AI-REML klatrer
        # deretter bare ca. faktor (1+α) per iterasjon tilbake (scorer i millioner).
        α = adaptive_damping ? alpha_next : min(Float64(damping), 1.0)
        α_start = α
        for _attempt in 1:20
            new_d = model.σ2_d + α * applied_delta[1]
            new_s = model.σ2_s + α * applied_delta[2]
            new_ds = model.σ_ds + α * applied_delta[3]
            new_e = model.σ2_e + α * applied_delta[4]
            env_ok = all(model.σ2_env[h] + α * applied_delta[4+h] > 1e-8 for h in 1:H)
            pe_ok = !model.peer_env || model.σ2_pe + α * applied_delta[pe_param_index(model)] > 1e-8
            ψ_ok = !estimate_ψ || ψ_bounds[1] <= model.ψ + α * applied_delta[ψ_idx] <= ψ_bounds[2]
            if new_d > 1e-6 && new_s > 1e-8 && new_e > 1e-4 &&
                new_d * new_s > new_ds^2 && env_ok && pe_ok && ψ_ok
                break
            end
            α *= 0.5
        end

        # 7. Oppdater parametere med safe steglengde (update_params! gulv er siste skanse)
        update_params!(model, applied_delta, α)
        if estimate_ψ
            model.ψ = clamp(model.ψ + α * applied_delta[ψ_idx], ψ_bounds...)
            _set_group_scale!(model; centered=true)
        end
        last_alpha = α
        new_params = get_params(model; include_ψ=estimate_ψ)

        # --- UTSKRIFT ---
        t_iter = time() - t_iter_start
        push!(iter_times, t_iter)
        cum_time = time() - t_start_total
        avg_iter = mean(iter_times)
        remaining_iters = max_iter - iter
        eta_min = remaining_iters * avg_iter / 60.0

        # G-matrise helse
        rho_ds, cond_G, _ = _g_health(model.σ2_d, model.σ2_s, model.σ_ds)
        score_norm = norm(score)

        # Use the undamped AI correction in units of its estimated standard errors;
        # a tiny feasible/damped step alone is not evidence of convergence.
        step_crit = maximum(abs.(applied_delta) ./ max.(se, eps()))
        converged = isfinite(step_crit) && step_crit < 1e-2

        # Iterasjonsrapport = samme rapport som display(model). model.fit settes ikke her:
        # sol_y hører til variansene før oppdateringen, så modellen er ikke ferdig tilpasset.
        println("\n", "="^90)
        @printf("ITERASJON %d  |  %.1fs (y:%.1f z:%.1f ai:%.1f)  cum:%.0fs  ETA:%.0fmin  |  ψ=%.8g (%s)  |  CG reltol %.0e\n",
            iter, t_iter, t_y, t_z, t_ai, cum_time, eta_min, model.ψ,
            estimate_ψ ? "estimated" : "fixed", iter_cg_reltol)
        println("="^90)
        iter_fit = AiRemlFit(AI, AI_inv, se, score, initial_params, ai_params,
            converged, iter, step_crit, α, settings, nothing)
        _show_model(stdout, model, iter_fit; in_progress=true)
        flush(stdout)

        # CSV-logg (med flush for ekstern overvåking)
        if use_log
            open(log_file, "a") do io
                @printf(io, "%d,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,",
                    iter, t_y, t_z, t_ai, t_iter, cum_time, eta_min)
                for p in new_params; @printf(io, "%.6f,", p); end
                !estimate_ψ && @printf(io, "%.6f,", model.ψ)
                for s in se; @printf(io, "%.6f,", s); end
                @printf(io, "%.6e,%.6e,%.6e,%.4f,%.1f,%s",
                    score_norm, step_crit, α, rho_ds, cond_G,
                    converged ? "true" : "false")
                estimate_ψ && @printf(io, ",%.6e", psi_mcse)
                println(io)
                flush(io)
            end
        end

        if converged
            println("\n✅ AI-REML har konvergert 😄.")
            break
        end
        # Adapt from the attempted α, not the feasibility-halved one: a boundary-limited
        # α≈1e-7 would otherwise become the next start, and from there the criterion can
        # never improve 10% per iteration, so α would never recover. Re-halving is cheap.
        alpha_next = adapt_damping(α_start, step_crit, previous_step_crit;
            enabled=adaptive_damping)
        previous_step_crit = step_crit
        if iter == max_iter
            println("\n⚠️ Maks iterasjoner nådd uten full konvergens.")
        end
    end

    # Final solve med endelige varianskomponenter → BLUE(β) og BLUP(a_d, a_s)
    println("\nBeregner endelige BLUE/BLUP-estimater...")
    if use_block_precond
        final_precond = get_preconditioner(model, workspaces[1])
    else
        final_precond = get_diag_preconditioner(model)
    end
    fill!(model.sol_y, 0.0)
    solve_mme_y!(model, workspaces[1], final_precond, reltol=cg_reltol, maxiter=cg_maxiter, verbose=verbose_cg)

    vcov = inv(AI + 1e-6 * I)
    model.fit = AiRemlFit(AI, vcov, sqrt.(abs.(diag(vcov))), score,
        initial_params, ai_params, isfinite(step_crit) && step_crit < 1e-2,
        last_iter, step_crit, last_alpha, settings, nothing)
    return model
end

"""
    run_aireml(model; kwargs...)

Compatibility interface: fit in place and return a snapshot as before.
Fixed-effect SEs are omitted to preserve the previous computational cost.
New code should use `fit_aireml!` and explicitly request `snapshot` if needed.
"""
function run_aireml(model::SGEModel; refit::Bool=true, kwargs...)
    fit_aireml!(model; refit, kwargs...)
    return snapshot(model; include_fixef_se=false)
end
