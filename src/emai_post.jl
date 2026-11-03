# =====================================================================
# POST-FIT DIAGNOSTICS
# Quantities evaluated at the model's current variance parameters (the estimates
# after fit_aireml!) that AI-REML itself never needs:
#   stochastic_loglik — restricted log-likelihood with a stochastic log|C|
#   loglik_difference — paired difference of two stochastic_loglik results
#   psi_score         — score and score test for the dilution exponent ψ
# Never forms dense A/V/MME. Included via emai.jl.
# =====================================================================

# Fresh BLUP solve at the current parameters, without touching model.sol_y.
function _post_solve_y(model::SGEModel, ws::ThreadWorkspace, precond; reltol, maxiter)
    sol = copy(model.sol_y)
    _solve_mme!(sol, model.y, model, ws, precond; reltol, maxiter, verbose=false, label="post-y")
    Py = zeros(length(model.y))
    compute_Px_from_sol!(Py, model, model.y, sol, ws)
    return sol, Py
end

function _post_setup(model::SGEModel)
    workspaces = create_thread_workspaces(model, Threads.maxthreadid())
    preconditioners = [get_preconditioner(model, ws) for ws in workspaces]
    return workspaces, preconditioners
end

function _post_probes(probes, len::Int, K::Int, seed::Int)
    if probes !== nothing
        size(probes, 1) == len ||
            throw(DimensionMismatch("probes must have $len rows, got $(size(probes, 1))"))
        return [Vector{Float64}(probes[:, j]) for j in axes(probes, 2)]
    end
    K >= 2 || throw(ArgumentError("K must be at least 2 to estimate a Monte Carlo SE"))
    # Rademacher probes, one RNG per probe: identical probes across models of equal size.
    return [rand(MersenneTwister(seed + j), (-1.0, 1.0), len) for j in 1:K]
end

_mc_se(x::AbstractVector) = length(x) > 1 ? std(x) / sqrt(length(x)) : NaN

# ---------------------------------------------------------------------
# Stochastic REML log-likelihood
# ---------------------------------------------------------------------

# Whitening factor W of the block preconditioner M = W W':
#   M = blockdiag(diag(X'X), σ²_e G⁻¹ ⊗ Ainv, diag(counts + σ²_e/σ²_h), diag(d_pe + σ²_e/σ²_pe))
#   genetic block: W_g = √σ²_e (U ⊗ L) with G⁻¹ = U U' and Ainv = L L'.
struct _Whitening
    sqrt_diag::Vector{Float64}   # √M for every diagonal entry (fixed, env, pe; genetic slots unused)
    Uinv::Matrix{Float64}        # U⁻¹ (2×2 lower)
    sqrt_σ2e::Float64
    logdet_M::Float64
end

function _whitening(model::SGEModel)
    lay = model.layout
    d = ones(lay.dim)
    for j in 1:lay.nf
        d[j] = max(dot(@view(model.X[:, j]), @view(model.X[:, j])), 1e-6)
    end
    for h in eachindex(model.env_ids)
        λ_h = model.σ2_e / model.σ2_env[h]
        d[lay.env[h]] .= model.env_counts[h] .+ λ_h
    end
    if model.peer_env
        d[lay.pe] .= model.pe_data_diag .+ model.σ2_e / model.σ2_pe
    end
    G = [model.σ2_d model.σ_ds; model.σ_ds model.σ2_s]
    U = cholesky(Symmetric(inv(G))).L
    N = lay.N
    logdet_Ainv = 2 * sum(log, diag(parent(model.L_Ainv)))
    logdet_M = sum(log, @view d[lay.fixed]) +
               sum(h -> sum(log, @view d[lay.env[h]]), eachindex(model.env_ids); init=0.0) +
               (model.peer_env ? sum(log, @view d[lay.pe]) : 0.0) +
               2N * log(model.σ2_e) - N * logdet(G) + 2 * logdet_Ainv
    return _Whitening(sqrt.(d), Matrix(inv(U)), sqrt(model.σ2_e), logdet_M), logdet_Ainv
end

# out = W⁻¹ x (transpose=false) or W⁻ᵀ x (transpose=true); t1, t2 are length-N buffers.
function _whiten!(out, x, Wh::_Whitening, model::SGEModel, t1, t2; transpose::Bool)
    lay = model.layout
    nf, N = lay.nf, lay.N
    L = model.L_Ainv
    Ui = Wh.Uinv
    @inbounds for i in eachindex(out)
        out[i] = x[i] / Wh.sqrt_diag[i]
    end
    xd = @view x[lay.direct]
    xs = @view x[lay.social]
    if !transpose
        # (U⁻¹ ⊗ L⁻¹) x / √σ²_e
        copyto!(t1, xd); ldiv!(L, t1)
        copyto!(t2, xs); ldiv!(L, t2)
        @inbounds for i in 1:N
            out[nf+i] = Ui[1, 1] * t1[i] / Wh.sqrt_σ2e
            out[nf+N+i] = (Ui[2, 1] * t1[i] + Ui[2, 2] * t2[i]) / Wh.sqrt_σ2e
        end
    else
        # (U⁻ᵀ ⊗ L⁻ᵀ) x / √σ²_e
        @inbounds for i in 1:N
            t1[i] = (Ui[1, 1] * xd[i] + Ui[2, 1] * xs[i]) / Wh.sqrt_σ2e
            t2[i] = Ui[2, 2] * xs[i] / Wh.sqrt_σ2e
        end
        ldiv!(L', t1); ldiv!(L', t2)
        copyto!(@view(out[lay.direct]), t1)
        copyto!(@view(out[lay.social]), t2)
    end
    return out
end

# Gauss quadrature estimate of z' log(B) z from the Lanczos tridiagonal (α, β).
function _lanczos_logquad(α, β)
    F = eigen(SymTridiagonal(α, β))
    minimum(F.values) > 0 || error("Lanczos produced a non-positive Ritz value; B is not positive definite")
    return sum(F.vectors[1, k]^2 * log(F.values[k]) for k in eachindex(F.values))
end

# Stochastic Lanczos quadrature for one probe; no reorthogonalisation (O(dim) memory).
function _slq_probe!(z, apply_B!, v, vprev, w; maxiter::Int, tol::Float64)
    nz2 = dot(z, z)
    v .= z ./ sqrt(nz2)
    fill!(vprev, 0.0)
    α, β = Float64[], Float64[]
    est_old = NaN
    est = NaN
    next_check = 5
    for j in 1:maxiter
        apply_B!(w, v)
        a = dot(w, v)
        push!(α, a)
        bprev = j > 1 ? β[end] : 0.0
        @. w = w - a * v - bprev * vprev
        b = norm(w)
        breakdown = b <= 1e-12 * abs(a)
        # Geometric check schedule keeps the O(j²) tridiagonal eigensolves cheap for long runs.
        if j >= next_check || breakdown || j == maxiter
            next_check = j + max(5, j ÷ 10)
            est = _lanczos_logquad(α, β)
            if breakdown || abs(est - est_old) <= tol * max(1.0, abs(est))
                return nz2 * est, j
            end
            est_old = est
        end
        push!(β, b)
        copyto!(vprev, v)
        @. v = w / b
    end
    @warn "Lanczos quadrature did not reach tolerance" maxiter tol
    return nz2 * est, maxiter
end

"""
    stochastic_loglik(model; K=30, seed=2026, probes=nothing, lanczos_maxiter=2000,
                      lanczos_tol=1e-8, reltol=1e-8, maxiter=2000)

Restricted log-likelihood at the model's current variance parameters, by the
Henderson identity

    -2ℓ = n log σ²_e + log|Γ| + log|C| + y'Py,      Γ = blockdiag(G⊗A, σ²_h I, σ²_pe I),

where C is the (unscaled) MME coefficient matrix. Everything is exact except
log|C| = log|M| + tr log(W⁻¹ C W⁻ᵀ), with M = WW' the block preconditioner;
the trace is estimated by stochastic Lanczos quadrature with `K` Rademacher
probes (or the columns of `probes`). Constants are omitted
(ℓ = -½(log|V| + log|X'V⁻¹X| + y'Py)), so only differences between models
with the same y and X are meaningful. Use [`loglik_difference`](@ref) with the
same `seed` to compare models with paired probes.
"""
function stochastic_loglik(model::SGEModel; K::Int=30, seed::Int=2026, probes=nothing,
    lanczos_maxiter::Int=2000, lanczos_tol::Float64=1e-8, reltol=1e-8, maxiter=2000)
    lay = model.layout
    n, N, dim = length(model.y), lay.N, lay.dim
    workspaces, preconditioners = _post_setup(model)
    _, Py = _post_solve_y(model, workspaces[1], preconditioners[1]; reltol, maxiter)
    yPy = dot(model.y, Py)

    Wh, logdet_Ainv = _whitening(model)
    zs = _post_probes(probes, dim, K, seed)
    K = length(zs)
    nt = length(workspaces)
    bufs = [(v=zeros(dim), vprev=zeros(dim), w=zeros(dim), t=zeros(dim), t1=zeros(N), t2=zeros(N))
            for _ in 1:nt]
    trlogB = zeros(K)
    steps = zeros(Int, K)
    Threads.@threads :static for j in 1:K
        tid = Threads.threadid()
        b = bufs[tid]
        op = make_mme_operator(model, workspaces[tid])
        apply_B! = (out, x) -> begin
            _whiten!(b.t, x, Wh, model, b.t1, b.t2; transpose=true)
            op(out, b.t)
            _whiten!(out, copy!(b.t, out), Wh, model, b.t1, b.t2; transpose=false)
        end
        trlogB[j], steps[j] = _slq_probe!(zs[j], apply_B!, b.v, b.vprev, b.w;
            maxiter=lanczos_maxiter, tol=lanczos_tol)
    end

    # The operator is residual-scaled: C_op = σ²_e C, so log|C| = log|C_op| - dim log σ²_e.
    logdet_C = Wh.logdet_M + mean(trlogB) - dim * log(model.σ2_e)
    G = [model.σ2_d model.σ_ds; model.σ_ds model.σ2_s]
    logdet_Γ = N * logdet(G) - 2 * logdet_Ainv +
               sum(h -> model.env_n_levels[h] * log(model.σ2_env[h]), eachindex(model.env_ids); init=0.0) +
               (model.peer_env ? length(model.pe_ids) * log(model.σ2_pe) : 0.0)
    loglik = -0.5 * (n * log(model.σ2_e) + logdet_Γ + logdet_C + yPy)
    return (; loglik, se=0.5 * _mc_se(trlogB), logdet_C, logdet_Γ, yPy,
        trlogB, lanczos_steps=steps, K, seed, dim, params=get_params(model), ψ=model.ψ)
end

"""
    loglik_difference(a, b)

`a.loglik - b.loglik` for two [`stochastic_loglik`](@ref) results, with a paired
Monte Carlo SE when both used the same probes (same `seed` and dimension), and
an independent SE otherwise.
"""
function loglik_difference(a, b)
    Δ = a.loglik - b.loglik
    paired = a.dim == b.dim && a.seed == b.seed && a.K == b.K
    se = paired ? 0.5 * _mc_se(a.trlogB .- b.trlogB) : sqrt(a.se^2 + b.se^2)
    return (; Δ, se, paired)
end

# ---------------------------------------------------------------------
# Score for the dilution exponent ψ
# ---------------------------------------------------------------------
# Z_s(ψ) = D_c Z_s1, c_g = (m_g-1)^(-ψ)  ⇒  ∂Z_s/∂ψ = Λ Z_s with Λ_ii = -log(m_{g(i)}-1).
# With Q = Z_s(σ_ds A Z_d' + σ²_s A Z_s' + σ²_pe Z_s'), V̇ = ΛQ + Q'Λ, so
#   y'PV̇Py = 2 (Py)'Λ ŝ  where ŝ = Q Py = Z_s â_s + Z_pe p̂ (fitted social part),
#   tr(PV̇)  = 2 E[(Pz)'Λ(Qz)].
# (Z_pe Z_pe' = Z_s Z_s', since Z_s has nonzero columns only for roster members.)

function _psi_log_weights(model::SGEModel; centered::Bool=_fit_estimates_psi(model))
    gs = model.groups
    ref_peers = centered ? _psi_reference_size(model) - 1 : 1.0
    lg = [length(m) <= 1 ? 0.0 :
          -log((length(m) - 1.0) / ref_peers) for m in gs.members_id]
    return [lg[g] for g in gs.obs_to_group]
end

# out = Z_s x_N (diluted, self-excluded peer sum)
function _apply_Zs!(out, x_N, model::SGEModel, ws::ThreadWorkspace)
    gs = model.groups
    compute_group_sums!(ws.group_sum, x_N, gs.members_id)
    @inbounds for (i, tid) in enumerate(gs.obs_to_id)
        g = gs.obs_to_group[i]
        out[i] = (ws.group_sum[g] - x_N[tid]) * model.group_scale[g]
    end
    return out
end

# out = Q z. Uses ws.tmp_N1..3 and ws.op_Ad/op_As.
function _apply_Q!(out, z, model::SGEModel, ws::ThreadWorkspace)
    s = ws.tmp_N1
    fill!(s, 0.0)
    apply_Zs_transpose!(s, z, model.groups, ws.group_sum, model.group_scale)
    x = ws.op_Ad
    @. x = model.σ2_s * s
    @inbounds for (i, id) in enumerate(model.groups.obs_to_id)
        x[id] += model.σ_ds * z[i]
    end
    Ax = ws.op_As
    apply_A!(Ax, model.L_Ainv, x, ws.tmp_N2)
    model.peer_env && (@. Ax += model.σ2_pe * s)
    return _apply_Zs!(out, Ax, model, ws)
end

# out = Q' u. Uses ws.tmp_N1..3, ws.op_As and ws.vi_out.
function _apply_Qt!(out, u, model::SGEModel, ws::ThreadWorkspace)
    s = ws.tmp_N1
    fill!(s, 0.0)
    apply_Zs_transpose!(s, u, model.groups, ws.group_sum, model.group_scale)
    As = ws.op_As
    apply_A!(As, model.L_Ainv, s, ws.tmp_N2)
    _apply_Zs!(out, As, model, ws)
    out .*= model.σ2_s
    @inbounds for (i, id) in enumerate(model.groups.obs_to_id)
        out[i] += model.σ_ds * As[id]
    end
    if model.peer_env
        _apply_Zs!(ws.vi_out, s, model, ws)
        @. out += model.σ2_pe * ws.vi_out
    end
    return out
end

"""
    psi_score(model; K=30, seed=2026, probes=nothing, reltol=1e-8, maxiter=2000)

Score of the restricted log-likelihood for the dilution exponent ψ at the
model's current parameters (ψ = `model.ψ`, held fixed during fitting):

    s_ψ = ½ (y'P V̇ P y - tr(P V̇)),   V̇ = ∂V/∂ψ.

The quadratic term is exact; the trace uses `K` Rademacher probes (or the
columns of `probes`, length n), and `score_se` is its Monte Carlo SE.
`AI` is the Average Information matrix augmented with ψ (last row/column),
`efficient_info` = AI_ψψ - AI_ψθ AI_θθ⁻¹ AI_θψ, and `z = score/√efficient_info`
is the score-test statistic for H₀: ψ = model.ψ (valid when the other scores are
≈ 0, i.e. at a converged fit). `ψ_onestep` is the one-step Newton update.
Positive scores mean the likelihood increases towards larger ψ (more dilution).
"""
function psi_score(model::SGEModel; K::Int=30, seed::Int=2026, probes=nothing,
    reltol=1e-8, maxiter=2000)
    n = length(model.y)
    dim = model.layout.dim
    workspaces, preconditioners = _post_setup(model)
    ws1 = workspaces[1]
    sol, Py = _post_solve_y(model, ws1, preconditioners[1]; reltol, maxiter)
    λ = _psi_log_weights(model)

    # ŝ = Q Py from the BLUPs: Z_s(â_s + scatter(p̂))
    soc = copy(@view sol[model.layout.social])
    if model.peer_env
        p = @view sol[model.layout.pe]
        @inbounds for (k, id) in enumerate(model.pe_ids)
            soc[id] += p[k]
        end
    end
    s_hat = zeros(n)
    _apply_Zs!(s_hat, soc, model, ws1)
    quad = 2 * sum(λ .* Py .* s_hat)

    zs = _post_probes(probes, n, K, seed)
    K = length(zs)
    nt = length(workspaces)
    tbufs = [(sol=zeros(dim), Pz=zeros(n), Qz=zeros(n)) for _ in 1:nt]
    t = zeros(K)
    Threads.@threads :static for j in 1:K
        tid = Threads.threadid()
        ws, b = workspaces[tid], tbufs[tid]
        fill!(b.sol, 0.0)
        _solve_mme!(b.sol, zs[j], model, ws, preconditioners[tid]; reltol, maxiter,
            verbose=false, label="psi-z")
        compute_Px_from_sol!(b.Pz, model, zs[j], b.sol, ws)
        _apply_Q!(b.Qz, zs[j], model, ws)
        t[j] = 2 * sum(λ .* b.Pz .* b.Qz)
    end
    score = 0.5 * (quad - mean(t))
    score_se = 0.5 * _mc_se(t)

    # Augmented AI: columns V_j Py (j = 1..p) and V̇ Py
    comps = get_variance_components(model)
    p = length(comps)
    Vpy = [zeros(n) for _ in 1:p+1]
    _threaded_vi_py!(view(Vpy, 1:p), Py, comps, model, workspaces)
    QPy = zeros(n)
    _apply_Q!(QPy, Py, model, ws1)
    _apply_Qt!(Vpy[p+1], λ .* Py, model, ws1)
    Vpy[p+1] .+= λ .* QPy
    PVpy = [zeros(n) for _ in 1:p+1]
    _threaded_ai_cg!(PVpy, Vpy, model, workspaces, preconditioners, reltol, maxiter)
    Qm, PQ = reduce(hcat, Vpy), reduce(hcat, PVpy)
    AI = Matrix(Symmetric(0.5 .* (Qm' * PQ .+ PQ' * Qm) ./ 2))
    θ, ψi = 1:p, p + 1
    efficient_info = AI[ψi, ψi] - dot(AI[ψi, θ], AI[θ, θ] \ AI[θ, ψi])
    z = score / sqrt(efficient_info)
    return (; score, score_se, quad_term=0.5 * quad, trace_term=0.5 * mean(t),
        AI, efficient_info, z, ψ=model.ψ, ψ_onestep=model.ψ + score / efficient_info,
        K, seed, params=get_params(model))
end
