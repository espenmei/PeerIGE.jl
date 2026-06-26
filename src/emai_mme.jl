# =====================================================================
# MIXED MODEL EQUATIONS (matrix-free)
# The MME operator (LHS * v), RHS construction, CG solves, V_i operators,
# and preconditioner builders. Never forms dense A/V/MME. Included via emai.jl.
# =====================================================================
# Operatøren som representerer LHS i MME (LHS * v)
struct MMEOperator{T}
    m::SGEModel
    workspace_n::Vector{T}    # y_hat buffer (length n)
    workspace_n2::Vector{T}   # tmp buffer (length n)
    workspace_dim::Vector{T}  # for 5-arg mul! (length dim)
    Ad::Vector{T}             # Ainv * v_d buffer (length N)
    As::Vector{T}             # Ainv * v_s buffer (length N)
    Ginv_scaled::Matrix{T}    # Ginv * σ²_e, cached (2×2)
    group_sum::Vector{T}      # length num_groups (workspace alias)
end

function make_mme_operator(model::SGEModel, ws::ThreadWorkspace)
    # Beregn Ginv * σ²_e uten allokering (2×2 inline)
    det_G = model.σ2_d * model.σ2_s - model.σ_ds^2
    inv_det = model.σ2_e / det_G
    λ = ws.op_Ginv
    λ[1, 1] = model.σ2_s * inv_det
    λ[1, 2] = -model.σ_ds * inv_det
    λ[2, 1] = λ[1, 2]
    λ[2, 2] = model.σ2_d * inv_det
    return MMEOperator{Float64}(model, ws.op_wn, ws.op_wn2, ws.op_wdim,
        ws.op_Ad, ws.op_As, λ, ws.group_sum)
end

# Disse to funksjonene er nødvendige for at IterativeSolvers skal fungere
eltype(op::MMEOperator{T}) where T = T
size(op::MMEOperator) = (d = get_dim(op.m); (d, d))
size(op::MMEOperator, i::Int) = size(op)[i]

# Denne funksjonen gjør at cg kan bruke operatoren effektivt
function mul!(y, op::MMEOperator, x)
    op(y, x)
end

function mul!(y, op::MMEOperator, x, α, β)
    op(op.workspace_dim, x)
    @. y = α * op.workspace_dim + β * y
end

function compute_Px_from_sol!(out::Vector{Float64}, model::SGEModel, y_vector::Vector{Float64}, sol_vector::Vector{Float64}, ws::ThreadWorkspace)
    n = length(model.y)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    gs = model.groups

    H = length(model.env_ids)

    # 1. Hent ut estimatene fra løsningsvektoren (sol_vector)
    beta = @view sol_vector[1:nf]
    a_d = @view sol_vector[nf+1:nf+N]
    a_s = @view sol_vector[nf+N+1:nf+2N]

    # 2. Beregn y_hat in-place i out, deretter Px
    mul!(out, model.X, beta)
    # Cache group_sum for a_s så vi unngår å summere medlemmer per obs
    compute_group_sums!(ws.group_sum, a_s, gs.members_id)
    @inbounds for i in 1:n
        target_id = gs.obs_to_id[i]
        out[i] += a_d[target_id]
        g_idx = gs.obs_to_group[i]
        out[i] += (ws.group_sum[g_idx] - a_s[target_id]) * model.group_scale[g_idx]  # scaled peer sum (excl. self)
    end

    # + env contributions: Zh * ch
    env_offset = nf + 2N
    for h in 1:H
        ids = model.env_ids[h]
        n_h = model.env_n_levels[h]
        c_h = @view sol_vector[env_offset+1:env_offset+n_h]
        @inbounds for i in 1:n
            out[i] += c_h[ids[i]]
        end
        env_offset += n_h
    end

    # 3. out = (y_vector - y_hat) / σ²_e
    inv_σ2e = 1.0 / model.σ2_e
    @inbounds @simd for i in eachindex(out)
        out[i] = (y_vector[i] - out[i]) * inv_σ2e
    end
    return out
end

# Beregner: group_sum[g] = Σ_{bid ∈ members_g} v_N[bid]
@inline function compute_group_sums!(group_sum::Vector{Float64}, v_N::AbstractVector{Float64},
    members_id::Vector{Vector{Int}})
    @inbounds for g in eachindex(members_id)
        s = 0.0
        for bid in members_id[g]
            s += v_N[bid]
        end
        group_sum[g] = s
    end
    return group_sum
end

# BEREGNER: Zs' * u (Brukes for bidrag fra faste effekter og direkte genetikk)
#   Zs[i, bid] = 1 hvis bid ∈ group(i) og bid ≠ obs_to_id[i], else 0
#   => (Zs' * u)[bid] = Σ_{i: bid ∈ group(i), bid ≠ self_i} u[i]
#                     = Σ_{i: group(i) = g(bid)} u[i]  -  u[i_self if obs_to_id[i]=bid]
# Implementasjon: først akkumuler obs_sum[g] = Σ u[i] per gruppe (O(n)),
# deretter spre obs_sum til alle medlemmer av g (O(Σ m_g)),
# og trekk fra self-bidraget per obs (O(n)).
function apply_Zs_transpose!(v_N, u_n, gs::GroupStructure, group_sum::Vector{Float64}, group_scale::Vector{Float64})
    # Steg 1: obs_sum[g] = Σ_{i: g(i)=g} u_n[i]
    fill!(group_sum, 0.0)
    @inbounds for i in eachindex(u_n)
        group_sum[gs.obs_to_group[i]] += u_n[i]
    end
    # Steg 2: spre obs_sum (skalert med c_g) til alle medlemmer
    @inbounds for g in eachindex(gs.members_id)
        s = group_sum[g] * group_scale[g]
        if s != 0.0
            for bid in gs.members_id[g]
                v_N[bid] += s
            end
        end
    end
    # Steg 3: trekk fra self-bidrag (skalert med c_{g(i)})
    @inbounds for (i, target_id) in enumerate(gs.obs_to_id)
        v_N[target_id] -= u_n[i] * group_scale[gs.obs_to_group[i]]
    end
    return v_N
end

# BEREGNER: Zs' * Zs * v_s
# Med missing-obs (o_g <= m_g): Zs har rader for observerte obs og kolonner for
# alle pedigree-IDer i en gruppe (også uobserverte medlemmer bidrar til peer-sum).
# La S_g^full = Σ_{b ∈ members_g} v[b],  S_g^obs = Σ_{b ∈ obs members_g} v[b],
#    o_g     = antall observerte i gruppe g.
# Da gjelder for hvert medlem b ∈ members_g:
#   (Zs'Zs v)[b] = is_obs[b] · (v[b] - S_g^full)  +  o_g · S_g^full  -  S_g^obs
# Når alle medlemmer er observerte (o_g = m_g, S_obs = S_full, is_obs[b]=1) reduserer
# dette til v[b] + (m_g - 2) S_full (gammel rask vei).
function apply_ZsZs_accumulate!(u_N, v_N, gs::GroupStructure, group_scale::Vector{Float64})
    @inbounds for g in eachindex(gs.members_id)
        members = gs.members_id[g]
        m_g = length(members)
        if m_g <= 1
            continue
        end
        o_g = gs.obs_count_per_group[g]
        if o_g == 0
            continue  # ingen obs i gruppen — Zs har ingen rader her, bidrag = 0
        end

        s_full = 0.0
        s_obs = 0.0
        for bid in members
            vb = v_N[bid]
            s_full += vb
            if gs.is_observed[bid]
                s_obs += vb
            end
        end

        c2_g = group_scale[g]^2
        const_add = c2_g * (o_g * s_full - s_obs)
        for bid in members
            base = const_add
            if gs.is_observed[bid]
                base += c2_g * (v_N[bid] - s_full)
            end
            u_N[bid] += base
        end
    end
    return u_N
end

# BEREGNER: (Ginv ⊗ Ainv) * v (Slektskapsbidraget)
# λ = Ginv * σ²_e (preallokkert), Ad/As = workspace-buffere
function apply_GinvAinv!(out_d, out_s, v_d, v_s, Ainv, λ, Ad, As)
    mul!(Ad, Ainv, v_d)
    mul!(As, Ainv, v_s)
    @. out_d += λ[1, 1] * Ad + λ[1, 2] * As
    @. out_s += λ[2, 1] * Ad + λ[2, 2] * As
end

function (op::MMEOperator{T})(res, v) where T
    m = op.m
    n = length(m.y)
    N = size(m.Ainv, 1)
    nf = size(m.X, 2)
    gs = m.groups
    H = length(m.env_ids)
    v_f = @view(v[1:nf])
    v_d = @view(v[nf+1:nf+N])
    v_s = @view(v[nf+N+1:nf+2N])
    res_f = @view(res[1:nf])
    res_d = @view(res[nf+1:nf+N])
    res_s = @view(res[nf+N+1:nf+2N])
    fill!(res, 0.0)

    # Env views (v_env[h] og res_env[h])
    env_offset = nf + 2N
    v_env = Vector{SubArray}(undef, H)
    res_env = Vector{SubArray}(undef, H)
    for h in 1:H
        n_h = m.env_n_levels[h]
        v_env[h] = @view(v[env_offset+1:env_offset+n_h])
        res_env[h] = @view(res[env_offset+1:env_offset+n_h])
        env_offset += n_h
    end

    # 1. Partiell y_hat = X*v_f + Zd*v_d + sum_h Zh*v_h  (uten Zs*v_s ennå)
    y_hat = op.workspace_n
    mul!(y_hat, m.X, v_f)
    for i in 1:n
        @inbounds y_hat[i] += v_d[gs.obs_to_id[i]]
    end
    for h in 1:H
        ids = m.env_ids[h]
        vh = v_env[h]
        @inbounds for i in 1:n
            y_hat[i] += vh[ids[i]]
        end
    end

    # 2. SOSIALE EFFEKTER (res_s)
    # Krydsledd: Zs' * y_hat  (X*v_f + Zd*v_d + sum Zh*v_h) — ingen re-beregning
    apply_Zs_transpose!(res_s, y_hat, gs, op.group_sum, m.group_scale)
    # Ledd: Zs' * Zs * vs
    apply_ZsZs_accumulate!(res_s, v_s, gs, m.group_scale)

    # 3. Fullfør y_hat: legg til Zs * v_s (gjenbruker workspace_n2)
    tmp_zs = op.workspace_n2
    fill!(tmp_zs, zero(T))
    # Cache group_sum for v_s
    compute_group_sums!(op.group_sum, v_s, gs.members_id)
    @inbounds for (i, target_id) in enumerate(gs.obs_to_id)
        tmp_zs[i] = (op.group_sum[gs.obs_to_group[i]] - v_s[target_id]) * m.group_scale[gs.obs_to_group[i]]
    end
    y_hat .+= tmp_zs

    # 4. FASTE EFFEKTER (res_f = X' * y_hat)
    mul!(res_f, m.X', y_hat)

    # 5. DIREKTE EFFEKTER (res_d = Zd' * y_hat)
    for i in 1:n
        @inbounds res_d[gs.obs_to_id[i]] += y_hat[i]
    end

    # 6. GENETISK PRIOR — cached Ginv_scaled og preallokkerte Ad/As
    apply_GinvAinv!(res_d, res_s, v_d, v_s, m.Ainv, op.Ginv_scaled, op.Ad, op.As)

    # 7. ENV BLOCKS: res_h = Zh' * y_hat + (σ²_e / σ²_h) * v_h
    for h in 1:H
        ids = m.env_ids[h]
        λ_h = m.σ2_e / m.σ2_env[h]
        rh = res_env[h]
        vh = v_env[h]
        @inbounds for i in 1:n
            rh[ids[i]] += y_hat[i]
        end
        @inbounds @simd for j in eachindex(rh)
            rh[j] += λ_h * vh[j]
        end
    end

    return res
end

# Bygger MME høyresiden: rhs = [X'*v; Zd'*v; Zs'*v; (Zh'*v for h=1..H)]
function build_rhs!(rhs::Vector{Float64}, v::AbstractVector{Float64}, model::SGEModel, N::Int, nf::Int, ws::ThreadWorkspace)
    n = length(v)
    fill!(rhs, 0.0)
    mul!(@view(rhs[1:nf]), model.X', v)
    for i in 1:n
        @inbounds rhs[nf+model.groups.obs_to_id[i]] += v[i]
    end
    apply_Zs_transpose!(@view(rhs[nf+N+1:nf+2N]), v, model.groups, ws.group_sum, model.group_scale)
    env_offset = nf + 2N
    for h in 1:length(model.env_ids)
        ids = model.env_ids[h]
        n_h = model.env_n_levels[h]
        rhs_h = @view(rhs[env_offset+1:env_offset+n_h])
        @inbounds for i in 1:n
            rhs_h[ids[i]] += v[i]
        end
        env_offset += n_h
    end
end

# Løser for y (faktiske data)
function solve_mme_y!(model::SGEModel, ws::ThreadWorkspace, P_inv; reltol=1e-7, maxiter=2000, verbose=true)
    n = length(model.y)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)

    rhs = ws.rhs_buffer
    build_rhs!(rhs, model.y, model, N, nf, ws)

    op = make_mme_operator(model, ws)
    if verbose
        _, ch = cg!(model.sol_y, op, rhs, Pl=P_inv, reltol=reltol, maxiter=maxiter, log=true)
        niter = ch.iters
        final_res = ch.data[:resnorm][end]
        flag = niter < maxiter ? "✓" : "⚠ MAXITER"
        @printf("    CG(y): %d iter, res=%.2e %s\n", niter, final_res, flag)
    else
        cg!(model.sol_y, op, rhs, Pl=P_inv, reltol=reltol, maxiter=maxiter)
    end
end

# Løser for z_i (stokastisk trace)
function solve_mme_z!(sol_z_i::Vector{Float64}, model::SGEModel,
    z_vector::Vector{Float64}, ws::ThreadWorkspace, P_inv;
    reltol=1e-7, maxiter=2000, verbose=false)
    n = length(z_vector)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)

    rhs = ws.rhs_buffer
    build_rhs!(rhs, z_vector, model, N, nf, ws)

    op = make_mme_operator(model, ws)
    if verbose
        _, ch = cg!(sol_z_i, op, rhs, Pl=P_inv, reltol=reltol, maxiter=maxiter, log=true)
        niter = ch.iters
        final_res = ch.data[:resnorm][end]
        flag = niter < maxiter ? "✓" : "⚠ MAXITER"
        @printf("    CG(z): %d iter, res=%.2e %s\n", niter, final_res, flag)
    else
        cg!(sol_z_i, op, rhs, Pl=P_inv, reltol=reltol, maxiter=maxiter)
    end
end

function apply_A!(out::Vector{Float64}, L_Ainv, x_N::Vector{Float64}, tmp::Vector{Float64})
    # A * x = Ainv⁻¹ * x via Cholesky (Ainv = L*L')
    copyto!(tmp, x_N)
    ldiv!(L_Ainv, tmp)           # tmp ← L⁻¹ * x_N
    copyto!(out, tmp)
    ldiv!(L_Ainv', out)          # out ← (L')⁻¹ * (L⁻¹ * x_N) = A * x_N
end

# In-place Vi * u — bruker workspace-buffere (trådsikkert og sekvensiellt)
function get_Vi_u!(out::Vector{Float64}, u_n, type::Symbol, model::SGEModel, ws::ThreadWorkspace)
    gs = model.groups
    N = size(model.Ainv, 1)
    n = length(u_n)

    # Env effects: :env1, :env2, ...
    type_str = String(type)
    if startswith(type_str, "env")
        h = parse(Int, type_str[4:end])
        return get_Vi_env_u!(out, u_n, h, model, ws)
    end

    if type == :d  # Zd * A * Zd' * u
        fill!(ws.tmp_N1, 0.0)
        for i in 1:n
            @inbounds ws.tmp_N1[gs.obs_to_id[i]] += u_n[i]
        end
        apply_A!(ws.tmp_N2, model.L_Ainv, ws.tmp_N1, ws.tmp_N3)
        for i in 1:n
            @inbounds out[i] = ws.tmp_N2[gs.obs_to_id[i]]
        end

    elseif type == :s # Zs_ψ * A * Zs_ψ' * u
        fill!(ws.tmp_N1, 0.0)
        apply_Zs_transpose!(ws.tmp_N1, u_n, gs, ws.group_sum, model.group_scale)
        apply_A!(ws.tmp_N2, model.L_Ainv, ws.tmp_N1, ws.tmp_N3)
        # out[i] = c_{g(i)} * ((Σ_{bid ∈ g(i)} tmp_N2[bid]) - tmp_N2[obs_to_id[i]])
        compute_group_sums!(ws.group_sum, ws.tmp_N2, gs.members_id)
        @inbounds for (i, tid) in enumerate(gs.obs_to_id)
            out[i] = (ws.group_sum[gs.obs_to_group[i]] - ws.tmp_N2[tid]) * model.group_scale[gs.obs_to_group[i]]
        end

    elseif type == :ds # (Zd * A * Zs_ψ' + Zs_ψ * A * Zd') * u
        # Del 1: Zd * A * Zs_ψ' * u
        fill!(ws.tmp_N1, 0.0)
        apply_Zs_transpose!(ws.tmp_N1, u_n, gs, ws.group_sum, model.group_scale)
        apply_A!(ws.tmp_N2, model.L_Ainv, ws.tmp_N1, ws.tmp_N3)
        for i in 1:n
            @inbounds out[i] = ws.tmp_N2[gs.obs_to_id[i]]
        end
        # Del 2: Zs * A * Zd' * u
        fill!(ws.tmp_N1, 0.0)
        for i in 1:n
            @inbounds ws.tmp_N1[gs.obs_to_id[i]] += u_n[i]
        end
        apply_A!(ws.tmp_N2, model.L_Ainv, ws.tmp_N1, ws.tmp_N3)
        # out[i] += c_{g(i)} * ((Σ_{bid ∈ g(i)} tmp_N2[bid]) - tmp_N2[obs_to_id[i]])
        compute_group_sums!(ws.group_sum, ws.tmp_N2, gs.members_id)
        @inbounds for (i, tid) in enumerate(gs.obs_to_id)
            out[i] += (ws.group_sum[gs.obs_to_group[i]] - ws.tmp_N2[tid]) * model.group_scale[gs.obs_to_group[i]]
        end

    else # :e
        copyto!(out, u_n)
    end
    return out
end

# Env Vi: Zh * Zh' * u (scatter-gather, O(n))
function get_Vi_env_u!(out::Vector{Float64}, u_n, h::Int, model::SGEModel, ws::ThreadWorkspace)
    ids = model.env_ids[h]
    n = length(u_n)
    tmp = ws.env_tmp[h]
    fill!(tmp, 0.0)
    # Gather: tmp[j] = sum u_n[i] for i with ids[i]==j
    @inbounds for i in 1:n
        tmp[ids[i]] += u_n[i]
    end
    # Scatter: out[i] = tmp[ids[i]]
    @inbounds for i in 1:n
        out[i] = tmp[ids[i]]
    end
    return out
end

function solve_mme_direct!(x_sol, model::SGEModel, v_vector::Vector{Float64},
    ws::ThreadWorkspace, P_inv; reltol=1e-5, maxiter=500)
    n = length(v_vector)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    rhs = ws.rhs_buffer
    build_rhs!(rhs, v_vector, model, N, nf, ws)

    op = make_mme_operator(model, ws)
    cg!(x_sol, op, rhs, Pl=P_inv, reltol=reltol, maxiter=maxiter)
end

function get_preconditioner(model::SGEModel, ws::ThreadWorkspace)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)

    # 1. Faste effekter: diag(X'X)
    diag_f_inv = Vector{Float64}(undef, nf)
    for j in 1:nf
        @inbounds diag_f_inv[j] = 1.0 / max(dot(@view(model.X[:, j]), @view(model.X[:, j])), 1e-6)
    end

    # 2. Genetisk blokk: M⁻¹ = (G ⊗ A) / σ²_e
    G = [model.σ2_d model.σ_ds; model.σ_ds model.σ2_s]
    G_scaled = G ./ model.σ2_e

    # 3. Env blocks: diag preconditioner  1 / (count_j + σ²_e / σ²_h)
    env_diag_inv = Vector{Float64}[]
    for h in 1:length(model.env_ids)
        counts = model.env_counts[h]
        λ_h = model.σ2_e / model.σ2_env[h]
        push!(env_diag_inv, [1.0 / (counts[j] + λ_h) for j in eachindex(counts)])
    end

    return BlockPreconditioner(diag_f_inv, G_scaled, model.L_Ainv, nf, N,
        ws.precond_tmp1, ws.precond_tmp2, env_diag_inv)
end

function get_diag_preconditioner(model::SGEModel)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    n = length(model.y)
    gs = model.groups

    diag_Ainv = diag(model.Ainv)
    Ginv = inv([model.σ2_d model.σ_ds; model.σ_ds model.σ2_s])
    λ_dd = Ginv[1, 1] * model.σ2_e
    λ_ss = Ginv[2, 2] * model.σ2_e

    dim = get_dim(model)
    d = zeros(dim)

    # 1. Faste effekter: diag(X'X)
    for j in 1:nf
        @inbounds d[j] = dot(@view(model.X[:, j]), @view(model.X[:, j]))
    end

    # 2. Direkte genetikk: diag(Zd'Zd) + Ainv * λ_dd
    diag_ZdZd = @view d[nf+1:nf+N]
    for i in 1:n
        @inbounds diag_ZdZd[gs.obs_to_id[i]] += 1.0
    end
    @. d[nf+1:nf+N] = diag_ZdZd + diag_Ainv * λ_dd

    # 3. Sosial genetikk: diag(Zs'Zs) + Ainv * λ_ss
    # diag(Zs'Zs)[bid] = o_g - is_observed[bid]   for bid ∈ members_g, m_g >= 2
    diag_ZsZs = @view d[nf+N+1:nf+2N]
    for g in eachindex(gs.members_id)
        members = gs.members_id[g]
        if length(members) > 1
            o_g = gs.obs_count_per_group[g]
            c2_g = model.group_scale[g]^2
            for id in members
                @inbounds diag_ZsZs[id] += (o_g - (gs.is_observed[id] ? 1 : 0)) * c2_g
            end
        end
    end
    @. d[nf+N+1:nf+2N] = diag_ZsZs + diag_Ainv * λ_ss

    # 4. Env blocks: diag(Zh'Zh) + σ²_e/σ²_h = count_j + λ_h
    env_offset = nf + 2N
    for h in 1:length(model.env_ids)
        counts = model.env_counts[h]
        λ_h = model.σ2_e / model.σ2_env[h]
        n_h = model.env_n_levels[h]
        @inbounds for j in 1:n_h
            d[env_offset+j] = counts[j] + λ_h
        end
        env_offset += n_h
    end

    return DiagPreconditioner(1.0 ./ max.(d, 1e-6))
end
