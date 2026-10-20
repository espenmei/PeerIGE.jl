# =====================================================================
# PRECONDITIONERS
# The diagonal and block preconditioners used to accelerate the CG solves,
# their `ldiv!` (M⁻¹ r) applications, and the builders that assemble them
# from the current variance components. Never forms dense A/V/MME.
# Depends only on SGEModel / ThreadWorkspace / get_dim from emai_core.jl.
# Included via emai.jl.
# =====================================================================
# Rask diagonal preconditioner — unngår findfirst zero-sjekk i standard Diagonal ldiv!
# OG fikser bug: Diagonal(1/d) + ldiv! ga z=r*d (feil retning).
# DiagPreconditioner lagrer 1/d og gjør z=x*(1/d), som er korrekt M^{-1}*r.
struct DiagPreconditioner
    inv_diag::Vector{Float64}
end

function ldiv!(y::AbstractVector, P::DiagPreconditioner, x::AbstractVector)
    d = P.inv_diag
    @inbounds @simd for i in eachindex(y, x, d)
        y[i] = d[i] * x[i]
    end
    return y
end

# Blokk-preconditioner: diag for fixed effects, (G ⊗ A)/σ²_e for genetisk blokk.
# Fanger ALL off-diagonal struktur i Ainv via Cholesky (4 sparse tri-solves per applikasjon).
struct BlockPreconditioner
    diag_f_inv::Vector{Float64}     # 1/diag(X'X), length nf
    G_over_σ2e::Matrix{Float64}     # G / σ²_e (2×2)
    L_Ainv::LowerTriangular{Float64,SparseMatrixCSC{Float64,Int64}}
    layout::SolutionLayout          # blokk-ranges for løsningsvektoren
    tmp1::Vector{Float64}           # workspace (length N)
    tmp2::Vector{Float64}           # workspace (length N)
    env_diag_inv::Vector{Vector{Float64}}  # H vectors, 1/(count_j + σ²_e/σ²_h)
end

function ldiv!(z::AbstractVector, P::BlockPreconditioner, r::AbstractVector)
    nf, N = P.layout.nf, P.layout.N
    g11, g12, g21, g22 = P.G_over_σ2e[1, 1], P.G_over_σ2e[1, 2], P.G_over_σ2e[2, 1], P.G_over_σ2e[2, 2]
    tmp1, tmp2 = P.tmp1, P.tmp2

    # Faste effekter: diagonal
    @inbounds @simd for i in 1:nf
        z[i] = P.diag_f_inv[i] * r[i]
    end

    # A * r_d → tmp2, deretter skriv G-blanding inn i z
    @inbounds for i in 1:N
        tmp1[i] = r[nf+i]
    end
    ldiv!(P.L_Ainv, tmp1)          # tmp1 = L⁻¹ r_d
    copyto!(tmp2, tmp1)
    ldiv!(P.L_Ainv', tmp2)         # tmp2 = A * r_d

    # Fused: skriv G-blend for r_d + kopier r_s inn i tmp1
    @inbounds @simd for i in 1:N
        z[nf+i] = g11 * tmp2[i]
        z[nf+N+i] = g21 * tmp2[i]
        tmp1[i] = r[nf+N+i]
    end

    ldiv!(P.L_Ainv, tmp1)
    copyto!(tmp2, tmp1)
    ldiv!(P.L_Ainv', tmp2)         # tmp2 = A * r_s

    @inbounds @simd for i in 1:N
        z[nf+i] += g12 * tmp2[i]
        z[nf+N+i] += g22 * tmp2[i]
    end

    # Env blocks: diagonal preconditioner
    for h in eachindex(P.env_diag_inv)
        dinv = P.env_diag_inv[h]
        rng = P.layout.env[h]
        @inbounds @simd for j in eachindex(dinv)
            z[rng[j]] = dinv[j] * r[rng[j]]
        end
    end

    return z
end

function get_preconditioner(model::SGEModel, ws::ThreadWorkspace)
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

    return BlockPreconditioner(diag_f_inv, G_scaled, model.L_Ainv, model.layout,
        ws.precond_tmp1, ws.precond_tmp2, env_diag_inv)
end

function get_diag_preconditioner(model::SGEModel)
    nf = size(model.X, 2)
    n = length(model.y)
    gs = model.groups

    diag_Ainv = diag(model.Ainv)
    Ginv = inv([model.σ2_d model.σ_ds; model.σ_ds model.σ2_s])
    λ_dd = Ginv[1, 1] * model.σ2_e
    λ_ss = Ginv[2, 2] * model.σ2_e

    dim = get_dim(model)
    d = zeros(dim)
    lay = model.layout

    # 1. Faste effekter: diag(X'X)
    for j in 1:nf
        @inbounds d[j] = dot(@view(model.X[:, j]), @view(model.X[:, j]))
    end

    # 2. Direkte genetikk: diag(Zd'Zd) + Ainv * λ_dd
    diag_ZdZd = @view d[lay.direct]
    for i in 1:n
        @inbounds diag_ZdZd[gs.obs_to_id[i]] += 1.0
    end
    @. d[lay.direct] = diag_ZdZd + diag_Ainv * λ_dd

    # 3. Sosial genetikk: diag(Zs'Zs) + Ainv * λ_ss
    # diag(Zs'Zs)[bid] = o_g - is_observed[bid]   for bid ∈ members_g, m_g >= 2
    diag_ZsZs = @view d[lay.social]
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
    @. d[lay.social] = diag_ZsZs + diag_Ainv * λ_ss

    # 4. Env blocks: diag(Zh'Zh) + σ²_e/σ²_h = count_j + λ_h
    for h in 1:length(model.env_ids)
        counts = model.env_counts[h]
        λ_h = model.σ2_e / model.σ2_env[h]
        rng = lay.env[h]
        @inbounds for j in eachindex(counts)
            d[rng[j]] = counts[j] + λ_h
        end
    end

    return DiagPreconditioner(1.0 ./ max.(d, 1e-6))
end
