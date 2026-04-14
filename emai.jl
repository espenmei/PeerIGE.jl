using Printf
using CSV, DataFrames, LinearAlgebra, SparseArrays, Statistics
using IterativeSolvers
using JLD2
using Random
import Base: *, eltype, size
import LinearAlgebra: mul!, ldiv!

# Forhindre OpenBLAS-tråder fra å krasje med Julia-tråder på Windows VM
LinearAlgebra.BLAS.set_num_threads(1)

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
    nf::Int
    N::Int
    tmp1::Vector{Float64}           # workspace (length N)
    tmp2::Vector{Float64}           # workspace (length N)
end

function ldiv!(z::AbstractVector, P::BlockPreconditioner, r::AbstractVector)
    nf, N = P.nf, P.N
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

    return z
end

struct GroupStructure
    members_id::Vector{Vector{Int}} # ID-er i stamtavlen (1..N)
    obs_to_id::Vector{Int}          # Hvilken stamtavle-ID har obs i? (1..n)
    id_to_group::Vector{Int}        # Hvilken gruppe tilhører stamtavle-ID i?
end

function GroupStructure(group_ids::Vector{Int}, animal_ids::Vector{Int}, N_pedigree::Int)
    unique_groups = unique(group_ids)
    group_map = Dict(g => i for (i, g) in enumerate(unique_groups))
    num_groups = length(unique_groups)

    members = [Int[] for _ in 1:num_groups]
    for (i, g_id) in enumerate(group_ids)
        g_idx = group_map[g_id]
        push!(members[g_idx], animal_ids[i])
    end
    # Mapping for alle dyr i stamtavlen (0 = ingen gruppe/ikke observert)
    id_to_group = zeros(Int, N_pedigree)
    for (i, g_id) in enumerate(group_ids)
        id_to_group[animal_ids[i]] = group_map[g_id]
    end

    return GroupStructure(members, animal_ids, id_to_group)
end

mutable struct SGEModel
    y::Vector{Float64}
    X::Matrix{Float64}
    Ainv::SparseMatrixCSC{Float64,Int64}
    L_Ainv::LowerTriangular{Float64,SparseMatrixCSC{Float64,Int64}}
    groups::GroupStructure
    σ2_d::Float64
    σ2_s::Float64
    σ_ds::Float64
    σ2_e::Float64
    # Warm start
    sol_y::Vector{Float64}
    sol_z::Vector{Vector{Float64}} # For Hutchinson trace estimat av Vi * Pz
    # Preallokkerte workspace-buffere
    rhs_buffer::Vector{Float64}
    tmp_N1::Vector{Float64}
    tmp_N2::Vector{Float64}
    tmp_N3::Vector{Float64}
    # MMEOperator workspace (gjenbrukes mellom CG-kall)
    op_wn::Vector{Float64}    # length n
    op_wn2::Vector{Float64}   # length n
    op_wdim::Vector{Float64}  # length dim
    op_Ad::Vector{Float64}    # length N
    op_As::Vector{Float64}    # length N
    op_Ginv::Matrix{Float64}  # 2×2 cached Ginv*σ²_e
end

function SGEModel(y, X, Ainv, L_Ainv, groups, σ2_d, σ2_s, σ_ds, σ2_e)
    N = size(Ainv, 1)
    n = length(y)
    nf = size(X, 2)
    dim = nf + 2 * N
    sol_y = zeros(dim)
    sol_z = Vector{Vector{Float64}}()  # Allokeres av run_aireml
    rhs_buffer = zeros(dim)
    tmp_N1 = zeros(N)
    tmp_N2 = zeros(N)
    tmp_N3 = zeros(N)
    # MMEOperator workspaces (gjenbrukes)
    op_wn = zeros(n)
    op_wn2 = zeros(n)
    op_wdim = zeros(dim)
    op_Ad = zeros(N)
    op_As = zeros(N)
    op_Ginv = zeros(2, 2)
    return SGEModel(y, X, Ainv, L_Ainv, groups, σ2_d, σ2_s, σ_ds, σ2_e,
        sol_y, sol_z, rhs_buffer, tmp_N1, tmp_N2, tmp_N3,
        op_wn, op_wn2, op_wdim, op_Ad, op_As, op_Ginv)
end

# Per-tråd workspace for parallell trace-estimering
# Inneholder alle mutable buffere som ellers deles via SGEModel
struct ThreadWorkspace
    rhs_buffer::Vector{Float64}    # length dim
    # MMEOperator workspace
    op_wn::Vector{Float64}         # length n
    op_wn2::Vector{Float64}        # length n
    op_wdim::Vector{Float64}       # length dim
    op_Ad::Vector{Float64}         # length N
    op_As::Vector{Float64}         # length N
    op_Ginv::Matrix{Float64}       # 2×2
    # get_Vi_u / apply_A! workspace
    tmp_N1::Vector{Float64}        # length N
    tmp_N2::Vector{Float64}        # length N
    tmp_N3::Vector{Float64}        # length N
    # get_Vi_u output buffer
    vi_out::Vector{Float64}        # length n
    # BlockPreconditioner workspace
    precond_tmp1::Vector{Float64}  # length N
    precond_tmp2::Vector{Float64}  # length N
end

function create_thread_workspaces(model::SGEModel, nt::Int)
    n = length(model.y)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    dim = nf + 2N
    return [ThreadWorkspace(
        zeros(dim),
        zeros(n), zeros(n), zeros(dim),
        zeros(N), zeros(N), zeros(2, 2),
        zeros(N), zeros(N), zeros(N),
        zeros(n),
        zeros(N), zeros(N)
    ) for _ in 1:nt]
end

# Lag et enkelt ThreadWorkspace som aliaser modellens egne buffere (for sekvensiell bruk)
function model_workspace(model::SGEModel)
    n = length(model.y)
    N = size(model.Ainv, 1)
    return ThreadWorkspace(
        model.rhs_buffer,
        model.op_wn, model.op_wn2, model.op_wdim,
        model.op_Ad, model.op_As, model.op_Ginv,
        model.tmp_N1, model.tmp_N2, model.tmp_N3,
        zeros(n),       # vi_out
        zeros(N), zeros(N)  # precond_tmp1, precond_tmp2
    )
end

# Operatøren som representerer LHS i MME (LHS * v)
struct MMEOperator{T}
    m::SGEModel
    workspace_n::Vector{T}    # y_hat buffer (length n)
    workspace_n2::Vector{T}   # tmp buffer (length n)
    workspace_dim::Vector{T}  # for 5-arg mul! (length dim)
    Ad::Vector{T}             # Ainv * v_d buffer (length N)
    As::Vector{T}             # Ainv * v_s buffer (length N)
    Ginv_scaled::Matrix{T}    # Ginv * σ²_e, cached (2×2)
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
        ws.op_Ad, ws.op_As, λ)
end

# Disse to funksjonene er nødvendige for at IterativeSolvers skal fungere
eltype(op::MMEOperator{T}) where T = T
size(op::MMEOperator) = (size(op.m.X, 2) + 2 * size(op.m.Ainv, 1), size(op.m.X, 2) + 2 * size(op.m.Ainv, 1))
size(op::MMEOperator, i::Int) = size(op)[i]

# Denne funksjonen gjør at cg kan bruke operatoren effektivt
function mul!(y, op::MMEOperator, x)
    op(y, x)
end

function mul!(y, op::MMEOperator, x, α, β)
    op(op.workspace_dim, x)
    @. y = α * op.workspace_dim + β * y
end

function compute_Px_from_sol!(out::Vector{Float64}, model::SGEModel, y_vector::Vector{Float64}, sol_vector::Vector{Float64})
    n = length(model.y)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    gs = model.groups

    # 1. Hent ut estimatene fra løsningsvektoren (sol_vector)
    beta = @view sol_vector[1:nf]
    a_d = @view sol_vector[nf+1:nf+N]
    a_s = @view sol_vector[nf+N+1:end]

    # 2. Beregn y_hat in-place i out, deretter Px
    mul!(out, model.X, beta)
    @inbounds for i in 1:n
        target_id = gs.obs_to_id[i]
        out[i] += a_d[target_id]
        g_idx = gs.id_to_group[target_id]
        s = 0.0
        for bid in gs.members_id[g_idx]
            s += a_s[bid]
        end
        out[i] += s - a_s[target_id]  # branchless
    end

    # 3. out = (y_vector - y_hat) / σ²_e
    inv_σ2e = 1.0 / model.σ2_e
    @inbounds @simd for i in eachindex(out)
        out[i] = (y_vector[i] - out[i]) * inv_σ2e
    end
    return out
end

# BEREGNER: Zs' * u (Brukes for bidrag fra faste effekter og direkte genetikk)
function apply_Zs_transpose!(v_N, u_n, gs::GroupStructure)
    @inbounds for (i, target_id) in enumerate(gs.obs_to_id)
        val = u_n[i]
        g_idx = gs.id_to_group[target_id]
        for bid in gs.members_id[g_idx]
            v_N[bid] += val
        end
        v_N[target_id] -= val  # Fjern selv-bidraget (branchless)
    end
end

# BEREGNER: Zs' * Zs * v_s (Den optimaliserte snarveien for sosiale effekter)
function apply_ZsZs_accumulate!(u_N, v_N, gs::GroupStructure)
    for members in gs.members_id
        ng = length(members)
        if ng <= 1
            continue
        end
        # Manuell sum — unngår SubArray-allokering fra @view med Vector{Int}-indeks
        group_sum = 0.0
        @inbounds for id in members
            group_sum += v_N[id]
        end
        @inbounds for id in members
            u_N[id] += (ng - 2) * group_sum + v_N[id]
        end
    end
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
    v_f, v_d, v_s = @view(v[1:nf]), @view(v[nf+1:nf+N]), @view(v[nf+N+1:end])
    res_f, res_d, res_s = @view(res[1:nf]), @view(res[nf+1:nf+N]), @view(res[nf+N+1:end])
    fill!(res, 0.0)

    # 1. FASTE EFFEKTER (res_f)
    y_hat = op.workspace_n
    mul!(y_hat, m.X, v_f)
    for i in 1:n
        @inbounds y_hat[i] += v_d[gs.obs_to_id[i]]
    end

    # + Zs * v_s (gjenbruker workspace_n2, ingen allokering)
    tmp_zs = op.workspace_n2
    fill!(tmp_zs, zero(T))
    @inbounds for (i, target_id) in enumerate(gs.obs_to_id)
        g_idx = gs.id_to_group[target_id]
        s = zero(T)
        for bid in gs.members_id[g_idx]
            s += v_s[bid]
        end
        tmp_zs[i] = s - v_s[target_id]  # Branchless: sum alle, trekk fra self
    end
    y_hat .+= tmp_zs
    mul!(res_f, m.X', y_hat)

    # 2. DIREKTE EFFEKTER (res_d = Zd' * y_hat)
    for i in 1:n
        @inbounds res_d[gs.obs_to_id[i]] += y_hat[i]
    end

    # 3. SOSIALE EFFEKTER (res_s)
    # Ledd: Zs' * (X*beta + Zd*ad) — gjenbruker workspace_n2
    tmp_n = op.workspace_n2
    mul!(tmp_n, m.X, v_f)
    for i in 1:n
        @inbounds tmp_n[i] += v_d[gs.obs_to_id[i]]
    end
    apply_Zs_transpose!(res_s, tmp_n, gs)

    # Ledd: Zs' * Zs * vs
    apply_ZsZs_accumulate!(res_s, v_s, gs)

    # 4. GENETISK PRIOR — cached Ginv_scaled og preallokkerte Ad/As
    apply_GinvAinv!(res_d, res_s, v_d, v_s, m.Ainv, op.Ginv_scaled, op.Ad, op.As)

    return res
end

# Løser for y (faktiske data)
function solve_mme_y!(model::SGEModel, ws::ThreadWorkspace, P_inv; reltol=1e-7, maxiter=2000, verbose=true)
    n = length(model.y)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)

    rhs = ws.rhs_buffer
    fill!(rhs, 0.0)
    mul!(@view(rhs[1:nf]), model.X', model.y)
    for i in 1:n
        @inbounds rhs[nf+model.groups.obs_to_id[i]] += model.y[i]
    end
    apply_Zs_transpose!(@view(rhs[nf+N+1:end]), model.y, model.groups)

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
    fill!(rhs, 0.0)
    mul!(@view(rhs[1:nf]), model.X', z_vector)
    for i in 1:n
        @inbounds rhs[nf+model.groups.obs_to_id[i]] += z_vector[i]
    end
    apply_Zs_transpose!(@view(rhs[nf+N+1:end]), z_vector, model.groups)

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

    if type == :d  # Zd * A * Zd' * u
        fill!(ws.tmp_N1, 0.0)
        for i in 1:n
            @inbounds ws.tmp_N1[gs.obs_to_id[i]] += u_n[i]
        end
        apply_A!(ws.tmp_N2, model.L_Ainv, ws.tmp_N1, ws.tmp_N3)
        for i in 1:n
            @inbounds out[i] = ws.tmp_N2[gs.obs_to_id[i]]
        end

    elseif type == :s # Zs * A * Zs' * u
        fill!(ws.tmp_N1, 0.0)
        apply_Zs_transpose!(ws.tmp_N1, u_n, gs)
        apply_A!(ws.tmp_N2, model.L_Ainv, ws.tmp_N1, ws.tmp_N3)
        fill!(out, 0.0)
        for (i, tid) in enumerate(gs.obs_to_id)
            g_idx = gs.id_to_group[tid]
            for bid in gs.members_id[g_idx]
                if bid != tid
                    @inbounds out[i] += ws.tmp_N2[bid]
                end
            end
        end

    elseif type == :ds # (Zd * A * Zs' + Zs * A * Zd') * u
        # Del 1: Zd * A * Zs' * u
        fill!(ws.tmp_N1, 0.0)
        apply_Zs_transpose!(ws.tmp_N1, u_n, gs)
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
        for (i, tid) in enumerate(gs.obs_to_id)
            g_idx = gs.id_to_group[tid]
            for bid in gs.members_id[g_idx]
                if bid != tid
                    @inbounds out[i] += ws.tmp_N2[bid]
                end
            end
        end

    else # :e
        copyto!(out, u_n)
    end
    return out
end

function compute_score_and_ai(model::SGEModel, Py, z_list, Pz_list, workspaces, preconditioners;
    ai_reltol=1e-5, ai_maxiter=500)
    types = [:d, :s, :ds, :e]
    n_p = length(types)
    score = zeros(n_p)
    AI = zeros(n_p, n_p)
    k = length(Pz_list)
    n = length(Py)
    use_threads = length(workspaces) > 1 && Threads.nthreads() > 1

    # 1. Forhåndsberegn alle Vi * Py
    println("    -> Beregner Vi * Py...")
    Vi_Py_list = [Vector{Float64}(undef, n) for _ in 1:n_p]
    if use_threads
        Threads.@threads :static for i in 1:n_p
            tid = Threads.threadid()
            get_Vi_u!(Vi_Py_list[i], Py, types[i], model, workspaces[tid])
        end
    else
        ws = workspaces[1]
        for i in 1:n_p
            get_Vi_u!(Vi_Py_list[i], Py, types[i], model, ws)
        end
    end

    # 2. SCORE: s_i = 0.5 * (y'P * Vi * Py - tr(P * Vi))
    #    Trace-estimering: parallelliser k get_Vi_u-kall per parameter
    println("    -> Beregner trace (k=$k)...")
    if use_threads
        n_trace = n_p * k
        trace_results = zeros(n_trace)
        Threads.@threads :static for flat_idx in 1:n_trace
            tid = Threads.threadid()
            ws = workspaces[tid]
            i = (flat_idx - 1) ÷ k + 1
            j = (flat_idx - 1) % k + 1
            get_Vi_u!(ws.vi_out, Pz_list[j], types[i], model, ws)
            trace_results[flat_idx] = dot(z_list[j], ws.vi_out) / k
        end
        for i in 1:n_p
            tr_val = sum(@view trace_results[(i-1)*k+1 : i*k])
            score[i] = 0.5 * (Py' * Vi_Py_list[i] - tr_val)
        end
    else
        ws = workspaces[1]
        for i in 1:n_p
            tr_val = 0.0
            for j in 1:k
                get_Vi_u!(ws.vi_out, Pz_list[j], types[i], model, ws)
                @inbounds tr_val += dot(z_list[j], ws.vi_out) / k
            end
            score[i] = 0.5 * (Py' * Vi_Py_list[i] - tr_val)
        end
    end

    # 3. AI-MATRISE (Average Information)
    #    Trenger bare n_p=4 CG-solves (én per V_j), ikke 10.
    #    AI[i,j] = 0.5 * Py' * Vi * P * Vj * Py
    println("    -> Beregner AI-matrise (4 CG-solves)...")

    # Steg 1: Beregn P * V_j * Py for j=1..4 (dette er de dyre CG-solvene)
    P_Vi_Py_list = [Vector{Float64}(undef, n) for _ in 1:n_p]
    if use_threads
        tmp_sols = [copy(model.sol_y) for _ in 1:n_p]
        Threads.@threads :static for j in 1:n_p
            tid = Threads.threadid()
            ws = workspaces[tid]
            solve_mme_direct!(tmp_sols[j], model, Vi_Py_list[j], ws, preconditioners[tid],
                reltol=ai_reltol, maxiter=ai_maxiter)
            compute_Px_from_sol!(P_Vi_Py_list[j], model, Vi_Py_list[j], tmp_sols[j])
        end
    else
        ws = workspaces[1]
        tmp_sol = zeros(length(model.sol_y))
        for j in 1:n_p
            copyto!(tmp_sol, model.sol_y)
            solve_mme_direct!(tmp_sol, model, Vi_Py_list[j], ws, preconditioners[1],
                reltol=ai_reltol, maxiter=ai_maxiter)
            compute_Px_from_sol!(P_Vi_Py_list[j], model, Vi_Py_list[j], tmp_sol)
        end
    end

    # Steg 2: Beregn AI[i,j] = 0.5 * Py' * Vi * P * Vj * Py  (billige mat-vec)
    pairs = [(i, j) for i in 1:n_p for j in 1:i]  # 10 par
    if use_threads
        ai_vals = zeros(length(pairs))
        Threads.@threads :static for idx in eachindex(pairs)
            tid = Threads.threadid()
            ws = workspaces[tid]
            i_idx, j_idx = pairs[idx]
            get_Vi_u!(ws.vi_out, P_Vi_Py_list[j_idx], types[i_idx], model, ws)
            ai_vals[idx] = 0.5 * dot(Py, ws.vi_out)
        end
        for (idx, (i_idx, j_idx)) in enumerate(pairs)
            AI[i_idx, j_idx] = ai_vals[idx]
            AI[j_idx, i_idx] = ai_vals[idx]
        end
    else
        ws = workspaces[1]
        for (i_idx, j_idx) in pairs
            get_Vi_u!(ws.vi_out, P_Vi_Py_list[j_idx], types[i_idx], model, ws)
            val = 0.5 * dot(Py, ws.vi_out)
            AI[i_idx, j_idx] = val
            AI[j_idx, i_idx] = val
        end
    end
    return score, AI
end

function solve_mme_direct!(x_sol, model::SGEModel, v_vector::Vector{Float64},
    ws::ThreadWorkspace, P_inv; reltol=1e-5, maxiter=500)
    n = length(v_vector)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    rhs = ws.rhs_buffer
    fill!(rhs, 0.0)
    mul!(@view(rhs[1:nf]), model.X', v_vector)
    for i in 1:n
        @inbounds rhs[nf+model.groups.obs_to_id[i]] += v_vector[i]
    end
    apply_Zs_transpose!(@view(rhs[nf+N+1:end]), v_vector, model.groups)
    op = make_mme_operator(model, ws)
    cg!(x_sol, op, rhs, Pl=P_inv, reltol=reltol, maxiter=maxiter)
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
    # BLUE og BLUP (fra sol_y)
    beta::Vector{Float64}      # Faste effekter (nf)
    a_d::Vector{Float64}       # Direkte genetiske effekter (N)
    a_s::Vector{Float64}       # Sosiale genetiske effekter (N)
    # Inferens
    AI::Matrix{Float64}        # Average Information matrise (4×4)
    se::Vector{Float64}        # Standardfeil (4)
    # Metadata
    converged::Bool
    n_iter::Int
    n_obs::Int                 # n (antall observasjoner)
    N_pedigree::Int            # N (antall i stamtavlen)
    k::Int                     # Antall Hutchinson-prober
    # Mapping: obs → stamtavle-ID (for å koble BLUP til individer)
    obs_to_id::Vector{Int}
end

function save_results(result::AiRemlResult, filename::String)
    jldsave(filename; result=result)
    sz_mb = filesize(filename) / 1024^2
    @printf("Resultat lagret til %s (%.1f MB)\n", filename, sz_mb)
end

function load_results(filename::String)
    return load(filename, "result")
end

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

    return BlockPreconditioner(diag_f_inv, G_scaled, model.L_Ainv, nf, N,
        ws.precond_tmp1, ws.precond_tmp2)
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

    d = zeros(nf + 2 * N)

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
    diag_ZsZs = @view d[nf+N+1:end]
    for members in gs.members_id
        ng = length(members)
        if ng > 1
            for id in members
                @inbounds diag_ZsZs[id] += ng - 1
            end
        end
    end
    @. d[nf+N+1:end] = diag_ZsZs + diag_Ainv * λ_ss

    return DiagPreconditioner(1.0 ./ max.(d, 1e-6))
end

function run_aireml(model::SGEModel; max_iter=15, k=5, damping=0.2,
    cg_maxiter=2000, cg_reltol=1e-7,
    ai_maxiter=500, ai_reltol=1e-5,
    use_block_precond=true, verbose_cg=true,
    threaded=true, log_file::String="",
    seed::Int=42)
    n = length(model.y)
    dim = length(model.sol_y)
    param_names = ["σ²_d", "σ²_s", "σ_ds", "σ²_e"]

    nt = Threads.nthreads()
    max_tid = Threads.maxthreadid()   # Julia 1.12: threadid() kan returnere > nthreads()
    use_threads = threaded && nt > 1

    # Allokere sol_z til riktig størrelse for denne kjøringen
    model.sol_z = [zeros(dim) for _ in 1:k]

    # Vi må bruke de SAMME z-vektorene i hver iterasjon
    # for at Hutchinson-estimatet skal være sammenlignbart
    Random.seed!(seed)
    z_fixed_list = [randn(n) for _ in 1:k]

    # Preallokkerte buffere for Px-beregninger (gjenbrukes per iterasjon)
    Py = Vector{Float64}(undef, n)
    Pz_list = [Vector{Float64}(undef, n) for _ in 1:k]

    # Workspaces for CG-solves og trace-estimering
    # Sekvensiell: ett workspace som aliaser modellens buffere
    # Parallell: ett workspace per tråd (indeksert med threadid())
    if use_threads
        workspaces = create_thread_workspaces(model, max_tid)
    else
        workspaces = [model_workspace(model)]
    end

    precond_name = use_block_precond ? "Block" : "Diag"
    thread_info = use_threads ? ", threads=$nt" : ""
    println("Starter AI-REML: k=$k, damping=$damping, precond=$precond_name, cg_maxiter=$cg_maxiter$thread_info")
    println("N=$( size(model.Ainv,1)), n=$n, dim=$dim")
    println("-"^70)
    flush(stdout)

    # CSV-logg for ekstern overvåking (tail -f log.csv)
    use_log = !isempty(log_file)
    if use_log
        open(log_file, "w") do io
            println(io, "iter,time_y,time_z,time_ai,time_total,cum_time,eta_min,",
                "s2d,s2s,sds,s2e,se_d,se_s,se_ds,se_e,",
                "score_norm,conv_crit,alpha,rho_ds,cond_G,converged")
        end
        println("Iterasjonslogg skrives til: $log_file")
    end

    # Variabler som trengs etter loopen (for AiRemlResult)
    AI = zeros(4, 4)
    conv_crit = Inf
    last_iter = 0
    iter_times = Float64[]   # for ETA-beregning
    t_start_total = time()

    for iter in 1:max_iter
        last_iter = iter
        t_iter_start = time()

        # A. Oppdater preconditionere basert på nye varianser
        if use_block_precond
            preconditioners = [get_preconditioner(model, ws) for ws in workspaces]
        else
            if use_threads
                preconditioners = fill(get_diag_preconditioner(model), length(workspaces))
            else
                preconditioners = [get_diag_preconditioner(model)]
            end
        end

        # 1. Oppdater modellens egne løsninger (Warm Start) — y-solve er sekvensiell
        t_y = @elapsed begin
            solve_mme_y!(model, workspaces[1], preconditioners[1], reltol=cg_reltol, maxiter=cg_maxiter, verbose=verbose_cg)
            compute_Px_from_sol!(Py, model, model.y, model.sol_y)
        end

        # 2. Parallelliser k z-solves
        if verbose_cg
            @printf("    Z-solves: k=%d %s...\n", k, use_threads ? "(threaded)" : "")
            flush(stdout)
        end
        t_z = @elapsed begin
            if use_threads
                Threads.@threads :static for i in 1:k
                    tid = Threads.threadid()
                    ws = workspaces[tid]
                    solve_mme_z!(model.sol_z[i], model, z_fixed_list[i], ws, preconditioners[tid],
                        reltol=cg_reltol, maxiter=cg_maxiter)
                    compute_Px_from_sol!(Pz_list[i], model, z_fixed_list[i], model.sol_z[i])
                end
            else
                ws = workspaces[1]
                for i in 1:k
                    solve_mme_z!(model.sol_z[i], model, z_fixed_list[i], ws, preconditioners[1],
                        reltol=cg_reltol, maxiter=cg_maxiter, verbose=(verbose_cg && i == 1))
                    compute_Px_from_sol!(Pz_list[i], model, z_fixed_list[i], model.sol_z[i])
                end
            end
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
            score, AI_new = compute_score_and_ai(model, Py, z_fixed_list, Pz_list,
                workspaces, preconditioners,
                ai_reltol=ai_reltol, ai_maxiter=ai_maxiter)
            copyto!(AI, AI_new)
        end
        if verbose_cg
            @printf("    Score + AI ferdig: %.1fs\n", t_ai)
            flush(stdout)
        end

        # 4. Løs systemet og beregn Standardfeil (SE)
        AI_reg = AI + 1e-6 * I
        delta = AI_reg \ score
        se = sqrt.(abs.(diag(inv(AI_reg))))

        # 5. Lagre gamle verdier for utskrift
        old_params = [model.σ2_d, model.σ2_s, model.σ_ds, model.σ2_e]

        # 6. Step-halving: finn steglengde som holder alle varianser > 0 og G positiv definitt
        α = damping
        for _attempt in 1:20
            new_d = model.σ2_d + α * delta[1]
            new_s = model.σ2_s + α * delta[2]
            new_ds = model.σ_ds + α * delta[3]
            new_e = model.σ2_e + α * delta[4]
            if new_d > 1e-6 && new_s > 1e-8 && new_e > 1e-4 && new_d * new_s > new_ds^2
                break
            end
            α *= 0.5
        end

        # 7. Oppdater parametere med safe steglengde
        update_params!(model, delta, α)
        new_params = [model.σ2_d, model.σ2_s, model.σ_ds, model.σ2_e]
        actual_change = new_params .- old_params

        # --- UTSKRIFT ---
        t_iter = time() - t_iter_start
        push!(iter_times, t_iter)
        cum_time = time() - t_start_total
        avg_iter = mean(iter_times)
        remaining_iters = max_iter - iter
        eta_min = remaining_iters * avg_iter / 60.0

        # G-matrise helse
        rho_ds = model.σ_ds / sqrt(model.σ2_d * model.σ2_s)
        G_mat = [model.σ2_d model.σ_ds; model.σ_ds model.σ2_s]
        eig_G = eigvals(G_mat)
        cond_G = maximum(eig_G) / max(minimum(eig_G), 1e-12)

        score_norm = norm(score)

        println("\n" * "="^70)
        @printf("ITERASJON %d  (α=%.3f)  |  %.1fs (y:%.1f z:%.1f ai:%.1f)  cum:%.0fs  ETA:%.0fmin\n",
            iter, α, t_iter, t_y, t_z, t_ai, cum_time, eta_min)
        println("="^70)
        @printf("%-10s %-12s %-12s %-12s %-12s\n", "Param", "Est", "SE", "Score", "Delta")
        println("-"^70)
        for i in 1:4
            @printf("%-10s %-12.6f %-12.6f %-+12.4f %-+12.6f\n",
                param_names[i], new_params[i], se[i], score[i], delta[i])
        end
        println("-"^70)
        @printf("G: ρ_ds=%.4f  cond(G)=%.1f  eig=[%.4f, %.4f]",
            rho_ds, cond_G, eig_G[1], eig_G[2])
        if abs(rho_ds) > 0.95
            print("  ⚠ ρ nær grense!")
        end
        if cond_G > 1e4
            print("  ⚠ dårlig kondisjonert!")
        end
        println()

        # Konvergens-sjekk
        conv_crit = norm(actual_change) / norm(old_params)
        @printf("|score|=%.4e  Δparam=%.6f", score_norm, conv_crit)
        flush(stdout)

        # CSV-logg (med flush for ekstern overvåking)
        if use_log
            open(log_file, "a") do io
                @printf(io, "%d,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,",
                    iter, t_y, t_z, t_ai, t_iter, cum_time, eta_min)
                @printf(io, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,",
                    new_params[1], new_params[2], new_params[3], new_params[4],
                    se[1], se[2], se[3], se[4])
                @printf(io, "%.6e,%.6e,%.4f,%.4f,%.1f,%s\n",
                    score_norm, conv_crit, α, rho_ds, cond_G,
                    conv_crit < 1e-5 ? "true" : "false")
                flush(io)
            end
        end

        if conv_crit < 1e-5
            println("\n\n✅ AI-REML har konvergert 😄.")
            break
        end
        println()
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

    # Pakk ut BLUE/BLUP fra løsningsvektoren
    nf = size(model.X, 2)
    N = size(model.Ainv, 1)
    beta = model.sol_y[1:nf]
    a_d = model.sol_y[nf+1:nf+N]
    a_s = model.sol_y[nf+N+1:end]

    # Beregn endelig AI og SE
    AI_final = AI + 1e-6 * I
    se_final = sqrt.(abs.(diag(inv(AI_final))))

    result = AiRemlResult(
        model.σ2_d, model.σ2_s, model.σ_ds, model.σ2_e,
        beta, a_d, a_s,
        AI, se_final,
        conv_crit < 1e-5, last_iter, n, N, k,
        copy(model.groups.obs_to_id)
    )
    return result
end



