# =============================================================================
# AI-REML for standard dyremodell (kun direkte genetisk effekt)
# Modell: y = Xβ + Zd·a_d + e
#   a_d ~ N(0, A σ²_d),  e ~ N(0, I σ²_e)
# Variansparametere: θ = (σ²_d, σ²_e)
# MME: [X'X       X'Zd           ] [β  ]   [X'y ]
#      [Zd'X   Zd'Zd + Ainv·λ   ] [a_d] = [Zd'y]
# der λ = σ²_e / σ²_d
# =============================================================================

using Printf
using CSV, DataFrames, LinearAlgebra, SparseArrays, Statistics
using IterativeSolvers
using JLD2
using Random
import Base: *, eltype, size
import LinearAlgebra: mul!, ldiv!

# Forhindre OpenBLAS-tråder fra å krasje med Julia-tråder på Windows VM
LinearAlgebra.BLAS.set_num_threads(1)

# =====================================================================
# PRECONDITIONERS
# =====================================================================

# Rask diagonal preconditioner — unngår findfirst zero-sjekk i standard Diagonal ldiv!
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

# Blokk-preconditioner: diag for faste effekter, (σ²_d/σ²_e)·A for genetisk blokk.
# Bruker Cholesky av Ainv for å anvende A = Ainv⁻¹ (2 sparse tri-solves per applikasjon).
struct BlockPreconditioner
    diag_f_inv::Vector{Float64}     # 1/diag(X'X), length nf
    h2_ratio::Float64               # σ²_d / σ²_e
    L_Ainv::LowerTriangular{Float64,SparseMatrixCSC{Float64,Int64}}
    nf::Int
    N::Int
    tmp1::Vector{Float64}           # workspace (length N)
    tmp2::Vector{Float64}           # workspace (length N)
end

function ldiv!(z::AbstractVector, P::BlockPreconditioner, r::AbstractVector)
    nf, N = P.nf, P.N
    h2r = P.h2_ratio
    tmp1, tmp2 = P.tmp1, P.tmp2

    # Faste effekter: diagonal
    @inbounds @simd for i in 1:nf
        z[i] = P.diag_f_inv[i] * r[i]
    end

    # Genetisk blokk: z_d = (σ²_d/σ²_e) · A · r_d
    # A · r_d via Cholesky: Ainv = L·L' → A = (L')⁻¹ · L⁻¹
    @inbounds for i in 1:N
        tmp1[i] = r[nf+i]
    end
    ldiv!(P.L_Ainv, tmp1)          # tmp1 = L⁻¹ r_d
    copyto!(tmp2, tmp1)
    ldiv!(P.L_Ainv', tmp2)         # tmp2 = A · r_d

    @inbounds @simd for i in 1:N
        z[nf+i] = h2r * tmp2[i]
    end

    return z
end

# =====================================================================
# MODEL
# =====================================================================

mutable struct DirectModel
    y::Vector{Float64}
    X::Matrix{Float64}
    Ainv::SparseMatrixCSC{Float64,Int64}
    L_Ainv::LowerTriangular{Float64,SparseMatrixCSC{Float64,Int64}}
    obs_to_id::Vector{Int}          # Hvilken stamtavle-ID har obs i? (1..n → 1..N)
    σ2_d::Float64
    σ2_e::Float64
    # Warm start
    sol_y::Vector{Float64}
    sol_z::Vector{Vector{Float64}}
    # Preallokkerte workspace-buffere
    rhs_buffer::Vector{Float64}
    tmp_N1::Vector{Float64}
    tmp_N2::Vector{Float64}
    tmp_N3::Vector{Float64}
    # MMEOperator workspace
    op_wn::Vector{Float64}     # length n
    op_wdim::Vector{Float64}   # length dim
    op_Ad::Vector{Float64}     # length N
end

function DirectModel(y, X, Ainv, L_Ainv, obs_to_id, σ2_d, σ2_e)
    N = size(Ainv, 1)
    n = length(y)
    nf = size(X, 2)
    dim = nf + N
    sol_y = zeros(dim)
    sol_z = Vector{Vector{Float64}}()
    rhs_buffer = zeros(dim)
    tmp_N1 = zeros(N)
    tmp_N2 = zeros(N)
    tmp_N3 = zeros(N)
    op_wn = zeros(n)
    op_wdim = zeros(dim)
    op_Ad = zeros(N)
    return DirectModel(y, X, Ainv, L_Ainv, obs_to_id, σ2_d, σ2_e,
        sol_y, sol_z, rhs_buffer, tmp_N1, tmp_N2, tmp_N3,
        op_wn, op_wdim, op_Ad)
end

# =====================================================================
# PER-TRÅD WORKSPACE
# =====================================================================

struct ThreadWorkspace
    rhs_buffer::Vector{Float64}    # length dim
    op_wn::Vector{Float64}         # length n
    op_wdim::Vector{Float64}       # length dim
    op_Ad::Vector{Float64}         # length N
    tmp_N1::Vector{Float64}        # length N
    tmp_N2::Vector{Float64}        # length N
    tmp_N3::Vector{Float64}        # length N
    vi_out::Vector{Float64}        # length n
    precond_tmp1::Vector{Float64}  # length N
    precond_tmp2::Vector{Float64}  # length N
end

function create_thread_workspaces(model::DirectModel, nt::Int)
    n = length(model.y)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    dim = nf + N
    return [ThreadWorkspace(
        zeros(dim),
        zeros(n), zeros(dim), zeros(N),
        zeros(N), zeros(N), zeros(N),
        zeros(n),
        zeros(N), zeros(N)
    ) for _ in 1:nt]
end

# =====================================================================
# MME OPERATOR:  LHS · v  (matrix-free)
# =====================================================================
# LHS = [X'X       X'Zd           ]
#       [Zd'X   Zd'Zd + Ainv·λ   ]
# der λ = σ²_e / σ²_d

struct MMEOperator{T}
    m::DirectModel
    workspace_n::Vector{T}     # y_hat buffer (length n)
    workspace_dim::Vector{T}   # for 5-arg mul! (length dim)
    Ad::Vector{T}              # Ainv * v_d buffer (length N)
    λ::T                       # σ²_e / σ²_d (cached scalar)
end

function make_mme_operator(model::DirectModel)
    λ = model.σ2_e / model.σ2_d
    return MMEOperator{Float64}(model, model.op_wn, model.op_wdim, model.op_Ad, λ)
end

function make_mme_operator(model::DirectModel, ws::ThreadWorkspace)
    λ = model.σ2_e / model.σ2_d
    return MMEOperator{Float64}(model, ws.op_wn, ws.op_wdim, ws.op_Ad, λ)
end

eltype(op::MMEOperator{T}) where T = T
size(op::MMEOperator) = (size(op.m.X, 2) + size(op.m.Ainv, 1),
                          size(op.m.X, 2) + size(op.m.Ainv, 1))
size(op::MMEOperator, i::Int) = size(op)[i]

function mul!(y, op::MMEOperator, x)
    op(y, x)
end

function mul!(y, op::MMEOperator, x, α, β)
    op(op.workspace_dim, x)
    @. y = α * op.workspace_dim + β * y
end

function (op::MMEOperator{T})(res, v) where T
    m = op.m
    n = length(m.y)
    N = size(m.Ainv, 1)
    nf = size(m.X, 2)
    obs_to_id = m.obs_to_id
    v_f = @view(v[1:nf])
    v_d = @view(v[nf+1:nf+N])
    res_f = @view(res[1:nf])
    res_d = @view(res[nf+1:nf+N])
    fill!(res, 0.0)

    # 1. Beregn y_hat = X·v_f + Zd·v_d
    y_hat = op.workspace_n
    mul!(y_hat, m.X, v_f)
    @inbounds for i in 1:n
        y_hat[i] += v_d[obs_to_id[i]]
    end

    # 2. Faste effekter: X'·y_hat
    mul!(res_f, m.X', y_hat)

    # 3. Direkte effekter: Zd'·y_hat + Ainv·λ·v_d
    @inbounds for i in 1:n
        res_d[obs_to_id[i]] += y_hat[i]
    end
    mul!(op.Ad, m.Ainv, v_d)
    @inbounds @simd for i in 1:N
        res[nf+i] += op.λ * op.Ad[i]
    end

    return res
end

# =====================================================================
# CG-SOLVES
# =====================================================================

function build_rhs!(rhs::Vector{Float64}, model::DirectModel, v::Vector{Float64})
    n = length(v)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    fill!(rhs, 0.0)
    mul!(@view(rhs[1:nf]), model.X', v)
    @inbounds for i in 1:n
        rhs[nf+model.obs_to_id[i]] += v[i]
    end
end

function solve_mme_y!(model::DirectModel, P_inv; reltol=1e-7, maxiter=2000, verbose=true)
    build_rhs!(model.rhs_buffer, model, model.y)
    op = make_mme_operator(model)
    if verbose
        _, ch = cg!(model.sol_y, op, model.rhs_buffer, Pl=P_inv,
            reltol=reltol, maxiter=maxiter, log=true)
        niter = ch.iters
        final_res = ch.data[:resnorm][end]
        flag = niter < maxiter ? "✓" : "⚠ MAXITER"
        @printf("    CG(y): %d iter, res=%.2e %s\n", niter, final_res, flag)
    else
        cg!(model.sol_y, op, model.rhs_buffer, Pl=P_inv, reltol=reltol, maxiter=maxiter)
    end
end

function solve_mme_z!(model::DirectModel, z_vector::Vector{Float64}, idx::Int, P_inv;
    reltol=1e-7, maxiter=2000, verbose=false)
    build_rhs!(model.rhs_buffer, model, z_vector)
    op = make_mme_operator(model)
    if verbose
        _, ch = cg!(model.sol_z[idx], op, model.rhs_buffer, Pl=P_inv,
            reltol=reltol, maxiter=maxiter, log=true)
        niter = ch.iters
        final_res = ch.data[:resnorm][end]
        flag = niter < maxiter ? "✓" : "⚠ MAXITER"
        @printf("    CG(z%d): %d iter, res=%.2e %s\n", idx, niter, final_res, flag)
    else
        cg!(model.sol_z[idx], op, model.rhs_buffer, Pl=P_inv, reltol=reltol, maxiter=maxiter)
    end
end

# Thread-safe CG-solve
function solve_mme_direct_threaded!(x_sol, model::DirectModel, v_vector::Vector{Float64},
    ws::ThreadWorkspace, P_inv; reltol=1e-5, maxiter=500)
    build_rhs_threaded!(ws.rhs_buffer, model, v_vector)
    op = make_mme_operator(model, ws)
    cg!(x_sol, op, ws.rhs_buffer, Pl=P_inv, reltol=reltol, maxiter=maxiter)
end

function build_rhs_threaded!(rhs::Vector{Float64}, model::DirectModel, v::Vector{Float64})
    n = length(v)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    fill!(rhs, 0.0)
    mul!(@view(rhs[1:nf]), model.X', v)
    @inbounds for i in 1:n
        rhs[nf+model.obs_to_id[i]] += v[i]
    end
end

function solve_mme_z_threaded!(sol_z_i::Vector{Float64}, model::DirectModel,
    z_vector::Vector{Float64}, ws::ThreadWorkspace, P_inv;
    reltol=1e-7, maxiter=2000)
    build_rhs_threaded!(ws.rhs_buffer, model, z_vector)
    op = make_mme_operator(model, ws)
    cg!(sol_z_i, op, ws.rhs_buffer, Pl=P_inv, reltol=reltol, maxiter=maxiter)
end

function solve_mme_direct!(x_sol, model::DirectModel, v_vector::Vector{Float64}, P_inv;
    reltol=1e-5, maxiter=500)
    build_rhs!(model.rhs_buffer, model, v_vector)
    op = make_mme_operator(model)
    cg!(x_sol, op, model.rhs_buffer, Pl=P_inv, reltol=reltol, maxiter=maxiter)
end

# =====================================================================
# Px = V⁻¹ · (y - Xβ̂ - Zd·â_d) / σ²_e   (fra MME-løsning)
# =====================================================================

function compute_Px_from_sol!(out::Vector{Float64}, model::DirectModel,
    y_vector::Vector{Float64}, sol_vector::Vector{Float64})
    n = length(model.y)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    obs_to_id = model.obs_to_id

    beta = @view sol_vector[1:nf]
    a_d = @view sol_vector[nf+1:nf+N]

    # y_hat = X·β̂ + Zd·â_d
    mul!(out, model.X, beta)
    @inbounds for i in 1:n
        out[i] += a_d[obs_to_id[i]]
    end

    # Px = (y - y_hat) / σ²_e
    inv_σ2e = 1.0 / model.σ2_e
    @inbounds @simd for i in eachindex(out)
        out[i] = (y_vector[i] - out[i]) * inv_σ2e
    end
    return out
end

# =====================================================================
# V_i · u  (variansderivat-matrise ganger vektor)
# =====================================================================
# dV/dσ²_d = Zd · A · Zd'   (type :d)
# dV/dσ²_e = I              (type :e)

function apply_A!(out::Vector{Float64}, model::DirectModel, x_N::Vector{Float64})
    copyto!(model.tmp_N3, x_N)
    ldiv!(model.L_Ainv, model.tmp_N3)
    copyto!(out, model.tmp_N3)
    ldiv!(model.L_Ainv', out)
end

function apply_A_threaded!(out::Vector{Float64}, L_Ainv, x_N::Vector{Float64}, tmp::Vector{Float64})
    copyto!(tmp, x_N)
    ldiv!(L_Ainv, tmp)
    copyto!(out, tmp)
    ldiv!(L_Ainv', out)
end

function get_Vi_u(u_n, type::Symbol, model::DirectModel)
    n = length(u_n)
    if type == :d  # Zd · A · Zd' · u
        fill!(model.tmp_N1, 0.0)
        @inbounds for i in 1:n
            model.tmp_N1[model.obs_to_id[i]] += u_n[i]
        end
        apply_A!(model.tmp_N2, model, model.tmp_N1)
        out = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            out[i] = model.tmp_N2[model.obs_to_id[i]]
        end
        return out
    else  # :e → I · u
        return copy(u_n)
    end
end

# In-place versjon for non-threaded bruk — null allokeringer
function get_Vi_u!(out::Vector{Float64}, u_n, type::Symbol, model::DirectModel)
    n = length(u_n)
    if type == :d  # Zd · A · Zd' · u
        fill!(model.tmp_N1, 0.0)
        @inbounds for i in 1:n
            model.tmp_N1[model.obs_to_id[i]] += u_n[i]
        end
        apply_A!(model.tmp_N2, model, model.tmp_N1)
        @inbounds for i in 1:n
            out[i] = model.tmp_N2[model.obs_to_id[i]]
        end
    else  # :e → I · u
        copyto!(out, u_n)
    end
    return out
end

function get_Vi_u!(out::Vector{Float64}, u_n, type::Symbol, model::DirectModel, ws::ThreadWorkspace)
    n = length(u_n)
    if type == :d  # Zd · A · Zd' · u
        fill!(ws.tmp_N1, 0.0)
        @inbounds for i in 1:n
            ws.tmp_N1[model.obs_to_id[i]] += u_n[i]
        end
        apply_A_threaded!(ws.tmp_N2, model.L_Ainv, ws.tmp_N1, ws.tmp_N3)
        @inbounds for i in 1:n
            out[i] = ws.tmp_N2[model.obs_to_id[i]]
        end
    else  # :e → I · u
        copyto!(out, u_n)
    end
    return out
end

# =====================================================================
# SCORE OG AI-MATRISE
# =====================================================================

function compute_score_and_ai(model::DirectModel, Py, z_list, Pz_list, P_inv;
    ai_reltol=1e-5, ai_maxiter=500, thread_ws=nothing, P_inv_threads=nothing)
    types = [:d, :e]
    n_p = 2
    score = zeros(n_p)
    AI = zeros(n_p, n_p)
    k = length(Pz_list)
    n = length(Py)
    use_threads = thread_ws !== nothing && Threads.nthreads() > 1

    # 1. Forhåndsberegn Vi · Py  (2 mat-vecs)
    println("    -> Beregner Vi * Py...")
    if use_threads
        Vi_Py_list = [Vector{Float64}(undef, n) for _ in 1:n_p]
        Threads.@threads :static for i in 1:n_p
            tid = Threads.threadid()
            get_Vi_u!(Vi_Py_list[i], Py, types[i], model, thread_ws[tid])
        end
    else
        Vi_Py_list = [Vector{Float64}(undef, n) for _ in 1:n_p]
        for i in 1:n_p
            get_Vi_u!(Vi_Py_list[i], Py, types[i], model)
        end
    end

    # 2. SCORE: s_i = 0.5 * (y'P · Vi · Py  -  tr(P · Vi))
    println("    -> Beregner trace (k=$k)...")
    if use_threads
        n_trace = n_p * k
        trace_results = zeros(n_trace)
        Threads.@threads :static for flat_idx in 1:n_trace
            tid = Threads.threadid()
            ws = thread_ws[tid]
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
        vi_tmp = Vector{Float64}(undef, n)  # Gjenbrukes for alle trace-beregninger
        for i in 1:n_p
            tr_val = 0.0
            for j in 1:k
                get_Vi_u!(vi_tmp, Pz_list[j], types[i], model)
                @inbounds tr_val += dot(z_list[j], vi_tmp) / k
            end
            score[i] = 0.5 * (Py' * Vi_Py_list[i] - tr_val)
        end
    end

    # 3. AI-MATRISE: AI[i,j] = 0.5 · Py' · Vi · P · Vj · Py
    #    Trenger n_p=2 CG-solves (én per V_j)
    println("    -> Beregner AI-matrise (2 CG-solves)...")
    P_Vi_Py_list = [Vector{Float64}(undef, n) for _ in 1:n_p]
    if use_threads
        tmp_sols = [copy(model.sol_y) for _ in 1:n_p]
        Threads.@threads :static for j in 1:n_p
            tid = Threads.threadid()
            ws = thread_ws[tid]
            P_t = P_inv_threads[tid]
            solve_mme_direct_threaded!(tmp_sols[j], model, Vi_Py_list[j], ws, P_t,
                reltol=ai_reltol, maxiter=ai_maxiter)
            compute_Px_from_sol!(P_Vi_Py_list[j], model, Vi_Py_list[j], tmp_sols[j])
        end
    else
        tmp_sol = zeros(length(model.sol_y))
        for j in 1:n_p
            copyto!(tmp_sol, model.sol_y)
            solve_mme_direct!(tmp_sol, model, Vi_Py_list[j], P_inv,
                reltol=ai_reltol, maxiter=ai_maxiter)
            compute_Px_from_sol!(P_Vi_Py_list[j], model, Vi_Py_list[j], tmp_sol)
        end
    end

    # Steg 2: AI[i,j] = 0.5 · Py' · Vi · (P · Vj · Py)  (billige mat-vecs)
    pairs = [(i, j) for i in 1:n_p for j in 1:i]  # 3 par: (1,1),(2,1),(2,2)
    if use_threads
        ai_vals = zeros(length(pairs))
        Threads.@threads :static for idx in eachindex(pairs)
            tid = Threads.threadid()
            ws = thread_ws[tid]
            i_idx, j_idx = pairs[idx]
            get_Vi_u!(ws.vi_out, P_Vi_Py_list[j_idx], types[i_idx], model, ws)
            ai_vals[idx] = 0.5 * dot(Py, ws.vi_out)
        end
        for (idx, (i_idx, j_idx)) in enumerate(pairs)
            AI[i_idx, j_idx] = ai_vals[idx]
            AI[j_idx, i_idx] = ai_vals[idx]
        end
    else
        vi_tmp_ai = Vector{Float64}(undef, n)  # Gjenbrukes for alle AI-par
        for (i_idx, j_idx) in pairs
            get_Vi_u!(vi_tmp_ai, P_Vi_Py_list[j_idx], types[i_idx], model)
            val = 0.5 * dot(Py, vi_tmp_ai)
            AI[i_idx, j_idx] = val
            AI[j_idx, i_idx] = val
        end
    end
    return score, AI
end

# =====================================================================
# PRECONDITIONER-BYGGING
# =====================================================================

function get_preconditioner(model::DirectModel)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    diag_f_inv = Vector{Float64}(undef, nf)
    for j in 1:nf
        @inbounds diag_f_inv[j] = 1.0 / max(dot(@view(model.X[:, j]), @view(model.X[:, j])), 1e-6)
    end
    h2r = model.σ2_d / model.σ2_e
    return BlockPreconditioner(diag_f_inv, h2r, model.L_Ainv, nf, N,
        Vector{Float64}(undef, N), Vector{Float64}(undef, N))
end

function get_thread_preconditioner(model::DirectModel, ws::ThreadWorkspace)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    diag_f_inv = Vector{Float64}(undef, nf)
    for j in 1:nf
        @inbounds diag_f_inv[j] = 1.0 / max(dot(@view(model.X[:, j]), @view(model.X[:, j])), 1e-6)
    end
    h2r = model.σ2_d / model.σ2_e
    return BlockPreconditioner(diag_f_inv, h2r, model.L_Ainv, nf, N,
        ws.precond_tmp1, ws.precond_tmp2)
end

function get_diag_preconditioner(model::DirectModel)
    N = size(model.Ainv, 1)
    nf = size(model.X, 2)
    n = length(model.y)
    obs_to_id = model.obs_to_id
    diag_Ainv = diag(model.Ainv)
    λ = model.σ2_e / model.σ2_d

    d = zeros(nf + N)

    # 1. Faste effekter: diag(X'X)
    for j in 1:nf
        @inbounds d[j] = dot(@view(model.X[:, j]), @view(model.X[:, j]))
    end

    # 2. Direkte genetikk: diag(Zd'Zd) + Ainv · λ
    diag_ZdZd = @view d[nf+1:nf+N]
    for i in 1:n
        @inbounds diag_ZdZd[obs_to_id[i]] += 1.0
    end
    @. d[nf+1:nf+N] = diag_ZdZd + diag_Ainv * λ

    return DiagPreconditioner(1.0 ./ max.(d, 1e-6))
end

# =====================================================================
# PARAMETER-OPPDATERING
# =====================================================================

function update_params!(model::DirectModel, delta, damping)
    new_d2 = model.σ2_d + damping * delta[1]
    new_e2 = model.σ2_e + damping * delta[2]
    model.σ2_d = max(1e-4, new_d2)
    model.σ2_e = max(1e-2, new_e2)
end

# =====================================================================
# RESULTAT-STRUCT OG LAGRING (JLD2)
# =====================================================================

struct DirectRemlResult
    # Varianskomponenter
    σ2_d::Float64
    σ2_e::Float64
    # BLUE og BLUP
    beta::Vector{Float64}      # Faste effekter (nf)
    a_d::Vector{Float64}       # Direkte genetiske effekter (N)
    # Inferens
    AI::Matrix{Float64}        # Average Information matrise (2×2)
    se::Vector{Float64}        # Standardfeil (2)
    # Metadata
    converged::Bool
    n_iter::Int
    n_obs::Int
    N_pedigree::Int
    k::Int
    obs_to_id::Vector{Int}
end

function save_results(result::DirectRemlResult, filename::String)
    jldsave(filename; result=result)
    sz_mb = filesize(filename) / 1024^2
    @printf("Resultat lagret til %s (%.1f MB)\n", filename, sz_mb)
end

function load_results(filename::String)
    return load(filename, "result")
end

# =====================================================================
# HOVEDALGORITME: AI-REML
# =====================================================================

function run_aireml(model::DirectModel; max_iter=15, k=5, damping=0.2,
    cg_maxiter=2000, cg_reltol=1e-7,
    ai_maxiter=500, ai_reltol=1e-5,
    use_block_precond=true, verbose_cg=true,
    threaded=true, log_file::String="",
    seed::Int=42)
    n = length(model.y)
    dim = length(model.sol_y)
    param_names = ["σ²_d", "σ²_e"]

    nt = Threads.nthreads()
    max_tid = Threads.maxthreadid()   # Julia 1.12: threadid() kan returnere > nthreads()
    use_threads = threaded && nt > 1

    model.sol_z = [zeros(dim) for _ in 1:k]

    # Vi må bruke de SAMME z-vektorene i hver iterasjon
    # for at Hutchinson-estimatet skal være sammenlignbart
    Random.seed!(seed)
    z_fixed_list = [randn(n) for _ in 1:k]

    Py = Vector{Float64}(undef, n)
    Pz_list = [Vector{Float64}(undef, n) for _ in 1:k]

    thread_ws = use_threads ? create_thread_workspaces(model, max_tid) : nothing

    precond_name = use_block_precond ? "Block" : "Diag"
    thread_info = use_threads ? ", threads=$nt" : ""
    println("Starter AI-REML (direkte modell): k=$k, damping=$damping, precond=$precond_name, cg_maxiter=$cg_maxiter$thread_info")
    println("N=$(size(model.Ainv,1)), n=$n, dim=$dim")
    println("-"^70)
    flush(stdout)

    use_log = !isempty(log_file)
    if use_log
        open(log_file, "w") do io
            println(io, "iter,time_y,time_z,time_ai,time_total,cum_time,eta_min,",
                "s2d,s2e,se_d,se_e,score_norm,conv_crit,alpha,converged")
        end
        println("Iterasjonslogg skrives til: $log_file")
    end

    AI = zeros(2, 2)
    conv_crit = Inf
    last_iter = 0
    iter_times = Float64[]
    t_start_total = time()

    for iter in 1:max_iter
        last_iter = iter
        t_iter_start = time()

        P_inv = use_block_precond ? get_preconditioner(model) : get_diag_preconditioner(model)

        P_inv_threads = nothing
        if use_threads
            if use_block_precond
                P_inv_threads = [get_thread_preconditioner(model, ws) for ws in thread_ws]
            else
                P_inv_threads = fill(get_diag_preconditioner(model), max_tid)
            end
        end

        # 1. Løs MME for y (warm start)
        t_y = @elapsed begin
            solve_mme_y!(model, P_inv, reltol=cg_reltol, maxiter=cg_maxiter, verbose=verbose_cg)
            compute_Px_from_sol!(Py, model, model.y, model.sol_y)
        end

        # 2. Løs MME for k z-vektorer (parallelliserbart)
        if verbose_cg
            @printf("    Z-solves: k=%d %s...\n", k, use_threads ? "(threaded)" : "")
            flush(stdout)
        end
        t_z = @elapsed begin
            if use_threads
                Threads.@threads :static for i in 1:k
                    tid = Threads.threadid()
                    ws = thread_ws[tid]
                    P_t = P_inv_threads[tid]
                    solve_mme_z_threaded!(model.sol_z[i], model, z_fixed_list[i], ws, P_t,
                        reltol=cg_reltol, maxiter=cg_maxiter)
                    compute_Px_from_sol!(Pz_list[i], model, z_fixed_list[i], model.sol_z[i])
                end
            else
                for i in 1:k
                    solve_mme_z!(model, z_fixed_list[i], i, P_inv,
                        reltol=cg_reltol, maxiter=cg_maxiter, verbose=(verbose_cg && i == 1))
                    compute_Px_from_sol!(Pz_list[i], model, z_fixed_list[i], model.sol_z[i])
                end
            end
        end
        if verbose_cg
            @printf("    Z-solves ferdig: %.1fs\n", t_z)
            flush(stdout)
        end

        # 3. Score og AI
        if verbose_cg
            println("    Score + AI...")
            flush(stdout)
        end
        t_ai = @elapsed begin
            score, AI_new = compute_score_and_ai(model, Py, z_fixed_list, Pz_list, P_inv,
                ai_reltol=ai_reltol, ai_maxiter=ai_maxiter,
                thread_ws=thread_ws, P_inv_threads=P_inv_threads)
            copyto!(AI, AI_new)
        end
        if verbose_cg
            @printf("    Score + AI ferdig: %.1fs\n", t_ai)
            flush(stdout)
        end

        # 4. Løs og beregn SE
        AI_reg = AI + 1e-6 * I
        delta = AI_reg \ score
        se = sqrt.(abs.(diag(inv(AI_reg))))

        # 5. Lagre gamle verdier
        old_params = [model.σ2_d, model.σ2_e]

        # 6. Step-halving
        α = damping
        for _attempt in 1:20
            new_d = model.σ2_d + α * delta[1]
            new_e = model.σ2_e + α * delta[2]
            if new_d > 1e-6 && new_e > 1e-4
                break
            end
            α *= 0.5
        end

        # 7. Oppdater
        update_params!(model, delta, α)
        new_params = [model.σ2_d, model.σ2_e]
        actual_change = new_params .- old_params

        # --- UTSKRIFT ---
        t_iter = time() - t_iter_start
        push!(iter_times, t_iter)
        cum_time = time() - t_start_total
        avg_iter = mean(iter_times)
        remaining_iters = max_iter - iter
        eta_min = remaining_iters * avg_iter / 60.0

        h2 = model.σ2_d / (model.σ2_d + model.σ2_e)
        score_norm = norm(score)

        println("\n" * "="^70)
        @printf("ITERASJON %d  (α=%.3f)  |  %.1fs (y:%.1f z:%.1f ai:%.1f)  cum:%.0fs  ETA:%.0fmin\n",
            iter, α, t_iter, t_y, t_z, t_ai, cum_time, eta_min)
        println("="^70)
        @printf("%-10s %-12s %-12s %-12s %-12s\n", "Param", "Est", "SE", "Score", "Delta")
        println("-"^70)
        for i in 1:2
            @printf("%-10s %-12.6f %-12.6f %-+12.4f %-+12.6f\n",
                param_names[i], new_params[i], se[i], score[i], delta[i])
        end
        println("-"^70)
        @printf("h² = %.4f\n", h2)

        # Konvergens-sjekk
        conv_crit = norm(actual_change) / norm(old_params)
        @printf("|score|=%.4e  Δparam=%.6f", score_norm, conv_crit)
        flush(stdout)

        if use_log
            open(log_file, "a") do io
                @printf(io, "%d,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,",
                    iter, t_y, t_z, t_ai, t_iter, cum_time, eta_min)
                @printf(io, "%.6f,%.6f,%.6f,%.6f,",
                    new_params[1], new_params[2], se[1], se[2])
                @printf(io, "%.6e,%.6e,%.4f,%s\n",
                    score_norm, conv_crit, α,
                    conv_crit < 1e-5 ? "true" : "false")
                flush(io)
            end
        end

        if conv_crit < 1e-5
            println("\n\n✅ AI-REML har konvergert.")
            break
        end
        println()
        if iter == max_iter
            println("\n⚠️ Maks iterasjoner nådd uten full konvergens.")
        end
    end

    # Final solve med endelige varianskomponenter → BLUE(β) og BLUP(a_d)
    println("\nBeregner endelige BLUE/BLUP-estimater...")
    P_inv = use_block_precond ? get_preconditioner(model) : get_diag_preconditioner(model)
    fill!(model.sol_y, 0.0)
    solve_mme_y!(model, P_inv, reltol=cg_reltol, maxiter=cg_maxiter, verbose=verbose_cg)

    nf = size(model.X, 2)
    N = size(model.Ainv, 1)
    beta = model.sol_y[1:nf]
    a_d = model.sol_y[nf+1:nf+N]

    AI_final = AI + 1e-6 * I
    se_final = sqrt.(abs.(diag(inv(AI_final))))

    result = DirectRemlResult(
        model.σ2_d, model.σ2_e,
        beta, a_d,
        AI, se_final,
        conv_crit < 1e-5, last_iter, n, N, k,
        copy(model.obs_to_id)
    )
    return result
end

function load_emai_direct_model(; yx_path::String="yX_obs.csv",
    Ainv_path::String="Ainv_triplet.csv", L_Ainv_path::String="LAinv_triplet.csv",
    σ2_d::Float64=1.0, σ2_e::Float64=2.0)
    yX = DataFrame(CSV.File(yx_path))
    @assert hasproperty(yX, :y) "Observed table must contain column :y"
    @assert hasproperty(yX, :id_ped) "Observed table must contain column :id_ped"
    @assert hasproperty(yX, :x) "Observed table must contain column :x"
    @assert !any(ismissing.(yX[:, :y])) "yX_obs.csv must contain observed y only (no missing values)"
    n = nrow(yX)
    X = hcat(ones(n), Float64.(yX[:, :x]))
    y = Float64.(yX[:, :y])

    Ainv_df = DataFrame(CSV.File(Ainv_path))
    i_vals = Ainv_df[:, "i"]
    j_vals = Ainv_df[:, "j"]
    x_vals = Float64.(Ainv_df[:, "x"])

    off_diag = i_vals .!= j_vals
    i_all = vcat(i_vals, j_vals[off_diag])
    j_all = vcat(j_vals, i_vals[off_diag])
    x_all = vcat(x_vals, x_vals[off_diag])

    Ainv = sparse(i_all, j_all, x_all)

    L_Ainv_df = DataFrame(CSV.File(L_Ainv_path))
    L_Ainv = sparse(L_Ainv_df[:, "i"], L_Ainv_df[:, "j"], Float64.(L_Ainv_df[:, "x"]))
    L_Ainv = LowerTriangular(L_Ainv)

    obs_to_id = Int.(yX[:, :id_ped])
    return DirectModel(y, X, Ainv, L_Ainv, obs_to_id, σ2_d, σ2_e)
end

function run_emai_direct(; yx_path::String="yX_obs.csv",
    Ainv_path::String="Ainv_triplet.csv", L_Ainv_path::String="LAinv_triplet.csv",
    output_path::String="aireml_direct_result.jld2", log_path::String="aireml_direct_log.csv",
    σ2_d::Float64=1.0, σ2_e::Float64=2.0, kwargs...)
    model = load_emai_direct_model(; yx_path=yx_path, Ainv_path=Ainv_path,
        L_Ainv_path=L_Ainv_path, σ2_d=σ2_d, σ2_e=σ2_e)
    result = run_aireml(model; log_file=log_path, kwargs...)
    save_results(result, output_path)
    return model, result
end
