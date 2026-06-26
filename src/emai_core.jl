# =====================================================================
# CORE TYPES & DATA CONSTRUCTION
# Shared structs (preconditioners, GroupStructure, SGEModel, ThreadWorkspace)
# and observed-data / group-structure builders used by the MME and AI-REML
# layers. Included via emai.jl.
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
    nf::Int
    N::Int
    tmp1::Vector{Float64}           # workspace (length N)
    tmp2::Vector{Float64}           # workspace (length N)
    env_diag_inv::Vector{Vector{Float64}}  # H vectors, 1/(count_j + σ²_e/σ²_h)
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

    # Env blocks: diagonal preconditioner
    env_offset = nf + 2N
    for h in eachindex(P.env_diag_inv)
        dinv = P.env_diag_inv[h]
        @inbounds @simd for j in eachindex(dinv)
            z[env_offset+j] = dinv[j] * r[env_offset+j]
        end
        env_offset += length(dinv)
    end

    return z
end

struct GroupStructure
    members_id::Vector{Vector{Int}} # ID-er i stamtavlen (1..N)
    obs_to_id::Vector{Int}          # Hvilken stamtavle-ID har obs i? (1..n)
    id_to_group::Vector{Int}        # Hvilken gruppe tilhører stamtavle-ID i?
    obs_to_group::Vector{Int}       # Precomputed: hvilken gruppe tilhører obs i? (1..num_groups)
    obs_count_per_group::Vector{Int} # o_g: antall observerte rader i gruppe g (1..num_groups)
    is_observed::Vector{Bool}       # length N_pedigree; true hvis pedigree-ID har obs-rad
end

function GroupStructure(group_ids::Vector{Int}, animal_ids::Vector{Int}, N_pedigree::Int;
    all_group_ids::Union{Nothing,Vector{Int}}=nothing,
    all_animal_ids::Union{Nothing,Vector{Int}}=nothing)
    @assert length(group_ids) == length(animal_ids) "group_ids and animal_ids must have same length"

    has_full_mapping = !(all_group_ids === nothing)
    @assert has_full_mapping == !(all_animal_ids === nothing) "Provide both all_group_ids and all_animal_ids, or neither"

    # Build social-group membership either from observed rows only (legacy behavior)
    # or from a full pedigree-level membership mapping that can include unobserved members.
    base_group_ids = has_full_mapping ? all_group_ids::Vector{Int} : group_ids
    base_animal_ids = has_full_mapping ? all_animal_ids::Vector{Int} : animal_ids

    @assert length(base_group_ids) == length(base_animal_ids) "all_group_ids and all_animal_ids must have same length"
    @assert !isempty(base_group_ids) "Group mapping cannot be empty"
    @assert minimum(base_animal_ids) >= 1 && maximum(base_animal_ids) <= N_pedigree "Animal IDs must be in 1..N_pedigree"
    @assert minimum(animal_ids) >= 1 && maximum(animal_ids) <= N_pedigree "Observed animal IDs must be in 1..N_pedigree"

    unique_groups = unique(base_group_ids)
    group_map = Dict(g => i for (i, g) in enumerate(unique_groups))
    num_groups = length(unique_groups)

    members = [Int[] for _ in 1:num_groups]
    for (i, g_id) in enumerate(base_group_ids)
        g_idx = group_map[g_id]
        push!(members[g_idx], base_animal_ids[i])
    end

    # Mapping for alle dyr i stamtavlen (0 = ingen gruppe)
    id_to_group = zeros(Int, N_pedigree)
    for (i, g_id) in enumerate(base_group_ids)
        id = base_animal_ids[i]
        g_idx = group_map[g_id]
        prev = id_to_group[id]
        @assert prev == 0 || prev == g_idx "Pedigree ID $id appears in multiple groups"
        id_to_group[id] = g_idx
    end

    # All observed rows must map to a valid group index.
    for id in animal_ids
        @assert id_to_group[id] != 0 "Observed pedigree ID $id has no group assignment"
    end

    obs_to_group = [id_to_group[id] for id in animal_ids]

    # Per-group observed count and per-pedigree observed flag.
    # Needed by apply_ZsZs_accumulate! to handle missing-obs members correctly.
    obs_count_per_group = zeros(Int, num_groups)
    is_observed = falses(N_pedigree)
    for (i, id) in enumerate(animal_ids)
        obs_count_per_group[obs_to_group[i]] += 1
        is_observed[id] = true
    end

    return GroupStructure(members, animal_ids, id_to_group, obs_to_group,
        obs_count_per_group, is_observed)
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
    # Shared environment effects (0..H grouping factors)
    env_ids::Vector{Vector{Int}}       # H vectors, each length n, obs → level (1..n_h)
    env_n_levels::Vector{Int}          # n_h per factor
    env_counts::Vector{Vector{Int}}    # obs per level, for preconditioner
    σ2_env::Vector{Float64}            # H scalar variances
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
    ψ::Float64                # Dilution exponent: social peer sum scaled by (m_g-1)^{-ψ}
    group_scale::Vector{Float64}  # Per-group dilution factor; 1.0 when ψ=0, 0.0 when m_g<=1
end

# Beregner total dimensjon for løsningsvektoren
function get_dim(model::SGEModel)
    nf = size(model.X, 2)
    N = size(model.Ainv, 1)
    return nf + 2N + sum(model.env_n_levels)
end

function SGEModel(y, X, Ainv, L_Ainv, groups, σ2_d, σ2_s, σ_ds, σ2_e;
    env_ids::Vector{Vector{Int}}=Vector{Int}[], σ2_env::Vector{Float64}=Float64[],
    ψ::Float64=0.0)
    N = size(Ainv, 1)
    n = length(y)
    nf = size(X, 2)
    H = length(env_ids)
    @assert length(σ2_env) == H "env_ids and σ2_env must have same length"

    # Valider og precompute env
    env_n_levels = Int[]
    env_counts = Vector{Int}[]
    for h in 1:H
        ids = env_ids[h]
        @assert length(ids) == n "env_ids[$h] must have length n=$n"
        n_h = maximum(ids)
        @assert minimum(ids) >= 1 "env_ids[$h] must be 1-based"
        push!(env_n_levels, n_h)
        counts = zeros(Int, n_h)
        for i in 1:n
            counts[ids[i]] += 1
        end
        push!(env_counts, counts)
    end

    dim = nf + 2N + sum(env_n_levels)
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
    group_scale = [length(m) <= 1 ? 0.0 : (ψ == 0.0 ? 1.0 : Float64(length(m) - 1)^(-ψ))
                   for m in groups.members_id]
    return SGEModel(y, X, Ainv, L_Ainv, groups, σ2_d, σ2_s, σ_ds, σ2_e,
        env_ids, env_n_levels, env_counts, σ2_env,
        sol_y, sol_z, rhs_buffer, tmp_N1, tmp_N2, tmp_N3,
        op_wn, op_wn2, op_wdim, op_Ad, op_As, op_Ginv,
        ψ, group_scale)
end

function load_observed_data(yX::DataFrame;
    X_df::Union{Nothing,DataFrame}=nothing,
    covariate_cols::Vector{Symbol}=Symbol[:x],
    add_intercept::Bool=true)
    @assert hasproperty(yX, :y) "yX must contain column :y"
    @assert hasproperty(yX, :id_ped) "yX must contain column :id_ped"

    obs_mask = .!ismissing.(yX[:, :y])
    n_obs = count(obs_mask)
    @assert n_obs > 0 "yX contains no observed outcomes"

    y = Float64.(yX[obs_mask, :y])

    if X_df === nothing
        for col in covariate_cols
            @assert hasproperty(yX, col) "yX must contain column :$col"
        end
        x_parts = Vector{Vector{Float64}}()
        if add_intercept
            push!(x_parts, ones(n_obs))
        end
        for col in covariate_cols
            push!(x_parts, Float64.(yX[obs_mask, col]))
        end
        X = hcat(x_parts...)
    else
        X = if nrow(X_df) == nrow(yX)
            Float64.(Matrix(X_df[obs_mask, :]))
        elseif nrow(X_df) == n_obs
            Float64.(Matrix(X_df))
        else
            error("X design has $(nrow(X_df)) rows, but yX has $(nrow(yX)) rows and $(n_obs) observed rows")
        end
    end

    obs_to_id = Int.(yX[obs_mask, :id_ped])
    return (; obs_mask, y, X, obs_to_id)
end

function build_group_structure(yX::DataFrame, N_pedigree::Int, obs_mask::AbstractVector{Bool})
    @assert hasproperty(yX, :group) "yX must contain column :group"
    @assert hasproperty(yX, :id_ped) "yX must contain column :id_ped"
    return GroupStructure(Int.(yX[obs_mask, :group]), Int.(yX[obs_mask, :id_ped]), N_pedigree;
        all_group_ids=Int.(yX[:, :group]), all_animal_ids=Int.(yX[:, :id_ped]))
end

function build_group_structure(obs_table::DataFrame, roster::DataFrame, N_pedigree::Int)
    @assert hasproperty(obs_table, :id_ped) "obs_table must contain column :id_ped"
    @assert hasproperty(roster, :group) "roster must contain column :group"
    @assert hasproperty(roster, :id_ped) "roster must contain column :id_ped"

    obs_to_id = Int.(obs_table[:, :id_ped])
    roster_ids = Int.(roster[:, :id_ped])
    roster_groups = Int.(roster[:, :group])

    obs_group_lookup = zeros(Int, N_pedigree)
    for (id, g) in zip(roster_ids, roster_groups)
        prev = obs_group_lookup[id]
        @assert prev == 0 || prev == g "Roster id_ped $id appears in multiple groups"
        obs_group_lookup[id] = g
    end

    obs_group_ids = [obs_group_lookup[id] for id in obs_to_id]
    @assert all(g -> g != 0, obs_group_ids) "Every observed id_ped must have a group in roster.csv"

    return GroupStructure(obs_group_ids, obs_to_id, N_pedigree;
        all_group_ids=roster_groups, all_animal_ids=roster_ids)
end

function observed_env_ids(yX::DataFrame, obs_mask::AbstractVector{Bool}, cols::Vector{Symbol})
    ids = Vector{Vector{Int}}()
    for col in cols
        @assert hasproperty(yX, col) "yX must contain column :$col"
        push!(ids, Int.(yX[obs_mask, col]))
    end
    return ids
end

function observed_env_ids(obs_table::DataFrame, cols::Vector{Symbol})
    ids = Vector{Vector{Int}}()
    for col in cols
        @assert hasproperty(obs_table, col) "obs_table must contain column :$col"
        push!(ids, Int.(obs_table[:, col]))
    end
    return ids
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
    # Shared environment workspace (one tmp per factor)
    env_tmp::Vector{Vector{Float64}}  # H vectors, each length n_h
    # Group-sum buffer (length num_groups) — used to cache Σ v[bid] per group
    group_sum::Vector{Float64}
end

function create_thread_workspaces(model::SGEModel, nt::Int)
    n = length(model.y)
    N = size(model.Ainv, 1)
    dim = get_dim(model)
    ng = length(model.groups.members_id)
    return [ThreadWorkspace(
        zeros(dim),
        zeros(n), zeros(n), zeros(dim),
        zeros(N), zeros(N), zeros(2, 2),
        zeros(N), zeros(N), zeros(N),
        zeros(n),
        zeros(N), zeros(N),
        [zeros(n_h) for n_h in model.env_n_levels],
        zeros(ng)
    ) for _ in 1:nt]
end

# Lag et enkelt ThreadWorkspace som aliaser modellens egne buffere (for sekvensiell bruk)
function model_workspace(model::SGEModel)
    n = length(model.y)
    N = size(model.Ainv, 1)
    ng = length(model.groups.members_id)
    return ThreadWorkspace(
        model.rhs_buffer,
        model.op_wn, model.op_wn2, model.op_wdim,
        model.op_Ad, model.op_As, model.op_Ginv,
        model.tmp_N1, model.tmp_N2, model.tmp_N3,
        zeros(n),       # vi_out
        zeros(N), zeros(N),  # precond_tmp1, precond_tmp2
        [zeros(n_h) for n_h in model.env_n_levels],
        zeros(ng)
    )
end
