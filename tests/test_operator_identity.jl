include(joinpath(@__DIR__, "..", "emai.jl"))
using LinearAlgebra, SparseArrays, Random, Test, Printf

# Small dense reference for the matrix-free MMEOperator and Z_s'Z_s contraction.
# Hand-built test exercising MISSING observations in the social design.
#
# Pedigree: N=6 individuals, two classes:
#   class 1 = {1, 2, 3}, class 2 = {4, 5, 6}
# Observed (have y): {1, 2, 4, 5, 6}; ID 3 is roster-only (missing y).
# This forces o_g != m_g in class 1 (o=2, m=3) — the regression we just fixed.

println("[operator-identity] building small reference problem with missing obs...")

const N = 6
const obs_to_id = [1, 2, 4, 5, 6]   # 5 observed rows
const all_group_ids  = [1, 1, 1, 2, 2, 2]
const all_animal_ids = [1, 2, 3, 4, 5, 6]
const obs_group_ids  = [all_group_ids[i] for i in obs_to_id]

gs = GroupStructure(obs_group_ids, obs_to_id, N;
    all_group_ids=all_group_ids, all_animal_ids=all_animal_ids)

n = length(obs_to_id)
group_scale_ones = ones(length(gs.members_id))  # ψ=0 → all ones

# Construct dense reference Z_s explicitly.
# Z_s[i, b] = 1 if b ∈ group(i) and b != obs_to_id[i].
Zs_dense = zeros(Float64, n, N)
for i in 1:n
    g = gs.obs_to_group[i]
    self = gs.obs_to_id[i]
    for b in gs.members_id[g]
        if b != self
            Zs_dense[i, b] = 1.0
        end
    end
end
ZtZ_ref = Zs_dense' * Zs_dense          # N×N reference

# ---------- Test 1: apply_ZsZs_accumulate! matches Z_s' Z_s ----------
Random.seed!(123)
@testset "apply_ZsZs_accumulate! identity (with missing obs)" begin
    # Standard basis sweep — recovers ZtZ_ref column by column
    ZtZ_op = zeros(Float64, N, N)
    for j in 1:N
        v = zeros(Float64, N); v[j] = 1.0
        u = zeros(Float64, N)
        apply_ZsZs_accumulate!(u, v, gs, group_scale_ones)
        ZtZ_op[:, j] = u
    end
    @test isapprox(ZtZ_op, ZtZ_ref; atol=1e-12)

    # Random vectors
    for _ in 1:5
        v = randn(N)
        u = zeros(Float64, N)
        apply_ZsZs_accumulate!(u, v, gs, group_scale_ones)
        @test isapprox(u, ZtZ_ref * v; atol=1e-10)
    end
end

# ---------- Test 2: apply_Zs_transpose! matches Z_s' u ----------
@testset "apply_Zs_transpose! identity (with missing obs)" begin
    group_sum_buf = zeros(Float64, length(gs.members_id))
    for _ in 1:5
        u = randn(n)
        v = zeros(Float64, N)
        fill!(group_sum_buf, 0.0)
        apply_Zs_transpose!(v, u, gs, group_sum_buf, group_scale_ones)
        @test isapprox(v, Zs_dense' * u; atol=1e-10)
    end
end

# ---------- Test 3: full MMEOperator equals dense (X X' + Z_d'Z_d + Z_s'Z_s + G⁻¹⊗A⁻¹) ----------
@testset "MMEOperator identity (with missing obs)" begin
    # Build small SGEModel
    X = hcat(ones(n), randn(n))
    nf = size(X, 2)

    # A = I + small noise → SPD; use its inverse so emai's representation is consistent.
    A = Matrix{Float64}(I, N, N) .+ 0.05 * (rand(N, N) .- 0.5)
    A = (A + A') / 2 + N * I            # SPD
    Ainv_dense = inv(A)
    Ainv = sparse(Ainv_dense)
    L_Ainv = LowerTriangular(sparse(cholesky(Symmetric(Ainv_dense)).L))

    σ2_d = 1.2; σ2_s = 0.4; σ_ds = 0.15; σ2_e = 1.0
    y = randn(n)
    model = SGEModel(y, X, Ainv, L_Ainv, gs, σ2_d, σ2_s, σ_ds, σ2_e)

    # Build dense reference MME.  dim = nf + 2N
    Z_d = zeros(Float64, n, N)
    for (i, id) in enumerate(obs_to_id); Z_d[i, id] = 1.0; end

    W = hcat(X, Z_d, Zs_dense)          # n × dim
    G = [σ2_d σ_ds; σ_ds σ2_s]
    Ginv = inv(G)
    # MME left-hand side WITHOUT residual scaling: W'W + diag(0_f, Ginv⊗Ainv) * σ²_e
    GA = kron(Ginv, Ainv_dense) .* σ2_e
    LHS = W' * W
    LHS[nf+1:nf+2N, nf+1:nf+2N] .+= GA

    # Build operator and apply on random vectors
    ws = model_workspace(model)
    op = make_mme_operator(model, ws)
    dim = size(W, 2)

    for _ in 1:5
        v = randn(dim)
        r_op = zeros(Float64, dim)
        op(r_op, v)
        r_ref = LHS * v
        @test isapprox(r_op, r_ref; atol=1e-9, rtol=1e-9)
    end
end

println("[operator-identity] all tests passed.")
