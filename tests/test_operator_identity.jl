include(joinpath(@__DIR__, "..", "src", "emai.jl"))
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

# ---------- Test 4: peer env block (with ψ dilution + classroom env) ----------
# Dense reference: W = [X Z_d Z_s Z_h Z_pe], Z_pe = Z_s[:, pe_ids],
# LHS = W'W + blockdiag(0, σ²_e G⁻¹⊗A⁻¹, (σ²_e/σ²_h) I, (σ²_e/σ²_pe) I).
@testset "peer env: MME operator, RHS, Px, V_pe and diag preconditioner" begin
    Random.seed!(7)
    X = hcat(ones(n), randn(n))
    nf = size(X, 2)
    A = Matrix{Float64}(I, N, N) .+ 0.05 * (rand(N, N) .- 0.5)
    A = (A + A') / 2 + N * I
    Ainv_dense = inv(A)
    Ainv = sparse(Ainv_dense)
    L_Ainv = LowerTriangular(sparse(cholesky(Symmetric(Ainv_dense)).L))

    σ2_d = 1.2; σ2_s = 0.4; σ_ds = 0.15; σ2_e = 1.3; σ2_h = 0.6; σ2_pe = 0.25
    env = copy(gs.obs_to_group)
    y = randn(n)
    model = SGEModel(y, X, Ainv, L_Ainv, gs, σ2_d, σ2_s, σ_ds, σ2_e;
        env_ids=[env], σ2_env=[σ2_h], ψ=0.5, peer_env=true, σ2_pe=σ2_pe)
    lay = model.layout
    @test sort(model.pe_ids) == collect(1:N)
    @test length(lay.pe) == N && last(lay.pe) == lay.dim

    Zs_psi = copy(Zs_dense)
    for i in 1:n
        Zs_psi[i, :] .*= model.group_scale[gs.obs_to_group[i]]
    end
    Z_d = zeros(n, N)
    for (i, id) in enumerate(obs_to_id); Z_d[i, id] = 1.0; end
    Z_h = zeros(n, maximum(env))
    for i in 1:n; Z_h[i, env[i]] = 1.0; end
    Z_pe = Zs_psi[:, model.pe_ids]
    W = hcat(X, Z_d, Zs_psi, Z_h, Z_pe)
    @test size(W, 2) == lay.dim

    G = [σ2_d σ_ds; σ_ds σ2_s]
    LHS = W' * W
    LHS[nf+1:nf+2N, nf+1:nf+2N] .+= kron(inv(G), Ainv_dense) .* σ2_e
    for j in lay.env[1]; LHS[j, j] += σ2_e / σ2_h; end
    for j in lay.pe; LHS[j, j] += σ2_e / σ2_pe; end

    ws = model_workspace(model)
    op = make_mme_operator(model, ws)
    for _ in 1:5
        v = randn(lay.dim)
        r_op = zeros(lay.dim)
        op(r_op, v)
        @test isapprox(r_op, LHS * v; atol=1e-9, rtol=1e-9)
    end

    u = randn(n)
    rhs = zeros(lay.dim)
    build_rhs!(rhs, u, model, ws)
    @test isapprox(rhs, W' * u; atol=1e-10)

    sol = randn(lay.dim)
    Px = zeros(n)
    compute_Px_from_sol!(Px, model, y, sol, ws)
    @test isapprox(Px, (y - W * sol) ./ σ2_e; atol=1e-10)

    out = zeros(n)
    get_Vi_u!(out, u, VarianceComponent(VC_PE, 0), model, ws)
    @test isapprox(out, Z_pe * (Z_pe' * u); atol=1e-10)

    Pd = get_diag_preconditioner(model)
    @test isapprox(1.0 ./ Pd.inv_diag[lay.pe], diag(LHS)[lay.pe]; atol=1e-10)

    # Block preconditioner applies the same diagonal on the pe block
    Pb = get_preconditioner(model, ws)
    r = randn(lay.dim)
    z = zeros(lay.dim)
    ldiv!(z, Pb, r)
    @test isapprox(z[lay.pe], r[lay.pe] ./ diag(LHS)[lay.pe]; atol=1e-10)

    # Exact P*y via dense V must match the matrix-free Px path at the exact MME solution
    V = σ2_e * I + Z_h * Z_h' .* σ2_h + Z_pe * Z_pe' .* σ2_pe +
        hcat(Z_d, Zs_psi) * kron(G, A) * hcat(Z_d, Zs_psi)'
    Vinv = inv(V)
    P = Vinv - Vinv * X * inv(X' * Vinv * X) * X' * Vinv
    sol_exact = LHS \ (W' * y)
    compute_Px_from_sol!(Px, model, y, sol_exact, ws)
    @test isapprox(Px, P * y; atol=1e-8)
end

@testset "peer_env=false keeps the original layout" begin
    X = hcat(ones(n), randn(n))
    A = Matrix{Float64}(I, N, N) * (N + 1.0)
    Ainv = sparse(inv(A))
    L_Ainv = LowerTriangular(sparse(cholesky(Symmetric(Matrix(Ainv))).L))
    m0 = SGEModel(randn(n), X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0)
    @test isempty(m0.layout.pe) && m0.layout.dim == 2 + 2N
    @test get_params(m0) == [1.0, 0.2, 0.0, 2.0]
    @test_throws AssertionError SGEModel(randn(n), X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0; peer_env=true)
end

println("[operator-identity] all tests passed.")
