if !@isdefined(SGEModel)
    include(joinpath(@__DIR__, "..", "src", "emai.jl"))
end
using LinearAlgebra, SparseArrays, Random, Test

@testset "pedigree factor and CG diagnostics" begin
    rng = MersenneTwister(41)
    N = 6
    groups = GroupStructure([1, 1, 2, 2, 2], [1, 2, 4, 5, 6], N;
        all_group_ids=[1, 1, 1, 2, 2, 2], all_animal_ids=collect(1:N))
    B = randn(rng, N, N)
    Ainv = sparse(B * B' + I)
    L = LowerTriangular(sparse(cholesky(Symmetric(Matrix(Ainv))).L))
    model = SGEModel(randn(rng, 5), hcat(ones(5), randn(rng, 5)),
        Ainv, L, groups, 1.0, 0.2, 0.05, 1.0; peer_env=true, σ2_pe=0.1)
    ws = model_workspace(model)

    @test validate_pedigree_factor(model, ws) < 1e-12
    Random.seed!(91)
    expected_random = rand()
    Random.seed!(91)
    validate_pedigree_factor(model, ws)
    @test rand() == expected_random
    @test all(iszero, model.sol_y)

    model.L_Ainv = LowerTriangular(2.0 .* parent(L))
    @test_throws r"Ainv and L_Ainv disagree" validate_pedigree_factor(model, ws)
    # A valid factor in a different pedigree order is still invalid for this model.
    perm = [6, 2, 3, 4, 5, 1]
    model.L_Ainv = LowerTriangular(sparse(cholesky(Symmetric(Matrix(Ainv[perm, perm]))).L))
    @test_throws r"Ainv and L_Ainv disagree" validate_pedigree_factor(model, ws)
    @test_throws r"Ainv and L_Ainv disagree" run_aireml(model; max_iter=1, k=1)
    model.L_Ainv = LowerTriangular(sparse(1.0I, N-1, N-1))
    @test_throws DimensionMismatch validate_pedigree_factor(model, ws)
    model.L_Ainv = LowerTriangular(spzeros(N, N))
    @test_throws r"positive diagonal" validate_pedigree_factor(model, ws)
    model.L_Ainv = L

    for block in (false, true)
        precond = block ? get_preconditioner(model, ws) : get_diag_preconditioner(model)
        fill!(model.sol_y, 0.0)
        @test_logs (:warn, r"MME CG did not converge") solve_mme_y!(
            model, ws, precond; maxiter=1, reltol=1e-14, verbose=false)
        fill!(model.sol_y, 0.0)
        @test_logs solve_mme_y!(
            model, ws, precond; maxiter=200, reltol=1e-12, verbose=false)
        # An already exact warm start can require zero CG iterations.
        exact = zeros(model.layout.dim)
        @test_logs solve_mme_direct!(
            exact, model, zeros(5), ws, precond; maxiter=200, reltol=1e-12)
        @test all(iszero, exact)
    end
end
