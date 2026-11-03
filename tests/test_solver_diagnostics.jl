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

@testset "active-set AI step at variance floors" begin
    rng = MersenneTwister(5)
    N = 6
    groups = GroupStructure([1, 1, 2, 2, 2], [1, 2, 4, 5, 6], N;
        all_group_ids=[1, 1, 1, 2, 2, 2], all_animal_ids=collect(1:N))
    Ainv = sparse(1.0I, N, N)
    L = LowerTriangular(sparse(1.0I, N, N))
    model = SGEModel(randn(rng, 5), ones(5, 1), Ainv, L, groups, 1.0, 0.2, 0.05, 1.0;
        env_ids=[[1, 1, 2, 2, 2]], σ2_env=[0.3], peer_env=true, σ2_pe=1e-6)
    B = randn(rng, 6, 6)
    AI = B * B' + 6I
    pe = pe_param_index(model)
    # Choose a score whose unconstrained step pushes σ²_pe below zero.
    full = zeros(6); full[pe] = -1.0; full[1] = 0.5
    score = AI * full

    delta, active = _boundary_constrained_delta(model, AI, score)
    @test active == [pe]
    @test delta[pe] == 0
    free = setdiff(1:6, pe)
    @test delta[free] ≈ AI[free, free] \ score[free]

    # Interior σ²_pe is never fixed, regardless of step direction.
    model.σ2_pe = 0.1
    delta, active = _boundary_constrained_delta(model, AI, score)
    @test isempty(active)
    @test delta ≈ AI \ score

    # An inward step from the floor releases the component.
    model.σ2_pe = 1e-6
    delta, active = _boundary_constrained_delta(model, AI, AI * (-full))
    @test isempty(active)

    # Positive σ²_s does not force σ_ds to zero; step-halving still protects G.
    model.σ2_s = 1e-6; model.σ_ds = 0.0
    full_s = zeros(6); full_s[2] = -1.0; full_s[3] = 0.4
    delta, active = _boundary_constrained_delta(model, AI, AI * full_s)
    @test 2 in active
    @test delta[2] == 0
    free_s = setdiff(1:6, active)
    @test delta[free_s] ≈ AI[free_s, free_s] \ (AI * full_s)[free_s]

    # Two outward full steps can still leave an inward reduced gradient. Pinning
    # both without releasing a constraint would falsely report convergence.
    model.σ2_s = 0.2
    model.σ2_env[1] = 1e-6
    coupled_AI = Matrix{Float64}(I, 6, 6)
    coupled_AI[5:6, 5:6] .= [2.0 -1.0; -1.0 2.0]
    coupled_score = [0.0, 0.0, 0.0, 0.0, 1.0, -5.0]
    delta, active = _boundary_constrained_delta(model, coupled_AI, coupled_score)
    @test active == [pe]
    @test delta[5] ≈ 0.5
    @test delta[pe] == 0.0
    @test (coupled_score - coupled_AI * delta)[pe] <= 0.0

    model.σ2_s = 1e-6
    for _ in 1:40
        B = randn(rng, 6, 6)
        H = B * B' + I
        s = randn(rng, 6)
        d, held = _boundary_constrained_delta(model, H, s)
        free = setdiff(1:6, held)
        gradient = s - H * d
        @test all(d[[2, 5, 6]] .>= 0)
        @test all(iszero, d[held])
        @test all(gradient[held] .<= 1e-10)
        @test norm(gradient[free], Inf) < 1e-10
    end
end

@testset "household floor does not freeze a model without peer environment" begin
    rng = MersenneTwister(9)
    N = 6
    groups = GroupStructure([1, 1, 2, 2, 2], [1, 2, 4, 5, 6], N;
        all_group_ids=[1, 1, 1, 2, 2, 2], all_animal_ids=collect(1:N))
    Ainv = sparse(1.0I, N, N)
    L = LowerTriangular(sparse(1.0I, N, N))
    model = SGEModel(randn(rng, 5), ones(5, 1), Ainv, L, groups,
        0.770086, 0.231012, 0.089220, 0.150866;
        env_ids=[[1, 1, 2, 2, 2], [1, 2, 1, 2, 3]],
        σ2_env=[0.011380, 1e-6], env_names=["group", "household"], ψ=1.0)
    AI = Matrix{Float64}(I, 6, 6)
    full = [0.137234, 0.616960, 0.199689, -0.093589, -0.025903, -0.037238]
    score = AI * full
    delta, active = _boundary_constrained_delta(model, AI, score)
    @test !model.peer_env
    @test active == [6]
    @test delta[6] == 0.0
    @test delta[1:5] ≈ full[1:5]
    before = get_params(model)
    update_params!(model, delta, 0.2)
    @test model.σ2_env[2] == 1e-6
    @test abs(model.σ2_d - before[1]) > 0.02
    @test model.σ2_d * model.σ2_s > model.σ_ds^2

    # The report must use the same reduced step as the optimizer, not label the
    # outward household step as a remaining convergence requirement.
    fit_aireml!(model; max_iter=1, k=4, damping=0.2,
        adaptive_damping=false, verbose_cg=false)
    @test isfinite(model.fit.step_crit)
    report = sprint(show, MIME"text/plain"(), model)
    @test occursin("last α=", report)
    @test occursin("e-", split(report, "last α=")[2])
end
