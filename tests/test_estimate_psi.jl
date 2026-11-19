if !@isdefined(SGEModel)
    include(joinpath(@__DIR__, "..", "src", "emai.jl"))
end
using LinearAlgebra, SparseArrays, Random, Test

@testset "AI-REML estimates ψ on variable-size groups" begin
    rng = MersenneTwister(3381)
    group_sizes = repeat(collect(3:18), 6)
    n = sum(group_sizes)
    group_ids = reduce(vcat, [fill(g, m) for (g, m) in enumerate(group_sizes)])
    ids = collect(1:n)
    groups = GroupStructure(group_ids, ids, n;
        all_group_ids=group_ids, all_animal_ids=ids)
    X = ones(n, 1)
    Ainv = sparse(1.0I, n, n)
    L_Ainv = LowerTriangular(sparse(1.0I, n, n))

    truth = (σ2_d=0.5, σ2_s=0.6, σ_ds=0.05, σ2_e=0.5, ψ=0.4)
    ref_peers = median(group_sizes) - 1
    G = [truth.σ2_d truth.σ_ds; truth.σ_ds truth.σ2_s]
    effects = cholesky(Symmetric(G)).L * randn(rng, 2, n)
    y = Vector{Float64}(undef, n)
    row = 0
    for (g, m) in enumerate(group_sizes)
        c = ((m - 1) / ref_peers)^(-truth.ψ)
        members = row + 1:row + m
        social_sum = sum(@view effects[2, members])
        for id in members
            y[id] = 2.0 + effects[1, id] + c * (social_sum - effects[2, id]) +
                    randn(rng) * sqrt(truth.σ2_e)
        end
        row += m
    end

    model = SGEModel(y, X, Ainv, L_Ainv, groups, truth.σ2_d, truth.σ2_s,
        truth.σ_ds, truth.σ2_e;
        ψ=0.15)
    fit_aireml!(model; max_iter=100, k=48, damping=0.25, adaptive_damping=false,
        cg_maxiter=500, ai_maxiter=500, cg_reltol=1e-10, ai_reltol=1e-10,
        verbose_cg=false, estimate_ψ=true, ψ_bounds=(0.0, 1.0), ψ_max_step=0.2,
        seed=183)

    @printf("[psi-recovery] truth=%.3f estimate=%.3f params=%s converged=%s iterations=%d\n",
        truth.ψ, model.ψ, round.(get_params(model), digits=3), model.fit.converged, model.fit.n_iter)
    @test model.fit !== nothing
    @test "ψ" in get_param_names(model)
    @test length(model.fit.score) == 5
    @test 0.0 <= model.ψ <= 1.0
    @test abs(model.ψ - truth.ψ) < 0.25
    @test size(model.fit.AI) == (5, 5)
    @test model.σ2_d * model.σ2_s > model.σ_ds^2

    fixed = SGEModel(y, X, Ainv, L_Ainv, groups, 0.7, 0.15, 0.02, 1.0; ψ=truth.ψ)
    fit_aireml!(fixed; max_iter=1, k=4, damping=0.1, verbose_cg=false)
    @test !("ψ" in get_param_names(fixed))
    @test length(fixed.fit.score) == 4
end

@testset "ψ estimation requires variable peer-group sizes" begin
    n = 20
    ids = collect(1:n)
    group_ids = repeat([1, 2], inner=10)
    groups = GroupStructure(group_ids, ids, n;
        all_group_ids=group_ids, all_animal_ids=ids)
    Ainv = sparse(1.0I, n, n)
    L_Ainv = LowerTriangular(sparse(1.0I, n, n))
    model = SGEModel(randn(MersenneTwister(8), n), ones(n, 1), Ainv, L_Ainv,
        groups, 1.0, 0.2, 0.0, 1.0)
    @test_throws ArgumentError fit_aireml!(model; estimate_ψ=true, verbose_cg=false)
end
