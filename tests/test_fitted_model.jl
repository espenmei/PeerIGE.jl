if !@isdefined(score_ai_dense_fixture)
    include("test_score_ai.jl")
end
using Test

@testset "fixed damping retries the requested starting step" begin
    f = score_ai_dense_fixture()
    make() = SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups, 1.2, 0.2, 0.12, 0.9)
    settings = (k=20, seed=3, damping=1.0, adaptive_damping=false,
        cg_reltol=1e-10, cg_reltol_early=1e-10, ai_reltol=1e-10,
        cg_maxiter=2000, ai_maxiter=2000, verbose_cg=false)
    reference = make()
    fit_aireml!(reference; settings..., max_iter=1)
    first_alpha = reference.fit.alpha
    @test 0 < first_alpha < settings.damping

    # A fresh one-step fit starts from the requested damping, not the last safe step.
    fit_aireml!(reference; settings..., max_iter=1, refit=true)
    @test reference.fit.alpha > first_alpha
    model = make()
    fit_aireml!(model; settings..., max_iter=2)
    @test model.fit.alpha == reference.fit.alpha
    @test get_params(model) ≈ get_params(reference) rtol=1e-7 atol=1e-10
end

@testset "ψ is visible in every iteration and CSV log" begin
    f = score_ai_dense_fixture()
    for estimate_ψ in (false, true)
        model = SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups,
            1.2, 0.2, 0.12, 0.9; ψ=0.5)
        mktempdir() do dir
            output_path = joinpath(dir, "output.txt")
            log_path = joinpath(dir, "iterations.csv")
            open(output_path, "w") do io
                redirect_stdout(io) do
                    fit_aireml!(model; max_iter=3, k=8, damping=0.1,
                        adaptive_damping=false, verbose_cg=false,
                        estimate_ψ=estimate_ψ, log_file=log_path)
                end
            end
            output = read(output_path, String)
            matches = collect(eachmatch(
                r"ITERASJON\s+(\d+).*?\|\s+ψ=([0-9.eE+-]+)\s+\((fixed|estimated)\)", output))
            log = DataFrame(CSV.File(log_path))
            @test length(matches) == model.fit.n_iter == nrow(log)
            @test :psi in propertynames(log)
            @test (:se_psi in propertynames(log)) == estimate_ψ
            @test (:psi_score_mcse in propertynames(log)) == estimate_ψ
            for (i, m) in enumerate(matches)
                @test parse(Int, m.captures[1]) == i
                @test m.captures[3] == (estimate_ψ ? "estimated" : "fixed")
                @test parse(Float64, m.captures[2]) ≈ log.psi[i] atol=5.1e-7
            end
            @test log.psi[end] ≈ model.ψ atol=5.1e-7
            if estimate_ψ
                @test any(abs.(log.psi .- 0.5) .> 1e-6)
            else
                @test all(log.psi .== 0.5)
            end
            @test occursin("$(model.ψ) ($(estimate_ψ ? "estimated" : "fixed"))",
                sprint(show, MIME"text/plain"(), model))
            result = snapshot(model; include_fixef_se=false)
            @test occursin("$(model.ψ) ($(estimate_ψ ? "estimated" : "fixed"))",
                sprint(show, MIME"text/plain"(), result))
        end
    end
end

@testset "household boundary allows a real fit to progress" begin
    f = score_ai_dense_fixture()
    y = randn(MersenneTwister(3), length(f.y))
    model = SGEModel(y, f.X, f.Ainv, f.L_Ainv, f.groups,
        0.77, 0.23, 0.089, 0.15; env_ids=f.env_ids,
        σ2_env=[0.011, 1e-6], env_names=["group", "household"], ψ=1.0)
    mktempdir() do dir
        output_path = joinpath(dir, "output.txt")
        log_path = joinpath(dir, "iterations.csv")
        open(output_path, "w") do io
            redirect_stdout(io) do
                fit_aireml!(model; max_iter=3, k=32, damping=0.2,
                    adaptive_damping=false, cg_reltol=1e-10, cg_reltol_early=1e-10,
                    ai_reltol=1e-10, cg_maxiter=2000, ai_maxiter=2000,
                    verbose_cg=false, log_file=log_path)
            end
        end
        output = read(output_path, String)
        log = DataFrame(CSV.File(log_path))
        @test !model.peer_env
        @test occursin("Fixed at lower bound this iteration: σ²_env2", output)
        @test occursin("Held near lower bound: σ²_env2", output)
        @test log.alpha[1] > 1e-3
        @test log.s2env2[1] == 1e-6
        @test abs(log.s2d[1] - 0.77) > 0.1
        @test all(log.s2d .* log.s2s .> log.sds .^ 2)
        @test all(log.s2env2 .>= 1e-6)
    end
end

@testset "fitted model and independent inference snapshots" begin
    f = score_ai_dense_fixture()
    make(H=0, peer=false, psi=0.0) = SGEModel(f.y, f.X, f.Ainv, f.L_Ainv,
        f.groups, 1.2, 0.2, 0.12, 0.9;
        env_ids=deepcopy(f.env_ids[1:H]), σ2_env=[0.4, 0.25][1:H],
        peer_env=peer, σ2_pe=0.15, ψ=psi,
        fixef_names=["(Intercept)", "x"], env_names=["factor$h" for h in 1:H])
    settings = (max_iter=2, k=5, cg_reltol=1e-10, cg_reltol_early=1e-10,
        ai_reltol=1e-10, cg_maxiter=2000, ai_maxiter=2000, verbose_cg=false)

    fresh = make()
    @test fresh.fit === nothing
    @test !hasproperty(fresh, :sol_z)
    fresh_display = sprint(show, MIME"text/plain"(), fresh)
    @test occursin("not fitted", fresh_display)
    @test occursin("$(length(f.y)) observations", fresh_display)
    @test occursin("2 fixed effects", fresh_display)
    @test_throws ArgumentError snapshot(fresh)
    @test_throws ArgumentError fixef(fresh)
    @test_throws ArgumentError ranef(fresh)
    @test_throws ArgumentError save_model(fresh, "unused.jld2")
    @test_throws ArgumentError fit_aireml!(fresh; max_iter=0)
    @test_throws ArgumentError fit_aireml!(fresh; k=0)
    @test_throws ArgumentError fit_aireml!(fresh; cg_reltol=NaN)
    @test fresh.fit === nothing
    @test_throws DimensionMismatch SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups,
        1.2, 0.2, 0.12, 0.9; fixef_names=["x"])
    @test_throws DimensionMismatch SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups,
        1.2, 0.2, 0.12, 0.9; env_names=["household"])

    obs = DataFrame(y=[1.0, 2.0], x=[0.1, 0.2], id_ped=[1, 2])
    @test load_observed_data(obs).fixef_names == ["(Intercept)", "x"]
    @test load_observed_data(obs; add_intercept=false).fixef_names == ["x"]
    @test load_observed_data(obs; X_df=DataFrame(age=[0.1, 0.2])).fixef_names == ["age"]

    for (H, peer, psi) in ((0, false, 0.0), (1, false, 0.5), (2, true, 1.0))
        for block in (false, true)
            @testset "H=$H peer=$peer psi=$psi block=$block" begin
                model, legacy = make(H, peer, psi), make(H, peer, psi)
                initial = get_params(model)
                @test fit_aireml!(model; settings..., use_block_precond=block) === model
                result = run_aireml(legacy; settings..., use_block_precond=block)
                @test get_params(model) == get_params(legacy)
                @test model.sol_y == legacy.sol_y
                @test model.fit.AI == result.AI
                @test model.fit.se == result.se
                @test model.fit.initial_params == initial
                @test model.fit.n_iter == 2
                @test model.fit.converged == (isfinite(model.fit.step_crit) && model.fit.step_crit < 1e-2)
                @test model.fit.beta_vcov === nothing
                @test result.beta_vcov === nothing
                @test result.beta_se === nothing
                @test model.fit.settings.use_block_precond == block
                @test parent(fixef(model)) === model.sol_y
                @test parent(ranef(model).a_d) === model.sol_y
                @test parent(ranef(model).a_s) === model.sol_y

                sol_before = copy(model.sol_y)
                params_before = get_params(model)
                model_display = sprint(show, MIME"text/plain"(), model)
                @test occursin("AI-REML fit", model_display)
                @test occursin("Iteration diagnostics", model_display)
                @test occursin("max|Δ_AI|/SE=", model_display)
                @test occursin("Sampling correlations", model_display)
                @test occursin("Settings: k=5", model_display)
                @test occursin("Iteration diagnostics", sprint(show, MIME"text/plain"(), snapshot(model; include_fixef_se=false)))
                @test occursin("Variance components", model_display)
                @test occursin("σ²_d", model_display)
                @test occursin("(Intercept)", model_display)
                @test occursin("Fixed-effect SEs have not been computed", model_display)
                @test model.fit.beta_vcov === nothing
                @test model.sol_y == sol_before
                @test get_params(model) == params_before
                @test fixed_effect_vcov!(model) === model
                covariance_before = model.fit.beta_vcov
                @test covariance_before isa Matrix{Float64}
                snap = snapshot(model)
                result_display = sprint(show, MIME"text/plain"(), snap)
                @test occursin("AiRemlResult", sprint(show, snap))
                @test occursin("Fixed effects", result_display)
                @test occursin("(Intercept)", result_display)
                @test !occursin("Fixed-effect SEs have not been computed", result_display)
                @test snap.fixef_names == ["(Intercept)", "x"]
                @test snap.env_names == model.env_names
                @test snap.param_names == get_param_names(model)
                @test snap.env_ids == model.env_ids
                @test snap.beta == fixef(model)
                @test snap.a_d == ranef(model).a_d
                @test snap.c_env == ranef(model).c_env
                @test snap.pe_ids == model.pe_ids
                @test snap.p_env == ranef(model).p_env
                @test snap.vcov ≈ inv(snap.AI + 1e-6I)
                @test snap.beta_se ≈ sqrt.(diag(snap.beta_vcov))
                @test fixed_effect_vcov!(model) === model
                @test model.fit.beta_vcov === covariance_before
                @test model.sol_y == sol_before
                @test get_params(model) == params_before

                # Test-only marginal reference, never used by the production solver.
                Zs = f.Zs .* [model.group_scale[g] for g in model.groups.obs_to_group]
                Zg = hcat(f.Zd, Zs)
                G = [model.σ2_d model.σ_ds; model.σ_ds model.σ2_s]
                V = Zg * kron(G, f.A) * Zg' + model.σ2_e * I
                for h in 1:H
                    V += model.σ2_env[h] .* (f.Zh[h] * f.Zh[h]')
                end
                peer && (V += model.σ2_pe .* (Zs * Zs'))
                @test snap.beta_vcov ≈ inv(f.X' * (V \ f.X)) rtol=1e-7 atol=1e-9
                @test snap.beta ≈ (f.X' * (V \ f.X)) \ (f.X' * (V \ f.y)) rtol=1e-7 atol=1e-9

                @test_throws ArgumentError fit_aireml!(model; settings...)
                @test model.sol_y == sol_before
                if !block
                    mktempdir() do dir
                        model_path = joinpath(dir, "model.jld2")
                        result_path = joinpath(dir, "result.jld2")
                        save_model(model, model_path)
                        save_results(snap, result_path)
                        restored, restored_snap = load_model(model_path), load_results(result_path)
                        @test restored isa SGEModel
                        @test restored_snap isa AiRemlResult
                        @test get_params(restored) == get_params(model)
                        @test restored.sol_y == model.sol_y
                        @test restored.X == model.X
                        @test restored.Ainv == model.Ainv
                        @test restored.fixef_names == model.fixef_names
                        @test restored.fit.AI == model.fit.AI
                        @test restored.fit.beta_vcov == model.fit.beta_vcov
                        @test restored.env_ids == model.env_ids
                        @test restored.pe_ids == model.pe_ids
                        @test restored.ψ == model.ψ
                        @test restored_snap.beta == snap.beta
                        @test restored_snap.beta_se == snap.beta_se
                        @test restored_snap.env_ids == snap.env_ids
                        @test restored_snap.pe_ids == snap.pe_ids
                        @test fit_aireml!(restored; settings..., refit=true) === restored
                    end
                end
                if H == 0 && !block
                    failed = deepcopy(model)
                    failed.L_Ainv = LowerTriangular(2.0 .* parent(failed.L_Ainv))
                    @test_throws r"Ainv and L_Ainv disagree" fit_aireml!(failed; settings..., refit=true)
                    @test failed.fit === nothing
                    failed = deepcopy(model)
                    failed.fit.beta_vcov = nothing
                    failed.fit.settings = AiRemlSettings(cg_maxiter=1, cg_reltol=1e-14)
                    @test_throws r"Fixed-effect covariance CG did not converge" snapshot(failed)
                    @test failed.fit.beta_vcov === nothing
                    @test failed.sol_y == sol_before
                end

                # Snapshots must not alias any mutable fitted state or metadata.
                model.sol_y .+= 1
                model.fit.AI .+= 1
                model.fit.vcov .+= 1
                model.fit.se .+= 1
                model.fit.score .+= 1
                model.fit.initial_params .+= 1
                model.fit.ai_params .+= 1
                model.fit.beta_vcov .+= 1
                model.fixef_names[1] = "changed"
                model.σ2_env .+= 1
                H > 0 && (model.env_names[1] = "changed"; model.env_ids[1][1] += 1)
                @test snap.beta == sol_before[model.layout.fixed]
                @test snap.AI == result.AI
                @test snap.vcov == result.vcov
                @test snap.se == result.se
                @test snap.score == result.score
                @test snap.initial_params == initial
                @test snap.ai_params == result.ai_params
                @test snap.fixef_names[1] == "(Intercept)"
                @test snap.σ2_env == result.σ2_env
                @test snap.env_names == result.env_names
                @test snap.env_ids == result.env_ids
                @test snap.beta_vcov != model.fit.beta_vcov
            end
        end
    end
end
