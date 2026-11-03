if !@isdefined(post_fit_dense_fixture)
    include("test_post_fit.jl")
end
include(joinpath(@__DIR__, "..", "src", "emai_post_compare.jl"))
if Base.find_package("Plots") !== nothing
    import Plots
end

function comparison_result(res; kwargs...)
    changes = (; kwargs...)
    return AiRemlResult((hasproperty(changes, name) ? getproperty(changes, name) :
        getfield(res, name) for name in fieldnames(AiRemlResult))...)
end

@testset "interactive setup in a fresh Julia session" begin
    if Base.find_package("Plots") !== nothing
        script = abspath(joinpath(@__DIR__, "..", "post_fit.jl"))
        project = abspath(joinpath(@__DIR__, ".."))
        check_setup = raw"""
            using Test
            script = ARGS[1]
            setup = only(eachmatch(r"(?ms)^## Setup\r?\n(.*?)(?=^## |\z)",
                read(script, String))).captures[1]
            Base.include_string(Main, setup, script)
            @test all(name -> isdefined(Main, name),
                (:SGEModel, :compare_fits, :comparison_plot, :save_comparison, :Plots))
            @test !isdefined(Main, :report)
            Base.include_string(Main, setup, script)
            @test length(runs) == 3
            """
        command = `$(Base.julia_cmd()) --project=$project --startup-file=no -e $check_setup $script`
        @test success(command)
    else
        @test_skip "Plots.jl is an optional global-environment dependency"
    end
end

@testset "three-fit post-fit comparison" begin
    for peer_env in (false, true)
        mktempdir() do dir
            fit_dir = mkpath(joinpath(dir, "fits"))
            f = post_fit_dense_fixture()
            original_ids = reduce(vcat, f.groups.members_id)
            original_groups = reduce(vcat, [fill(g, length(ids)) for (g, ids) in enumerate(f.groups.members_id)])
            # Unobserved singleton groups must not change the centering median.
            groups = GroupStructure(f.groups.obs_to_group, f.groups.obs_to_id, size(f.Ainv, 1);
                all_group_ids=vcat(original_groups, collect(6:9)),
                all_animal_ids=vcat(original_ids, collect(21:24)))
            f = merge(f, (; groups))
            household = [(i - 1) ÷ 2 + 1 for i in eachindex(f.y)]
            Zh = Float64.(household .== permutedims(collect(1:maximum(household))))
            f = merge(f, (env_ids=[f.env_ids[1], household], Zh=[f.Zh[1], Zh]))
            obs = DataFrame(y=f.y, id_ped=f.groups.obs_to_id, group=f.groups.obs_to_group)
            roster_ids = reduce(vcat, f.groups.members_id)
            roster_groups = reduce(vcat, [fill(g, length(ids)) for (g, ids) in enumerate(f.groups.members_id)])
            CSV.write(joinpath(dir, "yX_obs.csv"), obs)
            CSV.write(joinpath(dir, "roster.csv"), DataFrame(id_ped=roster_ids, group=roster_groups))
            configs = ["Mean" => "mean", "Sum" => "sum", "Estimated" => "estimated"]
            models = SGEModel[]
            for ((_, name), ψ) in zip(configs, (1.0, 0.0, 0.4))
                model = SGEModel(f.y, copy(f.X), f.Ainv, f.L_Ainv, f.groups,
                    1.2, 0.2, 0.08, 0.9; env_ids=deepcopy(f.env_ids), σ2_env=[0.4, 0.25],
                    env_names=["class", "household"], fixef_names=["(Intercept)", "x"],
                    ψ, peer_env, σ2_pe=0.15)
                open(joinpath(dir, "fit_output.txt"), "w") do io
                    redirect_stdout(io) do
                        fit_aireml!(model; max_iter=2, k=8, damping=1e-4, adaptive_damping=false,
                            estimate_ψ=name == "estimated", verbose_cg=false,
                            cg_reltol=1e-10, cg_reltol_early=1e-10, ai_reltol=1e-10,
                            log_file=joinpath(fit_dir, "$name.csv"))
                    end
                end
                push!(models, model)
                save_model(model, joinpath(fit_dir, "$name.jld2"))
            end
            fixed_effect_vcov!(models[1])
            save_model(models[1], joinpath(fit_dir, "mean.jld2"))
            old_log = DataFrame(CSV.File(joinpath(fit_dir, "mean.csv")))
            select!(old_log, Not(:psi))
            CSV.write(joinpath(fit_dir, "mean.csv"), old_log)
            report = compare_fits(dir, configs; plot_sample=7)
            @test report.summary.fit == ["Mean", "Sum", "Estimated"]
            @test report.summary.psi_estimated == [false, false, true]
            @test report.summary.psi[1:2] == [1.0, 0.0]
            @test all(ismissing, report.summary.psi_se[1:2])
            @test report.summary.psi_se[3] == models[3].fit.se[end]
            @test report.reference_m == report.center_m == 4
            @test report.group_sizes == [2, 3, 4, 5, 6, 1, 1, 1, 1]
            @test report.fits[1].result.beta_se !== nothing
            @test report.fits[2].result.beta_se === nothing
            @test length(report.sample) == 7
            @test nrow(report.blup_comparison) == 9
            @test all(report.blup_comparison.n .== length(f.y))
            @test all(x -> ismissing(x) || -1 <= x <= 1, report.blup_comparison.correlation)
            @test all(x -> ismissing(x) || 0 <= x <= 1, report.blup_comparison.top10_overlap)
            @test nrow(report.curves) == 18
            @test "σ²_household" in report.common_parameters.effect
            @test "c(m₀)² × σ²_s" in report.common_parameters.effect
            @test ("σ²_pe" in report.common_parameters.parameter) == peer_env
            @test _post_log_columns(models[1])["σ²_env2"] == "s2env2"
            for (i, fit) in enumerate(report.fits)
                res = fit.result
                c = dil(report.reference_m, res.ψ, res, report.center_m)
                row = only(eachrow(report.common_parameters[
                    (report.common_parameters.fit .== fit.label) .&
                    (report.common_parameters.parameter .== "σ²_s"), :]))
                @test row.estimate ≈ c^2 * res.σ2_s
                @test report.effects[i].social_per_peer ≈ c .* res.a_s[res.obs_to_id]
                @test report.effects[i].tbv ≈ res.a_d[res.obs_to_id] +
                    (report.reference_m - 1) .* c .* res.a_s[res.obs_to_id]
            end

            @testset "equivalent raw and centered parameterizations" begin
                raw = report.fits[1].result
                p = length(raw.param_names)
                raw = comparison_result(raw; vcov=0.01Matrix{Float64}(I, p, p))
                factor = (report.center_m - 1)^raw.ψ
                scales = [name in ("σ²_s", "σ²_pe") ? 1 / factor^2 :
                    name == "σ_ds" ? 1 / factor : 1.0 for name in raw.param_names]
                J = vcat(Matrix(Diagonal(scales)), zeros(1, p))
                centered = comparison_result(raw;
                    σ2_s=raw.σ2_s / factor^2, σ_ds=raw.σ_ds / factor,
                    σ2_pe=raw.σ2_pe / factor^2, a_s=raw.a_s / factor, p_env=raw.p_env / factor,
                    param_names=vcat(raw.param_names, ["ψ"]), vcov=J * raw.vcov * J',
                    se=vcat(raw.se .* scales, 0.0))
                for m in (1, 2, 4, 6, 9)
                    a = derived_table(raw, m, report.center_m)
                    b = derived_table(centered, m, report.center_m)
                    @test a.quantity == b.quantity
                    @test a.estimate ≈ b.estimate rtol=1e-10
                    @test a.se ≈ b.se rtol=1e-7 atol=1e-9
                    c_raw = dil(m, raw.ψ, raw, report.center_m)
                    c_center = dil(m, centered.ψ, centered, report.center_m)
                    @test c_raw .* raw.a_s ≈ c_center .* centered.a_s
                    @test c_raw .* raw.p_env ≈ c_center .* centered.p_env
                end
                @test comparison_quantities(param_vector(raw), raw, 1, report.center_m).social_received == 0
                @test comparison_quantities(param_vector(raw), raw, 1, report.center_m).tbv == raw.σ2_d

                V = 0.01Matrix{Float64}(I, p + 1, p + 1)
                si = findfirst(==("σ²_s"), centered.param_names)
                V[si, end] = V[end, si] = 0.005
                θ = param_vector(centered)
                m = 7
                c = dil(m, centered.ψ, centered, report.center_m)
                gradient = zeros(p + 1)
                gradient[si] = c^2
                gradient[end] = -2log((m - 1) / (report.center_m - 1)) * c^2 * centered.σ2_s
                se = delta_se(t -> dil(m, ψ_of(t, centered), centered, report.center_m)^2 *
                    getp(t, centered, "σ²_s"), θ, V)
                @test se ≈ sqrt(dot(gradient, V * gradient)) rtol=1e-7
                @test !isapprox(se, abs(gradient[si]) * sqrt(V[si, si]); rtol=1e-3)
                @test_logs (:warn, r"Invalid delta-method variance") @test ismissing(
                    delta_se(t -> t[1], [1.0], fill(-1.0, 1, 1)))
            end

            @testset "input and inference guards" begin
                @test_throws ArgumentError compare_fits(dir, configs; plot_sample=0)
                @test_throws ArgumentError compare_fits(dir, [configs[1], configs[1]])
                @test_throws ArgumentError _post_full_model(report.fits[1])
                CSV.write(joinpath(dir, "yX_obs.csv"), reverse(obs))
                @test_throws ArgumentError compare_fits(dir, configs)
                CSV.write(joinpath(dir, "yX_obs.csv"), obs)
                models[2].X[1, 2] += 0.1
                save_model(models[2], joinpath(fit_dir, "sum.jld2"))
                @test_throws ArgumentError compare_fits(dir, configs)
                models[2].X .= f.X
                save_model(models[2], joinpath(fit_dir, "sum.jld2"))
                invalid_log = DataFrame(CSV.File(joinpath(fit_dir, "estimated.csv")))
                select!(invalid_log, Not(:psi))
                bad_path = joinpath(dir, "bad_log.csv")
                CSV.write(bad_path, invalid_log)
                @test_throws ArgumentError _post_log(bad_path, report.fits[3].result)
                stale_log = DataFrame(CSV.File(joinpath(fit_dir, "sum.csv")))
                stale_log.s2s[end] += 0.1
                CSV.write(bad_path, stale_log)
                @test_throws ArgumentError _post_log(bad_path, report.fits[2].result)
                bad_result = comparison_result(report.fits[2].result; env_names=["other", "household"])
                @test_throws ArgumentError _post_validate(bad_result, report.fits[1].result, obs)
                bad_result = comparison_result(report.fits[2].result; a_s=fill(NaN, length(f.groups.id_to_group)))
                @test_throws ArgumentError _post_validate(bad_result, report.fits[1].result, obs)
            end

            if !peer_env
                @testset "interactive cells" begin
                    if isdefined(@__MODULE__, :Plots)
                        script = joinpath(@__DIR__, "..", "post_fit.jl")
                        text = read(script, String)
                        cells = Dict(String(m.captures[1]) => String(m.captures[2]) for m in
                            eachmatch(r"(?ms)^## ([^\r\n]+)\r?\n(.*?)(?=^## |\z)", text))
                        run_cell(name) = Base.include_string(@__MODULE__, cells[name], script)
                        run_cell("Setup")
                        Core.eval(@__MODULE__, :(data_dir = $dir))
                        Core.eval(@__MODULE__, :(runs = $configs))
                        run_cell("Load and check the fits")
                        interactive_report = Base.invokelatest(getfield, @__MODULE__, :report)
                        interactive_figures = Base.invokelatest(getfield, @__MODULE__, :figures)
                        @test interactive_report.summary.fit == ["Mean", "Sum", "Estimated"]
                        @test isempty(interactive_figures)
                        @test !isdir(report.output_dir)
                        run_cell("Full fit diagnostics")
                        run_cell("Inspect one fit")
                        run_cell("Raw parameters")
                        run_cell("Convergence and psi")
                        @test Set(keys(interactive_figures)) == Set(["convergence"])
                        run_cell("Group-size curves")
                        @test Set(keys(interactive_figures)) == Set(["convergence", "group_size_curves"])
                        @test all(p -> p isa Plots.Plot, values(interactive_figures))
                        @test !isdir(report.output_dir)
                        @test_throws ArgumentError comparison_plot(report, "unknown")
                        run_cell("Save tables and the figures created so far")
                        @test length(readdir(report.output_dir)) == 9
                        @test isfile(joinpath(report.output_dir, "convergence.png"))
                        @test isfile(joinpath(report.output_dir, "group_size_curves.png"))
                        @test !isfile(joinpath(report.output_dir, "blup_comparison.png"))
                        for (cell, name) in (
                            "Common-scale variance components" => "components_reference",
                            "Derived quantities" => "derived_reference",
                            "Fixed effects" => "fixed_effects",
                            "BLUP comparisons" => "blup_comparison",
                            "Sampling correlations" => "estimate_correlations",
                        )
                            run_cell(cell)
                            @test haskey(interactive_figures, name)
                        end
                        @test length(interactive_figures) == 7
                        run_cell("Save tables and the figures created so far")
                        @test length(readdir(report.output_dir)) == 14
                        run_cell("Optional: stochastic likelihood comparison")
                        run_cell("Optional: psi score tests")
                        @test !isfile(joinpath(report.output_dir, "likelihood.csv"))
                        @test !isfile(joinpath(report.output_dir, "psi_scores.csv"))
                    else
                        @test_skip "Plots.jl is an optional global-environment dependency"
                    end
                end
                @testset "CSV and rendered plot outputs" begin
                    if isdefined(@__MODULE__, :Plots)
                        comparison = run_post_fit(dir, configs; plot_sample=7)
                        @test length(comparison.figures) == 7
                        @test length(readdir(report.output_dir)) == 14
                        for name in keys(comparison.figures)
                            @test filesize(joinpath(report.output_dir, "$name.png")) > 1000
                        end
                        for name in (:summary, :raw_parameters, :common_parameters, :derived,
                            :fixed_effects, :curves, :blup_comparison)
                            saved = DataFrame(CSV.File(joinpath(report.output_dir, "$name.csv")))
                            expected = getproperty(comparison.report, name)
                            @test names(saved) == names(expected)
                            @test nrow(saved) == nrow(expected)
                        end
                    else
                        @test_skip "Plots.jl is an optional global-environment dependency"
                    end
                end
                @testset "optional likelihood plumbing against dense reference" begin
                    # Mark only these tiny fixtures as converged to exercise the opt-in path;
                    # this does not assert that a two-iteration fit has actually converged.
                    for (model, (_, name)) in zip(models, configs)
                        model.fit.converged = true
                        save_model(model, joinpath(fit_dir, "$name.jld2"))
                    end
                    converged_report = compare_fits(dir, configs; plot_sample=7)
                    dim = get_dim(models[1])
                    probes = sqrt(dim) .* Matrix{Float64}(I, dim, dim)
                    ll = compare_likelihood(converged_report; probes, lanczos_maxiter=2dim)
                    @test all(ll.paired)
                    @test all(ll.K .== dim)
                    @test all(ll.lanczos_limit .== 2dim)
                    @test all(ll.lanczos_limit_hits .== 0)
                    @test ll.delta_vs_first ≈ ll.loglik .- first(ll.loglik)
                    @test all(ll.reference_fit .== "Mean")
                    for (i, model) in enumerate(models)
                        raw_model = deepcopy(model)
                        i == 3 && _convert_psi_scale!(raw_model; to_centered=false)
                        case = (H=2, peer_env=false, psi=raw_model.ψ)
                        exact = post_fit_dense_loglik(f, case,
                            get_params(raw_model; include_ψ=false), raw_model.ψ)
                        @test ll.loglik[i] ≈ exact atol=1e-6
                    end
                    @test isfile(joinpath(report.output_dir, "likelihood.csv"))
                    scores = compare_psi_scores(converged_report;
                        probes=sqrt(length(f.y)) .* Matrix{Float64}(I, length(f.y), length(f.y)))
                    @test scores.fit == ["Mean", "Sum"]
                    @test isfile(joinpath(report.output_dir, "psi_scores.csv"))
                end
            end

            @testset "saved snapshots" begin
                for (model, (_, name)) in zip(models, configs)
                    save_results(snapshot(model; include_fixef_se=false), joinpath(fit_dir, "$name.jld2"))
                end
                snapshots = compare_fits(dir, configs; plot_sample=7)
                @test all(fit -> fit.signature === nothing, snapshots.fits)
                @test snapshots.sample == report.sample
                @test snapshots.common_parameters.estimate ≈ report.common_parameters.estimate
                @test_throws ArgumentError compare_likelihood(snapshots)
                @test_throws ArgumentError compare_psi_scores(snapshots)
            end
        end
    end
end
