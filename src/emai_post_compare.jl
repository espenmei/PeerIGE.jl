# Comparison helpers used by the interactive post_fit.jl script.
using SHA

## Common-scale estimates and uncertainty
function param_vector(res::AiRemlResult)
    vals = Dict("σ²_d" => res.σ2_d, "σ²_s" => res.σ2_s, "σ_ds" => res.σ_ds,
        "σ²_e" => res.σ2_e, "σ²_pe" => res.σ2_pe, "ψ" => res.ψ)
    for h in eachindex(res.σ2_env)
        vals["σ²_env$h"] = res.σ2_env[h]
    end
    return [vals[p] for p in res.param_names]
end

function getp(θ, res, name)
    i = findfirst(==(name), res.param_names)
    i !== nothing && return θ[i]
    name == "ψ" && return res.ψ
    name == "σ²_pe" && !res.peer_env && return 0.0
    throw(ArgumentError("Parameter $name is absent from the saved fit"))
end

ψ_of(θ, res) = getp(θ, res, "ψ")
function dil(m, ψ, res, center_m)
    m <= 1 && return 0.0
    ref_peers = "ψ" in res.param_names ? center_m - 1.0 : 1.0
    return ((m - 1.0) / ref_peers)^(-ψ)
end

function comparison_quantities(θ, res, m, center_m)
    d, s, ds, e = (getp(θ, res, p) for p in ("σ²_d", "σ²_s", "σ_ds", "σ²_e"))
    c = dil(m, ψ_of(θ, res), res, center_m)
    social_g = (m - 1) * c^2 * s
    social_pe = (m - 1) * c^2 * getp(θ, res, "σ²_pe")
    # Reference group of unrelated individuals; not the empirical variance of y.
    vp = d + e + sum(getp(θ, res, "σ²_env$h") for h in eachindex(res.σ2_env); init=0.0) +
         social_g + social_pe
    q = (m - 1) * c
    tbv = d + 2q * ds + q^2 * s
    return (; r_ds=ds / sqrt(d * s), social_genetic_received=social_g,
        social_peer_env_received=social_pe, social_received=social_g + social_pe,
        phenotype_reference=vp, h2_direct=d / vp, tbv, T2=tbv / vp)
end

function delta_se(f, θ, V; quantity="")
    g = map(eachindex(θ)) do i
        h = 1e-5 * max(abs(θ[i]), 1e-3)
        plus, minus = copy(θ), copy(θ)
        plus[i] += h
        minus[i] -= h
        (f(plus) - f(minus)) / (2h)
    end
    v = dot(g, V * g)
    tolerance = 100eps(Float64) * max(1.0, sum(abs.(g .* (V * g))))
    if !isfinite(v) || v < -tolerance
        @warn "Invalid delta-method variance; reporting missing SE" quantity variance=v
        return missing
    end
    return sqrt(max(v, 0.0))
end

function derived_table(res, m, center_m)
    θ = param_vector(res)
    qs = comparison_quantities(θ, res, m, center_m)
    return DataFrame(quantity=collect(String.(keys(qs))), estimate=collect(values(qs)),
        se=[delta_se(t -> getproperty(comparison_quantities(t, res, m, center_m), k),
            θ, res.vcov; quantity=String(k)) for k in keys(qs)])
end

_post_digest(x) = SHA.sha256(reinterpret(UInt8, vec(x)))
function _post_signature(model)
    A = model.Ainv
    return (y=_post_digest(model.y), X_size=size(model.X), X=_post_digest(model.X),
        A_size=size(A), A_cols=_post_digest(A.colptr), A_rows=_post_digest(A.rowval),
        A_values=_post_digest(A.nzval), roster=_post_digest(model.groups.id_to_group),
        env=map(_post_digest, model.env_ids), pe_ids=_post_digest(model.pe_ids),
        fixef_names=copy(model.fixef_names), env_names=copy(model.env_names))
end

function _post_load(path)
    return jldopen(path, "r") do file
        if haskey(file, "model")
            model = file["model"]
            model isa SGEModel || throw(ArgumentError(
                "$path contains an incompatible model type; use the matching source version"))
            centered = _fit_estimates_psi(model)
            center = centered ? _psi_reference_size(model) : 2.0
            expected = [length(ids) <= 1 ? 0.0 :
                ((length(ids) - 1.0) / (center - 1.0))^(-model.ψ)
                for ids in model.groups.members_id]
            isapprox(model.group_scale, expected; rtol=1e-10) ||
                throw(ArgumentError("$path has inconsistent dilution scaling"))
            # Keep cached fixed-effect SEs without triggering any additional MME solves.
            result = snapshot(model; include_fixef_se=fitted_state(model).beta_vcov !== nothing)
            return (; result, signature=_post_signature(model))
        end
        haskey(file, "result") || throw(ArgumentError("$path contains neither model nor result"))
        result = file["result"]
        result isa AiRemlResult || throw(ArgumentError("$path contains an incompatible snapshot type"))
        return (; result, signature=nothing)
    end
end

function _post_validate(res, reference, obs)
    res.n_obs == nrow(obs) && res.obs_to_id == obs.id_ped ||
        throw(ArgumentError("Saved observations do not match yX_obs.csv, including row order"))
    res.N_pedigree == reference.N_pedigree ||
        throw(ArgumentError("Fits use different pedigree dimensions"))
    res.fixef_names == reference.fixef_names && res.env_names == reference.env_names &&
        res.env_ids == reference.env_ids && res.peer_env == reference.peer_env &&
        res.pe_ids == reference.pe_ids ||
        throw(ArgumentError("Fits must have identical fixed/random effects and level ordering"))
    filter(!=("ψ"), res.param_names) == filter(!=("ψ"), reference.param_names) ||
        throw(ArgumentError("Fits have different variance components"))
    p = length(res.param_names)
    size(res.vcov) == size(res.AI) == (p, p) && length(res.se) == p &&
        length(res.ai_params) == length(res.score) == p ||
        throw(ArgumentError("Saved inference dimensions do not match param_names"))
    all(isfinite, param_vector(res)) && all(isfinite, res.vcov) ||
        throw(ArgumentError("Saved estimates or covariance contain non-finite values"))
    length(res.a_d) == length(res.a_s) == res.N_pedigree &&
        length(res.beta) == length(res.fixef_names) &&
        length(res.env_names) == length(res.σ2_env) ||
        throw(ArgumentError("Saved effect dimensions do not match the model metadata"))
    all(isfinite, res.a_d) && all(isfinite, res.a_s) && all(isfinite, res.beta) ||
        throw(ArgumentError("Saved genetic or fixed-effect predictions are non-finite"))
    isposdef(Symmetric([res.σ2_d res.σ_ds; res.σ_ds res.σ2_s])) &&
        res.σ2_e > 0 && all(>(0), res.σ2_env) && (!res.peer_env || res.σ2_pe > 0) ||
        throw(ArgumentError("Saved variances must be positive and G positive definite"))
end

function _post_log(path, res)
    if !isfile(path)
        @warn "Iteration log not found; omitting its convergence traces" path
        return nothing
    end
    df = DataFrame(CSV.File(path))
    required = ["iter", "s2d", "s2s", "sds", "s2e", "step_crit", "alpha"]
    append!(required, ["s2env$h" for h in eachindex(res.σ2_env)])
    res.peer_env && push!(required, "s2pe")
    "ψ" in res.param_names && push!(required, "psi")
    absent = setdiff(required, names(df))
    isempty(absent) || throw(ArgumentError("$path is missing columns: $(join(absent, ", "))"))
    nrow(df) == res.n_iter && df.iter == collect(1:res.n_iter) ||
        throw(ArgumentError("$path does not cover the saved fit's iterations"))
    θ = param_vector(res)
    for (p, column) in _post_log_columns(res)
        p in res.param_names || p == "ψ" || continue
        column in names(df) || continue  # Older fixed-ψ logs do not contain psi.
        all(x -> x isa Real && isfinite(x), df[!, column]) ||
            throw(ArgumentError("$path contains invalid values in $column"))
        isapprox(df[end, column], getp(θ, res, p); atol=5.1e-7, rtol=1e-10) ||
            throw(ArgumentError("$path final $column disagrees with the saved estimates"))
    end
    return df
end

function _post_effect_name(res, p)
    h = findfirst(h -> p == "σ²_env$h", eachindex(res.env_names))
    return h === nothing ? p : "σ²_$(res.env_names[h])"
end

function _post_log_columns(res)
    cols = Dict("σ²_d" => "s2d", "σ²_s" => "s2s", "σ_ds" => "sds",
        "σ²_e" => "s2e", "σ²_pe" => "s2pe", "ψ" => "psi")
    for h in eachindex(res.σ2_env)
        cols["σ²_env$h"] = "s2env$h"
    end
    return cols
end

function _post_blups(fits, ids, m, center_m)
    effects = map(fits) do fit
        res = fit.result
        c = dil(m, res.ψ, res, center_m)
        direct = res.a_d[ids]
        social = c .* res.a_s[ids]
        (; direct, social_per_peer=social, tbv=direct .+ (m - 1) .* social)
    end
    rows = NamedTuple[]
    top_n = max(1, ceil(Int, 0.1length(ids)))
    for key in keys(first(effects)), i in eachindex(fits), j in i+1:length(fits)
        a, b = getproperty(effects[i], key), getproperty(effects[j], key)
        varying = std(a) > 0 && std(b) > 0
        !varying && @warn "Constant BLUPs; correlation and top-tail overlap are undefined" quantity=key
        overlap = varying ? length(intersect(
            partialsortperm(a, 1:top_n; rev=true), partialsortperm(b, 1:top_n; rev=true))) / top_n : missing
        push!(rows, (fit_a=fits[i].label, fit_b=fits[j].label, quantity=String(key),
            n=length(ids), correlation=varying ? cor(a, b) : missing, top10_overlap=overlap,
            rmse=sqrt(mean(abs2, a .- b))))
    end
    return effects, DataFrame(rows)
end

## Load full models one at a time; retain only independent snapshots.
function compare_fits(data_dir, runs; plot_sample=20_000)
    length(runs) >= 2 || throw(ArgumentError("Configure at least two runs"))
    allunique(first.(runs)) && allunique(last.(runs)) ||
        throw(ArgumentError("Run labels and filenames must be unique"))
    plot_sample > 0 || throw(ArgumentError("plot_sample must be positive"))
    obs = DataFrame(CSV.File(joinpath(data_dir, "yX_obs.csv")))
    roster = DataFrame(CSV.File(joinpath(data_dir, "roster.csv")))
    all(c -> c in names(obs), ["y", "id_ped", "group"]) &&
        all(c -> c in names(roster), ["id_ped", "group"]) ||
        throw(ArgumentError("Expected y/id_ped/group in yX_obs.csv and id_ped/group in roster.csv"))
    allunique(roster.id_ped) && allunique(obs.id_ped) ||
        throw(ArgumentError("Expected one row per individual in observations and roster"))
    fits = NamedTuple[]
    groups = nothing
    signature = nothing
    y_digest = _post_digest(Float64.(obs.y))
    for (label, run) in runs
        path = joinpath(data_dir, "fits", "$run.jld2")
        loaded = _post_load(path)
        res = loaded.result
        _post_validate(res, isempty(fits) ? res : first(fits).result, obs)
        if groups === nothing
            groups = GroupStructure(Int.(obs.group), Int.(obs.id_ped), res.N_pedigree;
                all_group_ids=Int.(roster.group), all_animal_ids=Int.(roster.id_ped))
            unique(roster.group)[groups.obs_to_group] == obs.group ||
                throw(ArgumentError("Observed group labels disagree with roster.csv"))
        end
        if loaded.signature === nothing
            @warn "Snapshot lacks y, X and pedigree: only metadata can be checked; original roster is required" run
        else
            loaded.signature.y == y_digest &&
                loaded.signature.roster == _post_digest(groups.id_to_group) ||
                throw(ArgumentError("$run does not match the observed data / original roster"))
            signature !== nothing && loaded.signature != signature &&
                throw(ArgumentError("$run differs in y, X, pedigree or social-group indexing"))
            signature = loaded.signature
        end
        !res.converged && @warn "Fit has NOT converged; comparisons and Wald SEs are exploratory" run step_crit=res.step_crit
        minimum(eigvals(Symmetric(res.vcov))) < 0 &&
            @warn "Parameter covariance is not positive semidefinite; inspect identifiability" run
        push!(fits, (; label, run, path, result=res, signature=loaded.signature,
            log=_post_log(joinpath(data_dir, "fits", "$run.csv"), res)))
        loaded = nothing
        GC.gc()
    end
    group_sizes = length.(groups.members_id)
    non_single = filter(>(1), group_sizes)
    isempty(non_single) && throw(ArgumentError("No non-singleton roster groups"))
    center_m = median(non_single)
    reference_m = round(Int, center_m)
    summary_rows, raw_rows, common_rows, derived_rows, fixed_rows, curve_rows = (NamedTuple[] for _ in 1:6)
    for fit in fits
        res = fit.result
        θ = param_vector(res)
        ψi = findfirst(==("ψ"), res.param_names)
        push!(summary_rows, (fit=fit.label, run=fit.run, converged=res.converged,
            iterations=res.n_iter, n_obs=res.n_obs, N_pedigree=res.N_pedigree, psi=res.ψ,
            psi_estimated=ψi !== nothing, psi_se=ψi === nothing ? missing : res.se[ψi],
            step_crit=res.step_crit, alpha=res.alpha, score_norm=norm(res.score),
            min_eigen_G=eigmin(Symmetric([res.σ2_d res.σ_ds; res.σ_ds res.σ2_s])),
            min_eigen_AI=eigmin(Symmetric(res.AI)),
            max_last_update=maximum(abs.(θ .- res.ai_params)), reference_m, center_m))
        for (i, p) in enumerate(res.param_names)
            display_name = _post_effect_name(res, p)
            push!(raw_rows, (fit=fit.label, parameter=p, effect=display_name, estimate=θ[i], se=res.se[i]))
            transform = t -> begin
                c = dil(reference_m, ψ_of(t, res), res, center_m)
                factor = p in ("σ²_s", "σ²_pe") ? c^2 : p == "σ_ds" ? c : 1.0
                factor * t[i]
            end
            common_name = p in ("σ²_s", "σ²_pe") ? "c(m₀)² × $display_name" :
                p == "σ_ds" ? "c(m₀) × $display_name" : display_name
            push!(common_rows, (fit=fit.label, parameter=p, effect=common_name,
                estimate=transform(θ), se=delta_se(transform, θ, res.vcov; quantity=p), reference_m))
        end
        if ψi === nothing
            push!(raw_rows, (fit=fit.label, parameter="ψ", effect="ψ", estimate=res.ψ, se=missing))
            push!(common_rows, (fit=fit.label, parameter="ψ", effect="ψ",
                estimate=res.ψ, se=missing, reference_m))
        end
        for row in eachrow(derived_table(res, reference_m, center_m))
            push!(derived_rows, (fit=fit.label, quantity=row.quantity, estimate=row.estimate, se=row.se, reference_m))
        end
        for i in eachindex(res.beta)
            push!(fixed_rows, (fit=fit.label, coefficient=res.fixef_names[i],
                estimate=res.beta[i], se=res.beta_se === nothing ? missing : res.beta_se[i]))
        end
        for m in sort(unique(group_sizes))
            qs = comparison_quantities(θ, res, m, center_m)
            push!(curve_rows, merge((fit=fit.label, m, dilution=dil(m, res.ψ, res, center_m)), qs,
                (social_received_se=delta_se(t -> comparison_quantities(t, res, m, center_m).social_received,
                    θ, res.vcov; quantity="social_received"),
                 tbv_se=delta_se(t -> comparison_quantities(t, res, m, center_m).tbv,
                    θ, res.vcov; quantity="tbv"))))
        end
    end
    ids = first(fits).result.obs_to_id
    effects, blup_comparison = _post_blups(fits, ids, reference_m, center_m)
    sample = randperm(MersenneTwister(2026), length(ids))[1:min(plot_sample, length(ids))]
    return (; fits, group_sizes, reference_m, center_m, effects, sample,
        summary=DataFrame(summary_rows), raw_parameters=DataFrame(raw_rows),
        common_parameters=DataFrame(common_rows), derived=DataFrame(derived_rows),
        fixed_effects=DataFrame(fixed_rows), curves=DataFrame(curve_rows), blup_comparison,
        output_dir=joinpath(data_dir, "fits", "comparison"))
end

## Plots (load Plots explicitly before calling).
function comparison_plots(report; selected=("components_reference", "derived_reference",
    "fixed_effects", "group_size_curves", "convergence", "blup_comparison", "estimate_correlations"))
    isdefined(@__MODULE__, :Plots) || throw(ArgumentError("Run `import Plots` before plotting"))
    available = ("components_reference", "derived_reference", "fixed_effects",
        "group_size_curves", "convergence", "blup_comparison", "estimate_correlations")
    all(name -> name in available, selected) ||
        throw(ArgumentError("Unknown comparison plot: $(join(setdiff(collect(selected), available), ", "))"))
    labels = [fit.label for fit in report.fits]
    legends = ["$(f.label), ψ=$(round(f.result.ψ; digits=3))" *
        (f.result.converged ? "" : " [NOT CONVERGED]") for f in report.fits]
    figures = Dict{String,Plots.Plot}()
    numeric_se(v) = Float64.(coalesce.(v, NaN))
    function estimates_plot(df, name_column)
        quantities = unique(df[!, name_column])
        panels = map(quantities) do q
            rows = df[df[!, name_column] .== q, :]
            x = [findfirst(==(s), labels) for s in rows.fit]
            Plots.scatter(x, rows.estimate; yerror=1.96 .* numeric_se(rows.se),
                xticks=(1:length(labels), labels), xlims=(0.5, length(labels) + 0.5),
                title=q, legend=false, markersize=4, titlefontsize=10)
        end
        columns = min(3, length(panels))
        return Plots.plot(panels...; layout=(cld(length(panels), columns), columns),
            size=(1200, 270cld(length(panels), columns)), margin=5Plots.mm)
    end
    if "components_reference" in selected
        figures["components_reference"] = estimates_plot(report.common_parameters, :effect)
    end
    if "derived_reference" in selected
        figures["derived_reference"] = estimates_plot(report.derived, :quantity)
    end
    if "fixed_effects" in selected
        figures["fixed_effects"] = estimates_plot(report.fixed_effects, :coefficient)
    end
    if "group_size_curves" in selected
        curve_panels = [Plots.plot(; xlabel="Roster group size m", title=t)
            for t in ("Social variance received (unrelated peers)", "TBV variance",
                "T2 (reference phenotype variance)", "Roster group-size distribution")]
        for (i, label) in enumerate(labels)
            rows = report.curves[report.curves.fit .== label, :]
            Plots.plot!(curve_panels[1], rows.m, rows.social_received;
                ribbon=1.96 .* numeric_se(rows.social_received_se), label=legends[i])
            Plots.plot!(curve_panels[2], rows.m, rows.tbv;
                ribbon=1.96 .* numeric_se(rows.tbv_se), label=legends[i])
            Plots.plot!(curve_panels[3], rows.m, rows.T2; label=legends[i])
        end
        Plots.histogram!(curve_panels[4], report.group_sizes; bins=min(50, length(unique(report.group_sizes))),
            label=false, ylabel="Number of groups")
        figures["group_size_curves"] = Plots.plot(curve_panels...; layout=(2, 2),
            size=(1200, 850), margin=5Plots.mm)
    end
    if "convergence" in selected
        parameters = unique(report.common_parameters.parameter)
        titles = [_post_effect_name(first(report.fits).result, p) for p in parameters]
        titles .= [p in ("σ²_s", "σ_ds", "σ²_pe") ? "$t (reference m)" : t
            for (p, t) in zip(parameters, titles)]
        append!(titles, ["max|Δ|/SE", "Step length α", "Score norm"])
        trace_panels = [Plots.plot(; xlabel="Iteration", title=t, legend=:best) for t in titles]
        for (i, fit) in enumerate(report.fits)
            df = fit.log
            df === nothing && continue
            psis = "psi" in names(df) ? df.psi : fill(fit.result.ψ, nrow(df))
            scales = [dil(report.reference_m, p, fit.result, report.center_m) for p in psis]
            cols = _post_log_columns(fit.result)
            for (k, p) in enumerate(parameters)
                y = p == "ψ" ? psis : df[!, cols[p]]
                factor = p in ("σ²_s", "σ²_pe") ? scales.^2 : p == "σ_ds" ? scales : 1.0
                Plots.plot!(trace_panels[k], df.iter, factor .* y; label=legends[i])
            end
            for (offset, column) in enumerate(("step_crit", "alpha", "score_norm"))
                column in names(df) || continue
                y = df[!, column]
                positive = isfinite.(y) .& (y .> 0)
                !all(positive) && @warn "Omitting nonpositive/nonfinite values from logarithmic trace" run=fit.run column
                Plots.plot!(trace_panels[length(parameters) + offset], df.iter[positive], y[positive];
                    label=legends[i], yscale=:log10)
            end
        end
        Plots.hline!(trace_panels[length(parameters) + 1], [1e-2];
            label="Convergence threshold", linestyle=:dash, color=:black)
        figures["convergence"] = Plots.plot(trace_panels...; layout=(cld(length(trace_panels), 3), 3),
            size=(1400, 300cld(length(trace_panels), 3)), margin=5Plots.mm)
    end
    if "blup_comparison" in selected
        blup_panels = Plots.Plot[]
        for key in keys(first(report.effects)), i in eachindex(report.fits), j in i+1:length(report.fits)
            x = getproperty(report.effects[i], key)[report.sample]
            y = getproperty(report.effects[j], key)[report.sample]
            p = Plots.scatter(x, y; markersize=1.2, markerstrokewidth=0, alpha=0.25,
                xlabel=labels[i], ylabel=labels[j], title=String(key), legend=false)
            lo, hi = extrema(vcat(x, y))
            Plots.plot!(p, [lo, hi], [lo, hi]; linestyle=:dash, color=:black)
            push!(blup_panels, p)
        end
        figures["blup_comparison"] = Plots.plot(blup_panels...;
            layout=(3, binomial(length(labels), 2)), size=(1400, 1100), margin=5Plots.mm)
    end
    if "estimate_correlations" in selected
        correlation_panels = map(report.fits) do fit
            res = fit.result
            variances = diag(res.vcov)
            if any(<=(0), variances)
                @warn "Cannot plot estimate correlations with nonpositive covariance diagonal" run=fit.run
                return Plots.plot(; title="$(fit.label): invalid covariance", legend=false)
            end
            D = Diagonal(1 ./ sqrt.(variances))
            Plots.heatmap(D * res.vcov * D; clims=(-1, 1), color=:RdBu,
                xticks=(eachindex(variances), res.param_names),
                yticks=(eachindex(variances), res.param_names), xrotation=60,
                title=fit.label, aspect_ratio=:equal)
        end
        figures["estimate_correlations"] = Plots.plot(correlation_panels...;
            layout=(1, length(labels)), size=(500length(labels), 500), margin=5Plots.mm)
    end
    header = "Reference group size m₀ = $(report.reference_m)"
    if any(f -> !f.result.converged, report.fits)
        header = "EXPLORATORY: fit(s) NOT converged | " * header
    end
    for (name, figure) in figures
        interval = name in ("components_reference", "derived_reference", "fixed_effects") ?
            " | approximate 95% Wald intervals" : ""
        Plots.plot!(figure; plot_title=header * interval, plot_titlefontsize=13)
    end
    return figures
end

comparison_plot(report, name::AbstractString) = comparison_plots(report; selected=(name,))[name]

function save_comparison(report, figures)
    mkpath(report.output_dir)
    for key in (:summary, :raw_parameters, :common_parameters, :derived, :fixed_effects, :curves, :blup_comparison)
        CSV.write(joinpath(report.output_dir, "$key.csv"), getproperty(report, key))
    end
    for (name, figure) in figures
        Plots.savefig(figure, joinpath(report.output_dir, "$name.png"))
    end
    println("Saved seven tables and $(length(figures)) figures to $(report.output_dir).")
    println("Other figures from earlier saves are left unchanged.")
    return report.output_dir
end

function run_post_fit(data_dir, runs; plot_sample=20_000)
    isdefined(@__MODULE__, :Plots) || throw(ArgumentError("Run `import Plots` before run_post_fit"))
    report = compare_fits(data_dir, runs; plot_sample)
    figures = comparison_plots(report)
    save_comparison(report, figures)
    display(select(report.summary, :fit, :psi, :psi_estimated, :psi_se, :converged, :iterations, :step_crit))
    display(report.derived)
    display(report.blup_comparison)
    println("Comparison saved to $(report.output_dir); reference group size m=$(report.reference_m).")
    println("Intervals are approximate 95% Wald intervals. Inspect convergence and boundary estimates.")
    return (; report, figures)
end

## Optional, EXPENSIVE: explicitly call in a suitably sized job, never automatically.
function _post_require_full(fit)
    fit.signature === nothing && throw(ArgumentError("$(fit.run): likelihood/score requires a saved full model"))
    fit.result.converged || throw(ArgumentError("$(fit.run): refit to convergence before likelihood/score comparison"))
    return nothing
end

function _post_full_model(fit)
    _post_require_full(fit)
    model = load_model(fit.path)
    fitted_state(model).converged && _post_signature(model) == fit.signature &&
        get_params(model) == param_vector(fit.result) ||
        throw(ArgumentError("$(fit.path) has changed since the comparison was loaded"))
    return model
end

_post_evaluate(fit, evaluate; kwargs...) = evaluate(_post_full_model(fit); kwargs...)

function compare_likelihood(report; K=64, seed=2026,
    lanczos_maxiter=2000, lanczos_tol=1e-8, kwargs...)
    foreach(_post_require_full, report.fits)
    values = map(report.fits) do fit
        value = _post_evaluate(fit, stochastic_loglik; K, seed, lanczos_maxiter, lanczos_tol, kwargs...)
        GC.gc()
        value
    end
    table = DataFrame(fit=[f.label for f in report.fits], psi=[f.result.ψ for f in report.fits],
        loglik=[v.loglik for v in values], mcse=[v.se for v in values],
        K=[v.K for v in values], lanczos_max_steps=[maximum(v.lanczos_steps) for v in values],
        lanczos_limit=fill(lanczos_maxiter, length(values)), lanczos_tol=fill(lanczos_tol, length(values)),
        lanczos_limit_hits=[count(>=(lanczos_maxiter), v.lanczos_steps) for v in values])
    differences = [loglik_difference(v, first(values)) for v in values]
    table.reference_fit = fill(first(report.fits).label, length(values))
    table.delta_vs_first = [d.Δ for d in differences]
    table.delta_mcse = [d.se for d in differences]
    table.paired = [d.paired for d in differences]
    any(>(0), table.lanczos_limit_hits) &&
        @warn "Lanczos reached the iteration limit; increase lanczos_maxiter before interpreting differences"
    @warn "Likelihood MCSE excludes Lanczos truncation error; repeat with tighter tolerances / more steps"
    mkpath(report.output_dir)
    CSV.write(joinpath(report.output_dir, "likelihood.csv"), table)
    return table
end

function compare_psi_scores(report; K=64, seed=2026, kwargs...)
    fixed_fits = filter(f -> !("ψ" in f.result.param_names), report.fits)
    isempty(fixed_fits) && throw(ArgumentError("No fixed-ψ fits configured for a score test"))
    foreach(_post_require_full, fixed_fits)
    values = map(fixed_fits) do fit
        value = _post_evaluate(fit, psi_score; K, seed, kwargs...)
        GC.gc()
        value
    end
    table = DataFrame(fit=[f.label for f in fixed_fits], psi=[s.ψ for s in values],
        score=[s.score for s in values], score_mcse=[s.score_se for s in values],
        z=[s.z for s in values], psi_onestep=[s.ψ_onestep for s in values])
    mkpath(report.output_dir)
    CSV.write(joinpath(report.output_dir, "psi_scores.csv"), table)
    return table
end
