# Interactive inspection of saved AI-REML fits.
# Open in VS Code and run cell by cell (Shift+Enter), or line by line in the REPL
# started with `julia --project=.`. Edit data_dir and run_name below first.
#
# ψ (dilution exponent, peer contribution scaled by (m−1)^−ψ) can differ between fits.
# σ²_s and σ²_pe are then on different scales, so compare fits through ψ-invariant
# quantities: the social variance received and σ²_TBV at a given group size m.

## Setup
include(joinpath(@__DIR__, "src", "emai.jl"))
using Plots
default(margin=5Plots.mm)
data_dir = "/ess/p805/data/durable/projects/40683815_Espen_IGEpeer/data/edu10_MAT_termin1"  # DATA_DIR
run_name = "ds_class_household_cohort_pe"                                                  # RUN_NAME
fit_dir = joinpath(data_dir, "fits")
group_sizes = combine(groupby(DataFrame(CSV.File(joinpath(data_dir, "roster.csv"))), :group), nrow => :m).m
m = round(Int, median(group_sizes))

## Load the fit (full model from save_model, or snapshot from save_results)
function load_fit(name)
    saved = load(joinpath(fit_dir, "$name.jld2"))
    return haskey(saved, "result") ? saved["result"] : snapshot(saved["model"]; include_fixef_se=false)
end
res = load_fit(run_name)

## Parameters and ψ-aware derived quantities
# θ follows res.param_names, so it matches res.vcov. ψ is taken from θ when the fit
# estimates it ("ψ" in param_names), otherwise it is the fixed res.ψ and contributes no SE.
function param_vector(res)
    vals = Dict("σ²_d" => res.σ2_d, "σ²_s" => res.σ2_s, "σ_ds" => res.σ_ds, "σ²_e" => res.σ2_e,
        "σ²_pe" => res.σ2_pe, "ψ" => res.ψ)
    for h in eachindex(res.σ2_env)
        vals["σ²_env$h"] = res.σ2_env[h]
    end
    return [vals[p] for p in res.param_names]
end
getp(θ, res, p, default=0.0) = (i = findfirst(==(p), res.param_names); i === nothing ? default : θ[i])
ψ_of(θ, res) = getp(θ, res, "ψ", res.ψ)
dil(m, ψ) = m <= 1 ? 0.0 : (m - 1.0)^(-ψ)

# Group with roster size m, ignoring relatedness among group mates.
# Social variance received by one individual from its m−1 group mates (genetic + peer env):
σ2_soc(θ, res, m) = dil(m, ψ_of(θ, res))^2 * (m - 1) * (getp(θ, res, "σ²_s") + getp(θ, res, "σ²_pe"))
σ2_P(θ, res, m) = getp(θ, res, "σ²_d") + getp(θ, res, "σ²_e") +
                  sum(getp(θ, res, "σ²_env$h") for h in eachindex(res.σ2_env); init=0.0) + σ2_soc(θ, res, m)
# σ²_TBV (Bijma et al. 2007); σ²_TBV/σ²_P can exceed 1 when ψ=0 and groups are large.
function σ2_TBV(θ, res, m)
    c = dil(m, ψ_of(θ, res)) * (m - 1)
    return getp(θ, res, "σ²_d") + 2c * getp(θ, res, "σ_ds") + c^2 * getp(θ, res, "σ²_s")
end

function delta_se(f, θ, V)
    g = map(eachindex(θ)) do i
        h = 1e-6 * max(abs(θ[i]), 1e-3)
        e = (1:length(θ)) .== i
        (f(θ .+ h .* e) - f(θ .- h .* e)) / 2h
    end
    return sqrt(max(g' * V * g, 0.0))
end

function derived_table(res, m)
    θ = param_vector(res)
    d = [
        "ψ" => θ -> ψ_of(θ, res),
        "r_ds" => θ -> getp(θ, res, "σ_ds") / sqrt(getp(θ, res, "σ²_d") * getp(θ, res, "σ²_s")),
        "σ²_soc (m=$m)" => θ -> σ2_soc(θ, res, m),
        "σ²_P (m=$m)" => θ -> σ2_P(θ, res, m),
        "h²_d" => θ -> getp(θ, res, "σ²_d") / σ2_P(θ, res, m),
        "σ²_TBV (m=$m)" => θ -> σ2_TBV(θ, res, m),
        "T² = σ²_TBV/σ²_P" => θ -> σ2_TBV(θ, res, m) / σ2_P(θ, res, m),
    ]
    for h in eachindex(res.σ2_env)
        push!(d, "$(res.env_names[h]) share" => θ -> getp(θ, res, "σ²_env$h") / σ2_P(θ, res, m))
    end
    return DataFrame(quantity=first.(d), estimate=[f(θ) for (_, f) in d],
        se=[delta_se(f, θ, res.vcov) for (_, f) in d])
end
θ = param_vector(res)
derived_table(res, m)

## Fit diagnostics
(converged=res.converged, n_iter=res.n_iter, step_crit=res.step_crit, ψ=res.ψ,
    ψ_estimated="ψ" in res.param_names, score_norm=norm(res.score),
    G_eigvals=eigvals(Symmetric([res.σ2_d res.σ_ds; res.σ_ds res.σ2_s])))

## Correlation of variance-component estimates (from AI⁻¹)
D = Diagonal(1 ./ sqrt.(diag(res.vcov)))
DataFrame(round.(D * res.vcov * D; digits=3), res.param_names)

## Convergence trace
log_df = DataFrame(CSV.File(joinpath(fit_dir, "$run_name.csv")))
pcols = filter(c -> startswith(c, "s2") || c == "sds", names(log_df))
scols = filter(c -> startswith(c, "se_"), names(log_df))
panels = [plot(log_df.iter, log_df[!, p]; ribbon=log_df[!, s], marker=:circle, title=p, legend=false)
          for (p, s) in zip(pcols, scols)]
push!(panels, plot(log_df.iter, log_df.step_crit; yscale=:log10, marker=:circle, title="max|Δ|/SE", legend=false))
hline!(panels[end], [1e-2]; ls=:dash, color=:red)
plot(panels...; size=(1100, 650))

## BLUPs for observed individuals
ids = unique(res.obs_to_id)
ad, as = res.a_d[ids], res.a_s[ids]
(sd_ad=std(ad), sd_as=std(as), cor=cor(ad, as),
    shrink_d=std(ad) / sqrt(res.σ2_d), shrink_s=std(as) / sqrt(res.σ2_s))

##
idx = randperm(MersenneTwister(1), length(ad))[1:min(20_000, length(ad))]
plot(scatter(ad[idx], as[idx]; ms=1.5, alpha=0.3, msw=0, xlabel="â_d", ylabel="â_s"),
    histogram(ad; bins=60, title="â_d"), histogram(as; bins=60, title="â_s");
    layout=(1, 3), size=(1200, 380), legend=false)

## Heritable and social variance by group size
ms = 2:maximum(group_sizes)
tbv_se = [delta_se(θ -> σ2_TBV(θ, res, k), θ, res.vcov) for k in ms]
soc_se = [delta_se(θ -> σ2_soc(θ, res, k), θ, res.vcov) for k in ms]
plot(plot(ms, [σ2_TBV(θ, res, k) for k in ms]; ribbon=1.96tbv_se, label="σ²_TBV", xlabel="group size m"),
    plot(ms, [σ2_soc(θ, res, k) for k in ms]; ribbon=1.96soc_se, label="σ²_soc received", xlabel="group size m"),
    histogram(group_sizes; legend=false, xlabel="group size m", title="roster group sizes");
    layout=(1, 3), size=(1300, 380))

## Compare fits with different ψ on a common scale
# Same data and model apart from ψ. The fit with the best size pattern of σ²_soc(m) is
# not decided by these plots; use the log-likelihood / score test in the next cell.
compare_runs = ["ds_class_household_cohort_pe", "ds_class_household_cohort_pe_mean"]
fits = [load_fit(r) for r in compare_runs]
compare_df = vcat([begin
    t = derived_table(f, m)
    DataFrame(run=r, ψ=f.ψ, quantity=t.quantity, estimate=t.estimate, se=t.se)
end for (r, f) in zip(compare_runs, fits)]...)
unstack(select(compare_df, :run, :quantity, :estimate), :quantity, :run, :estimate)

##
p_tbv = plot(xlabel="group size m", title="σ²_TBV")
p_soc = plot(xlabel="group size m", title="σ²_soc received")
for (r, f) in zip(compare_runs, fits)
    θf = param_vector(f)
    lab = "$r (ψ=$(round(f.ψ, digits=2)))"
    plot!(p_tbv, ms, [σ2_TBV(θf, f, k) for k in ms];
        ribbon=1.96 .* [delta_se(θ -> σ2_TBV(θ, f, k), θf, f.vcov) for k in ms], label=lab)
    plot!(p_soc, ms, [σ2_soc(θf, f, k) for k in ms];
        ribbon=1.96 .* [delta_se(θ -> σ2_soc(θ, f, k), θf, f.vcov) for k in ms], label=lab)
end
plot(p_tbv, p_soc; layout=(1, 2), size=(1200, 420))

## Model comparison: stochastic REML log-likelihood and ψ score test
# Needs full models saved with save_model (run_emai_hpc.jl writes fits/RUN_NAME.jld2), the same
# y and X in all fits, and enough memory (≈ 0.6 GB per thread on the real data). Each probe
# costs about one CG solve, so run this in a Slurm job rather than interactively on real data.
models = [load_model(joinpath(fit_dir, "$r.jld2")) for r in compare_runs]
lls = [stochastic_loglik(mod; K=64, seed=1) for mod in models]  # same seed ⇒ paired probes
DataFrame(run=compare_runs, ψ=[mod.ψ for mod in models], loglik=[l.loglik for l in lls],
    se=[l.se for l in lls], Δ_vs_first=[loglik_difference(l, lls[1]).Δ for l in lls],
    Δ_se=[l === lls[1] ? 0.0 : loglik_difference(l, lls[1]).se for l in lls])

##
# Score test for H₀: ψ = model.ψ at each converged fit. z > 0 favours more dilution (larger ψ).
# Check that score_se is small relative to score; otherwise increase K.
scores = [psi_score(mod; K=64, seed=1) for mod in models]
DataFrame(run=compare_runs, ψ=[s.ψ for s in scores], score=[s.score for s in scores],
    score_se=[s.score_se for s in scores], z=[s.z for s in scores], ψ_onestep=[s.ψ_onestep for s in scores])