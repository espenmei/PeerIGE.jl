# Interactive inspection of a saved AI-REML fit.
# Open in VS Code and run cell by cell (Shift+Enter), or line by line in the REPL
# started with `julia --project=.`. Edit the paths below first.

## Setup
include(joinpath(@__DIR__, "src", "emai.jl"))
using Plots
default(margin=5Plots.mm)
project_dir = "/ess/p805/data/durable/projects/40683815_Espen_IGEpeer"
input = "edu10_MAT_termin1"
result_file = joinpath(project_dir, "outputs", "results", "aireml_ds_e_$input.jld2")
log_file = joinpath(project_dir, "outputs", "logs", "aireml_ds_e_$input.csv")
roster_file = joinpath(project_dir, "data", "roster_$input.csv")

## Load the fit (snapshot from save_results or full model from save_model)
saved = load(result_file)
res = haskey(saved, "result") ? saved["result"] : snapshot(saved["model"]; include_fixef_se=false)
group_sizes = combine(groupby(DataFrame(CSV.File(roster_file)), :group), nrow => :m).m
m = round(Int, median(group_sizes))
res

## Derived quantities with delta-method SEs
# Parameters: [σ²_d, σ²_s, σ_ds, σ²_e, σ²_env1..H, (σ²_pe)]
θ = vcat([res.σ2_d, res.σ2_s, res.σ_ds, res.σ2_e], res.σ2_env, res.peer_env ? [res.σ2_pe] : Float64[])
H = length(res.σ2_env)

function delta_se(f, θ, V)
    g = map(eachindex(θ)) do i
        h = 1e-6 * max(abs(θ[i]), 1e-3)
        e = (1:length(θ)) .== i
        (f(θ .+ h .* e) - f(θ .- h .* e)) / 2h
    end
    return sqrt(max(g' * V * g, 0.0))
end

# Group with roster size m, dilution c = (m-1)^-ψ, ignoring relatedness among group mates.
# σ²_TBV (Bijma et al. 2007) / σ²_P can exceed 1 when ψ=0 and groups are large.
dil(m) = m <= 1 ? 0.0 : (m - 1.0)^(-res.ψ)
σ2_P(θ, m) = θ[1] + θ[4] + sum(θ[5:4+H]; init=0.0) +
             dil(m)^2 * (m - 1) * (θ[2] + (res.peer_env ? θ[end] : 0.0))
σ2_TBV(θ, m) = θ[1] + 2dil(m) * (m - 1) * θ[3] + dil(m)^2 * (m - 1)^2 * θ[2]

derived = [
    "r_ds" => θ -> θ[3] / sqrt(θ[1] * θ[2]),
    "σ²_P (m=$m)" => θ -> σ2_P(θ, m),
    "h²_d" => θ -> θ[1] / σ2_P(θ, m),
    "σ²_TBV (m=$m)" => θ -> σ2_TBV(θ, m),
    "T² = σ²_TBV/σ²_P" => θ -> σ2_TBV(θ, m) / σ2_P(θ, m),
]
for h in 1:H
    push!(derived, "$(res.env_names[h]) share" => θ -> θ[4+h] / σ2_P(θ, m))
end
derived_df = DataFrame(quantity=first.(derived),
    estimate=[f(θ) for (_, f) in derived],
    se=[delta_se(f, θ, res.vcov) for (_, f) in derived])

## Fit diagnostics
(converged=res.converged, n_iter=res.n_iter, step_crit=res.step_crit,
    score_norm=norm(res.score), G_eigvals=eigvals(Symmetric([res.σ2_d res.σ_ds; res.σ_ds res.σ2_s])))

## Correlation of variance-component estimates (from AI⁻¹)
D = Diagonal(1 ./ sqrt.(diag(res.vcov)))
DataFrame(round.(D * res.vcov * D; digits=3), res.param_names)

## Convergence trace
log_df = DataFrame(CSV.File(log_file))
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

## Heritable variance by group size
ms = 2:maximum(group_sizes)
tbv_se = [delta_se(θ -> σ2_TBV(θ, k), θ, res.vcov) for k in ms]
plot(plot(ms, σ2_TBV.(Ref(θ), ms); ribbon=1.96tbv_se, label="σ²_TBV", xlabel="group size m"),
    histogram(group_sizes; legend=false, xlabel="group size m", title="roster group sizes");
    size=(1000, 380))
