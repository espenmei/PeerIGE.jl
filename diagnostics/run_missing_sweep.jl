include("../emai.jl")
using CSV, DataFrames, SparseArrays, LinearAlgebra, Printf

const TRUTH = (σ2_d=1.0, σ2_s=0.1, σ_ds=0.1, σ2_e=1.5625)
const FRACS = ["frac00", "frac05", "frac10", "frac20"]

# Build Ainv / L_Ainv once (same across fracs)
df = DataFrame(CSV.File("Ainv_triplet.csv"))
i = df[:,1]; j = df[:,2]; v = Float64.(df[:,3])
od = i .!= j
Ainv = sparse(vcat(i, j[od]), vcat(j, i[od]), vcat(v, v[od]))
N = size(Ainv, 1)
Ldf = DataFrame(CSV.File("LAinv_triplet.csv"))
LA = LowerTriangular(sparse(Ldf[:,1], Ldf[:,2], Float64.(Ldf[:,3])))

results = DataFrame(frac=String[], n_obs=Int[],
    s2d=Float64[], s2s=Float64[], sds=Float64[], s2e=Float64[],
    converged=Bool[], iter=Int[])

for tag in FRACS
    obs = DataFrame(CSV.File(joinpath("diagnostics","missing_sweep_data","yX_obs_$(tag).csv")))
    rost = DataFrame(CSV.File(joinpath("diagnostics","missing_sweep_data","roster_$(tag).csv")))

    obs_to_id = Int.(obs[:, :id_ped])
    y = Float64.(obs[:, :y])
    X = hcat(ones(length(y)), Float64.(obs[:, :x]))

    rids = Int.(rost[:, :id_ped]); rgrp = Int.(rost[:, :group])
    lookup = zeros(Int, N)
    for (id, g) in zip(rids, rgrp); lookup[id] = g; end
    obs_grp = [lookup[id] for id in obs_to_id]

    gs = GroupStructure(obs_grp, obs_to_id, N;
        all_group_ids=rgrp, all_animal_ids=rids)

    m = SGEModel(y, X, Ainv, LA, gs, 1.0, 0.2, 0.0, 2.0)
    r = run_aireml(m; max_iter=30, k=20, damping=0.5,
        cg_maxiter=5000, cg_reltol_early=1e-5, cg_early_iters=4,
        cg_reltol=1e-7, threaded=true, verbose_cg=false)

    push!(results, (tag, length(y), m.σ2_d, m.σ2_s, m.σ_ds, m.σ2_e,
        r.converged, r.n_iter))
end

println("\n========================================================")
println("Missing-fraction sweep — TRUTH: σ²_d=1.0  σ²_s=0.1  σ_ds=0.1  σ²_e=1.5625")
println("========================================================")
@printf("%-7s %7s %9s %9s %9s %9s  %s\n", "frac", "n_obs",
    "s2d", "s2s", "sds", "s2e", "conv/iter")
for r in eachrow(results)
    @printf("%-7s %7d %9.4f %9.4f %9.4f %9.4f  %s/%d\n",
        r.frac, r.n_obs, r.s2d, r.s2s, r.sds, r.s2e,
        r.converged ? "yes" : "no", r.iter)
end

println("\nPercent error (vs truth):")
@printf("%-7s %9s %9s %9s %9s\n", "frac", "s2d%", "s2s%", "sds%", "s2e%")
for r in eachrow(results)
    @printf("%-7s %8.1f%% %8.1f%% %8.1f%% %8.1f%%\n", r.frac,
        100*(r.s2d-TRUTH.σ2_d)/TRUTH.σ2_d,
        100*(r.s2s-TRUTH.σ2_s)/TRUTH.σ2_s,
        100*(r.sds-TRUTH.σ_ds)/TRUTH.σ_ds,
        100*(r.s2e-TRUTH.σ2_e)/TRUTH.σ2_e)
end

CSV.write("diagnostics/missing_sweep_results.csv", results)
println("\nWrote diagnostics/missing_sweep_results.csv")
