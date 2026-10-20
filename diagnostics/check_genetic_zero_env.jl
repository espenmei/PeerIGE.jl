include(joinpath(@__DIR__, "..", "src", "emai.jl"))
using Printf
using CSV, DataFrames, SparseArrays, LinearAlgebra

const X_COLS = [:x]

obs_table = DataFrame(CSV.File("yX_obs.csv"))
roster = DataFrame(CSV.File("roster.csv"))

obs_to_id = Int.(obs_table[:, :id_ped])
y_obs = Float64.(obs_table[:, :y])
X_obs = hcat(ones(length(y_obs)), [Float64.(obs_table[:, col]) for col in X_COLS]...)

Ainv_df = DataFrame(CSV.File("Ainv_triplet.csv"))
i_vals = Ainv_df[:, "i"]
j_vals = Ainv_df[:, "j"]
x_vals = Float64.(Ainv_df[:, "x"])

off_diag = i_vals .!= j_vals
i_all = vcat(i_vals, j_vals[off_diag])
j_all = vcat(j_vals, i_vals[off_diag])
x_all = vcat(x_vals, x_vals[off_diag])

Ainv = sparse(i_all, j_all, x_all)
N_pedigree = size(Ainv, 1)

L_Ainv_df = DataFrame(CSV.File("LAinv_triplet.csv"))
L_Ainv = sparse(L_Ainv_df[:, "i"], L_Ainv_df[:, "j"], Float64.(L_Ainv_df[:, "x"]))
L_Ainv = LowerTriangular(L_Ainv)

roster_ids = Int.(roster[:, :id_ped])
roster_groups = Int.(roster[:, :group])
obs_group_lookup = zeros(Int, N_pedigree)
for (id, g) in zip(roster_ids, roster_groups)
    obs_group_lookup[id] = g
end
obs_group_ids = [obs_group_lookup[id] for id in obs_to_id]

gs = GroupStructure(obs_group_ids, obs_to_id, N_pedigree;
    all_group_ids=roster_groups, all_animal_ids=roster_ids)

model = SGEModel(y_obs, X_obs, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0)

result = run_aireml(
    model,
    max_iter=15,
    k=20,
    damping=0.5,
    cg_maxiter=5000,
    cg_reltol_early=1e-5,
    cg_early_iters=4,
    cg_reltol=1e-7,
)

truth = Dict(
    :σ2_d => 1.0,
    :σ2_s => 0.1,
    :σ_ds => 0.1,
    :σ2_e => 1.2,
)

est = Dict(
    :σ2_d => result.σ2_d,
    :σ2_s => result.σ2_s,
    :σ_ds => result.σ_ds,
    :σ2_e => result.σ2_e,
)

println("\nGenetic-only recovery check (env truth = 0):")
@printf("%-8s %10s %10s %10s %10s\n", "param", "truth", "est", "abs_err", "pct_err")
for p in (:σ2_d, :σ2_s, :σ_ds, :σ2_e)
    t = truth[p]
    e = est[p]
    ae = abs(e - t)
    pe = 100 * ae / t
    @printf("%-8s %10.4f %10.4f %10.4f %9.1f%%\n", String(p), t, e, ae, pe)
end
