include("../emai.jl")
using CSV, DataFrames, SparseArrays, LinearAlgebra, Printf

obs_table = DataFrame(CSV.File("yX_obs.csv"))
roster = DataFrame(CSV.File("roster.csv"))
obs = load_observed_data(obs_table; covariate_cols=[:x])
X = obs.X; y = obs.y

df = DataFrame(CSV.File("Ainv_triplet.csv"))
i = df[:,1]; j = df[:,2]; v = Float64.(df[:,3])
od = i .!= j
Ainv = sparse(vcat(i, j[od]), vcat(j, i[od]), vcat(v, v[od]))
N = size(Ainv,1)

Ldf = DataFrame(CSV.File("LAinv_triplet.csv"))
LA = LowerTriangular(sparse(Ldf[:,1], Ldf[:,2], Float64.(Ldf[:,3])))

gs = build_group_structure(obs_table, roster, N)
m = SGEModel(y, X, Ainv, LA, gs, 1.0, 0.2, 0.0, 2.0)
r = run_aireml(m; max_iter=15, k=20, damping=0.5,
    cg_maxiter=5000, cg_reltol_early=1e-5, cg_early_iters=4,
    cg_reltol=1e-7, threaded=true)

@printf("FINAL: s2d=%.6f s2s=%.6f sds=%.6f s2e=%.6f\n",
    m.σ2_d, m.σ2_s, m.σ_ds, m.σ2_e)
