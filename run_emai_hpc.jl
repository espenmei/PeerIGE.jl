include(joinpath(@__DIR__, "src", "emai.jl"))

# Usage: julia --threads N --project=. run_emai_hpc.jl   (no arguments)
#
# Everything that changes between runs is set here and in the model block below.
# DATA_DIR is the folder written by the R setup script (out_dir), as seen from Linux:
#   N:/durable/... on Windows is /ess/p805/data/durable/... on Colossus.
# Results go to DATA_DIR/fits/RUN_NAME.{csv,jld2,jl}; the .jl is a copy of this
# script, so the exact model behind each result is kept with it.
const DATA_DIR = "/ess/p805/data/durable/projects/40683815_Espen_IGEpeer/data/edu10_MAT_termin1"
const RUN_NAME = "ds_class_household_cohort_pe"

function read_csv(name)
    path = joinpath(DATA_DIR, name)
    isfile(path) || error("Input file not found: $path")
    return DataFrame(CSV.File(path))
end
println("Run: $RUN_NAME, data: $DATA_DIR, threads: $(Threads.nthreads())")

obs_table = read_csv("yX_obs.csv")
roster = read_csv("roster.csv")
X_df = read_csv("X_obs.csv")

# X_obs.csv and yX_obs.csv must describe the same rows in the same order.
@assert nrow(X_df) == nrow(obs_table) "X_obs.csv and yX_obs.csv have different row counts"
@assert X_df.id_ped == obs_table.id_ped "X_obs.csv and yX_obs.csv are not in the same row order"
@assert !any(ismissing, obs_table.y) "yX_obs.csv must contain observed outcomes only"
obs = load_observed_data(obs_table; X_df=select(X_df, Not(:id_ped)))

# Ainv triplets hold the upper triangle; mirror the off-diagonal entries.
A = read_csv("Ainv_triplet.csv")
@assert all(A.i .<= A.j) "Ainv_triplet.csv must contain the upper triangle only"
od = A.i .!= A.j
N = max(maximum(A.i), maximum(A.j))
Ainv = sparse(vcat(A.i, A.j[od]), vcat(A.j, A.i[od]), Float64.(vcat(A.x, A.x[od])), N, N)
L = read_csv("LAinv_triplet.csv")
@assert all(L.i .>= L.j) "LAinv_triplet.csv must contain a lower-triangular factor"
L_Ainv = LowerTriangular(sparse(L.i, L.j, Float64.(L.x), N, N))

gs = build_group_structure(obs_table, roster, N)
println("Observed: $(length(obs.y)), roster: $(nrow(roster)), pedigree: $N, ",
    "fixed effects: $(size(obs.X, 2)), var(y)=$(var(obs.y))")

# ---- Model: edit by hand ---------------------------------------------------
# Start values: σ²_d, σ²_s, σ_ds, σ²_e.
# env_cols are columns in yX_obs.csv; σ2_env needs one start value per column,
# in the same order (printed as σ²_env1, σ²_env2, ...).
# For a pure direct + social genetic model, drop env_ids, σ2_env, env_names,
# peer_env and σ2_pe.
# For a direct-only model (direct + household + residual, no social effects):
#   model = SGEModel(obs.y, obs.X, Ainv, L_Ainv, gs, 0.70, 0.0, 0.0, 0.20;
#       social=false, fixef_names=obs.fixef_names,
#       env_ids=observed_env_ids(obs_table, [:household]), σ2_env=[0.02],
#       env_names=["household"])
# and remove estimate_ψ/ψ_bounds/ψ_max_step from fit_aireml! below.
env_cols = [:group, :household, :cohort]
model = SGEModel(obs.y, obs.X, Ainv, L_Ainv, gs, 0.70, 0.02, 0.01, 0.20;
    fixef_names=obs.fixef_names,
    env_ids=observed_env_ids(obs_table, env_cols),
    σ2_env=[0.02, 0.02, 0.02],
    env_names=string.(env_cols),
    peer_env=true, σ2_pe=0.01)
# ---------------------------------------------------------------------------

fit_dir = joinpath(DATA_DIR, "fits")
mkpath(fit_dir)
cp(@__FILE__, joinpath(fit_dir, "$RUN_NAME.jl"); force=true)
@time fit_aireml!(model; max_iter=100, k=64, damping=0.2, cg_maxiter=5000,
    cg_reltol_early=1e-5, cg_early_iters=4, cg_reltol=1e-7,
    estimate_ψ=true, ψ_bounds=(0.0, 1.0), ψ_max_step=0.25,
    log_file=joinpath(fit_dir, "$RUN_NAME.csv"))

display(model)
save_model(model, joinpath(fit_dir, "$RUN_NAME.jld2"))
