include("emai.jl")

const OUT_LOG_DIR = joinpath("outputs", "logs")
const OUT_RES_DIR = joinpath("outputs", "results")
mkpath(OUT_LOG_DIR)
mkpath(OUT_RES_DIR)

# Reminder: Julia worker thread count is fixed at startup.
println("[Thread reminder] Julia threads = $(Threads.nthreads()). Start with --threads <N> to control this, e.g.:")
println("  julia --threads 8 --project=. run_emai.jl")

# =====================================================================
# SHARED DATA LOADING
# =====================================================================
yX = DataFrame(CSV.File("yX.csv"))
obs = load_observed_data(yX; covariate_cols=[:x])
X = obs.X
y = obs.y

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

gs = build_group_structure(yX, N_pedigree, obs.obs_mask)
println("Loaded yX master table: $(nrow(yX)) total rows, $(length(y)) observed outcomes")

# =====================================================================
# MODEL 1: Base SGE (no shared environment)
# =====================================================================
println("\n" * "="^70)
println("MODEL 1: Base SGE (σ²_d, σ²_s, σ_ds, σ²_e)")
println("="^70)

model1 = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0)

@time result1 = run_aireml(
    model1,
    max_iter=20,
    k=20,
    damping=0.2,
    cg_maxiter=5000,
    cg_reltol_early=1e-5,
    cg_early_iters=4,
    cg_reltol=1e-7,
    threaded=true,
    log_file=joinpath(OUT_LOG_DIR, "aireml_log_base.csv"),
)

# save_results(result1, joinpath(OUT_RES_DIR, "aireml_result_base.jld2"))  # JLD2 unavailable on offline HPC
println("Model 1 estimates: σ²_d=$(model1.σ2_d), σ²_s=$(model1.σ2_s), σ_ds=$(model1.σ_ds), σ²_e=$(model1.σ2_e)")

# =====================================================================
# MODEL 2: SGE + household shared environment
# =====================================================================
println("\n" * "="^70)
println("MODEL 2: SGE + household (σ²_d, σ²_s, σ_ds, σ²_e, σ²_c)")
println("="^70)

household_ids = observed_env_ids(yX, obs.obs_mask, [:household])[1]
model2 = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0;
    env_ids=[household_ids], σ2_env=[0.5])

@time result2 = run_aireml(
    model2,
    max_iter=20,
    k=20,
    damping=0.2,
    cg_maxiter=5000,
    cg_reltol_early=1e-5,
    cg_early_iters=4,
    cg_reltol=1e-7,
    threaded=true,
    log_file=joinpath(OUT_LOG_DIR, "aireml_log_env.csv"),
)

# save_results(result2, joinpath(OUT_RES_DIR, "aireml_result_env.jld2"))  # JLD2 unavailable on offline HPC
println("Model 2 estimates: σ²_d=$(model2.σ2_d), σ²_s=$(model2.σ2_s), σ_ds=$(model2.σ_ds), σ²_e=$(model2.σ2_e), σ²_c=$(model2.σ2_env)")

# =====================================================================
# MODEL 3: SGE + household + classroom + school
# =====================================================================
println("\n" * "="^70)
println("MODEL 3: SGE + household + classroom + school")
println("="^70)

for col in (:household, :classroom, :school)
    @assert hasproperty(yX, col) "Missing column '$col' in yX.csv required for Model 3"
end

env_ids3 = observed_env_ids(yX, obs.obs_mask, [:household, :classroom, :school])
classroom_ids = env_ids3[2]
school_ids = env_ids3[3]
model3 = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0;
    env_ids=env_ids3,
    σ2_env=[0.5, 0.3, 0.2])

@time result3 = run_aireml(
    model3,
    max_iter=20,
    k=20,
    damping=0.2,
    cg_maxiter=5000,
    cg_reltol_early=1e-5,
    cg_early_iters=4,
    cg_reltol=1e-7,
    threaded=true,
    log_file=joinpath(OUT_LOG_DIR, "aireml_log_env3.csv"),
)

# save_results(result3, joinpath(OUT_RES_DIR, "aireml_result_env3.jld2"))  # JLD2 unavailable on offline HPC
println("Model 3 estimates: σ²_d=$(model3.σ2_d), σ²_s=$(model3.σ2_s), σ_ds=$(model3.σ_ds), σ²_e=$(model3.σ2_e), σ²_household=$(model3.σ2_env[1]), σ²_classroom=$(model3.σ2_env[2]), σ²_school=$(model3.σ2_env[3])")

# =====================================================================
# COMPARISON
# =====================================================================
println("\n" * "="^70)
println("COMPARISON")
println("="^70)
@printf("%-14s %12s %12s %12s\n", "Parameter", "Base", "+Household", "+Hh+Class+School")
@printf("%-14s %12.6f %12.6f %12.6f\n", "σ²_d", result1.σ2_d, result2.σ2_d, result3.σ2_d)
@printf("%-14s %12.6f %12.6f %12.6f\n", "σ²_s", result1.σ2_s, result2.σ2_s, result3.σ2_s)
@printf("%-14s %12.6f %12.6f %12.6f\n", "σ_ds", result1.σ_ds, result2.σ_ds, result3.σ_ds)
@printf("%-14s %12.6f %12.6f %12.6f\n", "σ²_e", result1.σ2_e, result2.σ2_e, result3.σ2_e)
@printf("%-14s %12s %12.6f %12.6f\n", "σ²_household", "—", result2.σ2_env[1], result3.σ2_env[1])
@printf("%-14s %12s %12s %12.6f\n", "σ²_classroom", "—", "—", result3.σ2_env[2])
@printf("%-14s %12s %12s %12.6f\n", "σ²_school", "—", "—", result3.σ2_env[3])