include("emai.jl")

const OUT_LOG_DIR = joinpath("outputs", "logs")
const OUT_RES_DIR = joinpath("outputs", "results")
mkpath(OUT_LOG_DIR)
mkpath(OUT_RES_DIR)

# Reminder: Julia worker thread count is fixed at startup.
println("[Thread reminder] Julia threads = $(Threads.nthreads()). Start with --threads <N> to control this, e.g.:")
println("  julia --threads 8 --project=. run_emai.jl")

# =====================================================================
# USER-SPECIFIED DESIGN
# Columns in yX_obs.csv to use as covariates in X (intercept always added).
const X_COLS = [:x]

# =====================================================================
# SHARED DATA LOADING
# =====================================================================
# Observed-analysis table: one row per observed outcome.
obs_table = DataFrame(CSV.File("yX_obs.csv"))
# Full roster table: social group membership for all relevant pedigree IDs.
roster = DataFrame(CSV.File("roster.csv"))

for col in (:y, :id_ped)
    @assert hasproperty(obs_table, col) "Missing column '$col' in yX_obs.csv"
end
for col in X_COLS
    @assert hasproperty(obs_table, col) "X_COLS references column '$col' not found in yX_obs.csv"
end
for col in (:id_ped, :group)
    @assert hasproperty(roster, col) "Missing column '$col' in roster.csv"
end

@assert !any(ismissing.(obs_table[:, :y])) "yX_obs.csv must contain observed y only (no missing values)"

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

@assert minimum(obs_to_id) >= 1 && maximum(obs_to_id) <= N_pedigree "Observed id_ped must be in 1..N_pedigree"
roster_ids = Int.(roster[:, :id_ped])
roster_groups = Int.(roster[:, :group])
@assert minimum(roster_ids) >= 1 && maximum(roster_ids) <= N_pedigree "Roster id_ped must be in 1..N_pedigree"

# Map observed IDs to their social group using the full roster.
obs_group_lookup = zeros(Int, N_pedigree)
for (id, g) in zip(roster_ids, roster_groups)
    prev = obs_group_lookup[id]
    @assert prev == 0 || prev == g "Roster id_ped $id appears in multiple groups"
    obs_group_lookup[id] = g
end
obs_group_ids = [obs_group_lookup[id] for id in obs_to_id]
@assert all(g -> g != 0, obs_group_ids) "Every observed id_ped must have a group in roster.csv"

gs = GroupStructure(obs_group_ids, obs_to_id, N_pedigree;
    all_group_ids=roster_groups, all_animal_ids=roster_ids)
println("Loaded tables: $(nrow(roster)) roster rows, $(length(y_obs)) observed outcomes")

# =====================================================================
# MODEL 1: Base SGE (no shared environment)
# =====================================================================
println("\n" * "="^70)
println("MODEL 1: Base SGE (σ²_d, σ²_s, σ_ds, σ²_e)")
println("="^70)

model1 = SGEModel(y_obs, X_obs, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0)

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

@assert hasproperty(obs_table, :household) "Missing column 'household' in yX_obs.csv required for Model 2"
household_ids = Int.(obs_table[:, :household])
model2 = SGEModel(y_obs, X_obs, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0;
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
    @assert hasproperty(obs_table, col) "Missing column '$col' in yX_obs.csv required for Model 3"
end

env_ids3 = [Int.(obs_table[:, :household]), Int.(obs_table[:, :classroom]), Int.(obs_table[:, :school])]
model3 = SGEModel(y_obs, X_obs, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0;
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