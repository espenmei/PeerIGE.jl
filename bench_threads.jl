include(joinpath(@__DIR__, "src", "emai.jl"))

const OUT_BENCH_DIR = joinpath("outputs", "benchmarks")
mkpath(OUT_BENCH_DIR)
const THR_LOG = joinpath(OUT_BENCH_DIR, "bench_thr.csv")

println("="^70)
println("THREADING BENCHMARK")
println("Julia threads: $(Threads.nthreads())")
println("="^70)

if Threads.nthreads() == 1
    @warn "Only 1 thread available. Restart with: julia --threads 8 --project=. bench_threads.jl"
end

# --- Load data ---
obs_table = DataFrame(CSV.File("yX_obs.csv"))
roster = DataFrame(CSV.File("roster.csv"))
obs = load_observed_data(obs_table; covariate_cols=[:x])
X = obs.X
y = obs.y

Ainv_df = DataFrame(CSV.File("Ainv_triplet.csv"))
i_vals = Ainv_df[:, "i"]; j_vals = Ainv_df[:, "j"]; x_vals = Float64.(Ainv_df[:, "x"])
off_diag = i_vals .!= j_vals
Ainv = sparse(vcat(i_vals, j_vals[off_diag]), vcat(j_vals, i_vals[off_diag]), vcat(x_vals, x_vals[off_diag]))
N_pedigree = size(Ainv, 1)

L_Ainv_df = DataFrame(CSV.File("LAinv_triplet.csv"))
L_Ainv = LowerTriangular(sparse(L_Ainv_df[:, "i"], L_Ainv_df[:, "j"], Float64.(L_Ainv_df[:, "x"])))

gs = build_group_structure(obs_table, roster, N_pedigree)

# Use env if household column exists
has_env = "household" in names(obs_table)

# --- Settings ---
# Run enough iterations for convergence so we can inspect final estimates.
# Use k=20 for enough parallel work to exercise threading.
bench_settings = (max_iter=10, k=20, damping=0.5,
                  cg_maxiter=1000, ai_maxiter=300,
                  use_block_precond=true, verbose_cg=false, seed=42)

function make_model()
    if has_env
        hh = observed_env_ids(obs_table, [:household])[1]
        SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0; env_ids=[hh], σ2_env=[0.5])
    else
        SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0)
    end
end

println("\nModel: $(has_env ? "SGE + household env (H=1)" : "Base SGE (H=0)")")
println("Settings: max_iter=$(bench_settings.max_iter), k=$(bench_settings.k)")
println("-"^70)

# --- Warmup: JIT compile ---
print("Warmup... ")
m_w = make_model()
run_aireml(m_w; max_iter=1, k=2, damping=0.5, cg_maxiter=200, ai_maxiter=100,
           verbose_cg=false, seed=42)
println("done")
GC.gc()  # clean up warmup allocations

# --- Timed run ---
@printf("\n--- RUN (%d threads) ---\n", Threads.nthreads())
m = make_model()
t_run = @elapsed result = run_aireml(m; bench_settings..., log_file=THR_LOG)
@printf("Total: %.2fs  (%d threads)\n", t_run, Threads.nthreads())

# --- Per-iteration timing from CSV log ---
println("\nPer-iteration timing (from CSV log):")
println("-"^70)
try
    thr_log = DataFrame(CSV.File(THR_LOG))
    @printf("%-5s  %8s %8s %8s | %8s\n", "Iter", "y", "z", "ai", "total")
    for r in 1:nrow(thr_log)
        total = thr_log[r, :time_y] + thr_log[r, :time_z] + thr_log[r, :time_ai]
        @printf("%-5d  %7.1fs %7.1fs %7.1fs | %7.1fs\n",
                r, thr_log[r, :time_y], thr_log[r, :time_z], thr_log[r, :time_ai], total)
    end
catch e
    println("  (could not parse CSV log: $e)")
end

# --- Final estimates ---
println("\n" * "="^70)
println("FINAL ESTIMATES (at convergence)")
println("="^70)
@printf("%-12s %12s\n", "Parameter", "Estimate")
@printf("%-12s %12.6f\n", "σ²_d", result.σ2_d)
@printf("%-12s %12.6f\n", "σ²_s", result.σ2_s)
@printf("%-12s %12.6f\n", "σ_ds", result.σ_ds)
@printf("%-12s %12.6f\n", "σ²_e", result.σ2_e)
for h in eachindex(result.σ2_env)
    @printf("%-12s %12.6f\n", "σ²_env$h", result.σ2_env[h])
end
println("-"^70)
println("Converged: $(result.converged) ($(result.n_iter) iter)")
