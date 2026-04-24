include("emai.jl")

const OUT_BENCH_DIR = joinpath("outputs", "benchmarks")
mkpath(OUT_BENCH_DIR)
const SEQ_LOG = joinpath(OUT_BENCH_DIR, "bench_seq.csv")
const THR_LOG = joinpath(OUT_BENCH_DIR, "bench_thr.csv")

println("="^70)
println("THREADING BENCHMARK")
println("Julia threads: $(Threads.nthreads())")
println("="^70)

if Threads.nthreads() == 1
    @warn "Only 1 thread available. Restart with: julia --threads 8 --project=. bench_threads.jl"
end

# --- Load data ---
yX = DataFrame(CSV.File("yX.csv"))
X_df = DataFrame(CSV.File("X.csv"))
obs = load_observed_data(yX; X_df=X_df)
X = obs.X
y = obs.y

Ainv_df = DataFrame(CSV.File("Ainv_triplet.csv"))
i_vals = Ainv_df[:, "i"]; j_vals = Ainv_df[:, "j"]; x_vals = Float64.(Ainv_df[:, "x"])
off_diag = i_vals .!= j_vals
Ainv = sparse(vcat(i_vals, j_vals[off_diag]), vcat(j_vals, i_vals[off_diag]), vcat(x_vals, x_vals[off_diag]))
N_pedigree = size(Ainv, 1)

L_Ainv_df = DataFrame(CSV.File("LAinv_triplet.csv"))
L_Ainv = LowerTriangular(sparse(L_Ainv_df[:, "i"], L_Ainv_df[:, "j"], Float64.(L_Ainv_df[:, "x"])))

gs = build_group_structure(yX, N_pedigree, obs.obs_mask)

# Use env if household column exists
has_env = "household" in names(yX)

# --- Settings ---
# Run enough iterations for convergence so we can compare final estimates.
# Use k=20 for enough parallel work to show threading benefit.
bench_settings = (max_iter=10, k=20, damping=0.5,
                  cg_maxiter=1000, ai_maxiter=300,
                  use_block_precond=true, verbose_cg=false, seed=42)

function make_model()
    if has_env
        hh = observed_env_ids(yX, obs.obs_mask, [:household])[1]
        SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0; env_ids=[hh], σ2_env=[0.5])
    else
        SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0)
    end
end

println("\nModel: $(has_env ? "SGE + household env (H=1)" : "Base SGE (H=0)")")
println("Settings: max_iter=$(bench_settings.max_iter), k=$(bench_settings.k)")
println("-"^70)

# --- Warmup: JIT compile BOTH code paths ---
print("Warmup sequential... ")
m_w1 = make_model()
run_aireml(m_w1; max_iter=1, k=2, damping=0.5, cg_maxiter=200, ai_maxiter=100,
           verbose_cg=false, threaded=false, seed=42)
println("done")

if Threads.nthreads() > 1
    print("Warmup threaded...   ")
    m_w2 = make_model()
    run_aireml(m_w2; max_iter=1, k=2, damping=0.5, cg_maxiter=200, ai_maxiter=100,
               verbose_cg=false, threaded=true, seed=42)
    println("done")
end
GC.gc()  # clean up warmup allocations

# --- Sequential ---
println("\n--- SEQUENTIAL RUN ---")
m_seq = make_model()
t_seq = @elapsed result_seq = run_aireml(m_seq; bench_settings..., threaded=false,
                                          log_file=SEQ_LOG)
@printf("Sequential total: %.2fs\n", t_seq)

# --- Threaded ---
if Threads.nthreads() > 1
    GC.gc()
    @printf("\n--- THREADED RUN (%d threads) ---\n", Threads.nthreads())
    m_thr = make_model()
    t_thr = @elapsed result_thr = run_aireml(m_thr; bench_settings..., threaded=true,
                                              log_file=THR_LOG)
    @printf("Threaded total:   %.2fs\n", t_thr)

    # --- Timing comparison ---
    speedup = t_seq / t_thr
    println("\n" * "="^70)
    println("TIMING RESULTS")
    println("="^70)
    @printf("Sequential: %.2fs\n", t_seq)
    @printf("Threaded:   %.2fs  (%d threads)\n", t_thr, Threads.nthreads())
    @printf("Speedup:    %.2fx\n", speedup)
    if speedup > 1.1
        @printf("→ Threading helps: %.0f%% faster\n", (speedup - 1) * 100)
    elseif speedup < 0.9
        @printf("→ Threading HURTS: %.0f%% slower\n", (1 - speedup) * 100)
    else
        println("→ No significant difference (within ±10%)")
    end

    # --- Per-iteration timing from CSV logs ---
    println("\nPer-iteration timing (from CSV logs):")
    println("-"^70)
    try
        seq_log = DataFrame(CSV.File(SEQ_LOG))
        thr_log = DataFrame(CSV.File(THR_LOG))
        nrows = min(nrow(seq_log), nrow(thr_log))
        @printf("%-5s  %8s %8s %8s | %8s %8s %8s | %6s\n",
                "Iter", "y_seq", "z_seq", "ai_seq", "y_thr", "z_thr", "ai_thr", "ratio")
        for r in 1:nrows
            total_seq = seq_log[r, :time_y] + seq_log[r, :time_z] + seq_log[r, :time_ai]
            total_thr = thr_log[r, :time_y] + thr_log[r, :time_z] + thr_log[r, :time_ai]
            ratio = total_seq / max(total_thr, 0.01)
            @printf("%-5d  %7.1fs %7.1fs %7.1fs | %7.1fs %7.1fs %7.1fs | %5.2fx\n",
                    r,
                    seq_log[r, :time_y], seq_log[r, :time_z], seq_log[r, :time_ai],
                    thr_log[r, :time_y], thr_log[r, :time_z], thr_log[r, :time_ai],
                    ratio)
        end
    catch e
        println("  (could not parse CSV logs: $e)")
    end

    # --- Parameter comparison ---
    println("\n" * "="^70)
    println("PARAMETER COMPARISON (at convergence)")
    println("="^70)
    @printf("%-12s %12s %12s %12s\n", "Parameter", "Sequential", "Threaded", "|Δ|")
    @printf("%-12s %12.6f %12.6f %12.2e\n", "σ²_d", result_seq.σ2_d, result_thr.σ2_d, abs(result_seq.σ2_d - result_thr.σ2_d))
    @printf("%-12s %12.6f %12.6f %12.2e\n", "σ²_s", result_seq.σ2_s, result_thr.σ2_s, abs(result_seq.σ2_s - result_thr.σ2_s))
    @printf("%-12s %12.6f %12.6f %12.2e\n", "σ_ds", result_seq.σ_ds, result_thr.σ_ds, abs(result_seq.σ_ds - result_thr.σ_ds))
    @printf("%-12s %12.6f %12.6f %12.2e\n", "σ²_e", result_seq.σ2_e, result_thr.σ2_e, abs(result_seq.σ2_e - result_thr.σ2_e))
    for h in eachindex(result_seq.σ2_env)
        Δ = abs(result_seq.σ2_env[h] - result_thr.σ2_env[h])
        @printf("%-12s %12.6f %12.6f %12.2e\n", "σ²_env$h", result_seq.σ2_env[h], result_thr.σ2_env[h], Δ)
    end

    all_Δ = [abs(result_seq.σ2_d - result_thr.σ2_d),
             abs(result_seq.σ2_s - result_thr.σ2_s),
             abs(result_seq.σ_ds - result_thr.σ_ds),
             abs(result_seq.σ2_e - result_thr.σ2_e)]
    append!(all_Δ, abs.(result_seq.σ2_env .- result_thr.σ2_env))
    max_Δ = maximum(all_Δ)
    println("-"^70)
    if max_Δ < 1e-4
        @printf("Max |Δ| = %.2e  ✓  Results match (threading is correct)\n", max_Δ)
    elseif max_Δ < 0.01
        @printf("Max |Δ| = %.2e  ≈  Close (CG tolerance noise, OK)\n", max_Δ)
    else
        @printf("Max |Δ| = %.2e  ⚠  Diverged — investigate!\n", max_Δ)
    end

    println("\nConverged: seq=$(result_seq.converged) ($(result_seq.n_iter) iter), thr=$(result_thr.converged) ($(result_thr.n_iter) iter)")
else
    println("\n" * "="^70)
    @printf("RESULTS:  sequential=%.2fs  (threaded benchmark skipped, only 1 thread)\n", t_seq)
    println("Restart with: julia --threads 8 --project=. bench_threads.jl")
end
