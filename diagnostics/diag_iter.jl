include("emai.jl")

# Single-iteration comparison: runs ONE AI-REML iteration both sequential and
# threaded, printing intermediate values to find where they first diverge.
# Also tests H=0 vs H=1 to check whether env code broke threading.

println("="^70)
println("SINGLE-ITERATION THREADING COMPARISON")
println("Julia threads: $(Threads.nthreads())")
println("="^70)

if Threads.nthreads() == 1
    error("Need >1 thread. Start with: julia --threads 8 --project=. diag_iter.jl")
end

# --- Load data ---
yX = DataFrame(CSV.File("yX.csv"))
X_df = DataFrame(CSV.File("X.csv"))
obs = load_observed_data(yX; X_df=X_df)
X = obs.X
y = obs.y
n = length(y)
nf = size(X, 2)

Ainv_df = DataFrame(CSV.File("Ainv_triplet.csv"))
i_vals = Ainv_df[:, "i"]; j_vals = Ainv_df[:, "j"]; x_vals = Float64.(Ainv_df[:, "x"])
off_diag = i_vals .!= j_vals
Ainv = sparse(vcat(i_vals, j_vals[off_diag]), vcat(j_vals, i_vals[off_diag]), vcat(x_vals, x_vals[off_diag]))
N_ped = size(Ainv, 1)

L_Ainv_df = DataFrame(CSV.File("LAinv_triplet.csv"))
L_Ainv = LowerTriangular(sparse(L_Ainv_df[:, "i"], L_Ainv_df[:, "j"], Float64.(L_Ainv_df[:, "x"])))

gs = build_group_structure(yX, N_ped, obs.obs_mask)
has_env = "household" in names(yX)

function run_one_iter(label, model; threaded)
    println("\n--- $label (threaded=$threaded) ---")
    k = 10
    dim = get_dim(model)
    max_tid = Threads.maxthreadid()
    use_threads = threaded && Threads.nthreads() > 1

    model.sol_z = [zeros(dim) for _ in 1:k]
    fill!(model.sol_y, 0.0)

    Random.seed!(42)
    z_fixed_list = [randn(n) for _ in 1:k]
    Py = Vector{Float64}(undef, n)
    Pz_list = [Vector{Float64}(undef, n) for _ in 1:k]

    if use_threads
        workspaces = create_thread_workspaces(model, max_tid)
    else
        workspaces = [model_workspace(model)]
    end

    preconditioners = [get_preconditioner(model, ws) for ws in workspaces]

    # 1. Y-solve (always sequential)
    solve_mme_y!(model, workspaces[1], preconditioners[1], reltol=1e-7, maxiter=1000, verbose=false)
    compute_Px_from_sol!(Py, model, model.y, model.sol_y)
    @printf("  sol_y norm      = %.10e\n", norm(model.sol_y))
    @printf("  Py norm         = %.10e\n", norm(Py))

    # 2. Z-solves
    if use_threads
        Threads.@threads :static for i in 1:k
            tid = Threads.threadid()
            ws = workspaces[tid]
            solve_mme_z!(model.sol_z[i], model, z_fixed_list[i], ws, preconditioners[tid],
                reltol=1e-7, maxiter=1000)
            compute_Px_from_sol!(Pz_list[i], model, z_fixed_list[i], model.sol_z[i])
        end
    else
        ws = workspaces[1]
        for i in 1:k
            solve_mme_z!(model.sol_z[i], model, z_fixed_list[i], ws, preconditioners[1],
                reltol=1e-7, maxiter=1000)
            compute_Px_from_sol!(Pz_list[i], model, z_fixed_list[i], model.sol_z[i])
        end
    end

    sol_z_norms = [norm(model.sol_z[i]) for i in 1:k]
    Pz_norms = [norm(Pz_list[i]) for i in 1:k]
    @printf("  sol_z norms     = [%s]\n", join([@sprintf("%.6e", x) for x in sol_z_norms], ", "))
    @printf("  Pz norms        = [%s]\n", join([@sprintf("%.6e", x) for x in Pz_norms], ", "))

    # 3. Score + AI
    score, AI = compute_score_and_ai(model, Py, z_fixed_list, Pz_list,
        workspaces, preconditioners, ai_reltol=1e-5, ai_maxiter=300)

    @printf("  score           = [%s]\n", join([@sprintf("%+.6e", s) for s in score], ", "))
    @printf("  AI diag         = [%s]\n", join([@sprintf("%.6e", AI[i,i]) for i in 1:size(AI,1)], ", "))

    AI_reg = AI + 1e-6 * I
    delta = AI_reg \ score
    @printf("  delta           = [%s]\n", join([@sprintf("%+.6e", d) for d in delta], ", "))

    return (sol_y_norm=norm(model.sol_y), Py_norm=norm(Py),
            sol_z_norms=sol_z_norms, Pz_norms=Pz_norms,
            score=score, AI_diag=[AI[i,i] for i in 1:size(AI,1)], delta=delta)
end

# =====================================================================
# TEST A: H=0 (original model, should be same as pre-refactoring)
# =====================================================================
println("\n" * "="^70)
println("TEST A: H=0 (Base SGE, no env)")
println("="^70)

model_a_seq = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0)
r_a_seq = run_one_iter("H=0 sequential", model_a_seq, threaded=false)

model_a_thr = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0)
r_a_thr = run_one_iter("H=0 threaded", model_a_thr, threaded=true)

println("\n  COMPARISON H=0:")
@printf("  |Δ sol_y|    = %.2e\n", abs(r_a_seq.sol_y_norm - r_a_thr.sol_y_norm))
@printf("  |Δ Py|       = %.2e\n", abs(r_a_seq.Py_norm - r_a_thr.Py_norm))
max_dz = maximum(abs.(r_a_seq.sol_z_norms .- r_a_thr.sol_z_norms))
max_dpz = maximum(abs.(r_a_seq.Pz_norms .- r_a_thr.Pz_norms))
@printf("  max|Δ sol_z|  = %.2e\n", max_dz)
@printf("  max|Δ Pz|     = %.2e\n", max_dpz)
max_ds = maximum(abs.(r_a_seq.score .- r_a_thr.score))
max_dd = maximum(abs.(r_a_seq.delta .- r_a_thr.delta))
@printf("  max|Δ score|  = %.2e\n", max_ds)
@printf("  max|Δ delta|  = %.2e\n", max_dd)
h0_ok = max_dd < 1e-3
println("  H=0 threading: $(h0_ok ? "✓ MATCH" : "✗ DIVERGED")")

# =====================================================================
# TEST B: H=1 (env model)
# =====================================================================
if has_env
    println("\n" * "="^70)
    println("TEST B: H=1 (SGE + household env)")
    println("="^70)

    hh = observed_env_ids(yX, obs.obs_mask, [:household])[1]

    model_b_seq = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0; env_ids=[hh], σ2_env=[0.5])
    r_b_seq = run_one_iter("H=1 sequential", model_b_seq, threaded=false)

    model_b_thr = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0; env_ids=[hh], σ2_env=[0.5])
    r_b_thr = run_one_iter("H=1 threaded", model_b_thr, threaded=true)

    println("\n  COMPARISON H=1:")
    @printf("  |Δ sol_y|    = %.2e\n", abs(r_b_seq.sol_y_norm - r_b_thr.sol_y_norm))
    @printf("  |Δ Py|       = %.2e\n", abs(r_b_seq.Py_norm - r_b_thr.Py_norm))
    max_dz = maximum(abs.(r_b_seq.sol_z_norms .- r_b_thr.sol_z_norms))
    max_dpz = maximum(abs.(r_b_seq.Pz_norms .- r_b_thr.Pz_norms))
    @printf("  max|Δ sol_z|  = %.2e\n", max_dz)
    @printf("  max|Δ Pz|     = %.2e\n", max_dpz)
    max_ds = maximum(abs.(r_b_seq.score .- r_b_thr.score))
    max_dd = maximum(abs.(r_b_seq.delta .- r_b_thr.delta))
    @printf("  max|Δ score|  = %.2e\n", max_ds)
    @printf("  max|Δ delta|  = %.2e\n", max_dd)
    h1_ok = max_dd < 1e-3
    println("  H=1 threading: $(h1_ok ? "✓ MATCH" : "✗ DIVERGED")")
else
    println("\nSkipping H=1 test (no household column in yX.csv)")
end
