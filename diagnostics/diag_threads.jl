include("emai.jl")

println("="^70)
println("THREADING DIAGNOSTIC")
println("Julia threads: $(Threads.nthreads()), BLAS threads: $(LinearAlgebra.BLAS.get_num_threads())")
println("BLAS vendor: $(LinearAlgebra.BLAS.vendor())")
println("="^70)

if Threads.nthreads() == 1
    error("Need >1 thread. Start with: julia --threads 8 --project=. diag_threads.jl")
end

PASS = 0
FAIL = 0
function check_test(name, ok, detail="")
    global PASS, FAIL
    if ok
        PASS += 1
        println("  ✓ $name")
    else
        FAIL += 1
        println("  ✗ $name  $detail")
    end
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
hh = has_env ? observed_env_ids(yX, obs.obs_mask, [:household])[1] : nothing

function make_model()
    if has_env
        SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0; env_ids=[hh], σ2_env=[0.5])
    else
        SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0)
    end
end

# =====================================================================
# TEST 1: Dense BLAS dgemv thread-safety  (mul!(y, X, v) and mul!(y, X', v))
# =====================================================================
println("\n--- TEST 1: Dense BLAS mul! thread-safety ---")
let
    nt = min(Threads.nthreads(), 8)
    x_vecs = [randn(nf) for _ in 1:nt]
    u_vecs = [randn(n) for _ in 1:nt]
    # Reference: sequential
    y_ref = [X * x_vecs[t] for t in 1:nt]
    xt_ref = [X' * u_vecs[t] for t in 1:nt]
    # Threaded
    y_thr = [similar(y_ref[1]) for _ in 1:nt]
    xt_thr = [similar(xt_ref[1]) for _ in 1:nt]

    max_err_fwd = 0.0
    max_err_adj = 0.0
    for trial in 1:50
        Threads.@threads :static for t in 1:nt
            mul!(y_thr[t], X, x_vecs[t])
            mul!(xt_thr[t], X', u_vecs[t])
        end
        for t in 1:nt
            max_err_fwd = max(max_err_fwd, norm(y_thr[t] - y_ref[t]) / max(norm(y_ref[t]), 1e-10))
            max_err_adj = max(max_err_adj, norm(xt_thr[t] - xt_ref[t]) / max(norm(xt_ref[t]), 1e-10))
        end
    end
    check_test("mul!(y, X, v)  max_err = $(Printf.@sprintf("%.2e", max_err_fwd))", max_err_fwd < 1e-10)
    check_test("mul!(y, X', v) max_err = $(Printf.@sprintf("%.2e", max_err_adj))", max_err_adj < 1e-10)
end

# =====================================================================
# TEST 2: Sparse mul! thread-safety  (mul!(y, Ainv, v))
# =====================================================================
println("\n--- TEST 2: Sparse mul!(y, Ainv, v) thread-safety ---")
let
    nt = min(Threads.nthreads(), 8)
    v_vecs = [randn(N_ped) for _ in 1:nt]
    ref = [Ainv * v_vecs[t] for t in 1:nt]
    out = [similar(ref[1]) for _ in 1:nt]
    max_err = 0.0
    for trial in 1:50
        Threads.@threads :static for t in 1:nt
            mul!(out[t], Ainv, v_vecs[t])
        end
        for t in 1:nt
            max_err = max(max_err, norm(out[t] - ref[t]) / max(norm(ref[t]), 1e-10))
        end
    end
    check_test("mul!(y, Ainv, v) max_err = $(Printf.@sprintf("%.2e", max_err))", max_err < 1e-10)
end

# =====================================================================
# TEST 3: Sparse ldiv! thread-safety  (ldiv!(L_Ainv, v) and ldiv!(L_Ainv', v))
# =====================================================================
println("\n--- TEST 3: Sparse ldiv! thread-safety ---")
let
    nt = min(Threads.nthreads(), 8)
    v_vecs = [randn(N_ped) for _ in 1:nt]
    # Reference
    ref_fwd = [let v=copy(v_vecs[t]); ldiv!(L_Ainv, v); v end for t in 1:nt]
    ref_adj = [let v=copy(v_vecs[t]); ldiv!(L_Ainv', v); v end for t in 1:nt]
    max_err_fwd = 0.0
    max_err_adj = 0.0
    for trial in 1:50
        bufs_fwd = [copy(v_vecs[t]) for t in 1:nt]
        bufs_adj = [copy(v_vecs[t]) for t in 1:nt]
        Threads.@threads :static for t in 1:nt
            ldiv!(L_Ainv, bufs_fwd[t])
            ldiv!(L_Ainv', bufs_adj[t])
        end
        for t in 1:nt
            max_err_fwd = max(max_err_fwd, norm(bufs_fwd[t] - ref_fwd[t]) / max(norm(ref_fwd[t]), 1e-10))
            max_err_adj = max(max_err_adj, norm(bufs_adj[t] - ref_adj[t]) / max(norm(ref_adj[t]), 1e-10))
        end
    end
    check_test("ldiv!(L, v)  max_err = $(Printf.@sprintf("%.2e", max_err_fwd))", max_err_fwd < 1e-10)
    check_test("ldiv!(L', v) max_err = $(Printf.@sprintf("%.2e", max_err_adj))", max_err_adj < 1e-10)
end

# =====================================================================
# TEST 4: MME Operator thread-safety
#   Apply the operator from multiple threads simultaneously, compare to sequential
# =====================================================================
println("\n--- TEST 4: MME Operator thread-safety ---")
let
    model = make_model()
    dim = get_dim(model)
    nt = min(Threads.nthreads(), 8)
    max_tid = Threads.maxthreadid()
    workspaces = create_thread_workspaces(model, max_tid)

    # Create nt different input vectors
    Random.seed!(123)
    v_vecs = [randn(dim) for _ in 1:nt]

    # Sequential reference: create ONE workspace, apply operator nt times
    ws_seq = model_workspace(model)
    op_seq = make_mme_operator(model, ws_seq)
    refs = [zeros(dim) for _ in 1:nt]
    for t in 1:nt
        op_seq(refs[t], v_vecs[t])
    end

    # Threaded
    outs = [zeros(dim) for _ in 1:nt]
    max_err = 0.0
    for trial in 1:20
        Threads.@threads :static for t in 1:nt
            tid = Threads.threadid()
            op_t = make_mme_operator(model, workspaces[tid])
            op_t(outs[t], v_vecs[t])
        end
        for t in 1:nt
            max_err = max(max_err, norm(outs[t] - refs[t]) / max(norm(refs[t]), 1e-10))
        end
    end
    check_test("MME operator max_err = $(Printf.@sprintf("%.2e", max_err))", max_err < 1e-10)
end

# =====================================================================
# TEST 5: Preconditioner ldiv! thread-safety
# =====================================================================
println("\n--- TEST 5: BlockPreconditioner ldiv! thread-safety ---")
let
    model = make_model()
    dim = get_dim(model)
    nt = min(Threads.nthreads(), 8)
    max_tid = Threads.maxthreadid()
    workspaces = create_thread_workspaces(model, max_tid)
    preconds = [get_preconditioner(model, ws) for ws in workspaces]

    Random.seed!(456)
    r_vecs = [randn(dim) for _ in 1:nt]

    # Sequential reference
    ws_seq = model_workspace(model)
    P_seq = get_preconditioner(model, ws_seq)
    refs = [zeros(dim) for _ in 1:nt]
    for t in 1:nt
        ldiv!(refs[t], P_seq, r_vecs[t])
    end

    # Threaded
    outs = [zeros(dim) for _ in 1:nt]
    max_err = 0.0
    for trial in 1:20
        Threads.@threads :static for t in 1:nt
            tid = Threads.threadid()
            ldiv!(outs[t], preconds[tid], r_vecs[t])
        end
        for t in 1:nt
            max_err = max(max_err, norm(outs[t] - refs[t]) / max(norm(refs[t]), 1e-10))
        end
    end
    check_test("BlockPreconditioner max_err = $(Printf.@sprintf("%.2e", max_err))", max_err < 1e-10)
end

# =====================================================================
# TEST 6: Parallel CG z-solves — the critical test
#   Same model, same starting point, same z-vectors.
#   Compare k z-solve results between sequential and threaded.
# =====================================================================
println("\n--- TEST 6: Parallel CG z-solves ---")
let
    k_test = min(Threads.nthreads(), 8)
    cg_reltol = 1e-7
    cg_maxiter = 1000

    # Shared y-solve (sequential)
    model = make_model()
    dim = get_dim(model)
    ws_main = model_workspace(model)
    P_main = get_preconditioner(model, ws_main)
    solve_mme_y!(model, ws_main, P_main, reltol=cg_reltol, maxiter=cg_maxiter, verbose=false)

    Random.seed!(42)
    z_list = [randn(n) for _ in 1:k_test]

    # --- Sequential z-solves ---
    sol_z_seq = [zeros(dim) for _ in 1:k_test]
    Pz_seq = [Vector{Float64}(undef, n) for _ in 1:k_test]
    for i in 1:k_test
        solve_mme_z!(sol_z_seq[i], model, z_list[i], ws_main, P_main,
            reltol=cg_reltol, maxiter=cg_maxiter)
        compute_Px_from_sol!(Pz_seq[i], model, z_list[i], sol_z_seq[i])
    end

    # --- Threaded z-solves (fresh workspaces, same model) ---
    max_tid = Threads.maxthreadid()
    workspaces = create_thread_workspaces(model, max_tid)
    preconds = [get_preconditioner(model, ws) for ws in workspaces]
    sol_z_thr = [zeros(dim) for _ in 1:k_test]
    Pz_thr = [Vector{Float64}(undef, n) for _ in 1:k_test]

    Threads.@threads :static for i in 1:k_test
        tid = Threads.threadid()
        solve_mme_z!(sol_z_thr[i], model, z_list[i], workspaces[tid], preconds[tid],
            reltol=cg_reltol, maxiter=cg_maxiter)
        compute_Px_from_sol!(Pz_thr[i], model, z_list[i], sol_z_thr[i])
    end

    # Compare
    println("  Per-vector comparison (k=$k_test):")
    max_sol_err = 0.0
    max_Pz_err = 0.0
    for i in 1:k_test
        sol_err = norm(sol_z_seq[i] - sol_z_thr[i]) / max(norm(sol_z_seq[i]), 1e-10)
        Pz_err = norm(Pz_seq[i] - Pz_thr[i]) / max(norm(Pz_seq[i]), 1e-10)
        max_sol_err = max(max_sol_err, sol_err)
        max_Pz_err = max(max_Pz_err, Pz_err)
        status = sol_err < 1e-4 ? "✓" : "✗"
        @printf("    z[%d]: sol_err=%.2e  Pz_err=%.2e  %s\n", i, sol_err, Pz_err, status)
    end
    check_test("CG sol max_err = $(Printf.@sprintf("%.2e", max_sol_err))", max_sol_err < 1e-4)
    check_test("Pz    max_err = $(Printf.@sprintf("%.2e", max_Pz_err))", max_Pz_err < 1e-4)
end

# =====================================================================
# TEST 7: Sequential CG from THREAD workspaces (no actual parallelism)
#   If this fails, the issue is workspace creation, not threading per se.
# =====================================================================
println("\n--- TEST 7: Sequential CG using thread-workspaces ---")
let
    model = make_model()
    dim = get_dim(model)
    ws_main = model_workspace(model)
    P_main = get_preconditioner(model, ws_main)
    solve_mme_y!(model, ws_main, P_main, reltol=1e-7, maxiter=1000, verbose=false)

    Random.seed!(42)
    z_list = [randn(n) for _ in 1:4]

    # Reference: model workspace
    sol_ref = [zeros(dim) for _ in 1:4]
    for i in 1:4
        solve_mme_z!(sol_ref[i], model, z_list[i], ws_main, P_main,
            reltol=1e-7, maxiter=1000)
    end

    # Test: use thread workspace[1] sequentially
    max_tid = Threads.maxthreadid()
    workspaces = create_thread_workspaces(model, max_tid)
    P_thr = get_preconditioner(model, workspaces[1])
    sol_thr = [zeros(dim) for _ in 1:4]
    for i in 1:4
        solve_mme_z!(sol_thr[i], model, z_list[i], workspaces[1], P_thr,
            reltol=1e-7, maxiter=1000)
    end

    max_err = maximum(norm(sol_ref[i] - sol_thr[i]) / max(norm(sol_ref[i]), 1e-10) for i in 1:4)
    check_test("Thread-workspace sequential max_err = $(Printf.@sprintf("%.2e", max_err))", max_err < 1e-4)
end

# =====================================================================
# SUMMARY
# =====================================================================
println("\n" * "="^70)
println("SUMMARY: $PASS passed, $FAIL failed")
println("="^70)
if FAIL == 0
    println("All low-level operations are thread-safe.")
    println("The bug may be in IterativeSolvers.cg! or in how results are assembled.")
else
    println("Thread-safety issues found! Fix the failing operations above.")
end
