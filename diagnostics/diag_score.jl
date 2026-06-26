include(joinpath(@__DIR__, "..", "src", "emai.jl"))

# Focused diagnostic for compute_score_and_ai:
# 1. Checks if workspace creation affects sequential results
# 2. Checks which sub-step of score/AI diverges under threading:
#    a) Vi * Py
#    b) Trace estimation
#    c) AI CG solves
#    d) AI mat-vec products

println("="^70)
println("COMPUTE_SCORE_AND_AI THREADING DIAGNOSTIC")
println("Julia threads: $(Threads.nthreads())")
println("="^70)

# --- Load data ---
obs_table = DataFrame(CSV.File("yX_obs.csv"))
roster = DataFrame(CSV.File("roster.csv"))
obs = load_observed_data(obs_table; covariate_cols=[:x])
X = obs.X
y = obs.y
n = length(y); nf = size(X, 2)

Ainv_df = DataFrame(CSV.File("Ainv_triplet.csv"))
i_vals = Ainv_df[:, "i"]; j_vals = Ainv_df[:, "j"]; x_vals = Float64.(Ainv_df[:, "x"])
off_diag = i_vals .!= j_vals
Ainv = sparse(vcat(i_vals, j_vals[off_diag]), vcat(j_vals, i_vals[off_diag]), vcat(x_vals, x_vals[off_diag]))
N_ped = size(Ainv, 1)

L_Ainv_df = DataFrame(CSV.File("LAinv_triplet.csv"))
L_Ainv = LowerTriangular(sparse(L_Ainv_df[:, "i"], L_Ainv_df[:, "j"], Float64.(L_Ainv_df[:, "x"])))

gs = build_group_structure(obs_table, roster, N_ped)
has_env = "household" in names(obs_table)

# Use H=0 model (simpler, fewer confounders)
model = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.1, 0.1, 2.0)
k = 10; dim = get_dim(model)
max_tid = Threads.maxthreadid()

# Generate z-vectors
Random.seed!(42)
z_fixed_list = [randn(n) for _ in 1:k]

# --- Prepare: run y-solve and z-solves sequentially (results are identical regardless) ---
model.sol_y = zeros(dim)
model.sol_z = [zeros(dim) for _ in 1:k]
ws_seq = model_workspace(model)
P_seq = get_preconditioner(model, ws_seq)
solve_mme_y!(model, ws_seq, P_seq, reltol=1e-7, maxiter=1000, verbose=false)
Py = Vector{Float64}(undef, n)
compute_Px_from_sol!(Py, model, model.y, model.sol_y)
Pz_list = [Vector{Float64}(undef, n) for _ in 1:k]
for i in 1:k
    solve_mme_z!(model.sol_z[i], model, z_fixed_list[i], ws_seq, P_seq, reltol=1e-7, maxiter=1000)
    compute_Px_from_sol!(Pz_list[i], model, z_fixed_list[i], model.sol_z[i])
end
println("Py norm = $(norm(Py))")
println("Pz[1] norm = $(norm(Pz_list[1]))")

types = get_variance_types(model)
n_p = length(types)
println("types = $types, n_p = $n_p")
println()

# =====================================================================
# TEST 1: Does workspace type matter for SEQUENTIAL score/AI?
# =====================================================================
println("="^70)
println("TEST 1: Sequential score with model_workspace vs fresh workspace")
println("="^70)

# 1a: model_workspace (aliases model buffers)
ws_a = [model_workspace(model)]
P_a = [get_preconditioner(model, ws_a[1])]
score_a, AI_a = compute_score_and_ai(model, Py, z_fixed_list, Pz_list, ws_a, P_a, ai_reltol=1e-5, ai_maxiter=300)

# 1b: single fresh workspace (not aliased)
ws_b = [create_thread_workspaces(model, 1)[1]]
P_b = [get_preconditioner(model, ws_b[1])]
score_b, AI_b = compute_score_and_ai(model, Py, z_fixed_list, Pz_list, ws_b, P_b, ai_reltol=1e-5, ai_maxiter=300)

println("Score (model_ws): ", [@sprintf("%+.6e", s) for s in score_a])
println("Score (fresh_ws): ", [@sprintf("%+.6e", s) for s in score_b])
println("max|Δ score| = $(maximum(abs.(score_a - score_b)))")
println("max|Δ AI|    = $(maximum(abs.(AI_a - AI_b)))")
println()

# =====================================================================
# TEST 2: Thread workspaces, but forced sequential (length=1 trick)
# =====================================================================
println("="^70)
println("TEST 2: Thread workspace[1] only (forced sequential)")
println("="^70)

all_ws = create_thread_workspaces(model, max_tid)
all_P  = [get_preconditioner(model, ws) for ws in all_ws]

# Pass only workspace[1] → use_threads=false inside compute_score_and_ai
score_c, AI_c = compute_score_and_ai(model, Py, z_fixed_list, Pz_list, [all_ws[1]], [all_P[1]], ai_reltol=1e-5, ai_maxiter=300)
println("Score (forced_seq): ", [@sprintf("%+.6e", s) for s in score_c])
println("vs model_ws:  max|Δ score| = $(maximum(abs.(score_a - score_c)))")
println()

# =====================================================================
# TEST 3: Full threaded
# =====================================================================
println("="^70)
println("TEST 3: Full threaded")
println("="^70)

score_d, AI_d = compute_score_and_ai(model, Py, z_fixed_list, Pz_list, all_ws, all_P, ai_reltol=1e-5, ai_maxiter=300)
println("Score (threaded):   ", [@sprintf("%+.6e", s) for s in score_d])
println("vs model_ws:  max|Δ score| = $(maximum(abs.(score_a - score_d)))")
println("vs forced_seq: max|Δ score| = $(maximum(abs.(score_c - score_d)))")
println()

# =====================================================================
# TEST 4: Isolate sub-steps — Vi*Py, trace, AI
#          Do each sub-step manually, comparing seq vs threaded
# =====================================================================
println("="^70)
println("TEST 4: Isolate sub-steps")
println("="^70)

# --- 4a: Vi * Py ---
println("\n--- 4a: Vi * Py ---")
Vi_Py_seq = [Vector{Float64}(undef, n) for _ in 1:n_p]
for i in 1:n_p
    get_Vi_u!(Vi_Py_seq[i], Py, types[i], model, all_ws[1])
end

Vi_Py_thr = [Vector{Float64}(undef, n) for _ in 1:n_p]
Threads.@threads :static for i in 1:n_p
    tid = Threads.threadid()
    get_Vi_u!(Vi_Py_thr[i], Py, types[i], model, all_ws[tid])
end

for i in 1:n_p
    d = maximum(abs.(Vi_Py_seq[i] - Vi_Py_thr[i]))
    println("  Vi_Py[$i] ($(types[i])): seq_norm=$(norm(Vi_Py_seq[i])), thr_norm=$(norm(Vi_Py_thr[i])), max|Δ|=$d")
end
max_vi_py_diff = maximum(maximum(abs.(Vi_Py_seq[i] - Vi_Py_thr[i])) for i in 1:n_p)
println("  Overall Vi*Py max|Δ| = $max_vi_py_diff")

# --- 4b: Trace estimation ---
println("\n--- 4b: Trace estimation ---")
# Sequential trace
trace_seq = zeros(n_p)
for i in 1:n_p
    tr_val = 0.0
    for j in 1:k
        get_Vi_u!(all_ws[1].vi_out, Pz_list[j], types[i], model, all_ws[1])
        tr_val += dot(z_fixed_list[j], all_ws[1].vi_out) / k
    end
    trace_seq[i] = tr_val
end

# Threaded trace
n_trace = n_p * k
trace_results_thr = zeros(n_trace)
Threads.@threads :static for flat_idx in 1:n_trace
    tid = Threads.threadid()
    ws = all_ws[tid]
    i = (flat_idx - 1) ÷ k + 1
    j = (flat_idx - 1) % k + 1
    get_Vi_u!(ws.vi_out, Pz_list[j], types[i], model, ws)
    trace_results_thr[flat_idx] = dot(z_fixed_list[j], ws.vi_out) / k
end
trace_thr = zeros(n_p)
for i in 1:n_p
    trace_thr[i] = sum(@view trace_results_thr[(i-1)*k+1 : i*k])
end

for i in 1:n_p
    d = abs(trace_seq[i] - trace_thr[i])
    println("  trace[$i] ($(types[i])): seq=$(trace_seq[i]), thr=$(trace_thr[i]), |Δ|=$d")
end
max_trace_diff = maximum(abs.(trace_seq - trace_thr))
println("  Overall trace max|Δ| = $max_trace_diff")

# Compute score from matched components
println("\n  Score components:")
for i in 1:n_p
    yViPy = dot(Py, Vi_Py_seq[i])
    score_seq_manual = 0.5 * (yViPy - trace_seq[i])
    score_thr_manual = 0.5 * (yViPy - trace_thr[i])
    println("  [$i] ($(types[i])): Py'ViPy=$(@sprintf("%.4e", yViPy))  " *
            "score_seq=$(@sprintf("%+.4e", score_seq_manual))  " *
            "score_thr=$(@sprintf("%+.4e", score_thr_manual))")
end

# --- 4c: AI CG solves ---
println("\n--- 4c: AI CG solves (P * Vi * Py) ---")
P_Vi_Py_seq = [Vector{Float64}(undef, n) for _ in 1:n_p]
let
    tmp_sol = zeros(length(model.sol_y))
    for j in 1:n_p
        copyto!(tmp_sol, model.sol_y)
        solve_mme_direct!(tmp_sol, model, Vi_Py_seq[j], all_ws[1], all_P[1],
            reltol=1e-5, maxiter=300)
        compute_Px_from_sol!(P_Vi_Py_seq[j], model, Vi_Py_seq[j], tmp_sol)
    end
end

P_Vi_Py_thr = [Vector{Float64}(undef, n) for _ in 1:n_p]
let
    tmp_sols = [copy(model.sol_y) for _ in 1:n_p]
    Threads.@threads :static for j in 1:n_p
        tid = Threads.threadid()
        ws = all_ws[tid]
        solve_mme_direct!(tmp_sols[j], model, Vi_Py_thr[j], ws, all_P[tid],
            reltol=1e-5, maxiter=300)
        compute_Px_from_sol!(P_Vi_Py_thr[j], model, Vi_Py_thr[j], tmp_sols[j])
    end
end

for j in 1:n_p
    d = maximum(abs.(P_Vi_Py_seq[j] - P_Vi_Py_thr[j]))
    println("  P_Vi_Py[$j] ($(types[j])): seq_norm=$(norm(P_Vi_Py_seq[j])), thr_norm=$(norm(P_Vi_Py_thr[j])), max|Δ|=$d")
end

# --- 4d: AI mat-vec ---
println("\n--- 4d: AI mat-vec products ---")
pairs = [(i, j) for i in 1:n_p for j in 1:i]

AI_seq = zeros(n_p, n_p)
for (i_idx, j_idx) in pairs
    get_Vi_u!(all_ws[1].vi_out, P_Vi_Py_seq[j_idx], types[i_idx], model, all_ws[1])
    val = 0.5 * dot(Py, all_ws[1].vi_out)
    AI_seq[i_idx, j_idx] = val
    AI_seq[j_idx, i_idx] = val
end

AI_thr = zeros(n_p, n_p)
ai_vals = zeros(length(pairs))
Threads.@threads :static for idx in eachindex(pairs)
    tid = Threads.threadid()
    ws = all_ws[tid]
    i_idx, j_idx = pairs[idx]
    get_Vi_u!(ws.vi_out, P_Vi_Py_thr[j_idx], types[i_idx], model, ws)
    ai_vals[idx] = 0.5 * dot(Py, ws.vi_out)
end
for (idx, (i_idx, j_idx)) in enumerate(pairs)
    AI_thr[i_idx, j_idx] = ai_vals[idx]
    AI_thr[j_idx, i_idx] = ai_vals[idx]
end

println("  AI diagonal (seq): ", [@sprintf("%.4e", AI_seq[i,i]) for i in 1:n_p])
println("  AI diagonal (thr): ", [@sprintf("%.4e", AI_thr[i,i]) for i in 1:n_p])
println("  max|Δ AI| = $(maximum(abs.(AI_seq - AI_thr)))")

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("  Vi*Py:  max|Δ| = $max_vi_py_diff  $(max_vi_py_diff < 1e-10 ? "✓ MATCH" : "✗ DIVERGED")")
println("  Trace:  max|Δ| = $max_trace_diff  $(max_trace_diff < 1e-6 ? "✓ MATCH" : "✗ DIVERGED")")
ai_cg_diff = maximum(maximum(abs.(P_Vi_Py_seq[j] - P_Vi_Py_thr[j])) for j in 1:n_p)
println("  AI CG:  max|Δ| = $ai_cg_diff  $(ai_cg_diff < 1e-6 ? "✓ MATCH" : "✗ DIVERGED")")
ai_diff = maximum(abs.(AI_seq - AI_thr))
println("  AI mat: max|Δ| = $ai_diff  $(ai_diff < 1e-6 ? "✓ MATCH" : "✗ DIVERGED")")
