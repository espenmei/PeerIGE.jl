include(joinpath(@__DIR__, "..", "src", "emai.jl"))
using Printf, Random

obs_table = DataFrame(CSV.File("yX_obs.csv"))
roster = DataFrame(CSV.File("roster.csv"))
obs = load_observed_data(obs_table; covariate_cols=[:x])
X = obs.X
y = obs.y
n = length(y); nf = size(X, 2)
Ainv_df = DataFrame(CSV.File("Ainv_triplet.csv"))
i_v = Ainv_df[:, "i"]; j_v = Ainv_df[:, "j"]; x_v = Float64.(Ainv_df[:, "x"])
od = i_v .!= j_v
Ainv = sparse(vcat(i_v, j_v[od]), vcat(j_v, i_v[od]), vcat(x_v, x_v[od]))
N_ped = size(Ainv, 1)
L_Ainv_df = DataFrame(CSV.File("LAinv_triplet.csv"))
L_Ainv = LowerTriangular(sparse(L_Ainv_df[:, "i"], L_Ainv_df[:, "j"], Float64.(L_Ainv_df[:, "x"])))
gs = build_group_structure(obs_table, roster, N_ped)
hh = observed_env_ids(obs_table, [:household])[1]

for (label, k_val) in [("k=20", 20), ("k=10", 10)]
    m1 = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0; env_ids=[hh], σ2_env=[0.5])
    m2 = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0; env_ids=[hh], σ2_env=[0.5])
    dim = get_dim(m1)
    m1.sol_y = zeros(dim); m2.sol_y = zeros(dim)
    m1.sol_z = [zeros(dim) for _ in 1:k_val]
    m2.sol_z = [zeros(dim) for _ in 1:k_val]
    Random.seed!(42); z1 = [randn(n) for _ in 1:k_val]
    Random.seed!(42); z2 = [randn(n) for _ in 1:k_val]
    max_tid = Threads.maxthreadid()

    # Sequential
    ws_s = [model_workspace(m1)]
    ps_s = [get_preconditioner(m1, ws_s[1])]
    solve_mme_y!(m1, ws_s[1], ps_s[1], reltol=1e-7, maxiter=1000, verbose=false)
    Py_s = Vector{Float64}(undef, n)
    compute_Px_from_sol!(Py_s, m1, m1.y, m1.sol_y)
    Pz_s = [Vector{Float64}(undef, n) for _ in 1:k_val]
    for i in 1:k_val
        solve_mme_z!(m1.sol_z[i], m1, z1[i], ws_s[1], ps_s[1], reltol=1e-7, maxiter=1000)
        compute_Px_from_sol!(Pz_s[i], m1, z1[i], m1.sol_z[i])
    end
    sc_s, AI_s = compute_score_and_ai(m1, Py_s, z1, Pz_s, ws_s, ps_s)

    # Threaded
    ws_t = create_thread_workspaces(m2, max_tid)
    ps_t = [get_preconditioner(m2, ws) for ws in ws_t]
    solve_mme_y!(m2, ws_t[1], ps_t[1], reltol=1e-7, maxiter=1000, verbose=false)
    Py_t = Vector{Float64}(undef, n)
    compute_Px_from_sol!(Py_t, m2, m2.y, m2.sol_y)
    Pz_t = [Vector{Float64}(undef, n) for _ in 1:k_val]
    Threads.@threads :static for i in 1:k_val
        tid = Threads.threadid()
        solve_mme_z!(m2.sol_z[i], m2, z2[i], ws_t[tid], ps_t[tid], reltol=1e-7, maxiter=1000)
        compute_Px_from_sol!(Pz_t[i], m2, z2[i], m2.sol_z[i])
    end
    sc_t, AI_t = compute_score_and_ai(m2, Py_t, z2, Pz_t, ws_t, ps_t)

    d_Py = maximum(abs.(Py_s - Py_t))
    d_Pz = maximum(maximum(abs.(Pz_s[i] - Pz_t[i])) for i in 1:k_val)
    d_sc = maximum(abs.(sc_s - sc_t))
    d_AI = maximum(abs.(AI_s - AI_t))
    println("$label: |dPy|=$(d_Py) |dPz|=$(d_Pz) |dscore|=$(d_sc) |dAI|=$(d_AI)")
    @printf("  score seq: [%s]\n", join([@sprintf("%.2f", s) for s in sc_s], ","))
    @printf("  score thr: [%s]\n", join([@sprintf("%.2f", s) for s in sc_t], ","))
end
