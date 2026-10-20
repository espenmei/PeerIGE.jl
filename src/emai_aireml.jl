# =====================================================================
# AI-REML DRIVER
# Score and Average Information assembly, parameter updates, the result
# struct, and the run_aireml iteration loop. Included via emai.jl.
# =====================================================================
# Returnerer varianstype-representasjon for gjeldende modell (d, s, ds, e, env_h)
function get_variance_components(model::SGEModel)
    comps = VarianceComponent[
        VarianceComponent(VC_D, 0),
        VarianceComponent(VC_S, 0),
        VarianceComponent(VC_DS, 0),
        VarianceComponent(VC_E, 0)
    ]
    for h in 1:length(model.env_ids)
        push!(comps, VarianceComponent(VC_ENV, h))
    end
    if model.peer_env
        push!(comps, VarianceComponent(VC_PE, 0))
    end
    return comps
end

function get_param_names(model::SGEModel)
    names = ["σ²_d", "σ²_s", "σ_ds", "σ²_e"]
    for h in 1:length(model.env_ids)
        push!(names, "σ²_env$h")
    end
    model.peer_env && push!(names, "σ²_pe")
    return names
end

function compute_score_and_ai(model::SGEModel, Py, z_list, Pz_list, workspaces, preconditioners;
    ai_reltol=1e-5, ai_maxiter=500)
    comps = get_variance_components(model)
    n_p = length(comps)
    score = zeros(n_p)
    AI = zeros(n_p, n_p)
    k = length(Pz_list)
    n = length(Py)

    # 1. Forhåndsberegn alle Vi * Py
    println("    -> Beregner Vi * Py...")
    Vi_Py_list = [Vector{Float64}(undef, n) for _ in 1:n_p]
    _threaded_vi_py!(Vi_Py_list, Py, comps, model, workspaces)

    # 2. SCORE: s_i = 0.5 * (y'P * Vi * Py - tr(P * Vi))
    #    Trace-estimering: parallelliser k get_Vi_u-kall per parameter
    println("    -> Beregner trace (k=$k)...")
    n_trace = n_p * k
    trace_results = zeros(n_trace)
    _threaded_trace!(trace_results, n_trace, k, Pz_list, z_list, comps, model, workspaces)
    for i in 1:n_p
        tr_val = sum(@view trace_results[(i-1)*k+1 : i*k])
        score[i] = 0.5 * (Py' * Vi_Py_list[i] - tr_val)
    end

    # 3. AI-MATRISE (Average Information)
    #    Trenger bare n_p CG-solves (én per V_j).
    #    AI[i,j] = 0.5 * Py' * Vi * P * Vj * Py
    println("    -> Beregner AI-matrise ($n_p CG-solves)...")

    # Steg 1: Beregn P * V_j * Py
    P_Vi_Py_list = [Vector{Float64}(undef, n) for _ in 1:n_p]
    _threaded_ai_cg!(P_Vi_Py_list, Vi_Py_list, model, workspaces, preconditioners, ai_reltol, ai_maxiter)

    # Steg 2: Beregn AI[i,j] = 0.5 * Py' * Vi * P * Vj * Py  (billige mat-vec)
    pairs = [(i, j) for i in 1:n_p for j in 1:i]
    ai_vals = zeros(length(pairs))
    _threaded_ai_matvec!(ai_vals, pairs, P_Vi_Py_list, Py, comps, model, workspaces)
    for (idx, (i_idx, j_idx)) in enumerate(pairs)
        AI[i_idx, j_idx] = ai_vals[idx]
        AI[j_idx, i_idx] = ai_vals[idx]
    end
    return score, AI
end

# --- Helper functions: each wraps a single @threads block to isolate closures ---

function _threaded_vi_py!(Vi_Py_list, Py, comps, model, workspaces)
    n_p = length(comps)
    Threads.@threads :static for i in 1:n_p
        tid = Threads.threadid()
        get_Vi_u!(Vi_Py_list[i], Py, comps[i], model, workspaces[tid])
    end
end

function _threaded_trace!(trace_results, n_trace, k, Pz_list, z_list, comps, model, workspaces)
    Threads.@threads :static for flat_idx in 1:n_trace
        tid = Threads.threadid()
        ws = workspaces[tid]
        i = (flat_idx - 1) ÷ k + 1
        j = (flat_idx - 1) % k + 1
        get_Vi_u!(ws.vi_out, Pz_list[j], comps[i], model, ws)
        trace_results[flat_idx] = dot(z_list[j], ws.vi_out) / k
    end
end

function _threaded_ai_cg!(P_Vi_Py_list, Vi_Py_list, model, workspaces, preconditioners, ai_reltol, ai_maxiter)
    n_p = length(Vi_Py_list)
    tmp_sols = [copy(model.sol_y) for _ in 1:n_p]
    Threads.@threads :static for j in 1:n_p
        tid = Threads.threadid()
        ws = workspaces[tid]
        solve_mme_direct!(tmp_sols[j], model, Vi_Py_list[j], ws, preconditioners[tid],
            reltol=ai_reltol, maxiter=ai_maxiter)
        compute_Px_from_sol!(P_Vi_Py_list[j], model, Vi_Py_list[j], tmp_sols[j], ws)
    end
end

function _threaded_ai_matvec!(ai_vals, pairs, P_Vi_Py_list, Py, comps, model, workspaces)
    Threads.@threads :static for idx in eachindex(pairs)
        tid = Threads.threadid()
        ws = workspaces[tid]
        i_idx, j_idx = pairs[idx]
        get_Vi_u!(ws.vi_out, P_Vi_Py_list[j_idx], comps[i_idx], model, ws)
        ai_vals[idx] = 0.5 * dot(Py, ws.vi_out)
    end
end

function _threaded_z_solves!(model, z_fixed_list, Pz_list, workspaces, preconditioners,
    k, cg_reltol, cg_maxiter)
    Threads.@threads :static for i in 1:k
        tid = Threads.threadid()
        ws = workspaces[tid]
        solve_mme_z!(model.sol_z[i], model, z_fixed_list[i], ws, preconditioners[tid],
            reltol=cg_reltol, maxiter=cg_maxiter)
        compute_Px_from_sol!(Pz_list[i], model, z_fixed_list[i], model.sol_z[i], ws)
    end
end

# =====================================================================
# RESULTAT-STRUCT OG LAGRING (JLD2)
# =====================================================================

struct AiRemlResult
    # Varianskomponenter
    σ2_d::Float64
    σ2_s::Float64
    σ_ds::Float64
    σ2_e::Float64
    σ2_env::Vector{Float64}        # H env variances (empty if H=0)
    # BLUE og BLUP (fra sol_y)
    beta::Vector{Float64}      # Faste effekter (nf)
    a_d::Vector{Float64}       # Direkte genetiske effekter (N)
    a_s::Vector{Float64}       # Sosiale genetiske effekter (N)
    c_env::Vector{Vector{Float64}} # Env BLUPs (H vectors, each n_h)
    # Inferens
    AI::Matrix{Float64}        # Average Information matrise (n_p × n_p)
    se::Vector{Float64}        # Standardfeil (n_p)
    # Metadata
    converged::Bool
    n_iter::Int
    n_obs::Int                 # n (antall observasjoner)
    N_pedigree::Int            # N (antall i stamtavlen)
    k::Int                     # Antall Hutchinson-prober
    # Mapping: obs → stamtavle-ID (for å koble BLUP til individer)
    obs_to_id::Vector{Int}
    ψ::Float64                 # Dilution exponent used for this fit
    # Social peer environmental effect (σ²_pe = 0 and empty vectors when peer_env=false)
    peer_env::Bool
    σ2_pe::Float64
    p_env::Vector{Float64}     # Peer env BLUPs, one per roster member
    pe_ids::Vector{Int}        # Pedigree ID for each entry of p_env
end

function save_results(result::AiRemlResult, filename::String)
    jldsave(filename; result=result)
    sz_mb = filesize(filename) / 1024^2
    @printf("Resultat lagret til %s (%.1f MB)\n", filename, sz_mb)
end

function load_results(filename::String)
    return load(filename, "result")
end

function update_params!(model::SGEModel, delta, damping)
    # Foreslåtte endringer
    new_d2 = model.σ2_d + damping * delta[1]
    new_s2 = model.σ2_s + damping * delta[2]
    new_ds = model.σ_ds + damping * delta[3]
    new_e2 = model.σ2_e + damping * delta[4]
    # Sikkerhetsmekanismer (Bending)
    model.σ2_d = max(1e-4, new_d2)
    model.σ2_s = max(1e-6, new_s2)
    model.σ2_e = max(1e-2, new_e2)
    # Hold korrelasjonen i sjakk: |rho| < 0.99
    max_cov = 0.99 * sqrt(model.σ2_d * model.σ2_s)
    model.σ_ds = clamp(new_ds, -max_cov, max_cov)
    # Env variances
    for h in 1:length(model.σ2_env)
        model.σ2_env[h] = max(1e-6, model.σ2_env[h] + damping * delta[4+h])
    end
    if model.peer_env
        model.σ2_pe = max(1e-6, model.σ2_pe + damping * delta[pe_param_index(model)])
    end
end

# Draw k fixed Gaussian probe vectors z_j ∈ ℝⁿ for the Hutchinson trace
# estimator tr(P*V_i) ≈ (1/k) Σ_j z_j' (V_i (P z_j)). Seeding here keeps the
# probes deterministic and identical across every iteration of run_aireml,
# which is required for the trace estimate to be comparable between iterations.
function make_trace_probes(n::Int, k::Int; seed::Int=42)
    Random.seed!(seed)
    return [randn(n) for _ in 1:k]
end

function run_aireml(model::SGEModel; max_iter=15, k=5, damping=0.2,
    cg_maxiter=2000, cg_reltol=1e-7,
    cg_reltol_early=1e-5, cg_early_iters=3,
    ai_maxiter=500, ai_reltol=1e-5,
    use_block_precond=true, verbose_cg=true,
    log_file::String="",
    seed::Int=42)
    n = length(model.y)
    N = size(model.Ainv, 1)
    dim = length(model.sol_y)
    H = length(model.env_ids)
    param_names = get_param_names(model)
    n_p = length(param_names)

    nt = Threads.nthreads()
    max_tid = Threads.maxthreadid()   # Julia 1.12: threadid() kan returnere > nthreads()

    # Allokere sol_z til riktig størrelse for denne kjøringen
    model.sol_z = [zeros(dim) for _ in 1:k]

    # Vi må bruke de SAMME z-vektorene i hver iterasjon
    # for at Hutchinson-estimatet skal være sammenlignbart (se make_trace_probes)
    z_fixed_list = make_trace_probes(n, k; seed=seed)

    # Preallokkerte buffere for Px-beregninger (gjenbrukes per iterasjon)
    Py = Vector{Float64}(undef, n)
    Pz_list = [Vector{Float64}(undef, n) for _ in 1:k]

    # Workspaces for CG-solves og trace-estimering: ett per tråd (indeksert med threadid())
    workspaces = create_thread_workspaces(model, max_tid)

    precond_name = use_block_precond ? "Block" : "Diag"
    println("Starter AI-REML: k=$k, damping=$damping, precond=$precond_name, cg_maxiter=$cg_maxiter, threads=$nt")
    println("CG reltol schedule: early=$cg_reltol_early (iter <= $cg_early_iters), late=$cg_reltol")
    println("N=$( size(model.Ainv,1)), n=$n, dim=$dim")
    println("-"^70)
    flush(stdout)

    # CSV-logg for ekstern overvåking (tail -f log.csv)
    use_log = !isempty(log_file)
    if use_log
        open(log_file, "w") do io
            pe_cols = model.peer_env ? ",s2pe" : ""
            pe_se_cols = model.peer_env ? ",se_pe" : ""
            param_cols = join(["s2d","s2s","sds","s2e", ["s2env$h" for h in 1:H]...], ",") * pe_cols
            se_cols = join(["se_d","se_s","se_ds","se_e", ["se_env$h" for h in 1:H]...], ",") * pe_se_cols
            println(io, "iter,time_y,time_z,time_ai,time_total,cum_time,eta_min,",
                param_cols, ",", se_cols, ",",
                "score_norm,conv_crit,alpha,rho_ds,cond_G,converged")
        end
        println("Iterasjonslogg skrives til: $log_file")
    end

    # Variabler som trengs etter loopen (for AiRemlResult)
    AI = zeros(n_p, n_p)
    conv_crit = Inf
    last_iter = 0
    iter_times = Float64[]   # for ETA-beregning
    t_start_total = time()

    for iter in 1:max_iter
        last_iter = iter
        t_iter_start = time()
        iter_cg_reltol = iter <= cg_early_iters ? cg_reltol_early : cg_reltol

        # A. Oppdater preconditionere basert på nye varianser
        if use_block_precond
            preconditioners = [get_preconditioner(model, ws) for ws in workspaces]
        else
            preconditioners = fill(get_diag_preconditioner(model), length(workspaces))
        end

        # 1. Oppdater modellens egne løsninger (Warm Start) — y-solve er sekvensiell
        t_y = @elapsed begin
            solve_mme_y!(model, workspaces[1], preconditioners[1], reltol=iter_cg_reltol, maxiter=cg_maxiter, verbose=verbose_cg)
            compute_Px_from_sol!(Py, model, model.y, model.sol_y, workspaces[1])
        end

        # 2. Parallelliser k z-solves
        if verbose_cg
            @printf("    Z-solves: k=%d (threaded)...\n", k)
            flush(stdout)
        end
        t_z = @elapsed begin
            _threaded_z_solves!(model, z_fixed_list, Pz_list, workspaces, preconditioners,
                k, iter_cg_reltol, cg_maxiter)
        end
        if verbose_cg
            @printf("    Z-solves ferdig: %.1fs\n", t_z)
            flush(stdout)
        end

        # 3. Beregn Score og AI
        if verbose_cg
            println("    Score + AI...")
            flush(stdout)
        end
        t_ai = @elapsed begin
            score, AI_new = compute_score_and_ai(model, Py, z_fixed_list, Pz_list,
                workspaces, preconditioners,
                ai_reltol=ai_reltol, ai_maxiter=ai_maxiter)
            copyto!(AI, AI_new)
        end
        if verbose_cg
            @printf("    Score + AI ferdig: %.1fs\n", t_ai)
            flush(stdout)
        end

        # 4. Løs systemet og beregn Standardfeil (SE)
        AI_reg = AI + 1e-6 * I
        delta = AI_reg \ score
        AI_inv = inv(AI_reg)
        se = sqrt.(abs.(diag(AI_inv)))

        # 5. Lagre gamle verdier for utskrift
        old_params = get_params(model)

        # 6. Step-halving: finn steglengde som holder alle varianser > 0 og G positiv definitt
        α = damping
        for _attempt in 1:20
            new_d = model.σ2_d + α * delta[1]
            new_s = model.σ2_s + α * delta[2]
            new_ds = model.σ_ds + α * delta[3]
            new_e = model.σ2_e + α * delta[4]
            env_ok = all(model.σ2_env[h] + α * delta[4+h] > 1e-8 for h in 1:H)
            pe_ok = !model.peer_env || model.σ2_pe + α * delta[pe_param_index(model)] > 1e-8
            if new_d > 1e-6 && new_s > 1e-8 && new_e > 1e-4 && new_d * new_s > new_ds^2 && env_ok && pe_ok
                break
            end
            α *= 0.5
        end

        # 7. Oppdater parametere med safe steglengde
        update_params!(model, delta, α)
        new_params = get_params(model)
        actual_change = new_params .- old_params

        # --- UTSKRIFT ---
        t_iter = time() - t_iter_start
        push!(iter_times, t_iter)
        cum_time = time() - t_start_total
        avg_iter = mean(iter_times)
        remaining_iters = max_iter - iter
        eta_min = remaining_iters * avg_iter / 60.0

        # G-matrise helse
        rho_ds = model.σ_ds / sqrt(model.σ2_d * model.σ2_s)
        G_mat = [model.σ2_d model.σ_ds; model.σ_ds model.σ2_s]
        eig_G = eigvals(G_mat)
        cond_G = maximum(eig_G) / max(minimum(eig_G), 1e-12)

        score_norm = norm(score)

        println("\n" * "="^70)
        @printf("ITERASJON %d  (α=%.3f)  |  %.1fs (y:%.1f z:%.1f ai:%.1f)  cum:%.0fs  ETA:%.0fmin\n",
            iter, α, t_iter, t_y, t_z, t_ai, cum_time, eta_min)
        println("="^70)
        @printf("CG reltol this iter: %.1e\n", iter_cg_reltol)
        @printf("%-10s %-12s %-12s %-12s %-12s\n", "Param", "Est", "SE", "Score", "Delta")
        println("-"^70)
        for i in 1:n_p
            @printf("%-10s %-12.6f %-12.6f %-+12.4f %-+12.6f\n",
                param_names[i], new_params[i], se[i], score[i], delta[i])
        end
        println("-"^70)
        @printf("G: ρ_ds=%.4f  cond(G)=%.1f  eig=[%.4f, %.4f]",
            rho_ds, cond_G, eig_G[1], eig_G[2])
        if abs(rho_ds) > 0.95
            print("  ⚠ ρ nær grense!")
        end
        if cond_G > 1e4
            print("  ⚠ dårlig kondisjonert!")
        end
        println()
        if model.peer_env
            # Sampling-korrelasjoner fra AI⁻¹: σ²_pe er nært konfundert med σ²_s og gruppe-env
            ip = pe_param_index(model)
            print("corr(σ²_pe, ·):")
            for i in 1:n_p
                i == ip && continue
                r = AI_inv[ip, i] / sqrt(abs(AI_inv[ip, ip] * AI_inv[i, i]))
                @printf("  %s=%+.2f", param_names[i], r)
            end
            println()
        end

        # Konvergens-sjekk
        conv_crit = norm(actual_change) / norm(old_params)
        @printf("|score|=%.4e  Δparam=%.6f", score_norm, conv_crit)
        flush(stdout)

        # CSV-logg (med flush for ekstern overvåking)
        if use_log
            open(log_file, "a") do io
                @printf(io, "%d,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,",
                    iter, t_y, t_z, t_ai, t_iter, cum_time, eta_min)
                for p in new_params; @printf(io, "%.6f,", p); end
                for s in se; @printf(io, "%.6f,", s); end
                @printf(io, "%.6e,%.6e,%.4f,%.4f,%.1f,%s\n",
                    score_norm, conv_crit, α, rho_ds, cond_G,
                    conv_crit < 1e-5 ? "true" : "false")
                flush(io)
            end
        end

        if conv_crit < 1e-5
            println("\n\n✅ AI-REML har konvergert 😄.")
            break
        end
        println()
        if iter == max_iter
            println("\n⚠️ Maks iterasjoner nådd uten full konvergens.")
        end
    end

    # Final solve med endelige varianskomponenter → BLUE(β) og BLUP(a_d, a_s)
    println("\nBeregner endelige BLUE/BLUP-estimater...")
    if use_block_precond
        final_precond = get_preconditioner(model, workspaces[1])
    else
        final_precond = get_diag_preconditioner(model)
    end
    fill!(model.sol_y, 0.0)
    solve_mme_y!(model, workspaces[1], final_precond, reltol=cg_reltol, maxiter=cg_maxiter, verbose=verbose_cg)

    # Pakk ut BLUE/BLUP fra løsningsvektoren
    lay = model.layout
    beta = model.sol_y[lay.fixed]
    a_d = model.sol_y[lay.direct]
    a_s = model.sol_y[lay.social]

    # Env BLUPs
    c_env = Vector{Float64}[]
    for h in 1:H
        push!(c_env, model.sol_y[lay.env[h]])
    end
    p_env = model.sol_y[lay.pe]

    # Beregn endelig AI og SE
    AI_final = AI + 1e-6 * I
    se_final = sqrt.(abs.(diag(inv(AI_final))))

    result = AiRemlResult(
        model.σ2_d, model.σ2_s, model.σ_ds, model.σ2_e, copy(model.σ2_env),
        beta, a_d, a_s, c_env,
        AI, se_final,
        conv_crit < 1e-5, last_iter, n, N, k,
        copy(model.groups.obs_to_id), model.ψ,
        model.peer_env, model.peer_env ? model.σ2_pe : 0.0, p_env, copy(model.pe_ids)
    )
    return result
end
