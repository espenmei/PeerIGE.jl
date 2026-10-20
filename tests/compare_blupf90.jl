using CSV, DataFrames, LinearAlgebra, Printf, SparseArrays, Statistics

if !@isdefined(SGEModel)
    include(joinpath(@__DIR__, "..", "src", "emai.jl"))
end

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTPUT_DIR = joinpath(REPO_ROOT, "outputs", "blupf90_compare")
const BLUPF90_EXE = get(ENV, "BLUPF90_EXE",
    Sys.iswindows() ? raw"C:\Users\espenmei\software\blupf90+.exe" : "blupf90+")

function blupf90_executable()
    exe = BLUPF90_EXE
    isfile(exe) && return abspath(exe)
    path_exe = Sys.which(exe)
    path_exe === nothing && error(
        "BLUPF90 executable not found: '$exe'. Set BLUPF90_EXE to the blupf90+ executable path.")
    return path_exe
end

function read_inputs(data_dir::String)
    observed = DataFrame(CSV.File(joinpath(data_dir, "yX_obs.csv")))
    roster = DataFrame(CSV.File(joinpath(data_dir, "roster.csv")))
    Ainv_df = DataFrame(CSV.File(joinpath(data_dir, "Ainv_triplet.csv")))
    LAinv_df = DataFrame(CSV.File(joinpath(data_dir, "LAinv_triplet.csv")))
    truth_df = DataFrame(CSV.File(joinpath(data_dir, "truth.csv")))

    required_columns = (
        (observed, (:y, :x, :id_ped), "yX_obs.csv"),
        (roster, (:id_ped, :group), "roster.csv"),
        (Ainv_df, (:i, :j, :x), "Ainv_triplet.csv"),
        (LAinv_df, (:i, :j, :x), "LAinv_triplet.csv"),
    )
    for (table, columns, name) in required_columns
        missing_columns = filter(col -> !hasproperty(table, col), columns)
        isempty(missing_columns) || error("$name is missing required columns: $missing_columns")
    end
    nrow(truth_df) == 1 || error("truth.csv must contain exactly one row")
    truth = truth_df[1, :]
    if hasproperty(truth, :psi) && !iszero(Float64(truth.psi))
        error("BLUPF90 comparison currently supports only ψ=0; found ψ=$(truth.psi)")
    end

    obs_ids = Int.(observed.id_ped)
    y = Float64.(observed.y)
    x = Float64.(observed.x)
    all(isfinite, y) && all(isfinite, x) || error("Observed outcomes and covariates must be finite")
    X = hcat(ones(length(y)), x)

    ai = Int.(Ainv_df.i)
    aj = Int.(Ainv_df.j)
    av = Float64.(Ainv_df.x)
    N = max(maximum(ai), maximum(aj))
    Ainv = sparse(vcat(ai, aj[ai .!= aj]), vcat(aj, ai[ai .!= aj]),
        vcat(av, av[ai .!= aj]), N, N)

    li = Int.(LAinv_df.i)
    lj = Int.(LAinv_df.j)
    lv = Float64.(LAinv_df.x)
    L_Ainv = LowerTriangular(sparse(li, lj, lv, N, N))

    roster_ids = Int.(roster.id_ped)
    roster_groups = Int.(roster.group)
    length(unique(roster_ids)) == length(roster_ids) ||
        error("roster.csv contains duplicate animal IDs")
    all(id -> 1 <= id <= N, roster_ids) ||
        error("roster IDs must be in 1:$N to match Ainv_triplet.csv")
    lookup = zeros(Int, N)
    lookup[roster_ids] .= roster_groups
    all(id -> lookup[id] != 0, obs_ids) ||
        error("Every observed animal must have a group in roster.csv")
    groups = GroupStructure(lookup[obs_ids], obs_ids, N;
        all_group_ids=roster_groups, all_animal_ids=roster_ids)

    return (; y, X, obs_ids, Ainv, L_Ainv, groups, truth)
end

function write_sparse_lower(io::IO, A::SparseMatrixCSC{Float64})
    rows, cols, vals = findnz(tril(A))
    for i in eachindex(vals)
        @printf(io, "%d %d %.17g\n", rows[i], cols[i], vals[i])
    end
end

function write_blupf90_case(run_dir::String, y, X, obs_ids, groups, Ainv;
    σ2_d::Float64, σ2_s::Float64, σ_ds::Float64, σ2_e::Float64,
    estimate_variances::Bool, max_rounds::Int=100)
    mkpath(run_dir)
    max_rounds >= 1 || error("BLUPF90 maximum rounds must be positive")
    N = size(Ainv, 1)
    max_peers = maximum(length(members) - 1 for members in groups.members_id)
    max_peers >= 1 || error("BLUPF90 social-effect setup requires at least one group with peers")
    length(y) == length(obs_ids) == size(X, 1) ||
        error("Observed data, IDs and X have incompatible row counts")

    open(joinpath(run_dir, "data.txt"), "w") do io
        for row in eachindex(y)
            id = obs_ids[row]
            peer_ids = filter(!=(id), groups.members_id[groups.obs_to_group[row]])
            length(peer_ids) <= max_peers || error("Internal peer-column sizing error")
            @printf(io, "%.17g 1 %.17g %d", y[row], X[row, 2], id)
            for col in 1:max_peers
                peer_id = col <= length(peer_ids) ? peer_ids[col] : 0
                @printf(io, " %d", peer_id)
            end
            print(io, '\n')
        end
    end

    open(joinpath(run_dir, "Ainv.txt"), "w") do io
        write_sparse_lower(io, Ainv)
    end

    parameter_path = joinpath(run_dir, "renf90.par")
    open(parameter_path, "w") do io
        n_effects = 3 + max_peers
        println(io, "DATAFILE\ndata.txt")
        println(io, "NUMBER_OF_TRAITS\n1")
        println(io, "NUMBER_OF_EFFECTS\n$n_effects")
        println(io, "OBSERVATION(S)\n1")
        println(io, "WEIGHT(S)\n")
        println(io, "EFFECTS:")
        println(io, "2 1 cross")
        println(io, "3 1 cov")
        println(io, "4 $N cross")
        for peer_col in 1:max_peers
            data_column = 4 + peer_col
            levels = peer_col == max_peers ? N : 0
            println(io, "$data_column $levels cross")
        end
        println(io, "RANDOM_RESIDUAL VALUES\n$σ2_e")
        println(io, "RANDOM_GROUP\n3 4")
        println(io, "RANDOM_TYPE\nuser_file")
        println(io, "FILE\nAinv.txt")
        println(io, "(CO)VARIANCES")
        @printf(io, "%.17g %.17g\n", σ2_d, σ_ds)
        @printf(io, "%.17g %.17g\n", σ_ds, σ2_s)
        if estimate_variances
            println(io, "OPTION method VCE")
            println(io, "OPTION conv_crit 1e-8")
            println(io, "OPTION maxrounds $max_rounds")
        end
    end
    return parameter_path
end

function run_blupf90(run_dir::String, exe::String; threads::Int=1, require_log::Bool=false)
    threads >= 1 || error("BLUPF90_THREADS must be positive")
    console_path = joinpath(run_dir, "blupf90_console.log")
    log_path = joinpath(run_dir, "blupf90.log")
    for stale_output in (log_path, joinpath(run_dir, "solutions"))
        isfile(stale_output) && rm(stale_output)
    end
    stdin_buffer = IOBuffer("renf90.par\n")
    cmd = addenv(Cmd(`$exe`; dir=run_dir),
        "OMP_NUM_THREADS" => string(threads),
        "OPENBLAS_NUM_THREADS" => "1")
    elapsed = @elapsed open(console_path, "w") do console
        run(pipeline(cmd; stdin=stdin_buffer, stdout=console, stderr=console))
    end
    if require_log && !isfile(log_path)
        error("BLUPF90 did not create its run log: $log_path; console output: $console_path")
    end
    return (; elapsed, log_path=console_path)
end

function parse_solutions(path::String)
    isfile(path) || error("BLUPF90 did not create solutions file: $path")
    solutions = Dict{Tuple{Int,Int},Float64}()
    for line in eachline(path)
        fields = split(strip(line))
        length(fields) >= 4 || continue
        trait = tryparse(Int, fields[1])
        effect = tryparse(Int, fields[2])
        level = tryparse(Int, fields[3])
        value = tryparse(Float64, fields[end])
        if trait !== nothing && effect !== nothing && level !== nothing && value !== nothing
            solutions[(effect, level)] = value
        end
    end
    isempty(solutions) && error("Could not parse any solutions from $path")
    return solutions
end

function peerige_fixed_solution(y, X, obs_ids, Ainv, L_Ainv, groups;
    σ2_d, σ2_s, σ_ds, σ2_e)
    model = SGEModel(y, X, Ainv, L_Ainv, groups, σ2_d, σ2_s, σ_ds, σ2_e)
    ws = model_workspace(model)
    preconditioner = get_preconditioner(model, ws)
    solve_mme_y!(model, ws, preconditioner; reltol=1e-14, maxiter=5000, verbose=true)
    return model.sol_y
end

function verify_tiny_fixed_variance_case(exe::String)
    N = 6
    obs_ids = collect(1:N)
    group_ids = [1, 1, 1, 2, 2, 3]
    groups = GroupStructure(group_ids, obs_ids, N;
        all_group_ids=group_ids, all_animal_ids=obs_ids)
    Ainv = sparse(1.0I, N, N)
    L_Ainv = LowerTriangular(copy(Ainv))
    X = hcat(ones(N), [-1.0, -0.4, 0.2, 0.3, 0.7, 1.1])
    y = [1.7, -0.2, 0.8, 1.4, -0.6, 2.2]
    variances = (σ2_d=1.1, σ2_s=0.45, σ_ds=0.12, σ2_e=0.8)
    run_dir = joinpath(OUTPUT_DIR, "tiny_fixed_check")
    write_blupf90_case(run_dir, y, X, obs_ids, groups, Ainv;
        σ2_d=variances.σ2_d, σ2_s=variances.σ2_s,
        σ_ds=variances.σ_ds, σ2_e=variances.σ2_e,
        estimate_variances=false)
    run_blupf90(run_dir, exe)
    blup = parse_solutions(joinpath(run_dir, "solutions"))
    julia_solution = peerige_fixed_solution(y, X, obs_ids, Ainv, L_Ainv, groups;
        variances...)
    layout = SolutionLayout(size(X, 2), N, Int[])
    expected_effects = ((1, first(layout.fixed)), (2, last(layout.fixed)),
        (3, layout.direct), (3 + maximum(length(members) - 1 for members in groups.members_id),
            layout.social))
    max_abs_error = 0.0
    worst_case = nothing
    for (effect_id, julia_positions) in expected_effects
        levels = julia_positions isa Integer ? (1:1) : eachindex(julia_positions)
        for level in levels
            key = (effect_id, level)
            haskey(blup, key) ||
                error("Missing BLUPF90 solution for effect=$effect_id level=$level")
            julia_index = julia_positions isa Integer ?
                julia_positions : first(julia_positions) + level - 1
            abs_error = abs(blup[key] - julia_solution[julia_index])
            if abs_error > max_abs_error
                max_abs_error = abs_error
                worst_case = (key, blup[key], julia_solution[julia_index])
            end
        end
    end
    max_abs_error <= 5e-6 ||
        error("Tiny fixed-variance solutions differ: max abs error=$max_abs_error at $worst_case")
    println("[blupf90-compare] Tiny fixed-variance MME check passed ",
        "(max |difference|=$(max_abs_error)).")
    return max_abs_error
end

function parse_variance_estimates(log_path::String)
    isfile(log_path) || error("Missing BLUPF90 log: $log_path")
    log_text = read(log_path, String)
    final_marker = findlast("Final Estimates", log_text)
    final_marker === nothing &&
        error("BLUPF90 log has no 'Final Estimates' block: $log_path")
    final_section = log_text[first(final_marker):end]
    number_rx = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"
    genetic_match = match(
        r"(?is)Genetic variance\(s\) for effect\s+\d+\s+(.*?)(?=\n\s*correlations)",
        final_section)
    genetic_match === nothing &&
        error("Could not parse the direct/social covariance matrix from $log_path")
    genetic = [parse(Float64, replace(String(m.match), 'D' => 'E', 'd' => 'e'))
        for m in eachmatch(number_rx, genetic_match.captures[1])]
    length(genetic) == 4 ||
        error("Expected a 2x2 direct/social covariance matrix in BLUPF90 log; got $(genetic)")
    residual_match = match(
        r"(?is)Residual variance\(s\)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?)",
        final_section)
    residual_match === nothing &&
        error("Could not parse residual variance from BLUPF90 final estimates")
    residual = parse(Float64,
        replace(residual_match.captures[1], 'D' => 'E', 'd' => 'e'))
    rounds = collect(eachmatch(
        r"In round\s+(\d+)\s+convergence\s*=\s*([-+0-9.eEdD]+)", log_text))
    isempty(rounds) && error("Could not parse BLUPF90 REML iterations from $log_path")
    deltas = collect(eachmatch(
        r"delta convergence\s*=\s*([-+0-9.eEdD]+)", log_text))
    isempty(deltas) && error("Could not parse BLUPF90 delta convergence from $log_path")
    last_round = rounds[end]
    convergence = parse(Float64,
        replace(last_round.captures[2], 'D' => 'E', 'd' => 'e'))
    delta_convergence = parse(Float64,
        replace(deltas[end].captures[1], 'D' => 'E', 'd' => 'e'))
    return (; σ2_d=genetic[1], σ_ds=genetic[2], σ2_s=genetic[4],
        σ2_e=residual, iterations=parse(Int, last_round.captures[1]),
        converged=convergence < 1e-8 || delta_convergence < 1e-4)
end

function run_peerige(inputs, run_dir::String; repeats::Int, k::Int,
    threads::Int, max_iter::Int, initial_σ_ds::Float64)
    rows = NamedTuple[]
    for seed in 1:repeats
        model = SGEModel(inputs.y, inputs.X, inputs.Ainv, inputs.L_Ainv,
            inputs.groups, 1.0, 0.2, initial_σ_ds, 2.0)
        log_path = joinpath(run_dir, "peerige_seed$(seed).csv")
        elapsed = @elapsed result = run_aireml(model;
            max_iter=max_iter, k=k, damping=0.5,
            cg_maxiter=5000, cg_reltol_early=1e-5,
            cg_early_iters=4, cg_reltol=1e-7,
            verbose_cg=false, seed=seed, log_file=log_path)
        log_table = DataFrame(CSV.File(log_path))
        push!(rows, (
            program="PeerIGE",
            n_obs=length(inputs.y),
            N=size(inputs.Ainv, 1),
            seed=seed,
            sigma2_d=result.σ2_d,
            sigma2_s=result.σ2_s,
            sigma_ds=result.σ_ds,
            sigma2_e=result.σ2_e,
            converged=result.converged,
            iterations=nrow(log_table),
            elapsed_seconds=elapsed,
            threads=threads,
        ))
        @printf("[blupf90-compare] PeerIGE seed=%d time=%.2fs iterations=%d\n",
            seed, elapsed, nrow(log_table))
    end
    return rows
end

save_rows(rows) = CSV.write(joinpath(OUTPUT_DIR, "compare.csv"), DataFrame(rows))

function run_comparison(data_dir::String, exe::String; repeats::Int, k::Int,
    threads::Int, max_iter::Int, max_rounds::Int, initial_σ_ds::Float64,
    prior_rows=NamedTuple[])
    inputs = read_inputs(data_dir)
    N = size(inputs.Ainv, 1)
    run_dir = joinpath(OUTPUT_DIR, "blupf90_N$(N)")
    write_blupf90_case(run_dir, inputs.y, inputs.X, inputs.obs_ids,
        inputs.groups, inputs.Ainv;
        σ2_d=1.0, σ2_s=0.2, σ_ds=initial_σ_ds, σ2_e=2.0,
        estimate_variances=true, max_rounds=max_rounds)
    truth = inputs.truth
    rows = NamedTuple[(
        program="Truth",
        n_obs=length(inputs.y),
        N=N,
        seed=missing,
        sigma2_d=Float64(truth.sigma2_d),
        sigma2_s=Float64(truth.sigma2_s),
        sigma_ds=Float64(truth.sigma_ds),
        sigma2_e=Float64(truth.sigma2_e),
        converged=missing,
        iterations=missing,
        elapsed_seconds=missing,
        threads=missing,
    )]

    # PeerIGE runs first and is saved immediately so a long BLUPF90+ run
    # cannot cost the PeerIGE results.
    append!(rows, run_peerige(inputs, run_dir;
        repeats=repeats, k=k, threads=threads, max_iter=max_iter,
        initial_σ_ds=initial_σ_ds))
    save_rows(vcat(prior_rows, rows))

    blup_run = run_blupf90(run_dir, exe; threads=threads, require_log=true)
    blup_vcs = parse_variance_estimates(blup_run.log_path)
    push!(rows, (
        program="BLUPF90+",
        n_obs=length(inputs.y),
        N=N,
        seed=0,
        sigma2_d=blup_vcs.σ2_d,
        sigma2_s=blup_vcs.σ2_s,
        sigma_ds=blup_vcs.σ_ds,
        sigma2_e=blup_vcs.σ2_e,
        converged=blup_vcs.converged,
        iterations=blup_vcs.iterations,
        elapsed_seconds=blup_run.elapsed,
        threads=threads,
    ))
    save_rows(vcat(prior_rows, rows))
    @printf("[blupf90-compare] BLUPF90+ time=%.2fs\n", blup_run.elapsed)
    return rows
end

function ensure_data(data_dir::String, ng::Int, missing_frac::Float64)
    required = ("yX_obs.csv", "roster.csv", "Ainv_triplet.csv",
        "LAinv_triplet.csv", "truth.csv")
    all(name -> isfile(joinpath(data_dir, name)), required) && return
    rscript = Sys.which("Rscript")
    rscript === nothing && error(
        "Dataset missing in $data_dir and Rscript is not on PATH; generate it with sim/sim_parameter_recovery.R")
    mkpath(data_dir)
    sim_script = joinpath(REPO_ROOT, "sim", "sim_parameter_recovery.R")
    run(setenv(`$rscript $sim_script`, merge(copy(ENV), Dict(
        "SIM_OUT_DIR" => data_dir,
        "SIM_NG" => string(ng),
        "SIM_MISSING_FRAC" => string(missing_frac),
    ))))
end

function main()
    exe = blupf90_executable()
    mkpath(OUTPUT_DIR)
    verify_tiny_fixed_variance_case(exe)
    get(ENV, "BLUPF90_VALIDATE_ONLY", "0") == "1" && return

    repeats = parse(Int, get(ENV, "BLUPF90_PEERIGE_REPEATS", "3"))
    k = parse(Int, get(ENV, "BLUPF90_PEERIGE_K", "20"))
    threads = parse(Int, get(ENV, "BLUPF90_THREADS", "1"))
    Threads.nthreads() == threads ||
        error("BLUPF90_THREADS=$threads but Julia has $(Threads.nthreads()) threads; launch Julia with --threads=$threads")
    max_iter = parse(Int, get(ENV, "BLUPF90_PEERIGE_MAX_ITER", "15"))
    max_rounds = parse(Int, get(ENV, "BLUPF90_MAX_ROUNDS", "100"))
    initial_σ_ds = parse(Float64, get(ENV, "BLUPF90_SIGMA_DS_INIT", "0.05"))
    abs(initial_σ_ds) < sqrt(0.2) ||
        error("BLUPF90_SIGMA_DS_INIT must satisfy |σ_ds| < sqrt(1.0 * 0.2)")
    ng = parse(Int, get(ENV, "BLUPF90_NG", "3750"))
    missing_frac = parse(Float64, get(ENV, "BLUPF90_MISSING_FRAC", "0.10"))
    data_dir = abspath(get(ENV, "BLUPF90_DATA_DIR",
        joinpath(OUTPUT_DIR, "data_ng$(ng)")))

    ensure_data(data_dir, ng, missing_frac)
    println("[blupf90-compare] Dataset: $data_dir")
    println("[blupf90-compare] Executable: $exe")
    rows = run_comparison(data_dir, exe; repeats=repeats, k=k,
        threads=threads, max_iter=max_iter, max_rounds=max_rounds,
        initial_σ_ds=initial_σ_ds)
    println("[blupf90-compare] Wrote $(length(rows)) timing/variance rows to ",
        joinpath(OUTPUT_DIR, "compare.csv"))

    ng_list = strip(get(ENV, "BLUPF90_NG_LIST", ""))
    if !isempty(ng_list)
        for scale_ng in parse.(Int, split(ng_list, ','))
            scale_dir = joinpath(OUTPUT_DIR, "data_ng$(scale_ng)")
            ensure_data(scale_dir, scale_ng, missing_frac)
            scale_rows = run_comparison(scale_dir, exe; repeats=repeats, k=k,
                threads=threads, max_iter=max_iter, max_rounds=max_rounds,
                initial_σ_ds=initial_σ_ds, prior_rows=rows)
            append!(rows, scale_rows)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
