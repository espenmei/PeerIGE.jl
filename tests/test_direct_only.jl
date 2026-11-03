# Direct-only model (social=false): y = Xβ + Z_d a_d + Σ_h Z_h c_h + e.
# Checks the MME operator, preconditioners, Px, score/AI and the stochastic
# log-likelihood against dense references, the guards, and parameter recovery.
if !@isdefined(SGEModel)
    include(joinpath(@__DIR__, "..", "src", "emai.jl"))
end
if !@isdefined(score_ai_dense_fixture)
    include(joinpath(@__DIR__, "test_score_ai.jl"))
end
using LinearAlgebra, SparseArrays, Random, Test, Printf

direct_only_model(f, H) = SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups,
    1.2, 0.0, 0.0, 0.9; social=false,
    env_ids=f.env_ids[1:H], σ2_env=[0.4, 0.25][1:H])

function direct_only_dense_covariance(f, H, θ)
    V = θ[1] .* (f.Zd * f.A * f.Zd') + θ[2] * I
    for h in 1:H
        V += θ[2+h] .* (f.Zh[h] * f.Zh[h]')
    end
    return V
end

function direct_only_dense_loglik(f, H, θ)
    F = cholesky(Symmetric(direct_only_dense_covariance(f, H, θ)))
    ViX, Viy = F \ f.X, F \ f.y
    FX = cholesky(Symmetric(f.X' * ViX))
    Py = Viy - ViX * (FX \ (f.X' * Viy))
    return -0.5 * (logdet(F) + logdet(FX) + dot(f.y, Py))
end

@testset "direct-only model (social=false)" begin
    f = score_ai_dense_fixture()
    n, N, nf = length(f.y), size(f.Ainv, 1), size(f.X, 2)

    @testset "construction and guards" begin
        model = direct_only_model(f, 2)
        @test !has_social(model)
        @test isempty(model.layout.social)
        @test get_dim(model) == nf + N + sum(model.env_n_levels)
        @test first(model.layout.env[1]) == nf + N + 1
        @test get_param_names(model) == ["σ²_d", "σ²_e", "σ²_env1", "σ²_env2"]
        @test get_params(model) == [1.2, 0.9, 0.4, 0.25]
        @test model.σ2_s == 0.0 && model.σ_ds == 0.0
        @test pe_param_index(model) == 5
        @test genetic_covariance(model) == fill(1.2, 1, 1)
        @test has_social(SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups, 1.2, 0.2, 0.1, 0.9))
        # σ2_s/σ_ds are ignored when social=false
        @test SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups, 1.2, 0.3, 0.1, 0.9;
            social=false).σ2_s == 0.0
        @test_throws ArgumentError SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups,
            1.2, 0.0, 0.0, 0.9; social=false, peer_env=true, σ2_pe=0.1)
        @test_throws ArgumentError SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups,
            1.2, 0.0, 0.0, 0.9; social=false, ψ=0.5)
        @test_throws ArgumentError fit_aireml!(direct_only_model(f, 1); estimate_ψ=true, k=3)
        @test_throws ArgumentError psi_score(direct_only_model(f, 1); K=2)
    end

    for H in (0, 1, 2)
        @testset "dense references H=$H" begin
            model = direct_only_model(f, H)
            θ = get_params(model)
            ws = model_workspace(model)

            # Residual-scaled MME: C = W'W + blockdiag(0, λ_d Ainv, λ_h I)
            W = hcat(f.X, f.Zd, f.Zh[1:H]...)
            prior = zeros(size(W, 2), size(W, 2))
            prior[nf+1:nf+N, nf+1:nf+N] .= (θ[2] / θ[1]) .* Matrix(f.Ainv)
            for h in 1:H
                r = model.layout.env[h]
                prior[r, r] .= (θ[2] / θ[2+h]) .* Matrix{Float64}(I, length(r), length(r))
            end
            C = W' * W + prior
            @test size(C, 1) == get_dim(model)
            op = make_mme_operator(model, ws)
            v = randn(MersenneTwister(5), get_dim(model))
            out = zeros(get_dim(model))
            op(out, v)
            @test maximum(abs.(out - C * v)) < 1e-10 * maximum(abs.(C * v))
            rhs = zeros(get_dim(model))
            build_rhs!(rhs, f.y, model, ws)
            @test rhs ≈ W' * f.y

            # Block preconditioner is the exact inverse of blockdiag(diag(X'X), λ_d Ainv, ...).
            Pb = get_preconditioner(model, ws)
            M = zeros(size(C))
            for j in 1:nf
                M[j, j] = max(dot(f.X[:, j], f.X[:, j]), 1e-6)
            end
            M[nf+1:nf+N, nf+1:nf+N] .= prior[nf+1:nf+N, nf+1:nf+N]
            for h in 1:H
                r = model.layout.env[h]
                M[r, r] .= Diagonal(model.env_counts[h] .+ θ[2] / θ[2+h])
            end
            z = zeros(get_dim(model))
            ldiv!(z, Pb, v)
            @test maximum(abs.(z - M \ v)) < 1e-9 * maximum(abs.(M \ v))
            Pd = get_diag_preconditioner(model)
            ldiv!(z, Pd, v)
            @test z ≈ v ./ diag(C)

            V = direct_only_dense_covariance(f, H, θ)
            Vinv = inv(V)
            P = Vinv - Vinv * f.X * ((f.X' * Vinv * f.X) \ (f.X' * Vinv))
            Py_ref = P * f.y
            derivatives = [f.Zd * f.A * f.Zd', Matrix{Float64}(I, n, n)]
            append!(derivatives, [f.Zh[h] * f.Zh[h]' for h in 1:H])
            score_ref = [0.5 * (dot(Py_ref, D * Py_ref) - tr(P * D)) for D in derivatives]
            Q = hcat([D * Py_ref for D in derivatives]...)
            AI_ref = 0.5 .* (Q' * P * Q)
            gradient_fd, _ = score_ai_finite_differences(θ -> direct_only_dense_loglik(f, H, θ), θ)
            @test maximum(abs.(gradient_fd - score_ref)) < 1e-6

            for block in (false, true)
                workspaces = create_thread_workspaces(model, Threads.maxthreadid())
                preconditioners = block ?
                    [get_preconditioner(model, w) for w in workspaces] :
                    [get_diag_preconditioner(model) for _ in workspaces]
                fill!(model.sol_y, 0.0)
                solve_mme_y!(model, workspaces[1], preconditioners[1];
                    reltol=1e-12, maxiter=2000, verbose=false)
                Py = zeros(n)
                compute_Px_from_sol!(Py, model, f.y, model.sol_y, workspaces[1])
                @test maximum(abs.(Py - Py_ref)) < 1e-9

                Z = sqrt(n) .* Matrix{Float64}(I, n, n)
                zl = [Z[:, j] for j in 1:n]
                Pz = [zeros(n) for _ in 1:n]
                sol_z = [zeros(get_dim(model)) for _ in 1:n]
                _threaded_z_solves!(sol_z, model, zl, Pz, workspaces, preconditioners,
                    n, 1e-12, 2000)
                score, AI = compute_score_and_ai(model, Py, zl, Pz, workspaces,
                    preconditioners; ai_reltol=1e-12, ai_maxiter=2000, verbose=false)
                @test length(score) == length(θ)
                @test maximum(abs.(score - score_ref)) < 1e-8
                @test maximum(abs.(AI - AI_ref)) < 1e-8
            end

            # sqrt(dim) times the full basis makes the stochastic log|C| exact.
            dim = get_dim(model)
            exact = direct_only_dense_loglik(f, H, θ)
            r = stochastic_loglik(model; probes=sqrt(dim) .* Matrix{Float64}(I, dim, dim),
                lanczos_maxiter=2dim)
            @test abs(r.loglik - exact) < 1e-6
            @printf("[direct-only] H=%d loglik exact=%.6f err=%.1e; max score err=%.1e\n",
                H, exact, abs(r.loglik - exact), maximum(abs.(gradient_fd - score_ref)))
        end
    end

    @testset "fit, report, snapshot and log" begin
        model = direct_only_model(f, 1)
        log_file = tempname() * ".csv"
        redirect_stdout(devnull) do
            fit_aireml!(model; max_iter=3, k=3, damping=0.5, verbose_cg=false, log_file)
        end
        @test model.σ2_s == 0.0 && model.σ_ds == 0.0
        @test length(model.fit.score) == 3
        header = split(readline(log_file), ",")
        @test "s2d" in header && "s2e" in header && "s2env1" in header && !("s2s" in header)
        @test length(split(readlines(log_file)[2], ",")) == length(header)
        rm(log_file)
        report = sprint(show, MIME"text/plain"(), model)
        @test occursin("direct-only", report) && !occursin("σ²_s", report)
        res = snapshot(model; include_fixef_se=false)
        @test isempty(res.a_s) && res.param_names == ["σ²_d", "σ²_e", "σ²_env1"]
        @test occursin("direct-only", sprint(show, MIME"text/plain"(), res))
        effects = ranef(model)
        @test length(effects.a_d) == N && isempty(effects.a_s)
    end
end

# ---------------------------------------------------------------------
# Recovery: σ²_d, household σ²_h and σ²_e on a simulated 3-generation pedigree.
# Pedigree IDs are ordered offspring before parents, so L_Ainv = T'D^{-1/2} is lower.
# ---------------------------------------------------------------------
function direct_only_recovery_data(; seed=2026, n_founder_pairs=1500, kids1=4, kids2=3,
    σ2_d=1.0, σ2_h=0.4, σ2_e=1.5)
    rng = MersenneTwister(seed)
    sire, dam, gen, family = Int[], Int[], Int[], Int[]
    add!(s, d, g, fam) = (push!(sire, s); push!(dam, d); push!(gen, g); push!(family, fam); length(sire))
    founders = [add!(0, 0, 0, 0) for _ in 1:2n_founder_pairs]
    gen1 = Int[]
    for p in 1:n_founder_pairs, _ in 1:kids1
        push!(gen1, add!(founders[2p-1], founders[2p], 1, p))
    end
    # Mate gen-1 individuals from different families: no inbreeding in gen 2.
    order = gen1[randperm(rng, length(gen1))]
    gen2_family = 0
    for k in 1:2:length(order)-1
        a, b = order[k], order[k+1]
        family[a] == family[b] && continue
        gen2_family += 1
        for _ in 1:kids2
            add!(a, b, 2, n_founder_pairs + gen2_family)
        end
    end
    N = length(sire)
    # Reverse order: offspring first, parents later.
    new_id = N + 1 .- (1:N)
    T = sparse(1.0I, N, N)
    d = [s == 0 ? 1.0 : 0.5 for s in sire]
    Ti, Tj, Tv = Int[], Int[], Float64[]
    for i in 1:N, p in (sire[i], dam[i])
        p == 0 && continue
        push!(Ti, new_id[i]); push!(Tj, new_id[p]); push!(Tv, -0.5)
    end
    T = T + sparse(Ti, Tj, Tv, N, N)
    dn = zeros(N)
    dn[new_id] .= d
    Lt = sparse(T' * Diagonal(1 ./ sqrt.(dn)))
    L_Ainv = LowerTriangular(Lt)
    Ainv = sparse(Lt * Lt')
    a = UpperTriangular(Matrix(T)) \ (sqrt.(dn) .* randn(rng, N)) .* sqrt(σ2_d)

    obs_old = [i for i in 1:N if gen[i] >= 1]
    obs = new_id[obs_old]
    hh = family[obs_old]
    hh_ids = indexin(hh, unique(hh))
    n = length(obs)
    c = sqrt(σ2_h) .* randn(rng, maximum(hh_ids))
    x = randn(rng, n)
    y = 1.0 .+ 0.3 .* x .+ a[obs] .+ c[hh_ids] .+ sqrt(σ2_e) .* randn(rng, n)
    groups = GroupStructure(collect(1:n), obs, N)  # singleton groups: obs → ID map only
    return (; y, X=hcat(ones(n), x), Ainv, L_Ainv, groups, env_ids=[Vector{Int}(hh_ids)],
        truth=[σ2_d, σ2_e, σ2_h])
end

@testset "direct-only parameter recovery" begin
    d = direct_only_recovery_data()
    model = SGEModel(d.y, d.X, d.Ainv, d.L_Ainv, d.groups, 0.5, 0.0, 0.0, 2.0;
        social=false, env_ids=d.env_ids, σ2_env=[0.2], env_names=["household"])
    redirect_stdout(devnull) do
        fit_aireml!(model; max_iter=30, k=20, damping=0.5, verbose_cg=false)
    end
    est = get_params(model)
    se = model.fit.se
    @printf("[direct-only recovery] n=%d N=%d truth=%s est=%s se=%s converged=%s iters=%d\n",
        length(d.y), size(d.Ainv, 1), string(d.truth), string(round.(est; digits=3)),
        string(round.(se; digits=3)), model.fit.converged, model.fit.n_iter)
    @test model.fit.converged
    @test all(abs.(est .- d.truth) .< 4 .* se)
end
