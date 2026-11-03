if !@isdefined(SGEModel)
    include(joinpath(@__DIR__, "..", "src", "emai.jl"))
end
using LinearAlgebra, SparseArrays, Random, Test, Printf

function score_ai_dense_fixture()
    rng = MersenneTwister(2026)
    group_ids = vcat([fill(g, m) for (g, m) in enumerate((2, 3, 4, 5, 6))]...)
    roster = collect(eachindex(group_ids))
    obs = setdiff(roster, [2, 7, 18])
    N = length(roster) + 4
    n = length(obs)
    groups = GroupStructure(group_ids[obs], obs, N;
        all_group_ids=group_ids, all_animal_ids=roster)

    # Tiny test-only pedigree: two sibling families spread across unequal classes.
    B = Matrix{Float64}(I, N, N)
    for i in roster
        parents = isodd(i) ? (N - 3, N - 2) : (N - 1, N)
        B[i, i] = sqrt(0.5)
        B[i, parents[1]] = B[i, parents[2]] = 0.5
    end
    A = B * B'
    Ainv = sparse(inv(A))
    L_Ainv = LowerTriangular(sparse(cholesky(Symmetric(Matrix(Ainv))).L))
    X = hcat(ones(n), randn(rng, n))
    y = randn(rng, n)
    Zd = zeros(n, N)
    Zs = zeros(n, N)
    for (i, id) in enumerate(obs)
        Zd[i, id] = 1.0
        for peer in roster
            if peer != id && group_ids[peer] == group_ids[id]
                Zs[i, peer] = 1.0
            end
        end
    end
    env_ids = [group_ids[obs], [isodd(id) ? 1 : 2 for id in obs]]
    Zh = [Float64.(ids .== permutedims(collect(1:maximum(ids)))) for ids in env_ids]
    return (; y, X, A, Ainv, L_Ainv, groups, Zd, Zs, env_ids, Zh)
end

function score_ai_finite_differences(loglik, params)
    h = 2e-4 .* max.(1.0, abs.(params))
    p = length(params)
    steps = Matrix(Diagonal(h))
    gradient = zeros(p)
    observed = zeros(p, p)
    f0 = loglik(params)
    for i in 1:p
        ei = steps[:, i]
        fp, fm = loglik(params + ei), loglik(params - ei)
        gradient[i] = (fp - fm) / (2h[i])
        observed[i, i] = -(fp - 2 * f0 + fm) / h[i]^2
        for j in 1:i-1
            ej = steps[:, j]
            observed[i, j] = observed[j, i] = -(
                loglik(params + ei + ej) - loglik(params + ei - ej) -
                loglik(params - ei + ej) + loglik(params - ei - ej)
            ) / (4h[i] * h[j])
        end
    end
    return gradient, observed
end

@testset "score and AI against dense REML" begin
    f = score_ai_dense_fixture()
    n = length(f.y)
    cases = (
        (H=0, peer_env=false, psi=0.0, covariance=0.12),
        (H=1, peer_env=true, psi=0.0, covariance=-0.12),
        (H=2, peer_env=true, psi=0.5, covariance=0.15),
        (H=1, peer_env=true, psi=1.0, covariance=0.08),
    )
    for case in cases
        @testset "H=$(case.H), peer=$(case.peer_env), psi=$(case.psi)" begin
            model = SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups,
                1.2, 0.2, case.covariance, 0.9;
                env_ids=f.env_ids[1:case.H], σ2_env=[0.4, 0.25][1:case.H],
                ψ=case.psi, peer_env=case.peer_env, σ2_pe=0.15)
            params = get_params(model)
            Zs = copy(f.Zs)
            for i in 1:n
                # Use the full roster, not the number of observed responses.
                m = length(f.groups.members_id[f.groups.obs_to_group[i]])
                Zs[i, :] .*= (m - 1)^(-case.psi)
            end
            Zg = hcat(f.Zd, Zs)
            function dense_covariance(theta)
                G = [theta[1] theta[3]; theta[3] theta[2]]
                V = Zg * kron(G, f.A) * Zg' + theta[4] * I
                for h in 1:case.H
                    V += theta[4+h] .* (f.Zh[h] * f.Zh[h]')
                end
                if case.peer_env
                    V += theta[end] .* (Zs * Zs')
                end
                return V
            end
            function dense_loglik(theta)
                F = cholesky(Symmetric(dense_covariance(theta)))
                ViX, Viy = F \ f.X, F \ f.y
                FX = cholesky(Symmetric(f.X' * ViX))
                Py = Viy - ViX * (FX \ (f.X' * Viy))
                return -0.5 * (logdet(F) + logdet(FX) + dot(f.y, Py))
            end

            V = dense_covariance(params)
            Vinv = inv(V)
            P = Vinv - Vinv * f.X * ((f.X' * Vinv * f.X) \ (f.X' * Vinv))
            Py_ref = P * f.y
            derivatives = [
                f.Zd * f.A * f.Zd',
                Zs * f.A * Zs',
                f.Zd * f.A * Zs' + Zs * f.A * f.Zd',
                Matrix{Float64}(I, n, n),
            ]
            append!(derivatives, [f.Zh[h] * f.Zh[h]' for h in 1:case.H])
            case.peer_env && push!(derivatives, Zs * Zs')
            score_ref = [0.5 * (dot(Py_ref, D * Py_ref) - tr(P * D)) for D in derivatives]
            Q = hcat([D * Py_ref for D in derivatives]...)
            AI_ref = 0.5 .* (Q' * P * Q)
            fisher = [0.5 * tr(P * Di * P * Dj) for Di in derivatives, Dj in derivatives]
            gradient_fd, observed_fd = score_ai_finite_differences(dense_loglik, params)
            @test maximum(abs.(gradient_fd - score_ref)) < 1e-6
            @test maximum(abs.(0.5 .* (observed_fd + fisher) - AI_ref)) < 1e-5
            @printf("[score-ai] H=%d peer=%s psi=%.1f finite differences: max score err=%.2e, max AI err=%.2e\n",
                case.H, case.peer_env, case.psi, maximum(abs.(gradient_fd - score_ref)),
                maximum(abs.(0.5 .* (observed_fd + fisher) - AI_ref)))

            for block in (false, true)
                @testset "$(block ? "block" : "diagonal") preconditioner" begin
                    workspaces = create_thread_workspaces(model, Threads.maxthreadid())
                    preconditioners = block ?
                        [get_preconditioner(model, ws) for ws in workspaces] :
                        [get_diag_preconditioner(model) for _ in workspaces]
                    fill!(model.sol_y, 0.0)
                    solve_mme_y!(model, workspaces[1], preconditioners[1];
                        reltol=1e-12, maxiter=2000, verbose=false)
                    Py = zeros(n)
                    compute_Px_from_sol!(Py, model, f.y, model.sol_y, workspaces[1])
                    @test maximum(abs.(Py - Py_ref)) < 1e-9

                    # sqrt(n) times the full basis makes (1/k) sum(z*z') = I exactly.
                    probes = (sqrt(n) .* Matrix{Float64}(I, n, n),
                        randn(MersenneTwister(77), n, 7))
                    for (probe_idx, Z) in enumerate(probes)
                        k = size(Z, 2)
                        z = [Z[:, j] for j in 1:k]
                        Pz = [zeros(n) for _ in 1:k]
                        sol_z = [zeros(get_dim(model)) for _ in 1:k]
                        _threaded_z_solves!(sol_z, model, z, Pz, workspaces, preconditioners,
                            k, 1e-12, 2000)
                        @test maximum(abs.(hcat(Pz...) - P * Z)) < 1e-9
                        score, AI = compute_score_and_ai(model, Py, z, Pz,
                            workspaces, preconditioners; ai_reltol=1e-12, ai_maxiter=2000)
                        score_expected = probe_idx == 1 ? score_ref : [
                            0.5 * (dot(Py_ref, D * Py_ref) -
                                sum(dot(Z[:, j], D * P * Z[:, j]) for j in 1:k) / k)
                            for D in derivatives
                        ]
                        @test length(score) == length(params)
                        @test size(AI) == (length(params), length(params))
                        @test maximum(abs.(score - score_expected)) < 1e-8
                        @test maximum(abs.(AI - AI_ref)) < 1e-8
                        @test issymmetric(AI)
                        @test eigmin(Symmetric(AI)) > -1e-10
                        @printf("[score-ai] H=%d peer=%s psi=%.1f block=%s probes=%s: max score err=%.2e, max AI err=%.2e\n",
                            case.H, case.peer_env, case.psi, block,
                            probe_idx == 1 ? "exact" : "Gaussian",
                            maximum(abs.(score - score_expected)), maximum(abs.(AI - AI_ref)))
                    end
                end
            end
        end
    end
end

    @testset "estimated ψ score and AI against dense REML" begin
                    f = score_ai_dense_fixture()
                    case = (H=1, peer_env=true, psi=0.45, covariance=0.08)
                    model = SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups,
                        1.2, 0.2, case.covariance, 0.9;
                        env_ids=f.env_ids[1:case.H], σ2_env=[0.4],
                        ψ=case.psi, peer_env=case.peer_env, σ2_pe=0.15)
                    raw_params = get_params(model; include_ψ=false)
                    raw_scales = copy(model.group_scale)
                    n = length(f.y)
                    ref_peers = _psi_reference_size(model) - 1

                    function dense_covariance_psi(theta, psi, group_scale)
                        Zs = copy(f.Zs)
                        for i in 1:n
                            g = f.groups.obs_to_group[i]
                            Zs[i, :] .*= group_scale[g]
                        end
                        Zg = hcat(f.Zd, Zs)
                        G = [theta[1] theta[3]; theta[3] theta[2]]
                        V = Zg * kron(G, f.A) * Zg' + theta[4] * I
                        V += theta[5] .* (f.Zh[1] * f.Zh[1]')
                        V += theta[6] .* (Zs * Zs')
                        return V
                    end
                    raw_V = dense_covariance_psi(raw_params, case.psi, raw_scales)
                    _convert_psi_scale!(model; to_centered=true)
                    params = get_params(model; include_ψ=false)
                    centered_V = dense_covariance_psi(params, case.psi, model.group_scale)
                    @test centered_V ≈ raw_V rtol=1e-12 atol=1e-12

                    function dense_loglik_psi(theta, psi)
                        scales = [((length(members) - 1.0) / ref_peers)^(-psi)
                                  for members in f.groups.members_id]
                        F = cholesky(Symmetric(dense_covariance_psi(theta, psi, scales)))
                        ViX, Viy = F \ f.X, F \ f.y
                        FX = cholesky(Symmetric(f.X' * ViX))
                        Py = Viy - ViX * (FX \ (f.X' * Viy))
                        return -0.5 * (logdet(F) + logdet(FX) + dot(f.y, Py))
                    end

                    V = centered_V
                    Vinv = inv(V)
                    P = Vinv - Vinv * f.X * ((f.X' * Vinv * f.X) \ (f.X' * Vinv))
                    Py_ref = P * f.y
                    h = 1e-5
                    Vdot = (dense_covariance_psi(params, case.psi + h,
                                [((length(members) - 1.0) / ref_peers)^(-(case.psi + h))
                                 for members in f.groups.members_id]) -
                            dense_covariance_psi(params, case.psi - h,
                                [((length(members) - 1.0) / ref_peers)^(-(case.psi - h))
                                 for members in f.groups.members_id])) ./ (2h)
                    # Form the ordinary variance derivatives directly at the centered design.
                    Zs = copy(f.Zs)
                    for i in 1:n
                        g = f.groups.obs_to_group[i]
                        Zs[i, :] .*= ((length(f.groups.members_id[g]) - 1.0) / ref_peers)^(-case.psi)
                    end
                    derivatives = [
                        f.Zd * f.A * f.Zd',
                        Zs * f.A * Zs',
                        f.Zd * f.A * Zs' + Zs * f.A * f.Zd',
                        Matrix{Float64}(I, n, n),
                        f.Zh[1] * f.Zh[1]',
                        Zs * Zs',
                        Vdot,
                    ]
                    score_ref = [0.5 * (dot(Py_ref, D * Py_ref) - tr(P * D)) for D in derivatives]
                    Q = hcat([D * Py_ref for D in derivatives]...)
                    AI_ref = 0.5 .* (Q' * P * Q)
                    all_params = vcat(params, case.psi)
                    gradient_fd, _ = score_ai_finite_differences(
                        p -> dense_loglik_psi(p[1:end-1], p[end]), all_params)

                    workspaces = create_thread_workspaces(model, Threads.maxthreadid())
                    preconditioners = [get_preconditioner(model, ws) for ws in workspaces]
                    solve_mme_y!(model, workspaces[1], preconditioners[1]; reltol=1e-12, maxiter=2000, verbose=false)
                    Py = zeros(n)
                    compute_Px_from_sol!(Py, model, model.y, model.sol_y, workspaces[1])
                    probes = sqrt(n) .* Matrix{Float64}(I, n, n)
                    z = [Vector{Float64}(probes[:, j]) for j in axes(probes, 2)]
                    Pz = [zeros(n) for _ in eachindex(z)]
                    sol_z = [zeros(get_dim(model)) for _ in eachindex(z)]
                    _threaded_z_solves!(sol_z, model, z, Pz, workspaces, preconditioners,
                        length(z), 1e-12, 2000)
                    score, AI = compute_score_and_ai(model, Py, z, Pz, workspaces, preconditioners;
                        ai_reltol=1e-12, ai_maxiter=2000, estimate_ψ=true, verbose=false)
                    @test maximum(abs.(score - score_ref)) < 1e-6
                    @test maximum(abs.(score - gradient_fd)) < 1e-5
                    @test maximum(abs.(AI - AI_ref)) < 1e-6
                    @test issymmetric(AI)
                    @test eigmin(Symmetric(AI)) > 0

                    fit_model = SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups,
                        1.2, 0.2, case.covariance, 0.9;
                        env_ids=f.env_ids[1:case.H], σ2_env=[0.4],
                        ψ=case.psi, peer_env=case.peer_env, σ2_pe=0.15)
                    fit_aireml!(fit_model; max_iter=1, k=8, damping=0.1, verbose_cg=false,
                        estimate_ψ=true, ψ_bounds=(0.0, 1.0))
                    @test length(fit_model.fit.score) == 7
                    @test last(get_param_names(fit_model)) == "ψ"
                    fit_result = snapshot(fit_model; include_fixef_se=false)
                    @test last(fit_result.param_names) == "ψ"
                    @test occursin("ψ", sprint(show, MIME"text/plain"(), fit_result))
    end
