if !@isdefined(SGEModel)
    include(joinpath(@__DIR__, "..", "src", "emai.jl"))
end
using LinearAlgebra, SparseArrays, Random, Test, Printf

# Tiny dense reference for stochastic_loglik and psi_score (same design as test_score_ai.jl).
function post_fit_dense_fixture()
    rng = MersenneTwister(2026)
    group_ids = vcat([fill(g, m) for (g, m) in enumerate((2, 3, 4, 5, 6))]...)
    roster = collect(eachindex(group_ids))
    obs = setdiff(roster, [2, 7, 18])
    N = length(roster) + 4
    n = length(obs)
    groups = GroupStructure(group_ids[obs], obs, N;
        all_group_ids=group_ids, all_animal_ids=roster)
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
    Zs1 = zeros(n, N)
    for (i, id) in enumerate(obs)
        Zd[i, id] = 1.0
        for peer in roster
            if peer != id && group_ids[peer] == group_ids[id]
                Zs1[i, peer] = 1.0
            end
        end
    end
    m_obs = [length(groups.members_id[groups.obs_to_group[i]]) for i in 1:n]
    env_ids = [group_ids[obs], [isodd(id) ? 1 : 2 for id in obs]]
    Zh = [Float64.(ids .== permutedims(collect(1:maximum(ids)))) for ids in env_ids]
    return (; y, X, A, Ainv, L_Ainv, groups, Zd, Zs1, m_obs, env_ids, Zh)
end

function post_fit_dense_covariance(f, case, θ, ψ)
    Zs = (f.m_obs .- 1.0) .^ (-ψ) .* f.Zs1
    Zg = hcat(f.Zd, Zs)
    G = [θ[1] θ[3]; θ[3] θ[2]]
    V = Zg * kron(G, f.A) * Zg' + θ[4] * I
    for h in 1:case.H
        V += θ[4+h] .* (f.Zh[h] * f.Zh[h]')
    end
    case.peer_env && (V += θ[end] .* (Zs * Zs'))
    return V
end

function post_fit_dense_loglik(f, case, θ, ψ)
    F = cholesky(Symmetric(post_fit_dense_covariance(f, case, θ, ψ)))
    ViX, Viy = F \ f.X, F \ f.y
    FX = cholesky(Symmetric(f.X' * ViX))
    Py = Viy - ViX * (FX \ (f.X' * Viy))
    return -0.5 * (logdet(F) + logdet(FX) + dot(f.y, Py))
end

post_fit_model(f, case) = SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups,
    1.2, 0.2, case.covariance, 0.9;
    env_ids=f.env_ids[1:case.H], σ2_env=[0.4, 0.25][1:case.H],
    ψ=case.psi, peer_env=case.peer_env, σ2_pe=0.15)

const POST_FIT_CASES = (
    (H=0, peer_env=false, psi=0.0, covariance=0.12),
    (H=1, peer_env=true, psi=0.5, covariance=-0.12),
    (H=2, peer_env=true, psi=1.0, covariance=0.08),
)

@testset "post-fit diagnostics against dense REML" begin
    f = post_fit_dense_fixture()
    n = length(f.y)
    for case in POST_FIT_CASES
        @testset "H=$(case.H), peer=$(case.peer_env), psi=$(case.psi)" begin
            model = post_fit_model(f, case)
            θ = get_params(model)
            dim = get_dim(model)
            exact = post_fit_dense_loglik(f, case, θ, case.psi)

            @testset "stochastic_loglik" begin
                # sqrt(dim) times the full basis makes the Hutchinson trace exact.
                r = stochastic_loglik(model; probes=sqrt(dim) .* Matrix{Float64}(I, dim, dim),
                    lanczos_maxiter=2dim)
                @test abs(r.loglik - exact) < 1e-6
                rr = stochastic_loglik(model; K=300, seed=11)
                @test abs(rr.loglik - exact) < 4 * rr.se
                @test rr.se < 0.5
                @printf("[post] H=%d peer=%s psi=%.1f loglik exact=%.6f basis err=%.1e rademacher=%.4f±%.4f\n",
                    case.H, case.peer_env, case.psi, exact, abs(r.loglik - exact), rr.loglik, rr.se)
            end

            @testset "psi_score" begin
                h = 1e-5
                score_fd = (post_fit_dense_loglik(f, case, θ, case.psi + h) -
                            post_fit_dense_loglik(f, case, θ, case.psi - h)) / 2h
                r = psi_score(model; probes=sqrt(n) .* Matrix{Float64}(I, n, n))
                @test abs(r.score - score_fd) < 1e-6

                # Augmented AI against dense ½ (V_i Py)' P (V_j Py), with V̇ by central differences.
                V = post_fit_dense_covariance(f, case, θ, case.psi)
                Vinv = inv(V)
                P = Vinv - Vinv * f.X * ((f.X' * Vinv * f.X) \ (f.X' * Vinv))
                Py = P * f.y
                Vdot = (post_fit_dense_covariance(f, case, θ, case.psi + h) -
                        post_fit_dense_covariance(f, case, θ, case.psi - h)) ./ 2h
                Zs = (f.m_obs .- 1.0) .^ (-case.psi) .* f.Zs1
                derivatives = [f.Zd * f.A * f.Zd', Zs * f.A * Zs',
                    f.Zd * f.A * Zs' + Zs * f.A * f.Zd', Matrix{Float64}(I, n, n)]
                append!(derivatives, [f.Zh[k] * f.Zh[k]' for k in 1:case.H])
                case.peer_env && push!(derivatives, Zs * Zs')
                push!(derivatives, Vdot)
                Q = hcat([D * Py for D in derivatives]...)
                AI_ref = 0.5 .* (Q' * P * Q)
                @test maximum(abs.(r.AI - AI_ref)) < 1e-6
                p = length(θ)
                eff_ref = AI_ref[end, end] - dot(AI_ref[end, 1:p], AI_ref[1:p, 1:p] \ AI_ref[1:p, end])
                # Near-cancelling difference in this tiny design: compare on an absolute scale.
                @test r.efficient_info ≈ eff_ref atol = 1e-6
                @test r.efficient_info > 0

                rr = psi_score(model; K=300, seed=5)
                @test abs(rr.score - score_fd) < 4 * rr.score_se
                @printf("[post] H=%d peer=%s psi=%.1f psi-score fd=%.6f basis err=%.1e rademacher=%.4f±%.4f z=%.2f\n",
                    case.H, case.peer_env, case.psi, score_fd, abs(r.score - score_fd),
                    rr.score, rr.score_se, r.z)
            end
        end
    end

    @testset "paired loglik difference (sum vs mean)" begin
        case0 = (H=1, peer_env=true, psi=0.0, covariance=0.08)
        case1 = (H=1, peer_env=true, psi=1.0, covariance=0.08)
        m0, m1 = post_fit_model(f, case0), post_fit_model(f, case1)
        θ = get_params(m0)
        exact = post_fit_dense_loglik(f, case1, θ, 1.0) - post_fit_dense_loglik(f, case0, θ, 0.0)
        a = stochastic_loglik(m1; K=200, seed=3)
        b = stochastic_loglik(m0; K=200, seed=3)
        d = loglik_difference(a, b)
        @test d.paired
        @test abs(d.Δ - exact) < 4 * d.se
        @test d.se < sqrt(a.se^2 + b.se^2)
        @printf("[post] Δloglik(ψ=1 - ψ=0) exact=%.4f paired=%.4f±%.4f (unpaired se %.4f)\n",
            exact, d.Δ, d.se, sqrt(a.se^2 + b.se^2))
    end
end
