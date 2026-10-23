if !@isdefined(score_ai_dense_fixture)
    include("test_score_ai.jl")
end
using Test

module DirectReturnContracts
    include(joinpath(@__DIR__, "..", "emai_direct.jl"))
end

@testset "mutating return contracts" begin
    f = score_ai_dense_fixture()
    @testset "SGE peer=$peer" for peer in (false, true)
        model = SGEModel(f.y, f.X, f.Ainv, f.L_Ainv, f.groups,
            1.2, 0.2, 0.12, 0.9; env_ids=deepcopy(f.env_ids),
            σ2_env=[0.4, 0.25], peer_env=peer, σ2_pe=0.15)
        @test update_params!(model, zeros(length(get_params(model))), 0.5) === model
        workspaces = create_thread_workspaces(model, Threads.maxthreadid())
        preconds = [get_preconditioner(model, ws) for ws in workspaces]
        ws, precond = workspaces[1], preconds[1]
        rhs = zeros(get_dim(model))
        @test build_rhs!(rhs, model.y, model, ws) === rhs
        @test solve_mme_y!(model, ws, precond; verbose=false, reltol=1e-12) === model
        op = make_mme_operator(model, ws)
        actual = similar(rhs)
        @test mul!(actual, op, model.sol_y) === actual
        @test actual ≈ rhs rtol=1e-9 atol=1e-10
        @test mul!(actual, op, model.sol_y, 1.0, 0.0) === actual
        @test ldiv!(actual, precond, rhs) === actual
        @test ldiv!(actual, get_diag_preconditioner(model), rhs) === actual

        N, n = size(model.Ainv, 1), length(model.y)
        out_d, out_s = zeros(N), zeros(N)
        blocks = apply_GinvAinv!(out_d, out_s, ones(N), ones(N),
            model.Ainv, ws.op_Ginv, ws.op_Ad, ws.op_As)
        @test blocks[1] === out_d
        @test blocks[2] === out_s
        @test apply_A!(out_d, model.L_Ainv, ones(N), ws.tmp_N1) === out_d
        @test compute_group_sums!(ws.group_sum, out_d, model.groups.members_id) === ws.group_sum
        @test apply_Zs_transpose!(out_d, model.y, model.groups,
            ws.group_sum, model.group_scale) === out_d
        @test apply_ZsZs_accumulate!(out_s, out_d, model.groups, model.group_scale) === out_s
        projected = zeros(n)
        @test compute_Px_from_sol!(projected, model, model.y, model.sol_y, ws) === projected
        comps = get_variance_components(model)
        for comp in comps
            @test get_Vi_u!(ws.vi_out, projected, comp, model, ws) === ws.vi_out
        end

        sol_z = [zeros(get_dim(model))]
        Pz = [zeros(n)]
        z = [copy(model.y)]
        @test solve_mme_z!(sol_z[1], model, z[1], ws, precond; reltol=1e-12) === sol_z[1]
        @test solve_mme_direct!(sol_z[1], model, z[1], ws, precond; reltol=1e-12) === sol_z[1]
        @test _threaded_z_solves!(sol_z, model, z, Pz, workspaces,
            preconds, 1, 1e-12, 2000) === sol_z
        @test Pz[1] ≈ projected rtol=1e-9 atol=1e-10
        Vi = [zeros(n) for _ in comps]
        @test _threaded_vi_py!(Vi, projected, comps, model, workspaces) === Vi
        traces = zeros(length(comps))
        @test _threaded_trace!(traces, length(comps), 1, Pz, z,
            comps, model, workspaces) === traces
        PVi = [zeros(n) for _ in comps]
        @test _threaded_ai_cg!(PVi, Vi, model, workspaces,
            preconds, 1e-12, 2000) === PVi
        pairs = [(i, j) for i in eachindex(comps) for j in 1:i]
        ai = zeros(length(pairs))
        @test _threaded_ai_matvec!(ai, pairs, PVi, projected,
            comps, model, workspaces) === ai
    end

    @testset "direct model" begin
        D = DirectReturnContracts
        model = D.DirectModel(f.y, f.X, f.Ainv, f.L_Ainv,
            f.groups.obs_to_id, 1.2, 0.9)
        @test D.update_params!(model, [0.2, -0.4], 0.5) === model
        @test model.σ2_d ≈ 1.3
        @test model.σ2_e ≈ 0.7
        dim = length(model.sol_y)
        rhs = zeros(dim)
        @test D.build_rhs!(rhs, model, model.y) === rhs
        @test D.build_rhs_threaded!(rhs, model, model.y) === rhs
        W = hcat(f.X, f.Zd)
        @test rhs ≈ W' * model.y
        precond = D.get_diag_preconditioner(model)
        push!(model.sol_z, zeros(dim))
        for verbose in (false, true)
            fill!(model.sol_y, 0.0)
            fill!(model.sol_z[1], 0.0)
            @test D.solve_mme_y!(model, precond; verbose, reltol=1e-12) === model
            @test D.solve_mme_z!(model, model.y, 1, precond; verbose, reltol=1e-12) === model
            @test model.sol_z[1] ≈ model.sol_y
        end
        dest = zeros(dim)
        ws = D.create_thread_workspaces(model, 1)[1]
        @test D.solve_mme_direct!(dest, model, model.y, precond; reltol=1e-12) === dest
        @test D.solve_mme_direct_threaded!(dest, model, model.y, ws, precond; reltol=1e-12) === dest
        @test D.solve_mme_z_threaded!(dest, model, model.y, ws, precond; reltol=1e-12) === dest
        @test dest ≈ model.sol_y
        out = zeros(size(model.Ainv, 1))
        @test D.apply_A!(out, model, ones(length(out))) === out
        @test D.apply_A_threaded!(out, model.L_Ainv, ones(length(out)), ws.tmp_N1) === out
    end
end
