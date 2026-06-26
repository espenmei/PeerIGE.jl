include(joinpath(@__DIR__, "..", "src", "emai.jl"))

println("[psi-dilution] Running dilution parameter ψ tests...")

function check(cond::Bool, msg::String)
    cond || error(msg)
end

# -----------------------------------------------------------------------
# Helper: build Zs densely for a GroupStructure so we can brute-force
# operator correctness.  Zs[i, bid] = 1 iff bid ∈ group(i) and bid ≠ self.
# -----------------------------------------------------------------------
function build_Zs_dense(gs::GroupStructure, n::Int, N::Int)
    Zs = zeros(n, N)
    for i in 1:n
        g = gs.obs_to_group[i]
        for bid in gs.members_id[g]
            if bid != gs.obs_to_id[i]
                Zs[i, bid] = 1.0
            end
        end
    end
    return Zs
end

# Build Zs with dilution factor c_g applied row-wise (each row i scaled by c_{g(i)})
function build_Zs_psi_dense(gs::GroupStructure, n::Int, N::Int, group_scale::Vector{Float64})
    Zs = build_Zs_dense(gs, n, N)
    for i in 1:n
        Zs[i, :] .*= group_scale[gs.obs_to_group[i]]
    end
    return Zs
end

# -----------------------------------------------------------------------
# Tiny 3-group example:
#   Group 1: pedigree IDs 1,2,3   (3 members, all observed as obs rows 1,2,3)
#   Group 2: pedigree IDs 4,5,6   (3 members; obs rows 4,5; ID 6 unobserved)
#   Group 3: pedigree IDs 7       (singleton — should contribute nothing)
# -----------------------------------------------------------------------
N_ped = 7
n_obs = 5
obs_group = [1, 1, 1, 2, 2]
obs_ids   = [1, 2, 3, 4, 5]
all_group = [1, 1, 1, 2, 2, 2, 3]
all_ids   = [1, 2, 3, 4, 5, 6, 7]

gs = GroupStructure(obs_group, obs_ids, N_ped;
    all_group_ids=all_group, all_animal_ids=all_ids)

group_scale_psi0 = [length(m) <= 1 ? 0.0 : 1.0 for m in gs.members_id]  # ψ=0 → all 1.0 (except singleton)
group_scale_psi1 = [length(m) <= 1 ? 0.0 : 1.0 / (length(m) - 1) for m in gs.members_id]  # ψ=1

println("[psi-dilution] Test 1: apply_Zs_transpose! at ψ=0 matches dense Zs'*u ...")
begin
    Zs_dense = build_Zs_psi_dense(gs, n_obs, N_ped, group_scale_psi0)
    u = randn(n_obs)
    expected = Zs_dense' * u

    group_sum = zeros(length(gs.members_id))
    v_N = zeros(N_ped)
    apply_Zs_transpose!(v_N, u, gs, group_sum, group_scale_psi0)

    diff = maximum(abs.(v_N .- expected))
    check(diff < 1e-12, "Test 1 FAIL: max diff = $diff")
    println("[psi-dilution] Test 1 PASS (apply_Zs_transpose! ψ=0, diff=$diff)")
end

println("[psi-dilution] Test 2: apply_Zs_transpose! at ψ=1 matches dense Zs_ψ'*u ...")
begin
    Zs_psi1 = build_Zs_psi_dense(gs, n_obs, N_ped, group_scale_psi1)
    u = randn(n_obs)
    expected = Zs_psi1' * u

    group_sum = zeros(length(gs.members_id))
    v_N = zeros(N_ped)
    apply_Zs_transpose!(v_N, u, gs, group_sum, group_scale_psi1)

    diff = maximum(abs.(v_N .- expected))
    check(diff < 1e-12, "Test 2 FAIL: max diff = $diff")
    println("[psi-dilution] Test 2 PASS (apply_Zs_transpose! ψ=1, diff=$diff)")
end

println("[psi-dilution] Test 3: apply_ZsZs_accumulate! at ψ=0 matches dense Zs'Zs*v ...")
begin
    Zs_dense = build_Zs_psi_dense(gs, n_obs, N_ped, group_scale_psi0)
    v = randn(N_ped)
    expected = Zs_dense' * (Zs_dense * v)

    u_N = zeros(N_ped)
    apply_ZsZs_accumulate!(u_N, v, gs, group_scale_psi0)

    diff = maximum(abs.(u_N .- expected))
    check(diff < 1e-12, "Test 3 FAIL: max diff = $diff")
    println("[psi-dilution] Test 3 PASS (apply_ZsZs_accumulate! ψ=0, diff=$diff)")
end

println("[psi-dilution] Test 4: apply_ZsZs_accumulate! at ψ=1 matches dense Zs_ψ'Zs_ψ*v ...")
begin
    Zs_psi1 = build_Zs_psi_dense(gs, n_obs, N_ped, group_scale_psi1)
    v = randn(N_ped)
    expected = Zs_psi1' * (Zs_psi1 * v)

    u_N = zeros(N_ped)
    apply_ZsZs_accumulate!(u_N, v, gs, group_scale_psi1)

    diff = maximum(abs.(u_N .- expected))
    check(diff < 1e-12, "Test 4 FAIL: max diff = $diff")
    println("[psi-dilution] Test 4 PASS (apply_ZsZs_accumulate! ψ=1, diff=$diff)")
end

println("[psi-dilution] Test 5: group_scale computed correctly in SGEModel constructor ...")
begin
    # Use the real data to build a model and check group_scale values
    obs_table = DataFrame(CSV.File(joinpath(@__DIR__, "..", "yX_obs.csv")))
    roster = DataFrame(CSV.File(joinpath(@__DIR__, "..", "roster.csv")))
    obs = load_observed_data(obs_table; covariate_cols=[:x])
    Ainv_df = DataFrame(CSV.File(joinpath(@__DIR__, "..", "Ainv_triplet.csv")))
    i_vals = Ainv_df[:, "i"]; j_vals = Ainv_df[:, "j"]; x_vals = Float64.(Ainv_df[:, "x"])
    off_diag = i_vals .!= j_vals
    Ainv = sparse(vcat(i_vals, j_vals[off_diag]), vcat(j_vals, i_vals[off_diag]),
                  vcat(x_vals, x_vals[off_diag]))
    L_Ainv_df = DataFrame(CSV.File(joinpath(@__DIR__, "..", "LAinv_triplet.csv")))
    L_Ainv = LowerTriangular(sparse(L_Ainv_df[:,"i"], L_Ainv_df[:,"j"], Float64.(L_Ainv_df[:,"x"])))
    gs_real = build_group_structure(obs_table, roster, size(Ainv,1))

    # ψ=0: all non-singleton groups should have group_scale=1.0
    m0 = SGEModel(obs.y, obs.X, Ainv, L_Ainv, gs_real, 1.0, 0.2, 0.0, 2.0; ψ=0.0)
    check(m0.ψ == 0.0, "Test 5: ψ field should be 0.0")
    for g in eachindex(gs_real.members_id)
        mg = length(gs_real.members_id[g])
        expected_scale = mg <= 1 ? 0.0 : 1.0
        check(m0.group_scale[g] == expected_scale,
              "Test 5: group_scale[$g] wrong for ψ=0 (m_g=$mg): got $(m0.group_scale[g])")
    end
    println("[psi-dilution] Test 5a PASS (ψ=0 group_scale all 1.0)")

    # ψ=1: group_scale[g] = 1/(m_g-1) for non-singletons
    m1 = SGEModel(obs.y, obs.X, Ainv, L_Ainv, gs_real, 1.0, 0.2, 0.0, 2.0; ψ=1.0)
    check(m1.ψ == 1.0, "Test 5: ψ field should be 1.0")
    for g in eachindex(gs_real.members_id)
        mg = length(gs_real.members_id[g])
        expected_scale = mg <= 1 ? 0.0 : 1.0 / (mg - 1)
        diff = abs(m1.group_scale[g] - expected_scale)
        check(diff < 1e-12,
              "Test 5: group_scale[$g] wrong for ψ=1 (m_g=$mg): got $(m1.group_scale[g]), expected $expected_scale")
    end
    println("[psi-dilution] Test 5b PASS (ψ=1 group_scale = 1/(m_g-1))")
end

println("[psi-dilution] Test 6: ψ=0 AiRemlResult matches two independent runs (reproducibility) ...")
begin
    obs_table = DataFrame(CSV.File(joinpath(@__DIR__, "..", "yX_obs.csv")))
    roster = DataFrame(CSV.File(joinpath(@__DIR__, "..", "roster.csv")))
    obs = load_observed_data(obs_table; covariate_cols=[:x])
    Ainv_df = DataFrame(CSV.File(joinpath(@__DIR__, "..", "Ainv_triplet.csv")))
    i_vals = Ainv_df[:, "i"]; j_vals = Ainv_df[:, "j"]; x_vals = Float64.(Ainv_df[:, "x"])
    off_diag = i_vals .!= j_vals
    Ainv = sparse(vcat(i_vals, j_vals[off_diag]), vcat(j_vals, i_vals[off_diag]),
                  vcat(x_vals, x_vals[off_diag]))
    L_Ainv_df = DataFrame(CSV.File(joinpath(@__DIR__, "..", "LAinv_triplet.csv")))
    L_Ainv = LowerTriangular(sparse(L_Ainv_df[:,"i"], L_Ainv_df[:,"j"], Float64.(L_Ainv_df[:,"x"])))
    gs_real = build_group_structure(obs_table, roster, size(Ainv,1))

    run_opts = (max_iter=2, k=3, damping=0.5, cg_maxiter=300, ai_maxiter=100,
                threaded=false, verbose_cg=false, seed=42)

    mA = SGEModel(obs.y, obs.X, Ainv, L_Ainv, gs_real, 1.0, 0.2, 0.0, 2.0; ψ=0.0)
    rA = run_aireml(mA; run_opts...)

    mB = SGEModel(obs.y, obs.X, Ainv, L_Ainv, gs_real, 1.0, 0.2, 0.0, 2.0; ψ=0.0)
    rB = run_aireml(mB; run_opts...)

    check(rA.σ2_d == rB.σ2_d, "Test 6: σ2_d differs between runs")
    check(rA.σ2_s == rB.σ2_s, "Test 6: σ2_s differs between runs")
    check(rA.σ_ds == rB.σ_ds, "Test 6: σ_ds differs between runs")
    check(rA.σ2_e == rB.σ2_e, "Test 6: σ2_e differs between runs")
    check(rA.a_d == rB.a_d, "Test 6: a_d differs between runs")
    check(rA.a_s == rB.a_s, "Test 6: a_s differs between runs")
    check(rA.ψ == 0.0, "Test 6: result.ψ should be 0.0")
    println("[psi-dilution] Test 6 PASS (ψ=0 reproducible)")
end

println("[psi-dilution] Test 7: ψ=0 and default (implicit ψ=0) give identical results ...")
begin
    obs_table = DataFrame(CSV.File(joinpath(@__DIR__, "..", "yX_obs.csv")))
    roster = DataFrame(CSV.File(joinpath(@__DIR__, "..", "roster.csv")))
    obs = load_observed_data(obs_table; covariate_cols=[:x])
    Ainv_df = DataFrame(CSV.File(joinpath(@__DIR__, "..", "Ainv_triplet.csv")))
    i_vals = Ainv_df[:, "i"]; j_vals = Ainv_df[:, "j"]; x_vals = Float64.(Ainv_df[:, "x"])
    off_diag = i_vals .!= j_vals
    Ainv = sparse(vcat(i_vals, j_vals[off_diag]), vcat(j_vals, i_vals[off_diag]),
                  vcat(x_vals, x_vals[off_diag]))
    L_Ainv_df = DataFrame(CSV.File(joinpath(@__DIR__, "..", "LAinv_triplet.csv")))
    L_Ainv = LowerTriangular(sparse(L_Ainv_df[:,"i"], L_Ainv_df[:,"j"], Float64.(L_Ainv_df[:,"x"])))
    gs_real = build_group_structure(obs_table, roster, size(Ainv,1))

    run_opts = (max_iter=2, k=3, damping=0.5, cg_maxiter=300, ai_maxiter=100,
                threaded=false, verbose_cg=false, seed=42)

    m_explicit = SGEModel(obs.y, obs.X, Ainv, L_Ainv, gs_real, 1.0, 0.2, 0.0, 2.0; ψ=0.0)
    r_explicit = run_aireml(m_explicit; run_opts...)

    m_default = SGEModel(obs.y, obs.X, Ainv, L_Ainv, gs_real, 1.0, 0.2, 0.0, 2.0)
    r_default = run_aireml(m_default; run_opts...)

    check(r_explicit.σ2_d == r_default.σ2_d, "Test 7: σ2_d: explicit ψ=0 ≠ default ψ=0")
    check(r_explicit.σ2_s == r_default.σ2_s, "Test 7: σ2_s: explicit ψ=0 ≠ default ψ=0")
    check(r_explicit.σ_ds == r_default.σ_ds, "Test 7: σ_ds: explicit ψ=0 ≠ default ψ=0")
    check(r_explicit.σ2_e == r_default.σ2_e, "Test 7: σ2_e: explicit ψ=0 ≠ default ψ=0")
    check(r_explicit.a_d == r_default.a_d, "Test 7: a_d: explicit ψ=0 ≠ default ψ=0")
    check(r_explicit.a_s == r_default.a_s, "Test 7: a_s: explicit ψ=0 ≠ default ψ=0")
    println("[psi-dilution] Test 7 PASS (explicit ψ=0 == default)")
end

println("[psi-dilution] ALL PASS")
