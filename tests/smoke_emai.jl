include(joinpath(@__DIR__, "..", "src", "emai.jl"))

println("[Smoke test] Running quick SGE model verification...")

# Keep this test short and deterministic.
const MAX_ITER = 2
const K_PROBES = 2
const CG_MAXITER = 300
const AI_MAXITER = 100
const LOG_FILE = joinpath("outputs", "smoke", "smoke_test_aireml_log.csv")
mkpath(dirname(LOG_FILE))

function check(cond::Bool, msg::String)
    cond || error(msg)
end

try
    # --- TEST 0: GroupStructure supports full membership mapping with unobserved members ---
    println("[Smoke test] Test 0: GroupStructure full membership mapping...")
    gs0 = GroupStructure([10, 10], [1, 2], 4;
        all_group_ids=[10, 10, 10], all_animal_ids=[1, 2, 3])
    gidx = gs0.id_to_group[1]
    check(gidx != 0, "Test 0: observed ID 1 should map to a group")
    check(3 in gs0.members_id[gidx], "Test 0: unobserved member ID 3 should be in group members")
    println("[Smoke test] Test 0 PASS (full membership mapping)")

    # --- TEST 0b: load_observed_data filters missing outcomes correctly ---
    println("[Smoke test] Test 0b: observed-row filtering from master table...")
    yX0 = DataFrame(y=Union{Missing,Float64}[1.0, missing, 3.0],
        x=[0.1, 0.2, 0.3], group=[10, 10, 10], id_ped=[1, 2, 3])
    obs0 = load_observed_data(yX0; covariate_cols=[:x])
    check(length(obs0.y) == 2, "Test 0b: expected 2 observed outcomes")
    check(size(obs0.X) == (2, 2), "Test 0b: expected intercept + one covariate")
    check(obs0.obs_to_id == [1, 3], "Test 0b: observed pedigree IDs mismatch")
    gs0b = build_group_structure(yX0, 3, obs0.obs_mask)
    check(2 in gs0b.members_id[gs0b.id_to_group[1]], "Test 0b: missing-outcome member should remain in group")
    println("[Smoke test] Test 0b PASS (observed filtering)")

    # --- TEST 0c: small groups keep missing-outcome peers in the social design ---
    println("[Smoke test] Test 0c: small-group peer counts with missing outcomes...")
    yX0c4 = DataFrame(y=Union{Missing,Float64}[1.0, 2.0, 3.0, missing],
        x=[0.1, 0.2, 0.3, 0.4], group=[10, 10, 10, 10], id_ped=[1, 2, 3, 4])
    obs0c4 = load_observed_data(yX0c4; covariate_cols=[:x])
    gs0c4 = build_group_structure(yX0c4, 4, obs0c4.obs_mask)
    a_s4 = collect(1.0:4.0)
    for id in obs0c4.obs_to_id
        members = gs0c4.members_id[gs0c4.id_to_group[id]]
        check(length(members) == 4, "Test 0c(4): expected full class membership for ID $id")
        check(length(members) - 1 == 3, "Test 0c(4): expected 3 peers for ID $id")
        social_sum = sum(a_s4[m] for m in members) - a_s4[id]
        check(social_sum == sum(setdiff(members, [id])), "Test 0c(4): social sum mismatch for ID $id")
    end

    yX0c3 = DataFrame(y=Union{Missing,Float64}[1.0, 2.0, missing],
        x=[0.1, 0.2, 0.3], group=[20, 20, 20], id_ped=[5, 6, 7])
    obs0c3 = load_observed_data(yX0c3; covariate_cols=[:x])
    gs0c3 = build_group_structure(yX0c3, 7, obs0c3.obs_mask)
    a_s3 = collect(1.0:7.0)
    for id in obs0c3.obs_to_id
        members = gs0c3.members_id[gs0c3.id_to_group[id]]
        check(length(members) == 3, "Test 0c(3): expected full class membership for ID $id")
        check(length(members) - 1 == 2, "Test 0c(3): expected 2 peers for ID $id")
        social_sum = sum(a_s3[m] for m in members) - a_s3[id]
        check(social_sum == sum(setdiff(members, [id])), "Test 0c(3): social sum mismatch for ID $id")
    end
    println("[Smoke test] Test 0c PASS (small-group peer counts)")

    # Load the same inputs as the main runner.
    obs_table = DataFrame(CSV.File("yX_obs.csv"))
    roster = DataFrame(CSV.File("roster.csv"))
    y = Float64.(obs_table[:, :y])
    X = hcat(ones(length(y)), Float64.(obs_table[:, :x]))
    obs_to_id = Int.(obs_table[:, :id_ped])

    Ainv_df = DataFrame(CSV.File("Ainv_triplet.csv"))
    i_vals = Ainv_df[:, "i"]
    j_vals = Ainv_df[:, "j"]
    x_vals = Float64.(Ainv_df[:, "x"])

    off_diag = i_vals .!= j_vals
    i_all = vcat(i_vals, j_vals[off_diag])
    j_all = vcat(j_vals, i_vals[off_diag])
    x_all = vcat(x_vals, x_vals[off_diag])

    Ainv = sparse(i_all, j_all, x_all)
    N_pedigree = size(Ainv, 1)

    L_Ainv_df = DataFrame(CSV.File("LAinv_triplet.csv"))
    L_Ainv = sparse(L_Ainv_df[:, "i"], L_Ainv_df[:, "j"], Float64.(L_Ainv_df[:, "x"]))
    L_Ainv = LowerTriangular(L_Ainv)

    roster_ids = Int.(roster[:, :id_ped])
    roster_groups = Int.(roster[:, :group])
    obs_group_lookup = zeros(Int, N_pedigree)
    for (id, g) in zip(roster_ids, roster_groups)
        obs_group_lookup[id] = g
    end
    obs_group_ids = [obs_group_lookup[id] for id in obs_to_id]
    gs = GroupStructure(obs_group_ids, obs_to_id, N_pedigree;
        all_group_ids=roster_groups, all_animal_ids=roster_ids)

    # --- TEST 1: H=0 (regression, same as before) ---
    println("[Smoke test] Test 1: H=0 (no env effects)...")
    model0 = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0)

    fit_aireml!(
        model0,
        max_iter=MAX_ITER,
        k=K_PROBES,
        damping=0.5,
        cg_maxiter=CG_MAXITER,
        ai_maxiter=AI_MAXITER,
        verbose_cg=false,
        log_file=LOG_FILE,
        seed=42,
    )
    result0 = snapshot(model0; include_fixef_se=false)

    check(isfinite(result0.σ2_d), "σ2_d is not finite")
    check(isfinite(result0.σ2_s), "σ2_s is not finite")
    check(isfinite(result0.σ_ds), "σ_ds is not finite")
    check(isfinite(result0.σ2_e), "σ2_e is not finite")
    check(result0.σ2_d > 0, "σ2_d must be > 0")
    check(result0.σ2_s > 0, "σ2_s must be > 0")
    check(result0.σ2_e > 0, "σ2_e must be > 0")
    check(result0.σ2_d * result0.σ2_s > result0.σ_ds^2, "G matrix is not positive definite")
    check(length(result0.beta) == size(X, 2), "beta length mismatch")
    check(length(result0.a_d) == size(Ainv, 1), "a_d length mismatch")
    check(length(result0.a_s) == size(Ainv, 1), "a_s length mismatch")
    check(isempty(result0.σ2_env), "H=0 should have empty σ2_env")
    check(isempty(result0.c_env), "H=0 should have empty c_env")
    check(size(result0.AI) == (4, 4), "AI matrix has wrong shape for H=0")
    check(length(result0.se) == 4, "SE vector has wrong length for H=0")
    println("[Smoke test] Test 1 PASS (H=0)")

    # --- TEST 2: H=1 (one env factor, using group as env_ids) ---
    println("[Smoke test] Test 2: H=1 (one env factor)...")
    env1_ids = obs_group_ids
    model1 = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0;
        env_ids=[env1_ids], σ2_env=[0.5])

    fit_aireml!(
        model1,
        max_iter=MAX_ITER,
        k=K_PROBES,
        damping=0.5,
        cg_maxiter=CG_MAXITER,
        ai_maxiter=AI_MAXITER,
        verbose_cg=false,
        log_file=LOG_FILE,
        seed=42,
    )
    result1 = snapshot(model1; include_fixef_se=false)

    check(isfinite(result1.σ2_d), "H=1: σ2_d is not finite")
    check(isfinite(result1.σ2_e), "H=1: σ2_e is not finite")
    check(result1.σ2_d > 0, "H=1: σ2_d must be > 0")
    check(result1.σ2_e > 0, "H=1: σ2_e must be > 0")
    check(length(result1.σ2_env) == 1, "H=1: should have 1 env variance")
    check(result1.σ2_env[1] > 0, "H=1: σ2_env[1] must be > 0")
    check(length(result1.c_env) == 1, "H=1: should have 1 env BLUP vector")
    n_levels_expected = maximum(env1_ids)
    check(length(result1.c_env[1]) == n_levels_expected, "H=1: env BLUP length mismatch")
    check(size(result1.AI) == (5, 5), "H=1: AI matrix has wrong shape")
    check(length(result1.se) == 5, "H=1: SE vector has wrong length")
    println("[Smoke test] Test 2 PASS (H=1)")

    println("[Smoke test] ALL PASS")
catch err
    println("[Smoke test] FAIL: " * sprint(showerror, err))
    Base.show_backtrace(stdout, catch_backtrace())
    exit(1)
end
