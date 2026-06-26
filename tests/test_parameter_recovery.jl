# Parameter recovery test on a dedicated zero-environment dataset.
if !@isdefined(SGEModel)
    include(joinpath(@__DIR__, "..", "src", "emai.jl"))
end
using Printf

const RECOVERY_DIR = joinpath(@__DIR__, "..", "outputs", "parameter_recovery")
const RECOVERY_SIM = joinpath(@__DIR__, "..", "sim_parameter_recovery.R")
const RECOVERY_NG = parse(Int, get(ENV, "PARAM_RECOVERY_NG", "5000"))
const RECOVERY_MISSING_FRAC = parse(Float64, get(ENV, "PARAM_RECOVERY_MISSING_FRAC", "0.10"))

function check(cond::Bool, msg::String)
    cond || error(msg)
end

function ensure_recovery_data()
    required = [
        joinpath(RECOVERY_DIR, "yX_obs.csv"),
        joinpath(RECOVERY_DIR, "roster.csv"),
        joinpath(RECOVERY_DIR, "Ainv_triplet.csv"),
        joinpath(RECOVERY_DIR, "LAinv_triplet.csv"),
        joinpath(RECOVERY_DIR, "truth.csv"),
    ]
    regenerate = get(ENV, "PARAM_RECOVERY_REGENERATE", "0") == "1"
    if regenerate || any(path -> !isfile(path), required)
        rscript = Sys.which("Rscript")
        rscript === nothing && error("Rscript not found; cannot generate parameter-recovery dataset")
        mkpath(RECOVERY_DIR)
        println("[param-recovery] Generating dedicated dataset...")
        cmd = setenv(`$(rscript) $(RECOVERY_SIM)`, merge(copy(ENV), Dict(
            "SIM_OUT_DIR" => RECOVERY_DIR,
            "SIM_NG" => string(RECOVERY_NG),
            "SIM_MISSING_FRAC" => string(RECOVERY_MISSING_FRAC),
        )))
        run(cmd)
    end
end

function load_recovery_inputs()
    obs_table = DataFrame(CSV.File(joinpath(RECOVERY_DIR, "yX_obs.csv")))
    roster = DataFrame(CSV.File(joinpath(RECOVERY_DIR, "roster.csv")))

    obs_to_id = Int.(obs_table[:, :id_ped])
    y_obs = Float64.(obs_table[:, :y])
    X_obs = hcat(ones(length(y_obs)), Float64.(obs_table[:, :x]))

    Ainv_df = DataFrame(CSV.File(joinpath(RECOVERY_DIR, "Ainv_triplet.csv")))
    i_v = Ainv_df[:, "i"]
    j_v = Ainv_df[:, "j"]
    x_v = Float64.(Ainv_df[:, "x"])
    od = i_v .!= j_v
    Ainv = sparse(vcat(i_v, j_v[od]), vcat(j_v, i_v[od]), vcat(x_v, x_v[od]))
    N_pedigree = size(Ainv, 1)

    L_df = DataFrame(CSV.File(joinpath(RECOVERY_DIR, "LAinv_triplet.csv")))
    L_Ainv = LowerTriangular(sparse(L_df[:, "i"], L_df[:, "j"], Float64.(L_df[:, "x"])))

    roster_ids = Int.(roster[:, :id_ped])
    roster_groups = Int.(roster[:, :group])
    lookup = zeros(Int, N_pedigree)
    for (id, g) in zip(roster_ids, roster_groups)
        lookup[id] = g
    end
    obs_group_ids = [lookup[id] for id in obs_to_id]

    gs = GroupStructure(obs_group_ids, obs_to_id, N_pedigree;
        all_group_ids=roster_groups, all_animal_ids=roster_ids)

    truth = DataFrame(CSV.File(joinpath(RECOVERY_DIR, "truth.csv")))
    check(nrow(truth) == 1, "truth.csv must contain exactly one row")

    return (; obs_to_id, y_obs, X_obs, Ainv, L_Ainv, gs, truth=truth[1, :])
end

ensure_recovery_data()
println("[param-recovery] Loading dedicated dataset...")
inputs = load_recovery_inputs()

@printf("[param-recovery] N_pedigree=%d  n_obs=%d  n_groups=%d  missing=%.2f\n",
    size(inputs.Ainv, 1), length(inputs.y_obs), length(inputs.gs.members_id), inputs.truth.missing_frac)

model = SGEModel(inputs.y_obs, inputs.X_obs, inputs.Ainv, inputs.L_Ainv, inputs.gs,
    1.0, 0.2, 0.0, 2.0)

println("[param-recovery] Running AI-REML (max_iter=15, k=20, threaded=true)...")
result = run_aireml(
    model,
    max_iter=15,
    k=20,
    damping=0.5,
    cg_maxiter=5000,
    cg_reltol_early=1e-5,
    cg_early_iters=4,
    cg_reltol=1e-7,
    threaded=true,
    verbose_cg=false,
)

true_σ2_d = Float64(inputs.truth.sigma2_d)
true_σ2_s = Float64(inputs.truth.sigma2_s)
true_σ_ds = Float64(inputs.truth.sigma_ds)
true_σ2_e = Float64(inputs.truth.sigma2_e)

println("\n[param-recovery] Results:")
@printf("  %-8s  %10s  %10s  %10s  %8s\n", "param", "truth", "estimate", "abs_err", "pct_err")
params = [
    (:σ2_d, true_σ2_d, result.σ2_d),
    (:σ2_s, true_σ2_s, result.σ2_s),
    (:σ_ds, true_σ_ds, result.σ_ds),
    (:σ2_e, true_σ2_e, result.σ2_e),
]
for (name, truth, est) in params
    ae = abs(est - truth)
    pe = 100.0 * ae / abs(truth)
    @printf("  %-8s  %10.4f  %10.4f  %10.4f  %7.1f%%\n", String(name), truth, est, ae, pe)
end

tol_d = 0.20 * true_σ2_d
tol_s = 0.30 * true_σ2_s
tol_ds = 0.30 * abs(true_σ_ds)
tol_e = 0.10 * true_σ2_e

check(abs(result.σ2_d - true_σ2_d) < tol_d,
    @sprintf("σ²_d recovery FAIL: est=%.4f truth=%.4f tol=%.4f", result.σ2_d, true_σ2_d, tol_d))
check(abs(result.σ2_s - true_σ2_s) < tol_s,
    @sprintf("σ²_s recovery FAIL: est=%.4f truth=%.4f tol=%.4f", result.σ2_s, true_σ2_s, tol_s))
check(abs(result.σ_ds - true_σ_ds) < tol_ds,
    @sprintf("σ_ds recovery FAIL: est=%.4f truth=%.4f tol=%.4f", result.σ_ds, true_σ_ds, tol_ds))
check(abs(result.σ2_e - true_σ2_e) < tol_e,
    @sprintf("σ²_e recovery FAIL: est=%.4f truth=%.4f tol=%.4f", result.σ2_e, true_σ2_e, tol_e))

println("\n[param-recovery] Checking ψ=0 invariance...")
model_psi = SGEModel(inputs.y_obs, inputs.X_obs, inputs.Ainv, inputs.L_Ainv, inputs.gs,
    1.0, 0.2, 0.0, 2.0; ψ=0.0)
result_psi = run_aireml(
    model_psi,
    max_iter=3,
    k=5,
    damping=0.5,
    cg_maxiter=500,
    cg_reltol=1e-6,
    threaded=false,
    verbose_cg=false,
    seed=99,
)
model_def = SGEModel(inputs.y_obs, inputs.X_obs, inputs.Ainv, inputs.L_Ainv, inputs.gs,
    1.0, 0.2, 0.0, 2.0)
result_def = run_aireml(
    model_def,
    max_iter=3,
    k=5,
    damping=0.5,
    cg_maxiter=500,
    cg_reltol=1e-6,
    threaded=false,
    verbose_cg=false,
    seed=99,
)
for field in (:σ2_d, :σ2_s, :σ_ds, :σ2_e)
    v1 = getfield(result_psi, field)
    v2 = getfield(result_def, field)
    check(abs(v1 - v2) < 1e-10,
        "ψ=0 invariance FAIL for $field: ψ=0 gives $v1, default gives $v2")
end

println("[param-recovery] ALL PASS")
