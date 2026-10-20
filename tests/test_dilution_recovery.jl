# Conditional parameter recovery test for dilution parameter ψ.
if !@isdefined(SGEModel)
    include(joinpath(@__DIR__, "..", "src", "emai.jl"))
end
using Printf

const DILUTION_BASE_DIR = joinpath(@__DIR__, "..", "outputs", "dilution_recovery")
const DILUTION_SIM = joinpath(@__DIR__, "..", "sim_dilution.R")
const DILUTION_NG = parse(Int, get(ENV, "DILUTION_RECOVERY_NG", "5000"))
const DILUTION_MISSING_FRAC = parse(Float64, get(ENV, "DILUTION_RECOVERY_MISSING_FRAC", "0.10"))
const DILUTION_MAX_ITER = parse(Int, get(ENV, "DILUTION_RECOVERY_MAX_ITER", "15"))
const DILUTION_K = parse(Int, get(ENV, "DILUTION_RECOVERY_K", "20"))
const DILUTION_CG_MAXITER = parse(Int, get(ENV, "DILUTION_RECOVERY_CG_MAXITER", "5000"))
const DILUTION_AI_MAXITER = parse(Int, get(ENV, "DILUTION_RECOVERY_AI_MAXITER", "500"))
const DILUTION_PSI_VALUES = begin
    raw = get(ENV, "DILUTION_RECOVERY_PSI_VALUES", "0.0,0.5,1.0")
    parts = split(replace(raw, "," => " "))
    Float64[parse(Float64, part) for part in parts if !isempty(part)]
end

function check(cond::Bool, msg::String)
    cond || error(msg)
end

function psi_dirname(psi::Float64)
    @sprintf("psi_%.1f", psi)
end

function ensure_dilution_data(psi::Float64)
    out_dir = joinpath(DILUTION_BASE_DIR, psi_dirname(psi))
    required = [
        joinpath(out_dir, "yX_obs.csv"),
        joinpath(out_dir, "roster.csv"),
        joinpath(out_dir, "Ainv_triplet.csv"),
        joinpath(out_dir, "LAinv_triplet.csv"),
        joinpath(out_dir, "truth.csv"),
    ]
    regenerate = get(ENV, "DILUTION_RECOVERY_REGENERATE", "0") == "1"
    if regenerate || any(path -> !isfile(path), required)
        rscript = Sys.which("Rscript")
        rscript === nothing && error("Rscript not found; cannot generate dilution-recovery dataset")
        mkpath(out_dir)
        println("[dilution-recovery] Generating ψ=$(psi) dataset...")
        cmd = setenv(`$(rscript) $(DILUTION_SIM)`, merge(copy(ENV), Dict(
            "SIM_OUT_DIR" => out_dir,
            "SIM_NG" => string(DILUTION_NG),
            "SIM_MISSING_FRAC" => string(DILUTION_MISSING_FRAC),
            "SIM_DILUTION_PSI" => string(psi),
        )))
        run(cmd)
    end
    return out_dir
end

function load_dilution_inputs(out_dir::String)
    obs_table = DataFrame(CSV.File(joinpath(out_dir, "yX_obs.csv")))
    roster = DataFrame(CSV.File(joinpath(out_dir, "roster.csv")))
    obs = load_observed_data(obs_table; covariate_cols=[:x])

    Ainv_df = DataFrame(CSV.File(joinpath(out_dir, "Ainv_triplet.csv")))
    i_v = Ainv_df[:, "i"]
    j_v = Ainv_df[:, "j"]
    x_v = Float64.(Ainv_df[:, "x"])
    od = i_v .!= j_v
    Ainv = sparse(vcat(i_v, j_v[od]), vcat(j_v, i_v[od]), vcat(x_v, x_v[od]))

    L_df = DataFrame(CSV.File(joinpath(out_dir, "LAinv_triplet.csv")))
    L_Ainv = LowerTriangular(sparse(L_df[:, "i"], L_df[:, "j"], Float64.(L_df[:, "x"])))

    gs = build_group_structure(obs_table, roster, size(Ainv, 1))
    truth = DataFrame(CSV.File(joinpath(out_dir, "truth.csv")))
    check(nrow(truth) == 1, "truth.csv must contain exactly one row")

    return (; obs, Ainv, L_Ainv, gs, truth=truth[1, :])
end

println("[dilution-recovery] Running conditional ψ recovery tests...")

tol_d = 0.20
tol_s = 0.35
tol_ds = 0.35
tol_e = 0.12

for psi in DILUTION_PSI_VALUES
    out_dir = ensure_dilution_data(psi)
    inputs = load_dilution_inputs(out_dir)
    true_psi = Float64(inputs.truth.psi)
    check(abs(true_psi - psi) < 1e-12,
        @sprintf("truth ψ mismatch: expected %.3f, got %.3f", psi, true_psi))

    @printf("[dilution-recovery] ψ=%.1f  N_pedigree=%d  n_obs=%d  n_groups=%d\n",
        psi, size(inputs.Ainv, 1), length(inputs.obs.y), length(inputs.gs.members_id))

    model = SGEModel(inputs.obs.y, inputs.obs.X, inputs.Ainv, inputs.L_Ainv, inputs.gs,
        1.0, 0.2, 0.0, 2.0; ψ=psi)

    result = run_aireml(
        model,
        max_iter=DILUTION_MAX_ITER,
        k=DILUTION_K,
        damping=0.5,
        cg_maxiter=DILUTION_CG_MAXITER,
        cg_reltol_early=1e-5,
        cg_early_iters=4,
        cg_reltol=1e-7,
        ai_maxiter=DILUTION_AI_MAXITER,
        verbose_cg=false,
        seed=42,
    )

    true_σ2_d = Float64(inputs.truth.sigma2_d)
    true_σ2_s = Float64(inputs.truth.sigma2_s)
    true_σ_ds = Float64(inputs.truth.sigma_ds)
    true_σ2_e = Float64(inputs.truth.sigma2_e)

    @printf("[dilution-recovery] ψ=%.1f results: d=%.4f s=%.4f ds=%.4f e=%.4f\n",
        psi, result.σ2_d, result.σ2_s, result.σ_ds, result.σ2_e)

    check(abs(result.ψ - psi) < 1e-12,
        @sprintf("result ψ mismatch: expected %.3f, got %.3f", psi, result.ψ))
    check(abs(result.σ2_d - true_σ2_d) < tol_d * true_σ2_d,
        @sprintf("ψ=%.1f σ²_d recovery FAIL: est=%.4f truth=%.4f tol=%.4f",
            psi, result.σ2_d, true_σ2_d, tol_d * true_σ2_d))
    check(abs(result.σ2_s - true_σ2_s) < tol_s * true_σ2_s,
        @sprintf("ψ=%.1f σ²_s recovery FAIL: est=%.4f truth=%.4f tol=%.4f",
            psi, result.σ2_s, true_σ2_s, tol_s * true_σ2_s))
    check(abs(result.σ_ds - true_σ_ds) < tol_ds * abs(true_σ_ds),
        @sprintf("ψ=%.1f σ_ds recovery FAIL: est=%.4f truth=%.4f tol=%.4f",
            psi, result.σ_ds, true_σ_ds, tol_ds * abs(true_σ_ds)))
    check(abs(result.σ2_e - true_σ2_e) < tol_e * true_σ2_e,
        @sprintf("ψ=%.1f σ²_e recovery FAIL: est=%.4f truth=%.4f tol=%.4f",
            psi, result.σ2_e, true_σ2_e, tol_e * true_σ2_e))
end

println("[dilution-recovery] ALL PASS")

# ── Mis-specification check ──────────────────────────────────────────────────
# Generate data at ψ_true=0.5, then fit with ψ_fit ∈ {0.0, 0.5, 1.0}.
# The matched fit (ψ_fit=0.5) must recover the truth better than either
# misspecified fit, measured by the sum of squared relative errors over
# σ²_d, σ²_s, σ²_e (the three positive-definite variance components).

const DILUTION_MISSPEC_ITER = parse(Int, get(ENV, "DILUTION_RECOVERY_MISSPEC_ITER", "8"))
const DILUTION_MISSPEC_PSI = Float64(0.5)

println("\n[dilution-misspec] Running ψ mis-specification check (ψ_true=$(DILUTION_MISSPEC_PSI))...")

let
    out_dir = ensure_dilution_data(DILUTION_MISSPEC_PSI)
    inputs  = load_dilution_inputs(out_dir)
    true_σ2_d = Float64(inputs.truth.sigma2_d)
    true_σ2_s = Float64(inputs.truth.sigma2_s)
    true_σ2_e = Float64(inputs.truth.sigma2_e)

    # Squared relative error for each variance component
    function ssre(r)
        ((r.σ2_d - true_σ2_d) / true_σ2_d)^2 +
        ((r.σ2_s - true_σ2_s) / true_σ2_s)^2 +
        ((r.σ2_e - true_σ2_e) / true_σ2_e)^2
    end

    misspec_psi_fits = [0.0, DILUTION_MISSPEC_PSI, 1.0]
    errors = Dict{Float64, Float64}()
    for ψ_fit in misspec_psi_fits
        model = SGEModel(inputs.obs.y, inputs.obs.X, inputs.Ainv, inputs.L_Ainv, inputs.gs,
            1.0, 0.2, 0.0, 2.0; ψ=ψ_fit)
        r = run_aireml(
            model,
            max_iter=DILUTION_MISSPEC_ITER,
            k=DILUTION_K,
            damping=0.5,
            cg_maxiter=DILUTION_CG_MAXITER,
            cg_reltol_early=1e-5,
            cg_early_iters=4,
            cg_reltol=1e-7,
            ai_maxiter=DILUTION_AI_MAXITER,
                verbose_cg=false,
                seed=42,
            )
            errors[ψ_fit] = ssre(r)
        @printf("[dilution-misspec] ψ_fit=%.1f  SSRE=%.6f  d=%.4f s=%.4f e=%.4f\n",
            ψ_fit, errors[ψ_fit], r.σ2_d, r.σ2_s, r.σ2_e)
    end

    e_match   = errors[DILUTION_MISSPEC_PSI]
    e_low     = errors[0.0]
    e_high    = errors[1.0]
    check(e_match < e_low,
        @sprintf("mis-spec FAIL: ψ_fit=%.1f SSRE=%.6f should be < ψ_fit=0.0 SSRE=%.6f",
            DILUTION_MISSPEC_PSI, e_match, e_low))
    check(e_match < e_high,
        @sprintf("mis-spec FAIL: ψ_fit=%.1f SSRE=%.6f should be < ψ_fit=1.0 SSRE=%.6f",
            DILUTION_MISSPEC_PSI, e_match, e_high))
end

println("[dilution-misspec] PASS")