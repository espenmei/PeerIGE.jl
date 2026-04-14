include("emai.jl")

println("[Smoke test] Running quick SGE model verification...")

# Keep this test short and deterministic.
const MAX_ITER = 2
const K_PROBES = 2
const CG_MAXITER = 300
const AI_MAXITER = 100
const LOG_FILE = "smoke_test_aireml_log.csv"

function check(cond::Bool, msg::String)
    cond || error(msg)
end

try
    # Load the same inputs as the main runner.
    yX = DataFrame(CSV.File("yX.csv"))
    X = Matrix(DataFrame(CSV.File("X.csv")))
    y = Vector{Float64}(yX[:, :y])

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

    gs = GroupStructure(yX[:, :group], yX[:, :id_ped], N_pedigree)
    model = SGEModel(y, X, Ainv, L_Ainv, gs, 1.0, 0.2, 0.0, 2.0)

    result = run_aireml(
        model,
        max_iter=MAX_ITER,
        k=K_PROBES,
        damping=0.5,
        cg_maxiter=CG_MAXITER,
        ai_maxiter=AI_MAXITER,
        threaded=false,
        verbose_cg=false,
        log_file=LOG_FILE,
        seed=42,
    )

    # Core sanity checks expected to hold after code changes.
    check(isfinite(result.σ2_d), "σ2_d is not finite")
    check(isfinite(result.σ2_s), "σ2_s is not finite")
    check(isfinite(result.σ_ds), "σ_ds is not finite")
    check(isfinite(result.σ2_e), "σ2_e is not finite")
    check(result.σ2_d > 0, "σ2_d must be > 0")
    check(result.σ2_s > 0, "σ2_s must be > 0")
    check(result.σ2_e > 0, "σ2_e must be > 0")
    check(result.σ2_d * result.σ2_s > result.σ_ds^2, "G matrix is not positive definite")
    check(length(result.beta) == size(X, 2), "beta length mismatch")
    check(length(result.a_d) == size(Ainv, 1), "a_d length mismatch")
    check(length(result.a_s) == size(Ainv, 1), "a_s length mismatch")
    check(size(result.AI) == (4, 4), "AI matrix has wrong shape")
    check(length(result.se) == 4, "SE vector has wrong length")

    println("[Smoke test] PASS")
catch err
    println("[Smoke test] FAIL: " * sprint(showerror, err))
    Base.show_backtrace(stdout, catch_backtrace())
    exit(1)
end
