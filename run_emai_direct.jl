include("emai_direct.jl")

const OUT_LOG_DIR = joinpath("outputs", "logs")
const OUT_RES_DIR = joinpath("outputs", "results")
mkpath(OUT_LOG_DIR)
mkpath(OUT_RES_DIR)

@time model, result = run_emai_direct(
    max_iter=10,
    k=10,
    damping=1.0,
    log_path=joinpath(OUT_LOG_DIR, "aireml_direct_log.csv"),
    output_path=joinpath(OUT_RES_DIR, "aireml_direct_result.jld2"),
)

@printf("\nEndelig: σ²_d = %.6f, σ²_e = %.6f, h² = %.4f\n",
    model.σ2_d, model.σ2_e, model.σ2_d / (model.σ2_d + model.σ2_e))