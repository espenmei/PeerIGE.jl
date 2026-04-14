include("emai_direct.jl")

@time model, result = run_emai_direct(max_iter=10, k=10, damping=1.0)

@printf("\nEndelig: σ²_d = %.6f, σ²_e = %.6f, h² = %.4f\n",
    model.σ2_d, model.σ2_e, model.σ2_d / (model.σ2_d + model.σ2_e))