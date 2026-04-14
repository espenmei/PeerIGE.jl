include("emai.jl")

# Reminder: Julia worker thread count is fixed at startup.
println("[Thread reminder] Julia threads = $(Threads.nthreads()). Start with --threads <N> to control this, e.g.:")
println("  julia --threads 8 --project=. run_emai.jl")

# Input files
yx_path = "yX.csv"
X_path = "X.csv"
Ainv_path = "Ainv_triplet.csv"
L_Ainv_path = "LAinv_triplet.csv"

# Initial variance values
σ2_d = 1.0
σ2_s = 0.2
σ_ds = 0.0
σ2_e = 2.0

# Load y and X
yX = DataFrame(CSV.File(yx_path))
X = Matrix(DataFrame(CSV.File(X_path)))
y = Vector{Float64}(yX[:, :y])

# Load and symmetrize Ainv
Ainv_df = DataFrame(CSV.File(Ainv_path))
i_vals = Ainv_df[:, "i"]
j_vals = Ainv_df[:, "j"]
x_vals = Float64.(Ainv_df[:, "x"])

off_diag = i_vals .!= j_vals
i_all = vcat(i_vals, j_vals[off_diag])
j_all = vcat(j_vals, i_vals[off_diag])
x_all = vcat(x_vals, x_vals[off_diag])

Ainv = sparse(i_all, j_all, x_all)
N_pedigree = size(Ainv, 1)

# Load L_Ainv
L_Ainv_df = DataFrame(CSV.File(L_Ainv_path))
L_Ainv = sparse(L_Ainv_df[:, "i"], L_Ainv_df[:, "j"], Float64.(L_Ainv_df[:, "x"]))
L_Ainv = LowerTriangular(L_Ainv)

gs = GroupStructure(yX[:, :group], yX[:, :id_ped], N_pedigree)
model = SGEModel(y, X, Ainv, L_Ainv, gs, σ2_d, σ2_s, σ_ds, σ2_e)

@time result = run_aireml(
	model,
	max_iter=20,
	k=20,
	damping=0.5,
	threaded=true,
	log_file="aireml_log.csv",
)

save_results(result, "aireml_result.jld2")

(model.σ2_d, model.σ2_s, model.σ_ds, model.σ2_e)