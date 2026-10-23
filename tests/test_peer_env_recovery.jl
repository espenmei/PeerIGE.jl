# Parameter recovery for the social peer environmental effect σ²_pe, fitted jointly with a
# classroom effect on the same groups. Identification of σ²_pe vs σ²_class relies on class-size
# variation (c_g²(m_g−2) slope); σ²_pe vs σ²_s relies on relatedness (siblings across classes).
# Self-contained Julia simulation (no R needed): exact sparse A⁻¹ via Henderson's rules.
if !@isdefined(SGEModel)
    include(joinpath(@__DIR__, "..", "src", "emai.jl"))
end
using Printf, Random, SparseArrays, LinearAlgebra, Statistics

const PE_NFAM = parse(Int, get(ENV, "PEER_ENV_RECOVERY_NFAM", "8000"))
const PE_MAX_ITER = parse(Int, get(ENV, "PEER_ENV_RECOVERY_MAX_ITER", "25"))
const PE_K = parse(Int, get(ENV, "PEER_ENV_RECOVERY_K", "20"))
const PE_SEED = parse(Int, get(ENV, "PEER_ENV_RECOVERY_SEED", "2026"))

# Pedigree with children indexed BEFORE their parents, so T = I − P is upper triangular and
# L = T' D^{-1/2} is lower triangular with L L' = T' D⁻¹ T = A⁻¹ (founders unrelated, no inbreeding).
function simulate_peer_env(; nfam, seed, σ2_d, σ2_s, σ_ds, σ2_e, σ2_class, σ2_pe,
    class_sizes=8:32, missing_frac=0.10, β=[1.0, 0.5])
    rng = MersenneTwister(seed)
    nkids = rand(rng, 2:4, nfam)
    n_child = sum(nkids)
    N = n_child + 2nfam
    sire = zeros(Int, n_child)
    dam = zeros(Int, n_child)
    c = 0
    for f in 1:nfam, _ in 1:nkids[f]
        c += 1
        sire[c] = n_child + 2f - 1
        dam[c] = n_child + 2f
    end

    # A⁻¹ = T' D⁻¹ T and its Cholesky factor L = T' D^{-1/2}
    Ti, Tj, Tv = collect(1:N), collect(1:N), ones(N)
    for i in 1:n_child
        append!(Ti, (i, i)); append!(Tj, (sire[i], dam[i])); append!(Tv, (-0.5, -0.5))
    end
    T = sparse(Ti, Tj, Tv, N, N)
    dvec = vcat(fill(0.5, n_child), ones(2nfam))
    Ainv = T' * spdiagm(0 => 1.0 ./ dvec) * T
    L_Ainv = LowerTriangular(sparse(T' * spdiagm(0 => 1.0 ./ sqrt.(dvec))))

    # Genetic effects: founders ~ N(0, G); children = parent mean + Mendelian N(0, G/2)
    G = [σ2_d σ_ds; σ_ds σ2_s]
    LG = cholesky(Symmetric(G)).L
    a = zeros(N, 2)
    for i in (n_child+1):N
        a[i, :] = LG * randn(rng, 2)
    end
    for i in 1:n_child
        a[i, :] = 0.5 .* (a[sire[i], :] .+ a[dam[i], :]) .+ sqrt(0.5) .* (LG * randn(rng, 2))
    end

    # Classes: random assignment of children (siblings mostly end up in different classes)
    order = randperm(rng, n_child)
    group_of = zeros(Int, n_child)
    pos, g = 0, 0
    while pos < n_child
        m = min(rand(rng, class_sizes), n_child - pos)
        if m < first(class_sizes) && g > 0
            group_of[order[pos+1:end]] .= g     # merge short remainder into last class
            break
        end
        g += 1
        group_of[order[pos+1:pos+m]] .= g
        pos += m
    end
    n_groups = g
    members = [Int[] for _ in 1:n_groups]
    for i in 1:n_child; push!(members[group_of[i]], i); end

    p = sqrt(σ2_pe) .* randn(rng, n_child)
    cls = sqrt(σ2_class) .* randn(rng, n_groups)
    soc = a[:, 2]
    x = randn(rng, n_child)
    y_all = zeros(n_child)
    for i in 1:n_child
        gi = group_of[i]
        peer = sum(soc[j] + p[j] for j in members[gi]) - (soc[i] + p[i])
        y_all[i] = β[1] + β[2] * x[i] + a[i, 1] + peer + cls[gi] + sqrt(σ2_e) * randn(rng)
    end

    observed = rand(rng, n_child) .>= missing_frac
    obs_ids = findall(observed)
    gs = GroupStructure(group_of[obs_ids], obs_ids, N;
        all_group_ids=group_of, all_animal_ids=collect(1:n_child))
    X = hcat(ones(length(obs_ids)), x[obs_ids])
    return (; y=y_all[obs_ids], X, Ainv, L_Ainv, gs, class_ids=gs.obs_to_group, N, n_groups)
end

pe_truth = (σ2_d=1.0, σ2_s=0.06, σ_ds=0.08, σ2_e=2.0, σ2_class=0.3, σ2_pe=0.08)

println("[peer-env-recovery] simulating (nfam=$PE_NFAM, seed=$PE_SEED)...")
sim = simulate_peer_env(; nfam=PE_NFAM, seed=PE_SEED, pe_truth...)
@printf("[peer-env-recovery] N=%d  n_obs=%d  n_classes=%d\n", sim.N, length(sim.y), sim.n_groups)

model = SGEModel(sim.y, sim.X, sim.Ainv, sim.L_Ainv, sim.gs, 1.0, 0.05, 0.0, 2.0;
    env_ids=[sim.class_ids], σ2_env=[0.5], peer_env=true, σ2_pe=0.05)

result = run_aireml(model; max_iter=PE_MAX_ITER, k=PE_K, damping=0.5,
    cg_maxiter=5000, cg_reltol_early=1e-5, cg_early_iters=4, cg_reltol=1e-7,
    ai_maxiter=500, verbose_cg=false, seed=42)

pe_est = [result.σ2_d, result.σ2_s, result.σ_ds, result.σ2_e, result.σ2_env[1], result.σ2_pe]
pe_tru = [pe_truth.σ2_d, pe_truth.σ2_s, pe_truth.σ_ds, pe_truth.σ2_e, pe_truth.σ2_class, pe_truth.σ2_pe]
pe_labels = ["σ²_d", "σ²_s", "σ_ds", "σ²_e", "σ²_class", "σ²_pe"]

println("[peer-env-recovery] parameter   truth      est        SE         z")
for i in eachindex(pe_est)
    @printf("[peer-env-recovery] %-10s %9.4f  %9.4f  %9.4f  %+6.2f\n",
        pe_labels[i], pe_tru[i], pe_est[i], result.se[i], (pe_est[i] - pe_tru[i]) / result.se[i])
end

check(cond::Bool, msg::String) = cond || error(msg)
check(result.peer_env && length(result.p_env) == length(result.pe_ids), "peer env outputs missing")
check(size(result.AI) == (6, 6), "AI should be 6×6 with one env factor + peer env")
for i in eachindex(pe_est)
    check(isfinite(pe_est[i]) && isfinite(result.se[i]), "$(pe_labels[i]) not finite")
    check(abs(pe_est[i] - pe_tru[i]) < 3.5 * result.se[i],
        @sprintf("%s recovery FAIL: pe_est=%.4f pe_truth=%.4f se=%.4f", pe_labels[i], pe_est[i], pe_tru[i], result.se[i]))
end
check(result.se[6] < pe_truth.σ2_pe, "σ²_pe SE too large to be informative in this design")

println("[peer-env-recovery] PASS")
