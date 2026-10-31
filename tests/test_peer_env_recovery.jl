# Parameter recovery with social genetic, peer environmental, classroom, household and cohort
# effects. Identification of σ²_pe vs σ²_class relies on class-size variation
# (c_g²(m_g−2) slope); σ²_pe vs σ²_s relies on relatedness (siblings across classes).
# Self-contained Julia simulation (no R needed): exact sparse A⁻¹ via Henderson's rules.
if !@isdefined(SGEModel)
    include(joinpath(@__DIR__, "..", "src", "emai.jl"))
end
using Printf, Random, SparseArrays, LinearAlgebra, Statistics

const PE_NFAM = parse(Int, get(ENV, "PEER_ENV_RECOVERY_NFAM", "1200"))
const PE_MAX_ITER = parse(Int, get(ENV, "PEER_ENV_RECOVERY_MAX_ITER", "20"))
const PE_K = parse(Int, get(ENV, "PEER_ENV_RECOVERY_K", "16"))
const PE_SEED = parse(Int, get(ENV, "PEER_ENV_RECOVERY_SEED", "2026"))

# Three-generation pedigree with descendants indexed BEFORE their parents, so T = I − P is upper triangular and
# L = T' D^{-1/2} is lower triangular with L L' = T' D⁻¹ T = A⁻¹ (founders unrelated, no inbreeding).
function simulate_peer_env(; nfam, seed, σ2_d, σ2_s, σ_ds, σ2_e, σ2_class,
    σ2_household, σ2_cohort, σ2_pe, class_sizes=8:32, cohort_size=60,
    missing_frac=0.10, β=[1.0, 0.5, 0.3, -0.2, 0.15])
    rng = MersenneTwister(seed)
    nkids = rand(rng, 2:4, nfam)
    n_gen1 = sum(nkids)
    gen1_family = zeros(Int, n_gen1)
    c = 0
    for f in 1:nfam, _ in 1:nkids[f]
        c += 1
        gen1_family[c] = f
    end

    unmatched = randperm(rng, n_gen1)
    couple_parents = Tuple{Int,Int}[]
    while length(unmatched) >= 2
        parent1 = pop!(unmatched)
        candidates = findall(j -> gen1_family[unmatched[j]] != gen1_family[parent1],
            eachindex(unmatched))
        isempty(candidates) && break
        parent2 = splice!(unmatched, rand(rng, candidates))
        push!(couple_parents, (parent1, parent2))
    end
    n_couples = length(couple_parents)
    @assert all(gen1_family[p1] != gen1_family[p2] for (p1, p2) in couple_parents)
    n_gen2_per_couple = rand(rng, 1:3, n_couples)
    n_gen2 = sum(n_gen2_per_couple)
    n_desc = n_gen1 + n_gen2
    N = n_desc + 2nfam
    sire = zeros(Int, n_desc)
    dam = zeros(Int, n_desc)
    household_of = zeros(Int, n_desc)
    for j in 1:n_gen1
        child_id = n_gen2 + j
        f = gen1_family[j]
        sire[child_id] = n_desc + 2f - 1
        dam[child_id] = n_desc + 2f
        household_of[child_id] = f
    end
    child_pos = 1
    for c in 1:n_couples, _ in 1:n_gen2_per_couple[c]
        sire[child_pos] = n_gen2 + couple_parents[c][1]
        dam[child_pos] = n_gen2 + couple_parents[c][2]
        household_of[child_pos] = nfam + c
        child_pos += 1
    end

    # A⁻¹ = T' D⁻¹ T and its Cholesky factor L = T' D^{-1/2}
    Ti, Tj, Tv = collect(1:N), collect(1:N), ones(N)
    for i in 1:n_desc
        append!(Ti, (i, i)); append!(Tj, (sire[i], dam[i])); append!(Tv, (-0.5, -0.5))
    end
    T = sparse(Ti, Tj, Tv, N, N)
    dvec = vcat(fill(0.5, n_desc), ones(2nfam))
    Ainv = T' * spdiagm(0 => 1.0 ./ dvec) * T
    L_Ainv = LowerTriangular(sparse(T' * spdiagm(0 => 1.0 ./ sqrt.(dvec))))

    # Genetic effects: founders ~ N(0, G); descendants = parent mean + Mendelian N(0, G/2)
    G = [σ2_d σ_ds; σ_ds σ2_s]
    LG = cholesky(Symmetric(G)).L
    a = zeros(N, 2)
    for i in (n_desc+1):N
        a[i, :] = LG * randn(rng, 2)
    end
    for j in 1:n_gen1
        i = n_gen2 + j
        a[i, :] = 0.5 .* (a[sire[i], :] .+ a[dam[i], :]) .+ sqrt(0.5) .* (LG * randn(rng, 2))
    end
    for i in 1:n_gen2
        a[i, :] = 0.5 .* (a[sire[i], :] .+ a[dam[i], :]) .+ sqrt(0.5) .* (LG * randn(rng, 2))
    end

    # Randomly assign descendants to balanced cohorts, then form variable-size classes
    # within each cohort. Siblings remain randomly distributed across classes/cohorts.
    order = randperm(rng, n_desc)
    group_of = zeros(Int, n_desc)
    cohort_of = zeros(Int, n_desc)
    group_cohort = Int[]
    n_cohorts = cld(n_desc, cohort_size)
    base_cohort_size, extra = divrem(n_desc, n_cohorts)
    cohort_sizes = [base_cohort_size + (h <= extra) for h in 1:n_cohorts]
    cohort_pos = 1
    g = 0
    for h in 1:n_cohorts
        cohort_end = cohort_pos + cohort_sizes[h] - 1
        cohort_members = order[cohort_pos:cohort_end]
        cohort_of[cohort_members] .= h
        pos = 1
        while pos <= length(cohort_members)
            remaining = length(cohort_members) - pos + 1
            m = min(rand(rng, class_sizes), remaining)
            if remaining > last(class_sizes) && remaining - m < first(class_sizes)
                m = remaining - first(class_sizes)
            end
            g += 1
            group_of[cohort_members[pos:pos+m-1]] .= g
            push!(group_cohort, h)
            pos += m
        end
        cohort_pos = cohort_end + 1
    end
    n_groups = g
    members = [Int[] for _ in 1:n_groups]
    for i in 1:n_desc; push!(members[group_of[i]], i); end

    p = sqrt(σ2_pe) .* randn(rng, n_desc)
    cls = sqrt(σ2_class) .* randn(rng, n_groups)
    household = sqrt(σ2_household) .* randn(rng, nfam + n_couples)
    cohort = sqrt(σ2_cohort) .* randn(rng, n_cohorts)
    soc = a[:, 2]
    x = randn(rng, n_desc)
    invkat = rand(rng, 1:4, n_desc)
    @assert length(β) == 5 "β must contain intercept, x, and three category effects"
    β_invkat = [0.0, β[3], β[4], β[5]]
    y_all = zeros(n_desc)
    for i in 1:n_desc
        gi = group_of[i]
        peer = sum(soc[j] + p[j] for j in members[gi]) - (soc[i] + p[i])
        y_all[i] = β[1] + β[2] * x[i] + β_invkat[invkat[i]] +
            a[i, 1] + peer + cls[gi] + household[household_of[i]] +
            cohort[group_cohort[gi]] + sqrt(σ2_e) * randn(rng)
    end

    observed = rand(rng, n_desc) .>= missing_frac
    obs_ids = findall(observed)
    gs = GroupStructure(group_of[obs_ids], obs_ids, N;
        all_group_ids=group_of, all_animal_ids=collect(1:n_desc))
    X = hcat(ones(length(obs_ids)), x[obs_ids],
        Float64.(invkat[obs_ids] .== 2), Float64.(invkat[obs_ids] .== 3),
        Float64.(invkat[obs_ids] .== 4))
    return (; y=y_all[obs_ids], X, Ainv, L_Ainv, gs,
        class_ids=group_of[obs_ids], household_ids=household_of[obs_ids],
        cohort_ids=cohort_of[obs_ids], N, n_groups, n_cohorts, n_desc,
        class_cohort=group_cohort, beta_truth=β)
end

pe_truth = (σ2_d=1.0, σ2_s=0.12, σ_ds=0.03, σ2_e=2.0, σ2_class=0.3,
    σ2_household=0.15, σ2_cohort=0.1, σ2_pe=0.08)

println("[peer-env-recovery] simulating (nfam=$PE_NFAM, seed=$PE_SEED)...")
sim = simulate_peer_env(; nfam=PE_NFAM, seed=PE_SEED, pe_truth...)
@printf("[peer-env-recovery] N=%d  n_obs=%d  n_classes=%d  n_cohorts=%d\n",
    sim.N, length(sim.y), sim.n_groups, sim.n_cohorts)
@assert length(unique(sim.class_cohort)) == sim.n_cohorts
@assert 1 < length(unique(sim.class_ids)) <= sim.n_groups
@assert all(count(==(h), sim.class_cohort) > 1 for h in 1:sim.n_cohorts)
observed_class_sizes = [count(==(g), sim.class_ids) for g in 1:sim.n_groups]
observed_household_sizes = [count(==(h), sim.household_ids) for h in unique(sim.household_ids)]
@assert length(unique(observed_class_sizes)) > 1
@assert any(==(1), observed_household_sizes)
@assert all(any(==(1.0), sim.X[:, j]) && any(==(0.0), sim.X[:, j]) for j in 3:5)

model = SGEModel(sim.y, sim.X, sim.Ainv, sim.L_Ainv, sim.gs, 1.0, 0.08, 0.01, 2.0;
    fixef_names=["(Intercept)", "x", "invkatB", "invkatC", "invkatF"],
    env_ids=[sim.class_ids, sim.household_ids, sim.cohort_ids],
    σ2_env=[0.5, 0.2, 0.2], env_names=["class", "household", "cohort"],
    peer_env=true, σ2_pe=0.05)

result = run_aireml(model; max_iter=PE_MAX_ITER, k=PE_K, damping=0.5,
    cg_maxiter=5000, cg_reltol_early=1e-5, cg_early_iters=4, cg_reltol=1e-7,
    ai_maxiter=500, verbose_cg=false, seed=42)

pe_est = [result.σ2_d, result.σ2_s, result.σ_ds, result.σ2_e,
    result.σ2_env..., result.σ2_pe]
pe_tru = [pe_truth.σ2_d, pe_truth.σ2_s, pe_truth.σ_ds, pe_truth.σ2_e,
    pe_truth.σ2_class, pe_truth.σ2_household, pe_truth.σ2_cohort, pe_truth.σ2_pe]
pe_labels = ["σ²_d", "σ²_s", "σ_ds", "σ²_e", "σ²_class", "σ²_household",
    "σ²_cohort", "σ²_pe"]

println("[peer-env-recovery] parameter   truth      est        SE         z")
for i in eachindex(pe_est)
    @printf("[peer-env-recovery] %-10s %9.4f  %9.4f  %9.4f  %+6.2f\n",
        pe_labels[i], pe_tru[i], pe_est[i], result.se[i], (pe_est[i] - pe_tru[i]) / result.se[i])
end

check(cond::Bool, msg::String) = cond || error(msg)
check(result.peer_env && length(result.p_env) == length(result.pe_ids), "peer env outputs missing")
check(size(result.AI) == (8, 8), "AI should be 8×8 with three env factors + peer env")
check(result.converged, "AI-REML did not converge after $(result.n_iter) iterations")
check(result.fixef_names == ["(Intercept)", "x", "invkatB", "invkatC", "invkatF"],
    "fixed-effect names/order do not match the simulated design")
check(result.env_names == ["class", "household", "cohort"],
    "environment effects are missing or in the wrong order")
check(maximum(abs.(result.beta .- sim.beta_truth)) < 0.25,
    "fixed-effect recovery failed: estimate=$(result.beta), truth=$(sim.beta_truth)")
for i in eachindex(pe_est)
    check(isfinite(pe_est[i]) && isfinite(result.se[i]), "$(pe_labels[i]) not finite")
    check(abs(pe_est[i] - pe_tru[i]) < 3.5 * result.se[i],
        @sprintf("%s recovery FAIL: pe_est=%.4f pe_truth=%.4f se=%.4f", pe_labels[i], pe_est[i], pe_tru[i], result.se[i]))
end
check(result.se[8] < pe_truth.σ2_pe, "σ²_pe SE too large to be informative in this design")

println("[peer-env-recovery] PASS")
