rm(list = ls())

set.seed(080318)

# =============================================================================
# Simulering av standard dyremodell (kun direkte genetisk effekt)
# Modell: y_i = x_i'beta + a_d_i + e_i
#   a_d ~ N(0, A * sigma2_d),  e ~ N(0, I * sigma2_e)
#
# Pedigree: blokk-diagonal A med identiske 4x4 familierelaterthetsblokker.
# Hvert dyr observeres en gang.
# =============================================================================

setwd("C:/Users/espenmei/OneDrive - Universitetet i Oslo/Prosjekter/schoolpeer_effect/peer_ige_Julia/emai")

library(Matrix)
library(tidyverse)

# --- Parametere ---
ng      = 2000000L             # antall genetiske grupper (familier)
nper    = 4L                  # individer per familie
nstudent = ng * nper          # totalt antall individer

sigma2_d = 1.0               # additiv genetisk varians
sigma2_e = 1.25^2            # residualvarians (= 1.5625)

cat(sprintf("Sanne verdier: sigma2_d = %.4f, sigma2_e = %.4f, h2 = %.4f\n",
    sigma2_d, sigma2_e, sigma2_d / (sigma2_d + sigma2_e)))

# --- Relatedness within each family group (4x4) ---
Astruc = matrix(c(1/1, 1/2, 1/8, 1/8,
                  1/2, 1/1, 1/8, 1/8,
                  1/8, 1/8, 1/1, 1/2,
                  1/8, 1/8, 1/2, 1/1), 4, 4, byrow = TRUE)

# --- 1. Simuler genetiske effekter ---
# Cov(a_d) = A * sigma2_d -> blokkvis: Astruc * sigma2_d
Lblock = t(chol(Astruc * sigma2_d))     # 4x4 lower-triangular

Z = matrix(rnorm(nper * ng), nrow = nper, ncol = ng)
U = Lblock %*% Z                         # 4 x ng
a_direct_orig = as.vector(U)             # length nstudent

# Permuter individer (tilsvarer A[A_samp, A_samp])
A_samp   = sample.int(nstudent)
a_direct = a_direct_orig[A_samp]

# --- 2. Simuler utfallsvariabel ---
x2    = runif(nstudent)
beta  = c(2.0, 0.5)                     # intercept + slope

e     = sqrt(sigma2_e) * rnorm(nstudent)
y_obs = beta[1] + beta[2] * x2 + a_direct + e

cat(sprintf("Simulert: var(a_d) = %.4f, var(e) = %.4f, var(y) = %.4f\n",
    var(a_direct), var(e), var(y_obs)))

# --- 3. Bygg Ainv og chol(Ainv) som triplets ---
Astruc_inv = solve(Astruc)

# Invers permutasjon: original indeks -> permutert posisjon
inv_perm = integer(nstudent)
inv_perm[A_samp] = seq_len(nstudent)

# Ainv triplets (kun nedre triangel -- Julia symmetriserer)
build_Ainv_triplets_vec = function(block_mat, ng, nper, inv_perm) {
  mask = lower.tri(block_mat, diag = TRUE)
  idx  = which(block_mat != 0 & mask, arr.ind = TRUE)
  bi   = idx[, 1]; bj = idx[, 2]
  bv   = block_mat[idx]
  nb   = length(bv)

  offsets = rep(seq.int(0L, by = nper, length.out = ng), each = nb)
  ii_orig = rep(bi, ng) + offsets
  jj_orig = rep(bj, ng) + offsets

  data.frame(i = inv_perm[ii_orig],
             j = inv_perm[jj_orig],
             x = rep(bv, ng))
}

# L_Ainv triplets (per-blokk Cholesky i sortert permutert rekkefølge)
build_LAinv_triplets = function(Astruc_inv, ng, nper, inv_perm) {
  max_nnz = nper * (nper + 1L) %/% 2L * ng
  tri_i = integer(max_nnz)
  tri_j = integer(max_nnz)
  tri_x = numeric(max_nnz)
  pos   = 0L

  for (g in seq_len(ng)) {
    orig_idx   = (g - 1L) * nper + seq_len(nper)
    perm_idx   = inv_perm[orig_idx]
    ord        = order(perm_idx)
    sorted_idx = perm_idx[ord]

    L_block = t(chol(Astruc_inv[ord, ord]))

    for (bi in seq_len(nper)) {
      for (bj in seq_len(bi)) {
        val = L_block[bi, bj]
        if (abs(val) > 1e-15) {
          pos = pos + 1L
          tri_i[pos] = sorted_idx[bi]
          tri_j[pos] = sorted_idx[bj]
          tri_x[pos] = val
        }
      }
    }
  }

  data.frame(i = tri_i[seq_len(pos)],
             j = tri_j[seq_len(pos)],
             x = tri_x[seq_len(pos)])
}

cat("Bygger Ainv triplets...\n")
Ainv_triplet  = build_Ainv_triplets_vec(Astruc_inv, ng, nper, inv_perm)
cat("Bygger LAinv triplets...\n")
LAinv_triplet = build_LAinv_triplets(Astruc_inv, ng, nper, inv_perm)

write_csv(Ainv_triplet,  "Ainv_triplet.csv")
write_csv(LAinv_triplet, "LAinv_triplet.csv")

# --- 4. Output data ---
# Inkluderer 'group' for kompatibilitet med peer-modellen (brukes ikke i emai_direct.jl)
nclass = nstudent %/% 20L
class  = sample.int(nclass, nstudent, replace = TRUE)

datao = data.frame(y = y_obs, x = x2, group = class, id_ped = seq_len(nstudent))

# New two-table contract:
# 1) observed table for REML response/design
# 2) full roster table for social group membership
obs_table = data.frame(y = y_obs, x = x2, id_ped = seq_len(nstudent))
roster = data.frame(id_ped = seq_len(nstudent), group = class)

write_csv(obs_table, "yX_obs.csv")
write_csv(roster, "roster.csv")

cat(sprintf("Ferdig! N = %d, n = %d, nclass = %d\n", nstudent, nstudent, nclass))
cat(sprintf("Forventet: sigma2_d = %.4f, sigma2_e = %.4f\n", sigma2_d, sigma2_e))
