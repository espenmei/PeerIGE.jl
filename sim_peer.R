rm(list = ls())

set.seed(080318)

# Peer effect trenger ulik gruppestC8rrelse?
# Direkte efect krever >= 2 obs pp
# Uten kovarians kan identifiseres ved >= 2 obs per person
# Kanskje identifikasjon krever ulik gruppestorrelse?
# Kanskje identifikasjon av kovarians krever genetisk data med slektninger within og between class?
# Kovarians krever at personer deltar i flere grupper -> genetisk korrelasjon ordner det.
# Krever slektninger i ulike klasser.
# Raske C% sortere innad i grupper for fit?
# Simulation this way, with relatives across all classes leads to very dense cov(y)
#
# OPTIMISED: All linear algebra is done on the small (4x4 / 8x8) block level.
# The block-diagonal structure of A (identical Astruc blocks) means we never
# need to form or factorise million-scale matrices.
#setwd("C:/Users/espenmei/OneDrive - Universitetet i Oslo/Prosjekter/schoolpeer_effect/peer_ige_Julia/emai")

library(Matrix)
library(tidyverse)

ng      = 15000L              # number of genetic groups (families)
nper    = 4L                  # individuals per genetic group
nstudent = ng * nper          # total individuals

# --- Relatedness within each family group (4x4) ---
Astruc = matrix(c(1/1, 1/2, 1/8, 1/8,
                  1/2, 1/1, 1/8, 1/8,
                  1/8, 1/8, 1/1, 1/2,
                  1/8, 1/8, 1/2, 1/1), 4, 4, byrow = TRUE)

# --- Variance components (direct + indirect) ---
vc_student = matrix(c(1.0, 0.1,
                      0.1, 0.1), 2, 2)

# --- 1. Simulate genetic effects using block structure ---
# V = vc_student b
# A is block-diagonal with identical (2x2 b
# 4x4 = 8x8) blocks.
# Cholesky only needs to be computed for the 8x8 block once.
Vblock = vc_student %x% Astruc          # 8 x 8
Lblock = t(chol(Vblock))                # 8 x 8, lower-triangular

# Generate all random effects at once: u_g = Lblock %*% z_g for each group g
Z = matrix(rnorm(8L * ng), nrow = 8L, ncol = ng)
U = Lblock %*% Z                         # 8 x ng, very fast

# Rows 1:4 = a_direct, rows 5:8 = a_indirect (in original group order)
a_direct_orig  = as.vector(U[1:nper, ])
a_indirect_orig = as.vector(U[(nper + 1):(2 * nper), ])

# Permute individuals (equivalent to A[A_samp, A_samp])
A_samp     = sample.int(nstudent)
a_direct   = a_direct_orig[A_samp]
a_indirect = a_indirect_orig[A_samp]

# --- 2. Assign to classes and compute outcome ---
nclass = nstudent %/% 20L
class  = sample.int(nclass, nstudent, replace = TRUE)

# Vectorised: y_i = sum(a_indirect in class_i) - a_indirect_i + a_direct_i
class_sum_indirect = numeric(nclass)
class_sum_indirect[sort(unique(class))] <-
  tapply(a_indirect, class, sum)       # sum of indirect effects per class
y = class_sum_indirect[class] - a_indirect + a_direct

y_obs = as.numeric(y) + 1.25 * rnorm(nstudent)
x2    = runif(nstudent)

# --- 3. Build Ainv and chol(Ainv) triplets using block structure ---
# Ainv is block-diagonal with identical inv(Astruc) blocks.
# After permutation: Ainv_perm[i,j] = Ainv_orig[A_samp[i], A_samp[j]]
Astruc_inv  = solve(Astruc)

# Inverse permutation: maps original index -> permuted position
inv_perm = integer(nstudent)
inv_perm[A_samp] = seq_len(nstudent)

# FIX 1: Output only LOWER-TRIANGLE entries (i >= j within each block).
# The Julia code symmetrizes by adding transposed off-diagonals.
# Previously ALL entries (both triangles) were emitted, causing doubling.
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

# FIX 2: Per-block Cholesky in permuted index order.
# Previously the block-Cholesky was symmetrically permuted, breaking
# lower-triangularity. Now we sort permuted indices per block and
# compute Cholesky of the reordered block so the result is globally
# lower-triangular.
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

    # Cholesky of Astruc_inv with rows/cols reordered to match sorted global indices
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

Ainv_triplet  = build_Ainv_triplets_vec(Astruc_inv, ng, nper, inv_perm)
LAinv_triplet = build_LAinv_triplets(Astruc_inv, ng, nper, inv_perm)

write_csv(Ainv_triplet,  "Ainv_triplet.csv")
write_csv(LAinv_triplet, "LAinv_triplet.csv")

# --- 4. Output data ---
# Master table: all individuals as rows; SIM_MISSING_FRAC controls fraction of
# individuals whose y is set to NA (peers without observed outcome).
# Default 0.0 keeps every outcome (validation-friendly recovery).
datao = data.frame(y = y_obs, x = x2, group = class, id_ped = seq_len(nstudent))
missing_frac = as.numeric(Sys.getenv("SIM_MISSING_FRAC", "0.0"))
stopifnot(missing_frac >= 0, missing_frac < 1)
set.seed(42)
n_missing = round(missing_frac * nstudent)
if (n_missing > 0) {
  missing_idx = sample.int(nstudent, n_missing)
  datao$y[missing_idx] = NA_real_
}
message(sprintf("Master yX table: %d total rows, %d observed, %d missing (frac=%.2f)",
                nstudent, nstudent - n_missing, n_missing, missing_frac))

# New two-table contract:
# 1) observed table for REML response/design and env mappings
# 2) full roster table for social group membership
obs_table = subset(datao, !is.na(y), select = c("y", "x", "id_ped"))
roster = data.frame(id_ped = seq_len(nstudent), group = class)

write_csv(obs_table, "yX_obs.csv")
write_csv(roster, "roster.csv")
