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

ng      = as.integer(Sys.getenv("SIM_NG", "5000"))  # number of genetic groups (families)
nper    = 4L                  # individuals per genetic group
nstudent = ng * nper          # total individuals

# --- Relatedness within each family group (4x4) ---
Astruc = matrix(c(1/1, 1/2, 1/8, 1/8,
                  1/2, 1/1, 1/8, 1/8,
                  1/8, 1/8, 1/1, 1/2,
                  1/8, 1/8, 1/2, 1/1), 4, 4, byrow = TRUE)

# --- Variance components (direct + indirect + shared env) ---
vc_student = matrix(c(1.0, 0.1,
                      0.1, 0.1), 2, 2)
s2_household = 0.3                       # household (sibling-pair) variance
s2_classroom = 0.3                       # classroom shared environment variance
s2_school = 0.1                          # school shared environment variance
s2_e = 1.2                               # residual variance

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

# --- 1b. Simulate household (sibling-pair) effects ---
# Each family of 4 has two sibling pairs: (1,2) and (3,4).
# Total households = 2 * ng.
nhousehold = 2L * ng
household_orig = rep(seq_len(nhousehold), each = 2L)  # 1,1,2,2,3,3,...
c_household = rnorm(nhousehold, sd = sqrt(s2_household))  # c_h ~ N(0, σ²_household)

# Permute individuals (equivalent to A[A_samp, A_samp])
A_samp     = sample.int(nstudent)
a_direct   = a_direct_orig[A_samp]
a_indirect = a_indirect_orig[A_samp]
household  = household_orig[A_samp]   # household IDs follow permuted individuals

# --- 2. Assign to classes and compute outcome ---
nclass = nstudent %/% 20L
class  = sample.int(nclass, nstudent, replace = TRUE)

# Classroom env uses the same grouping as the peer effect class.
classroom = class

# Build school IDs so each school has 3-5 classrooms.
class_to_school = integer(nclass)
school_id = 1L
cls = 1L
while (cls <= nclass) {
  block_size = sample(3L:5L, 1L)
  cls_end = min(nclass, cls + block_size - 1L)
  class_to_school[cls:cls_end] = school_id
  school_id = school_id + 1L
  cls = cls_end + 1L
}
school = class_to_school[class]

c_classroom = rnorm(nclass, sd = sqrt(s2_classroom))
c_school = rnorm(max(class_to_school), sd = sqrt(s2_school))

# Vectorised: y_i = sum(a_indirect in class_i) - a_indirect_i + a_direct_i
class_sum_indirect = numeric(nclass)
class_sum_indirect[sort(unique(class))] <-
  tapply(a_indirect, class, sum)       # sum of indirect effects per class
y = class_sum_indirect[class] - a_indirect + a_direct +
  c_household[household] + c_classroom[classroom] + c_school[school]

y_obs = as.numeric(y) + sqrt(s2_e) * rnorm(nstudent)
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
# Remap household IDs to 1..n_levels (dense) for Julia env_ids
household_dense = as.integer(factor(household))
classroom_dense = as.integer(factor(classroom))
school_dense = as.integer(factor(school))
datao = data.frame(y = y_obs, x = x2, group = class, id_ped = seq_len(nstudent),
                   household = household_dense,
                   classroom = classroom_dense,
                   school = school_dense)

# New two-table contract:
# 1) observed table for REML response/design and environment effects
# 2) full roster table for social group membership
obs_table = subset(datao, !is.na(y),
                   select = c("y", "x", "id_ped", "household", "classroom", "school"))
roster = data.frame(id_ped = seq_len(nstudent), group = class)

write_csv(obs_table, "yX_obs.csv")
write_csv(roster, "roster.csv")
cat(sprintf(
  "True variances: household=%.2f classroom=%.2f school=%.2f | n_households=%d n_classrooms=%d n_schools=%d\n",
  s2_household, s2_classroom, s2_school,
  length(unique(household_dense)), length(unique(classroom_dense)), length(unique(school_dense))
))
