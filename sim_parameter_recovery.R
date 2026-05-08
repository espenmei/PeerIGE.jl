rm(list = ls())

library(Matrix)
library(tidyverse)

set.seed(080318)

out_dir = Sys.getenv("SIM_OUT_DIR", unset = file.path("outputs", "parameter_recovery"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

ng       = as.integer(Sys.getenv("SIM_NG", "5000"))
nper     = 4L
nstudent = ng * nper

sigma2_d = 1.0
sigma2_s = 0.1
sigma_ds = 0.1
sigma2_e = 1.25^2

Astruc = matrix(c(1/1, 1/2, 1/8, 1/8,
                  1/2, 1/1, 1/8, 1/8,
                  1/8, 1/8, 1/1, 1/2,
                  1/8, 1/8, 1/2, 1/1), 4, 4, byrow = TRUE)

vc_student = matrix(c(sigma2_d, sigma_ds,
                      sigma_ds, sigma2_s), 2, 2)

Vblock = vc_student %x% Astruc
Lblock = t(chol(Vblock))

Z = matrix(rnorm(8L * ng), nrow = 8L, ncol = ng)
U = Lblock %*% Z

a_direct_orig   = as.vector(U[1:nper, ])
a_indirect_orig = as.vector(U[(nper + 1):(2 * nper), ])

A_samp     = sample.int(nstudent)
a_direct   = a_direct_orig[A_samp]
a_indirect = a_indirect_orig[A_samp]

nclass = nstudent %/% 20L
class  = sample.int(nclass, nstudent, replace = TRUE)

class_sum_indirect = numeric(nclass)
class_sum_indirect[sort(unique(class))] <- tapply(a_indirect, class, sum)
y = class_sum_indirect[class] - a_indirect + a_direct

y_obs = as.numeric(y) + sqrt(sigma2_e) * rnorm(nstudent)
x2    = runif(nstudent)

Astruc_inv = solve(Astruc)

inv_perm = integer(nstudent)
inv_perm[A_samp] = seq_len(nstudent)

build_Ainv_triplets_vec = function(block_mat, ng, nper, inv_perm) {
  mask = lower.tri(block_mat, diag = TRUE)
  idx  = which(block_mat != 0 & mask, arr.ind = TRUE)
  bi   = idx[, 1]
  bj   = idx[, 2]
  bv   = block_mat[idx]
  nb   = length(bv)

  offsets = rep(seq.int(0L, by = nper, length.out = ng), each = nb)
  ii_orig = rep(bi, ng) + offsets
  jj_orig = rep(bj, ng) + offsets

  data.frame(i = inv_perm[ii_orig],
             j = inv_perm[jj_orig],
             x = rep(bv, ng))
}

build_L_Ainv_triplets = function(block_mat, ng, nper, inv_perm) {
  Lblock = t(chol(block_mat))
  mask = lower.tri(Lblock, diag = TRUE)
  idx  = which(abs(Lblock) > 0 & mask, arr.ind = TRUE)
  bi   = idx[, 1]
  bj   = idx[, 2]
  bv   = Lblock[idx]
  nb   = length(bv)

  offsets = rep(seq.int(0L, by = nper, length.out = ng), each = nb)
  ii_orig = rep(bi, ng) + offsets
  jj_orig = rep(bj, ng) + offsets

  data.frame(i = inv_perm[ii_orig],
             j = inv_perm[jj_orig],
             x = rep(bv, ng))
}

Ainv_triplet  = build_Ainv_triplets_vec(Astruc_inv, ng, nper, inv_perm)
LAinv_triplet = build_L_Ainv_triplets(Astruc_inv, ng, nper, inv_perm)

write_csv(Ainv_triplet,  file.path(out_dir, "Ainv_triplet.csv"))
write_csv(LAinv_triplet, file.path(out_dir, "LAinv_triplet.csv"))

datao = data.frame(y = y_obs, x = x2, group = class, id_ped = seq_len(nstudent))
missing_frac = as.numeric(Sys.getenv("SIM_MISSING_FRAC", "0.10"))
stopifnot(missing_frac >= 0, missing_frac < 1)
set.seed(42)
n_missing = round(missing_frac * nstudent)
if (n_missing > 0) {
  missing_idx = sample.int(nstudent, n_missing)
  datao$y[missing_idx] = NA_real_
}

obs_table = subset(datao, !is.na(y), select = c("y", "x", "id_ped"))
roster = data.frame(id_ped = seq_len(nstudent), group = class)

write_csv(obs_table, file.path(out_dir, "yX_obs.csv"))
write_csv(roster, file.path(out_dir, "roster.csv"))

truth = data.frame(
  sigma2_d = sigma2_d,
  sigma2_s = sigma2_s,
  sigma_ds = sigma_ds,
  sigma2_e = sigma2_e,
  missing_frac = missing_frac,
  ng = ng,
  nstudent = nstudent,
  sim_name = "sim_parameter_recovery"
)
write_csv(truth, file.path(out_dir, "truth.csv"))

cat(sprintf(
  "Wrote parameter-recovery data to %s (ng=%d, N=%d, missing_frac=%.2f)\n",
  out_dir, ng, nstudent, missing_frac
))