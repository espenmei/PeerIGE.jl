#!/usr/bin/env Rscript
# Sweep missing-fraction in sim_peer.R simulator and check SGE recovery.
# Writes diagnostics/missing_sweep_results.csv

rm(list = ls())
set.seed(080318)

library(Matrix)
suppressPackageStartupMessages(library(tidyverse))

ng       = 5000L
nper     = 4L
nstudent = ng * nper

Astruc = matrix(c(1/1, 1/2, 1/8, 1/8,
                  1/2, 1/1, 1/8, 1/8,
                  1/8, 1/8, 1/1, 1/2,
                  1/8, 1/8, 1/2, 1/1), 4, 4, byrow = TRUE)

vc_student = matrix(c(1.0, 0.1,
                      0.1, 0.1), 2, 2)

Vblock = vc_student %x% Astruc
Lblock = t(chol(Vblock))
Z = matrix(rnorm(8L * ng), nrow = 8L, ncol = ng)
U = Lblock %*% Z

a_direct_orig   = as.vector(U[1:nper, ])
a_indirect_orig = as.vector(U[(nper + 1):(2 * nper), ])

A_samp = sample.int(nstudent)
a_direct   = a_direct_orig[A_samp]
a_indirect = a_indirect_orig[A_samp]

nclass = nstudent %/% 20L
class  = sample.int(nclass, nstudent, replace = TRUE)

class_sum_indirect = numeric(nclass)
class_sum_indirect[sort(unique(class))] <- tapply(a_indirect, class, sum)
y = class_sum_indirect[class] - a_indirect + a_direct
y_obs_full = as.numeric(y) + 1.25 * rnorm(nstudent)
x2 = runif(nstudent)

# Build A inverse triplets once (independent of missingness)
Astruc_inv = solve(Astruc)
inv_perm = integer(nstudent)
inv_perm[A_samp] = seq_len(nstudent)

build_Ainv_triplets_vec = function(block_mat, ng, nper, inv_perm) {
  mask = lower.tri(block_mat, diag = TRUE)
  idx  = which(block_mat != 0 & mask, arr.ind = TRUE)
  bi   = idx[, 1]; bj = idx[, 2]; bv = block_mat[idx]
  nb   = length(bv)
  offsets = rep(seq.int(0L, by = nper, length.out = ng), each = nb)
  ii_orig = rep(bi, ng) + offsets
  jj_orig = rep(bj, ng) + offsets
  data.frame(i = inv_perm[ii_orig], j = inv_perm[jj_orig], x = rep(bv, ng))
}
build_LAinv_triplets = function(Astruc_inv, ng, nper, inv_perm) {
  max_nnz = nper * (nper + 1L) %/% 2L * ng
  tri_i = integer(max_nnz); tri_j = integer(max_nnz); tri_x = numeric(max_nnz); pos = 0L
  for (g in seq_len(ng)) {
    orig_idx = (g - 1L) * nper + seq_len(nper)
    perm_idx = inv_perm[orig_idx]
    ord = order(perm_idx); sorted_idx = perm_idx[ord]
    L_block = t(chol(Astruc_inv[ord, ord]))
    for (bi in seq_len(nper)) for (bj in seq_len(bi)) {
      val = L_block[bi, bj]
      if (abs(val) > 1e-15) {
        pos = pos + 1L
        tri_i[pos] = sorted_idx[bi]; tri_j[pos] = sorted_idx[bj]; tri_x[pos] = val
      }
    }
  }
  data.frame(i = tri_i[seq_len(pos)], j = tri_j[seq_len(pos)], x = tri_x[seq_len(pos)])
}
write_csv(build_Ainv_triplets_vec(Astruc_inv, ng, nper, inv_perm), "Ainv_triplet.csv")
write_csv(build_LAinv_triplets(Astruc_inv, ng, nper, inv_perm), "LAinv_triplet.csv")

# Sweep
fracs = c(0.00, 0.05, 0.10, 0.20)
out_dir = "diagnostics/missing_sweep_data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

for (f in fracs) {
  set.seed(42)
  datao = data.frame(y = y_obs_full, x = x2, group = class, id_ped = seq_len(nstudent))
  if (f > 0) {
    n_missing = round(f * nstudent)
    missing_idx = sample.int(nstudent, n_missing)
    datao$y[missing_idx] = NA_real_
  }
  obs_table = subset(datao, !is.na(y), select = c("y", "x", "id_ped"))
  roster = data.frame(id_ped = seq_len(nstudent), group = class)
  tag = sprintf("frac%02d", round(f * 100))
  write_csv(obs_table, file.path(out_dir, sprintf("yX_obs_%s.csv", tag)))
  write_csv(roster,    file.path(out_dir, sprintf("roster_%s.csv", tag)))
  message(sprintf("frac=%.2f -> %d observed", f, nrow(obs_table)))
}
