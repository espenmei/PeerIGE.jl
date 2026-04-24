# illustrate_covariance.R
# Graphical illustration of covariance structures in social genetic effects models:
#   A     – additive relationship matrix (sparse, follows pedigree)
#   ZZ'   – pen membership (block-diagonal)
#   ZAZ'  – social genetic covariance (connects pen blocks via shared ancestry)
#
# Toy setup: 10 pen animals in pens of size 3, 3, and 4.
# Two sibling links bridge pens: one sibling pair across pens 1 and 2,
# and one sibling pair across pens 2 and 3.

# ── Pedigree ────────────────────────────────────────────────────────────────
# Row indices are used as IDs. Parents must appear before offspring.
# Rows 1-2:  external founders for pen1-pen2 sibling pair
# Rows 3-4:  external founders for pen2-pen3 sibling pair
# Rows 5-7:  pen 1 animals (row 5 is sibling with row 8)
# Rows 8-10: pen 2 animals (row 8 siblings with row 5; row 10 siblings with row 14)
# Rows 11-14: pen 3 animals (row 14 is sibling with row 10)
ped <- matrix(c(
   1,  0,  0,   # external founder
   2,  0,  0,   # external founder
   3,  0,  0,   # external founder
   4,  0,  0,   # external founder
   5,  1,  2,   # pen 1 sibling (with row 8)
   6,  0,  0,   # pen 1 founder
   7,  0,  0,   # pen 1 founder
   8,  1,  2,   # pen 2 sibling (with row 5)
   9,  0,  0,   # pen 2 founder
  10,  3,  4,   # pen 2 sibling (with row 14)
  11,  0,  0,   # pen 3 founder
  12,  0,  0,   # pen 3 founder
  13,  0,  0,   # pen 3 founder
  14,  3,  4    # pen 3 sibling (with row 10)
), ncol = 3, byrow = TRUE)

n_ped <- nrow(ped)          # 14

# pen animals by row index (excludes external founders 1:4)
pen_rows <- 5:14
n <- length(pen_rows)       # 10

# ── Numerator Relationship Matrix A (tabular method) ────────────────────────
A_full <- matrix(0, n_ped, n_ped)
for (i in seq_len(n_ped)) {
  s <- ped[i, 2]; d <- ped[i, 3]
  if (s > 0 && d > 0) {
    A_full[i, i] <- 1 + 0.5 * A_full[s, d]
  } else {
    A_full[i, i] <- 1
  }
  for (j in seq_len(i - 1)) {
    val <- 0
    if (s > 0) val <- val + 0.5 * A_full[s, j]
    if (d > 0) val <- val + 0.5 * A_full[d, j]
    A_full[i, j] <- val
    A_full[j, i] <- val
  }
}
A <- A_full[pen_rows, pen_rows]

# ── Z matrix (pen-mate incidence) ────────────────────────────────────────────
# pens defined by original row indices; external founders (1:4) excluded
pens <- list(c(5, 6, 7), c(8, 9, 10), c(11, 12, 13, 14))
Z_full <- matrix(0, n_ped, n_ped)
for (pen in pens) {
  for (i in pen) for (j in pen) if (i != j) Z_full[i, j] <- 1
}

ZZt_full  <- Z_full %*% t(Z_full)
ZAZt_full <- Z_full %*% A_full %*% t(Z_full)

ZZt  <- ZZt_full[pen_rows,  pen_rows]
ZAZt <- ZAZt_full[pen_rows, pen_rows]

# ── Plotting ─────────────────────────────────────────────────────────────────
plot_mat <- function(mat, title) {
  # image() plots column-major with (1,1) at bottom-left; we flip rows so the
  # matrix reads top-to-bottom as expected.
  image(1:n, 1:n, t(mat[n:1, ]),
        col  = hcl.colors(64, "YlOrRd", rev = TRUE),
        zlim = c(0, max(ZAZt)),
        xlab = "", ylab = "", axes = FALSE,
        main = title)
  axis(1, at = 1:n, labels = 1:n, cex.axis = 0.8, tick = FALSE)
  axis(2, at = 1:n, labels = n:1, cex.axis = 0.8, tick = FALSE, las = 1)
  # pen boundary lines (after animal 3 and 6)
  abline(v = c(3.5, 6.5),         col = "grey40", lty = 2, lwd = 1.5)
  abline(h = n + 1 - c(3.5, 6.5), col = "grey40", lty = 2, lwd = 1.5)
  box()
}

png("covariance_illustration.png", width = 1600, height = 600, res = 150)
  par(mfrow = c(1, 3), mar = c(2, 2, 3, 1), oma = c(0, 0, 3, 0))
  plot_mat(A,    expression(bold(A)))
  plot_mat(ZZt,  expression(bold(ZZ)*"'"))
  plot_mat(ZAZt, expression(bold(ZAZ)*"'"))
  mtext("Covariance structures in social genetic effects models",
        outer = TRUE, cex = 1.1, font = 2)
dev.off()

pdf("covariance_illustration.pdf", width = 12, height = 4.5)
  par(mfrow = c(1, 3), mar = c(2, 2, 3, 1), oma = c(0, 0, 3, 0))
  plot_mat(A,    expression(bold(A)))
  plot_mat(ZZt,  expression(bold(ZZ)*"'"))
  plot_mat(ZAZt, expression(bold(ZAZ)*"'"))
  mtext("Covariance structures in social genetic effects models",
        outer = TRUE, cex = 1.1, font = 2)
dev.off()

message("Saved covariance_illustration.pdf and .png")
