suppressPackageStartupMessages({
  library(tidyverse)
  library(Matrix)
  library(pedigreeTools)
})

setup_file <- normalizePath("setup_edu_data.R", mustWork = TRUE)
diagnostic_file <- normalizePath(file.path("diagnostics", "diagnose_h2_household.R"),
                                 mustWork = TRUE)

run_setup <- function(root) {
  previous <- setwd(root)
  on.exit(setwd(previous))
  env <- new.env(parent = globalenv())
  env$edu_file <- file.path(root, "edu.csv")
  env$household_dir <- file.path(root, "households")
  for (expr in parse(setup_file)) {
    if (is.call(expr) && identical(expr[[1L]], as.name("setwd"))) next
    if (is.call(expr) && identical(expr[[1L]], as.name("<-")) &&
        is.symbol(expr[[2L]]) &&
        as.character(expr[[2L]]) %in% c("edu_file", "household_dir")) next
    eval(expr, envir = env)
  }
  env
}

expect_error <- function(action, text) {
  error <- tryCatch({
    force(action)
    NULL
  }, error = function(e) e)
  stopifnot(inherits(error, "error"), grepl(text, conditionMessage(error), fixed = TRUE))
}

set.seed(42)
root <- tempfile("edu_setup_")
dir.create(file.path(root, "data"), recursive = TRUE)
dir.create(file.path(root, "households"))
n <- 72L
ids <- sprintf("%05d", seq_len(n) + 10L)
classes <- tibble(
  w19_0634_lnr = ids,
  w19_0634_lopenr_orgnr = rep(c("school1", "school2"), each = 36L),
  lopenr_gruppenr_NPLES = rep(rep(c("same", "other", "same"), each = 12L), 2L),
  utd_skoleaar_start = rep(rep(c(2012L, 2012L, 2015L), each = 12L), 2L),
  trinn = 9L
)
ped <- tibble(
  w19_0634_lnr = c("00001", "00002", "unused", ids),
  mid = c(0L, 0L, 0L, rep(1L, n)),
  fid = c(0L, 0L, 0L, rep(2L, n)),
  foedsels_aar_mnd = c(197001L, 197001L, 190001L,
    (classes$utd_skoleaar_start - 14L + sample(-1:1, n, TRUE)) * 100L + 1L),
  kjoenn = c(2L, 1L, 1L, sample(1:2, n, TRUE)),
  is_twin = c(0L, 0L, 0L, 1L, rep(0L, n - 1L)),
  invkat = c("A", "A", "A", rep(LETTERS[1:7], length.out = n))
)
ped$kjoenn[6L] <- NA_integer_
ped$foedsels_aar_mnd[7L] <- NA_integer_
ped$invkat[10L] <- NA_character_
edu <- tibble(
  w19_0634_lnr = ids,
  avgdato = paste0(classes$utd_skoleaar_start + 2L, "-06-01"),
  fagkode = "MAT0010",
  stp = as.character(sample(1:6, n, TRUE))
)
edu$stp[2L] <- NA_character_
write_csv(classes, file.path(root, "data", "classes.csv"))
write_csv(ped, file.path(root, "data", "ped.csv"))
write_csv(edu, file.path(root, "edu.csv"))
for (year in c(2015L, 2016L, 2017L)) {
  household <- tibble(
    w19_0634_lnr = ids,
    w19_0634_hushnr_lnr = paste0("house", rep(seq_len(n / 2L), each = 2L))
  )
  household$w19_0634_hushnr_lnr[5:6] <- NA_character_
  write_csv(household, file.path(root, "households",
    paste0("LIVING_CONDITIONS_BOFORHOLD_", year, ".csv")))
}

invisible(capture.output(env <- suppressMessages(run_setup(root))))
out <- file.path(root, "data", "edu10_MAT_checked")
obs <- read_csv(file.path(out, "yX_obs.csv"), show_col_types = FALSE)
roster <- read_csv(file.path(out, "roster.csv"), show_col_types = FALSE)
Xe <- read_csv(file.path(out, "X_obs.csv"), show_col_types = FALSE)
trim <- read_csv(file.path(out, "ped_trim.csv"),
                 col_types = cols(w19_0634_lnr = col_character()))
a <- read_csv(file.path(out, "Ainv_triplet.csv"), show_col_types = FALSE)
l <- read_csv(file.path(out, "LAinv_triplet.csv"), show_col_types = FALSE)
source_ids <- trim$w19_0634_lnr[obs$id_ped]
expected_ids <- setdiff(ids, ids[1:4])
expected_grades <- as.numeric(edu$stp[match(source_ids, ids)])
expected_y <- (expected_grades - mean(expected_grades)) / sd(expected_grades)
missing_household <- source_ids %in% ids[5:6]
stopifnot(
  nrow(roster) == n, nrow(obs) == n - 4L,
  setequal(source_ids, expected_ids),
  n_distinct(roster$group) == 6L, n_distinct(obs$cohort) == 4L,
  nrow(trim) == n + 2L, !("unused" %in% trim$w19_0634_lnr),
  identical(obs$id_ped, Xe$id_ped),
  max(abs(obs$y - expected_y)) < 1e-12,
  identical(obs$household_imputed, missing_household),
  identical(obs$household_observed, !missing_household),
  n_distinct(obs$household[missing_household]) == 2L,
  all(obs$household_year == pmax(obs$household_target_year, 2015)),
  all(obs$household_year_shifted == (obs$household_target_year < 2015)),
  any(obs$household_observed & obs$household_year_shifted),
  !("termin1_nu" %in% names(obs)),
  all(table(roster$group) == 12L)
)
obs$invkat <- factor(obs$invkat, levels = levels(env$obs_table$invkat))
expected_X <- model.matrix(~ kjoenn_c + byear_c + invkat, data = obs)
stopifnot(
  identical(colnames(expected_X), names(Xe)[-1L]),
  max(abs(expected_X - as.matrix(Xe[-1L]))) < 1e-12
)

# Independent sparse precision construction in the final row order.
sparse_precision <- function(pedigree, d) {
  N <- nrow(pedigree)
  known_m <- which(pedigree$mid > 0L)
  known_f <- which(pedigree$fid > 0L)
  Tinv <- sparseMatrix(
    i = c(seq_len(N), known_m, known_f),
    j = c(seq_len(N), pedigree$mid[known_m], pedigree$fid[known_f]),
    x = c(rep(1, N), rep(-0.5, length(known_m) + length(known_f))),
    dims = c(N, N)
  )
  crossprod(Tinv, Diagonal(x = 1 / d) %*% Tinv)
}
N <- nrow(trim)
d <- 1 - 0.25 * ((trim$mid > 0L) + (trim$fid > 0L))
expected_precision <- sparse_precision(trim, d)
upper <- sparseMatrix(i = a$i, j = a$j, x = a$x, dims = c(N, N))
precision <- upper + t(upper) - Diagonal(x = diag(upper))
factor <- sparseMatrix(i = l$i, j = l$j, x = l$x, dims = c(N, N))
stopifnot(
  max(abs(precision - expected_precision)) < 1e-10,
  max(abs(precision - tcrossprod(factor))) < 1e-10,
  all(a$i <= a$j), all(l$i >= l$j)
)
cat("PASS: education setup, identity mapping, X alignment, sparse precision and household flags\n")

# Conflicting same-year grades must not depend on source row order.
conflicting <- bind_rows(edu, mutate(edu[5L, ], stp = ifelse(stp == "6", "5", "6")))
write_csv(conflicting, file.path(root, "edu.csv"))
invisible(capture.output(expect_error(suppressMessages(run_setup(root)), "conflicting grades")))
write_csv(bind_rows(edu, edu[5L, ]), file.path(root, "edu.csv"))
invisible(capture.output(duplicate_env <- suppressMessages(run_setup(root))))
stopifnot(identical(env$obs_table, duplicate_env$obs_table))
cat("PASS: ambiguous grades stop; identical duplicates do not change the sample\n")

earlier_record <- mutate(edu[25L, ], avgdato = "2016-06-01", stp = "3")
write_csv(bind_rows(edu, earlier_record), file.path(root, "edu.csv"))
invisible(capture.output(earlier_env <- suppressMessages(run_setup(root))))
stopifnot(nrow(earlier_env$roster) == n - 12L,
          nrow(earlier_env$obs_table) == n - 16L)
missing_offyear <- edu
missing_offyear$avgdato[25L] <- "2016-06-01"
missing_offyear$stp[25L] <- NA_character_
write_csv(missing_offyear, file.path(root, "edu.csv"))
invisible(capture.output(missing_year_env <- suppressMessages(run_setup(root))))
stopifnot(nrow(missing_year_env$roster) == n - 12L,
          sum(with(missing_year_env$year_audit,
            has_grade & !all_years_match & observed_years_match)) == 1L)
write_csv(edu, file.path(root, "edu.csv"))
cat("PASS: original earliest-year and whole-class exclusion rules are preserved and audited\n")

# Two inbred sibling parents each have F=1/4; their children's D is 3/8.
ancestors <- ped[rep(1L, 7L), ]
ancestors$w19_0634_lnr <- c("00001", "00002", "unused", "ancS", "ancT", "ancU", "ancV")
ancestors$mid <- c(0L, 0L, 0L, 1L, 1L, 4L, 4L)
ancestors$fid <- c(0L, 0L, 0L, 2L, 2L, 5L, 5L)
children <- ped[-seq_len(3L), ]
children$mid <- 6L
children$fid <- 7L
write_csv(bind_rows(ancestors, children), file.path(root, "data", "ped.csv"))
invisible(capture.output(inbred_env <- suppressMessages(run_setup(root))))
inbred_trim <- inbred_env$ped_trim
inbred_d <- 1 - 0.25 * ((inbred_trim$mid > 0L) + (inbred_trim$fid > 0L))
inbred_d[inbred_trim$w19_0634_lnr %in% ids] <- 3 / 8
stopifnot(
  nrow(inbred_trim) == n + 6L,
  max(abs(inbred_env$Ag - sparse_precision(inbred_trim, inbred_d))) < 1e-10,
  max(abs(inbred_env$Ag - tcrossprod(inbred_env$Lt))) < 1e-10,
  identical(inbred_trim$w19_0634_lnr[inbred_env$obs_table$id_ped], source_ids)
)
write_csv(ped, file.path(root, "data", "ped.csv"))
cat("PASS: sparse pedigree precision and remapping with inbred parents\n")
bad_ped <- ped
bad_ped$fid[4L] <- bad_ped$mid[4L]
write_csv(bad_ped, file.path(root, "data", "ped.csv"))
invisible(capture.output(expect_error(suppressMessages(run_setup(root)),
                                    "same known mother and father")))
write_csv(ped, file.path(root, "data", "ped.csv"))
cat("PASS: biologically invalid repeated parent is rejected\n")

# Compare the diagnostic against explicitly enumerated pairs on a tiny pedigree.
diagnostic_root <- tempfile("household_diag_")
dir.create(diagnostic_root)
diag_ped <- data.frame(mid = integer(20L), fid = integer(20L))
diag_ped$fid[1:10] <- c(20L, 20L, 20L, 20L, 19L, 19L, 0L, 18L, 18L, 18L)
diag_ped$mid[1:10] <- c(17L, 17L, 16L, 15L, 17L, 14L, 13L, 12L, 12L, 11L)
diag_obs <- data.frame(
  y = c(2, 5, 4, 1, 7, 3, 8, 6, 9, 2),
  id_ped = 1:10, household = c(1L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, NA),
  household_observed = c(rep(TRUE, 5L), FALSE, rep(TRUE, 4L)),
  household_imputed = c(rep(FALSE, 5L), TRUE, rep(FALSE, 4L)),
  household_year_shifted = c(rep(FALSE, 3L), TRUE, rep(FALSE, 6L))
)
diag_X <- data.frame(id_ped = 1:10, intercept = 1, covariate = rep(0:1, 5L))
write.csv(diag_ped, file.path(diagnostic_root, "ped_trim.csv"), row.names = FALSE)
write.csv(diag_obs, file.path(diagnostic_root, "yX_obs.csv"), row.names = FALSE)
write.csv(diag_X, file.path(diagnostic_root, "X_obs.csv"), row.names = FALSE)
old_data_dir <- Sys.getenv("PEERIGE_DATA_DIR", unset = NA_character_)
Sys.setenv(PEERIGE_DATA_DIR = diagnostic_root)
run_diagnostic <- function() {
  result <- new.env(parent = globalenv())
  invisible(capture.output(sys.source(diagnostic_file, envir = result)))
  result
}
diag_env <- run_diagnostic()
z <- lm.fit(as.matrix(diag_X[-1L]), diag_obs$y)$residuals
pair_index <- combn(1:10, 2L)
i <- pair_index[1L, ]
j <- pair_index[2L, ]
sire <- diag_ped$fid[1:10]
dam <- diag_ped$mid[1:10]
full <- sire[i] > 0 & dam[i] > 0 & sire[i] == sire[j] & dam[i] == dam[j]
known <- diag_obs$household_observed & !diag_obs$household_year_shifted &
  !is.na(diag_obs$household)
eligible <- known[i] & known[j]
same_hh <- diag_obs$household[i] == diag_obs$household[j]
paternal <- sire[i] > 0 & sire[i] == sire[j] &
  dam[i] > 0 & dam[j] > 0 & dam[i] != dam[j]
maternal <- dam[i] > 0 & dam[i] == dam[j] &
  sire[i] > 0 & sire[j] > 0 & sire[i] != sire[j]
selectors <- list(full, full & eligible & same_hh, full & eligible & !same_hh,
                  paternal & eligible & !same_hh, maternal & eligible & !same_hh)
for (k in seq_along(selectors)) {
  selected <- which(selectors[[k]])
  expected_pairs <- length(selected)
  stopifnot(diag_env$kin_results$pairs[k] == expected_pairs)
  if (expected_pairs) {
    expected_cor <- cor(c(z[i[selected]], z[j[selected]]),
                        c(z[j[selected]], z[i[selected]]))
    stopifnot(abs(diag_env$kin_results$residual_correlation[k] - expected_cor) < 1e-12)
  } else {
    stopifnot(is.na(diag_env$kin_results$residual_correlation[k]))
  }
}
cat("PASS: all five sibling pair counts AND correlations match explicit enumeration\n")

write.csv(select(diag_obs, -household_observed),
          file.path(diagnostic_root, "yX_obs.csv"), row.names = FALSE)
imputed_only_env <- run_diagnostic()
stopifnot(identical(diag_env$kin_results, imputed_only_env$kin_results))
write.csv(select(diag_obs, -household_observed, -household_imputed),
          file.path(diagnostic_root, "yX_obs.csv"), row.names = FALSE)
no_status_env <- run_diagnostic()
stopifnot(all(is.na(no_status_env$kin_results$pairs[3:5])))
bad_flags <- diag_obs
bad_flags$household_imputed[1L] <- TRUE
write.csv(bad_flags, file.path(diagnostic_root, "yX_obs.csv"), row.names = FALSE)
expect_error(run_diagnostic(), "must be complements")
repeated_obs <- diag_obs
repeated_X <- diag_X
repeated_obs$id_ped[2L] <- repeated_obs$id_ped[1L]
repeated_X$id_ped[2L] <- repeated_X$id_ped[1L]
write.csv(repeated_obs, file.path(diagnostic_root, "yX_obs.csv"), row.names = FALSE)
write.csv(repeated_X, file.path(diagnostic_root, "X_obs.csv"), row.names = FALSE)
repeated_env <- run_diagnostic()
stopifnot(is.null(repeated_env$kin_results))
cat("PASS: imputed-only status, missing/invalid-status guards and duplicate-ID guard\n")
if (is.na(old_data_dir)) {
  Sys.unsetenv("PEERIGE_DATA_DIR")
} else {
  Sys.setenv(PEERIGE_DATA_DIR = old_data_dir)
}
cat("ALL PASS\n")
