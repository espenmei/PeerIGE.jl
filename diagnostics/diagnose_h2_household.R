# %% Setup
# Run interactively in VS Code, editing data_dir for the fit being checked.
# Requires yX_obs.csv, X_obs.csv, and ped_trim.csv from the same data setup.
# For between-household comparisons, yX_obs.csv should also include
# household_observed (TRUE only when the household ID came from the register).
data_dir <- Sys.getenv(
  "PEERIGE_DATA_DIR",
  unset = "/ess/p805/data/durable/projects/40683815_Espen_IGEpeer/data/edu10_MAT_termin1"
)

read_input <- function(name) {
  path <- file.path(data_dir, name)
  if (!file.exists(path)) stop("Input file not found: ", path, call. = FALSE)
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}

read_flag <- function(x, name) {
  if (is.logical(x) && !anyNA(x)) return(x)
  values <- tolower(as.character(x))
  if (anyNA(values) || any(!values %in% c("true", "false", "1", "0"))) {
    stop(name, " must contain only 0/1 or TRUE/FALSE values.", call. = FALSE)
  }
  values %in% c("true", "1")
}

# %% Load and validate
obs <- read_input("yX_obs.csv")
X_obs <- read_input("X_obs.csv")
ped <- read_input("ped_trim.csv")

require_columns <- function(x, columns, file) {
  missing <- setdiff(columns, names(x))
  if (length(missing)) {
    stop(file, " is missing required column(s): ",
         paste(missing, collapse = ", "), call. = FALSE)
  }
}

require_columns(obs, c("y", "id_ped", "household"), "yX_obs.csv")
require_columns(X_obs, "id_ped", "X_obs.csv")
require_columns(ped, c("mid", "fid"), "ped_trim.csv")

if (nrow(obs) != nrow(X_obs) || !identical(obs$id_ped, X_obs$id_ped)) {
  stop("yX_obs.csv and X_obs.csv must have identical id_ped values in the same row order.",
       call. = FALSE)
}
if (anyNA(obs$id_ped) || any(!is.finite(obs$id_ped)) ||
    any(obs$id_ped != as.integer(obs$id_ped)) ||
    any(obs$id_ped < 1L | obs$id_ped > nrow(ped))) {
  stop("yX_obs.csv contains invalid id_ped values for ped_trim.csv.",
       call. = FALSE)
}
if (anyNA(ped$mid) || anyNA(ped$fid) ||
    any(!is.finite(ped$mid)) || any(!is.finite(ped$fid)) ||
    any(ped$mid != as.integer(ped$mid)) ||
    any(ped$fid != as.integer(ped$fid)) ||
    any(ped$mid < 0L | ped$mid > nrow(ped)) ||
    any(ped$fid < 0L | ped$fid > nrow(ped))) {
  stop("ped_trim.csv must contain valid 0-or-row-index parent IDs in mid and fid.",
       call. = FALSE)
}

x_names <- setdiff(names(X_obs), "id_ped")
if (!length(x_names) ||
    !all(vapply(X_obs[x_names], is.numeric, logical(1)))) {
  stop("X_obs.csv must contain numeric fixed-effect design columns besides id_ped.",
       call. = FALSE)
}
if (!is.numeric(obs$y) || any(!is.finite(obs$y)) ||
    any(!vapply(X_obs[x_names], function(x) all(is.finite(x)), logical(1)))) {
  stop("The outcome and fixed-effect design matrix must contain only finite numbers.",
       call. = FALSE)
}

# %% Repeated-ID and household checks
id_counts <- table(obs$id_ped)
repeated_ids <- id_counts[id_counts > 1L]
cat("Observed rows:", nrow(obs), "\n")
cat("Unique observed pedigree IDs:", length(id_counts), "\n")
cat("IDs with repeated rows:", length(repeated_ids), "\n")
cat("Rows beyond the first per repeated ID:",
    sum(as.integer(repeated_ids) - 1L), "\n")
if (length(repeated_ids)) {
  cat("Most repeated IDs and row counts:\n")
  print(head(sort(repeated_ids, decreasing = TRUE), 10))
}

hh_known <- !is.na(obs$household) & nzchar(as.character(obs$household))
hh_sizes <- table(as.character(obs$household[hh_known]))
cat("Rows with known household:", sum(hh_known), "\n")
cat("Households with at least two observed people:",
    sum(hh_sizes >= 2L), "of", length(hh_sizes), "\n")
if (length(hh_sizes)) {
  cat("Observed household size (median / 90th percentile / maximum):",
      median(as.numeric(hh_sizes)), "/",
      unname(quantile(as.numeric(hh_sizes), 0.9)), "/",
      max(as.numeric(hh_sizes)), "\n")
}
has_household_status <- any(
  c("household_observed", "household_imputed") %in% names(obs)
)
if (has_household_status) {
  if ("household_observed" %in% names(obs)) {
    household_observed <- read_flag(obs$household_observed, "household_observed")
  } else {
    household_observed <- !read_flag(obs$household_imputed, "household_imputed")
  }
  if ("household_imputed" %in% names(obs) &&
      any(household_observed != !read_flag(obs$household_imputed,
                                          "household_imputed"))) {
    stop("household_observed and household_imputed must be complements.",
         call. = FALSE)
  }
  cat("Rows with a register-observed household ID:",
      sum(household_observed & hh_known), "\n")
} else {
  household_observed <- rep(NA, nrow(obs))
  message(
    "No household status column: cross-household sibling comparisons are ",
    "suppressed because imputed singleton household IDs cannot be distinguished."
  )
}
household_year_shifted <- if ("household_year_shifted" %in% names(obs)) {
  read_flag(obs$household_year_shifted, "household_year_shifted")
} else {
  message("No household_year_shifted column: household timing cannot be checked.")
  rep(FALSE, nrow(obs))
}
if (any(household_year_shifted)) {
  message(sum(household_year_shifted),
          " rows use a household from another year; excluding them from household-stratified correlations.")
}

# %% Fixed-effect residuals and family correlations
# Kinship correlations require one observed row per person. Do not silently
# count repeated records as independent relatives.
if (length(repeated_ids)) {
  kin_results <- NULL
  message("Skipping sibling correlations: resolve repeated id_ped rows first.")
} else {
  X <- as.matrix(X_obs[x_names])
  storage.mode(X) <- "double"
  fixed_fit <- lm.fit(x = X, y = obs$y)
  if (any(!is.finite(fixed_fit$residuals))) {
    stop("Fixed-effect residualization failed; inspect X_obs.csv for rank or data problems.",
         call. = FALSE)
  }

  dat <- data.frame(
    id_ped = as.integer(obs$id_ped),
    household = obs$household,
    z = fixed_fit$residuals - mean(fixed_fit$residuals)
  )
  dat$sire <- as.integer(ped$fid[dat$id_ped])
  dat$dam <- as.integer(ped$mid[dat$id_ped])
  dat$z2 <- dat$z * dat$z

  group_key <- function(d, by) {
    pieces <- lapply(d[by], as.character)
    do.call(paste, c(pieces, sep = "\034"))
  }

  group_counts <- function(d, by) {
    if (!nrow(d)) return(integer())
    key <- group_key(d, by)
    index <- match(key, unique(key))
    tabulate(index)[index]
  }

  pair_totals <- function(d, by) {
    if (nrow(d) < 2L) return(list(pairs = 0, crossprod = 0))
    key <- group_key(d, by)
    index <- match(key, unique(key))
    n <- tabulate(index)
    sum_z <- as.numeric(rowsum(d$z, index, reorder = FALSE))
    sum_z2 <- as.numeric(rowsum(d$z2, index, reorder = FALSE))
    list(
      pairs = sum(n * (n - 1) / 2),
      crossprod = sum((sum_z * sum_z - sum_z2) / 2)
    )
  }

  subtract_pairs <- function(a, b) {
    pairs <- a$pairs - b$pairs
    if (pairs < -1e-8) stop("Internal error: negative pair count.")
    list(pairs = max(0, round(pairs)),
         crossprod = a$crossprod - b$crossprod)
  }

  correlation_row <- function(label, d, degree, totals) {
    if (length(degree) != nrow(d) || any(degree < 0)) {
      stop("Internal error while counting relatives for ", label, ".")
    }
    if (abs(sum(degree) - 2 * totals$pairs) > 1e-6) {
      stop("Internal pair-count mismatch for ", label, ".")
    }
    n_endpoints <- 2 * totals$pairs
    if (n_endpoints < 2) {
      return(data.frame(relationship = label, pairs = totals$pairs,
                        residual_correlation = NA_real_))
    }
    # Each endpoint uses degree copies per person; each pair product occurs twice.
    sum_endpoint <- sum(degree * d$z)
    sum_endpoint2 <- sum(degree * d$z2)
    sum_product <- 2 * totals$crossprod
    centered_product <- sum_product - sum_endpoint^2 / n_endpoints
    centered_squares <- sum_endpoint2 - sum_endpoint^2 / n_endpoints
    r <- if (centered_squares > 0) centered_product / centered_squares else NA_real_
    data.frame(relationship = label, pairs = totals$pairs,
               residual_correlation = r)
  }

  sib <- dat[dat$sire > 0L & dat$dam > 0L, ]
  full_all <- pair_totals(sib, c("sire", "dam"))
  full_degree <- group_counts(sib, c("sire", "dam")) - 1L
  results <- list(correlation_row("Full siblings (all households)",
                                  sib, full_degree, full_all))

  hh_pair_valid <- if (has_household_status) {
    hh_known & household_observed & !household_year_shifted
  } else {
    hh_known & !household_year_shifted
  }
  sib_hh <- sib[hh_pair_valid[dat$sire > 0L & dat$dam > 0L], ]
  full_hh_all <- pair_totals(sib_hh, c("sire", "dam"))
  full_same <- pair_totals(sib_hh, c("sire", "dam", "household"))
  results[[2L]] <- correlation_row(
    "Full siblings (same household)", sib_hh,
    group_counts(sib_hh, c("sire", "dam", "household")) - 1L,
    full_same
  )
  if (has_household_status) {
    full_diff <- subtract_pairs(full_hh_all, full_same)
    results[[3L]] <- correlation_row(
      "Full siblings (different households)", sib_hh,
      group_counts(sib_hh, c("sire", "dam")) -
        group_counts(sib_hh, c("sire", "dam", "household")),
      full_diff
    )
  } else {
    results[[3L]] <- data.frame(
      relationship = "Full siblings (different households)",
      pairs = NA_real_, residual_correlation = NA_real_
    )
  }

  half_sib <- function(shared_parent, other_parent, label) {
    if (!has_household_status) {
      return(data.frame(relationship = label, pairs = NA_real_,
                        residual_correlation = NA_real_))
    }
    valid <- dat[
      dat[[shared_parent]] > 0L & dat[[other_parent]] > 0L & hh_pair_valid,
    ]
    across_households <- subtract_pairs(
      pair_totals(valid, shared_parent),
      pair_totals(valid, c(shared_parent, "household"))
    )
    full_across_households <- subtract_pairs(
      pair_totals(valid, c(shared_parent, other_parent)),
      pair_totals(valid, c(shared_parent, other_parent, "household"))
    )
    totals <- subtract_pairs(across_households, full_across_households)
    degree <- group_counts(valid, shared_parent) -
      group_counts(valid, c(shared_parent, "household")) -
      group_counts(valid, c(shared_parent, other_parent)) +
      group_counts(valid, c(shared_parent, other_parent, "household"))
    correlation_row(label, valid, degree, totals)
  }

  results[[4L]] <- half_sib("sire", "dam",
                            "Paternal half-siblings (different households)")
  results[[5L]] <- half_sib("dam", "sire",
                            "Maternal half-siblings (different households)")
  kin_results <- do.call(rbind, results)
  cat("OLS residualization used", fixed_fit$rank, "of", ncol(X),
      "fixed-effect columns.\n")
  print(kin_results, row.names = FALSE, digits = 4)
}

# %% Interpretation notes
# These are descriptive correlations of fixed-effect-adjusted OLS residuals,
# not REML estimates or independent-pair tests. In a direct-only model with no
# shared environment, full-sib and half-sib correlations are roughly 0.5*h2
# and 0.25*h2; household and social-peer effects can change those expectations.
