library(tidyverse)
library(Matrix)
library(pedigreeTools)

# %% Innstillinger
setwd("N:/durable/projects/40683815_Espen_IGEpeer")

edu_file <- "N:/durable/data/registers/SSB/01_data/data_v6.1/EDUCATION_VGS_GRS/csv/EDUCATION_TAB_KAR_GRS.csv"
household_dir <- "N:/durable/data/registers/SSB/01_data/data_v6.1/LIVING_CONDITIONS/csv"
fag <- "MAT"
inv_col <- "invkat"
out_dir <- file.path("data", paste0("edu10_", fag, "_checked"))

id_col <- "w19_0634_lnr"
school_col <- "w19_0634_lopenr_orgnr"
household_col <- "w19_0634_hushnr_lnr"

# %% Hjelpefunksjoner
# mid/fid er radnumre i ped, 0 = ukjent. Behold roster og alle deres aner.
pedigree_keep_rows <- function(ped, keep_rows, mid = "mid", fid = "fid") {
  n <- nrow(ped)
  validate <- function(x, lower, name) {
    if (!is.numeric(x) || any(!is.finite(x)) ||
        any(x != trunc(x)) || any(x < lower | x > n)) {
      stop(name, " must contain valid integer row indices")
    }
  }
  mother <- ped[[mid]]
  father <- ped[[fid]]
  if (is.null(mother) || is.null(father)) stop("Parent columns not found")
  validate(mother, 0, mid)
  validate(father, 0, fid)
  validate(keep_rows, 1, "keep_rows")

  frontier <- unique(keep_rows)
  keep <- logical(n)
  keep[frontier] <- TRUE
  while (length(frontier)) {
    parents <- unique(c(mother[frontier], father[frontier]))
    parents <- parents[parents != 0]
    frontier <- parents[!keep[parents]]
    keep[frontier] <- TRUE
  }
  which(keep)
}

recode_grade <- function(x) {
  x <- as.character(x)
  as.integer(ifelse(x %in% as.character(1:6), x, NA))
}

codes <- switch(fag,
  MAT = c("MAT0010", "MAT0015"),
  KRO = c("KRO0010", "KRO0020"),
  stop("Unknown subject: ", fag)
)

# %% 1. Klasser (9. trinn): gruppenummer x skole x skoleaar
classes <- read_csv("data/classes.csv", col_types = cols(
  w19_0634_lnr = col_character(),
  w19_0634_lopenr_orgnr = col_character(),
  lopenr_gruppenr_NPLES = col_character()
))
class <- classes %>%
  select(!contains(c("NPREG", "NPENG"))) %>%
  filter(trinn == 9,
         !is.na(lopenr_gruppenr_NPLES),
         !is.na(.data[[school_col]]),
         !is.na(utd_skoleaar_start),
         !is.na(.data[[id_col]]))

class %>%
  group_by(lopenr_gruppenr_NPLES) %>%
  summarise(flere_skoler = n_distinct(.data[[school_col]]) > 1,
            flere_aar = n_distinct(utd_skoleaar_start) > 1,
            .groups = "drop") %>%
  count(flere_skoler, flere_aar) %>%
  print()

class <- class %>%
  group_by(lopenr_gruppenr_NPLES, .data[[school_col]], utd_skoleaar_start) %>%
  mutate(group = cur_group_id(), ng = n_distinct(.data[[id_col]])) %>%
  ungroup() %>%
  filter(ng > 10 & ng < 35)
if (!nrow(class)) stop("No classes remain after the class-size filter (11..34).")

dup_pupils <- class[[id_col]][duplicated(class[[id_col]])]
if (length(dup_pupils)) {
  stop(length(unique(dup_pupils)),
       " pupils have repeated class rows; check duplicate rows and multiple memberships.")
}
class <- class %>%
  group_by(.data[[school_col]], utd_skoleaar_start) %>%
  mutate(cohort = cur_group_id()) %>%
  ungroup()

# %% 2. Standpunktkarakter 10. trinn
edu <- read_csv(edu_file, col_types = cols(w19_0634_lnr = col_character())) %>%
  mutate(avgaar = as.integer(substr(as.character(avgdato), 1, 4)))
min_forste_skoleaar_start <- min(class$utd_skoleaar_start)

edu_candidates <- edu %>%
  filter(.data[[id_col]] %in% class[[id_col]], fagkode %in% codes) %>%
  mutate(stp_nu = recode_grade(stp)) %>%
  filter(avgaar >= min_forste_skoleaar_start + 2)

# Behold regelen "foerste registrerte aar", men ikke velg vilkaarig ved konflikt.
earliest <- edu_candidates %>%
  group_by(.data[[id_col]]) %>%
  filter(avgaar == min(avgaar)) %>%
  ungroup()
ambiguous <- earliest %>%
  group_by(.data[[id_col]], avgaar) %>%
  summarise(n_grades = n_distinct(stp_nu, na.rm = FALSE), .groups = "drop") %>%
  filter(n_grades > 1L)
if (nrow(ambiguous)) {
  stop(nrow(ambiguous),
       " pupils have conflicting grades (including missing vs observed) in their earliest year. ",
       "Resolve subject-code/date priority before selecting one education row.")
}
edu_f <- earliest %>%
  group_by(.data[[id_col]]) %>%
  slice_head(n = 1L) %>%
  ungroup() %>%
  select(all_of(id_col), avgaar, stp_nu)

# %% 3. Koble klasser og karakterer
npccu <- left_join(class, edu_f, by = id_col, relationship = "many-to-one") %>%
  mutate(yeardiff = avgaar - utd_skoleaar_start)

# Samme utvalg som i originalen: hele klassen forkastes ved et avvikende aar,
# ogsaa hvis den aktuelle eleven ikke har en gyldig standpunktkarakter.
year_audit <- npccu %>%
  group_by(group) %>%
  summarise(has_grade = any(!is.na(stp_nu)),
            all_years_match = all(yeardiff == 2, na.rm = TRUE),
            observed_years_match = all(yeardiff[!is.na(stp_nu)] == 2),
            .groups = "drop")
cat("Classes without a valid grade:", sum(!year_audit$has_grade), "\n")
cat("Classes with an off-year education record:",
    sum(year_audit$has_grade & !year_audit$all_years_match), "\n")
cat("Of these, off-year records only have missing grades:",
    sum(year_audit$has_grade & !year_audit$all_years_match &
          year_audit$observed_years_match), "\n")
npccu <- npccu %>%
  group_by(group) %>%
  filter(any(!is.na(stp_nu)), all(yeardiff == 2, na.rm = TRUE)) %>%
  ungroup()
if (!nrow(npccu)) stop("No classes remain after education-year matching.")
print(table(npccu$yeardiff, useNA = "ifany"))

# %% 4. Pedigree: foreldre foer avkom, 0 = ukjent
ped <- read_csv("data/ped.csv", col_types = cols(w19_0634_lnr = col_character()))
stopifnot(all(c(id_col, "mid", "fid", "foedsels_aar_mnd",
                "kjoenn", "is_twin", inv_col) %in% names(ped)),
          !anyNA(ped[[id_col]]), !anyDuplicated(ped[[id_col]]),
          is.numeric(ped$mid), is.numeric(ped$fid),
          all(is.finite(ped$mid)), all(is.finite(ped$fid)),
          all(ped$mid == trunc(ped$mid)), all(ped$fid == trunc(ped$fid)),
          all(ped$mid >= 0 & ped$mid <= nrow(ped)),
          all(ped$fid >= 0 & ped$fid <= nrow(ped)),
          all(ped$mid == 0 | ped$mid < seq_len(nrow(ped))),
          all(ped$fid == 0 | ped$fid < seq_len(nrow(ped))))
if (any(ped$mid > 0L & ped$fid > 0L & ped$mid == ped$fid)) {
  stop("The human pedigree contains a person with the same known mother and father.")
}

n_not_in_ped <- sum(!(unique(npccu[[id_col]]) %in% ped[[id_col]]))
cat("Class members missing from pedigree:", n_not_in_ped, "\n")
stopifnot(n_not_in_ped == 0)
npccu <- npccu %>%
  mutate(pos_in_ped = match(.data[[id_col]], ped[[id_col]]))
p <- npccu$pos_in_ped
npccu$byear <- as.integer(substr(as.character(ped$foedsels_aar_mnd[p]), 1, 4))
npccu$kjoenn_c <- ped$kjoenn[p] - 1
npccu$is_twin <- ped$is_twin[p]
npccu$invkat <- as.character(ped[[inv_col]][p])
stopifnot(all(na.omit(npccu$invkat) %in% LETTERS[1:7]),
          all(npccu$kjoenn_c %in% c(0, 1, NA)))
npccu$invkat[is.na(npccu$invkat)] <- "ukjent"

# Tvillinger mister utfallet, men beholdes i rosteren.
npccu$stp_nu[which(npccu$is_twin == 1)] <- NA
if (all(is.na(npccu$byear))) stop("No valid birth years in the selected classes.")
npccu$byear_c <- npccu$byear - min(npccu$byear, na.rm = TRUE)

# %% 5. Husholdning i eksamensaaret, med opprinnelig 2015-fallback
npccu$household_target_year <- coalesce(
  npccu$avgaar, npccu$utd_skoleaar_start + 2L
)
npccu$hushyear <- pmax(npccu$household_target_year, 2015L)
npccu$household_year_shifted <- npccu$hushyear != npccu$household_target_year
aary <- sort(unique(npccu$hushyear))
hushd <- vector("list", length(aary))
for (k in seq_along(aary)) {
  j <- aary[k]
  hushj <- read_csv(
    file.path(household_dir, paste0("LIVING_CONDITIONS_BOFORHOLD_", j, ".csv")),
    col_select = all_of(c(id_col, household_col)),
    col_types = cols(.default = col_character())
  )
  hushj <- hushj[hushj[[id_col]] %in% npccu[[id_col]], ]
  hushj$hushyear <- j
  hushd[[k]] <- hushj
}
hushd <- bind_rows(hushd)
npccu2 <- left_join(npccu, hushd, by = c(id_col, "hushyear"),
                   relationship = "many-to-one")
npccu2$household_imputed <- is.na(npccu2[[household_col]]) |
  trimws(npccu2[[household_col]]) == ""
npccu2$household_observed <- !npccu2$household_imputed
hh_key <- paste0("hh:", npccu2[[household_col]])
hh_key[npccu2$household_imputed] <- paste0(
  "imp:", npccu2[[id_col]][npccu2$household_imputed]
)
# Samme husholdningskode paa tvers av aar er fortsatt samme random effect.
# Dette krever at registerkoden identifiserer samme husholdning over tid.
npccu2$household <- as.integer(factor(hh_key))

# %% 6. Observerte utfall og full roster
obs_table <- npccu2 %>%
  filter(!is.na(stp_nu), !is.na(kjoenn_c), !is.na(byear_c)) %>%
  arrange(group, pos_in_ped) %>%
  select(y = stp_nu, kjoenn_c, byear_c, invkat,
         group, cohort, household, household_observed, household_imputed,
         household_year = hushyear, household_target_year,
         household_year_shifted, id_ped = pos_in_ped)
if (nrow(obs_table) < 2L) stop("Fewer than two observed outcomes remain.")
my <- mean(obs_table$y)
sdy <- sd(obs_table$y)
if (!is.finite(sdy) || sdy <= 0) stop("Outcome has zero or invalid variance.")
obs_table$y <- (obs_table$y - my) / sdy
roster <- npccu2 %>%
  filter(group %in% obs_table$group) %>%
  arrange(group, pos_in_ped) %>%
  select(id_ped = pos_in_ped, group)

stopifnot(all(obs_table$id_ped %in% roster$id_ped),
          !anyNA(roster$group), !anyDuplicated(roster$id_ped),
          !anyDuplicated(obs_table$id_ped),
          all(obs_table$household_observed == !obs_table$household_imputed))
cat("Observed outcomes:", nrow(obs_table), "; full roster:", nrow(roster), "\n")
obs_table %>%
  count(household_target_year, household_year,
        household_year_shifted, household_imputed) %>%
  print(n = Inf)
klasser_per_kull <- obs_table %>% distinct(cohort, group) %>% count(cohort)
print(table(klasser_per_kull$n))

# %% 7. Designmatrise X: samme utvalg og faste effekter som i skjermbildene
inv_levels <- c(LETTERS[1:7], "ukjent")
obs_table$invkat <- factor(
  obs_table$invkat, levels = inv_levels[inv_levels %in% obs_table$invkat]
)
if (nlevels(obs_table$invkat) < 2L || levels(obs_table$invkat)[1] != "A") {
  stop("The intended immigration contrast requires reference A and at least one other level.")
}
X <- model.matrix(~ kjoenn_c + byear_c + invkat, data = obs_table)
stopifnot(nrow(X) == nrow(obs_table), all(is.finite(X)),
          qr(X)$rank == ncol(X))
print(colnames(X))

# %% 8. Trim og reverser pedigree; observasjonsrekkefolgen endres IKKE
obs_id_before <- ped[[id_col]][obs_table$id_ped]
roster_id_before <- ped[[id_col]][roster$id_ped]
obs_values_before <- select(obs_table, -id_ped)
roster_groups_before <- roster$group

row_keep <- pedigree_keep_rows(ped, roster$id_ped)
ped_trim <- ped[row_keep, ]
ped_trim$mid <- match(ped_trim$mid, row_keep, nomatch = 0L)
ped_trim$fid <- match(ped_trim$fid, row_keep, nomatch = 0L)
roster$id_ped <- match(roster$id_ped, row_keep)
obs_table$id_ped <- match(obs_table$id_ped, row_keep)
stopifnot(all((ped_trim$mid == 0L) == (ped$mid[row_keep] == 0L)),
          all((ped_trim$fid == 0L) == (ped$fid[row_keep] == 0L)),
          !anyNA(roster$id_ped), !anyNA(obs_table$id_ped))

# getLInv gir M med Ainv = M'M. I avkom-foerst rekkefolge er R M' R' nedre.
N <- nrow(ped_trim)
pedd <- pedigree(sire = as.integer(ped_trim$fid),
                 dam = as.integer(ped_trim$mid),
                 label = as.character(seq_len(N)))
r <- rev(seq_len(N))
Ainv <- getAInv(pedd)[r, r]
Linv <- t(getLInv(pedd))[r, r]

flip <- function(x) ifelse(x == 0L, 0L, N + 1L - x)
ped_trim <- ped_trim[r, ]
ped_trim$mid <- flip(ped_trim$mid)
ped_trim$fid <- flip(ped_trim$fid)
roster$id_ped <- flip(roster$id_ped)
obs_table$id_ped <- flip(obs_table$id_ped)

Ag <- as(Ainv, "generalMatrix")
Lt <- as(Linv, "generalMatrix")
if (nnzero(triu(Lt, k = 1L)) != 0L) {
  stop("The permuted factor is not lower triangular; do not truncate it.")
}
Ainv_triplet <- as.data.frame(mat2triplet(triu(Ag)))
Linv_triplet <- as.data.frame(mat2triplet(Lt))

# %% 9. Kontroller identiteter, foreldreforbindelser og faktor
stopifnot(all(ped_trim$mid == 0 | ped_trim$mid > seq_len(N)),
          all(ped_trim$fid == 0 | ped_trim$fid > seq_len(N)),
          all(obs_table$id_ped %in% roster$id_ped),
          identical(ped_trim[[id_col]][obs_table$id_ped], obs_id_before),
          identical(ped_trim[[id_col]][roster$id_ped], roster_id_before),
          identical(select(obs_table, -id_ped), obs_values_before),
          identical(roster$group, roster_groups_before))

old_rows_after <- row_keep[r]
stopifnot(identical(select(ped_trim, -mid, -fid),
                    select(ped[old_rows_after, ], -mid, -fid)))
for (parent in c("mid", "fid")) {
  known <- ped_trim[[parent]] > 0L
  stopifnot(all(old_rows_after[ped_trim[[parent]][known]] ==
                 ped[[parent]][old_rows_after[known]]))
}
set.seed(1)
for (k in 1:3) {
  x <- rnorm(N)
  ax <- Ag %*% x
  rel <- max(abs(ax - Lt %*% crossprod(Lt, x))) / max(abs(ax))
  cat("Factor relative residual:", rel, "\n")
  stopifnot(is.finite(rel), rel < 1e-10)
}
cat("Identity, parent-edge and observation-order checks passed.\n")
cat("N pedigree:", N, "n obs:", nrow(obs_table), "n roster:", nrow(roster),
    "classes:", n_distinct(roster$group), "cohorts:", n_distinct(obs_table$cohort),
    "X columns:", ncol(X), "\n")

# %% 10. Eksporter; X har intercept og samme radrekkefolge som obs_table
X_export <- bind_cols(tibble(id_ped = obs_table$id_ped), as_tibble(X))
stopifnot(identical(X_export$id_ped, obs_table$id_ped))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
write_csv(obs_table, file.path(out_dir, "yX_obs.csv"))
write_csv(roster, file.path(out_dir, "roster.csv"))
write_csv(X_export, file.path(out_dir, "X_obs.csv"))
write_csv(tibble(variable = "y", center = my, scale = sdy),
          file.path(out_dir, "scaling.csv"))
write_csv(ped_trim, file.path(out_dir, "ped_trim.csv"))
write_csv(Ainv_triplet, file.path(out_dir, "Ainv_triplet.csv"))
write_csv(Linv_triplet, file.path(out_dir, "LAinv_triplet.csv"))
