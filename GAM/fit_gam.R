#!/usr/bin/env Rscript
# Fit one mgcv GAM from a station parquet and write model + station fitted values.
# Called from gam_core.py. Columns required on input:
#   y, x, ycoord, elev [, slope, sinasp, cosasp, clc]
# Args: train.parquet out_dir formula_id family [n_splines]
# family: gaussian | binomial | Gamma

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: fit_gam.R train.parquet out_dir formula_id [family] [n_splines]")
}

train_path <- args[[1]]
out_dir <- args[[2]]
formula_id <- args[[3]]
family_name <- if (length(args) >= 4) args[[4]] else "gaussian"
n_splines <- if (length(args) >= 5) as.integer(args[[5]]) else 10L
if (!is.finite(n_splines) || n_splines < 4L) {
  n_splines <- 10L
}

suppressPackageStartupMessages({
  library(mgcv)
})

if (!requireNamespace("arrow", quietly = TRUE)) {
  stop("install.packages('arrow') required")
}

df <- arrow::read_parquet(train_path)
names(df) <- tolower(names(df))
if ("ycoord" %in% names(df)) {
  df$yc <- df$ycoord
} else {
  df$yc <- df$y
}

has <- function(col) col %in% names(df)
nuniq <- function(v) length(unique(v[is.finite(v)]))
k_ok <- function(v, k) max(4L, min(as.integer(k), nuniq(v) - 1L))

kx <- k_ok(df$x, n_splines)
ky <- k_ok(df$yc, n_splines)
ke <- k_ok(df$elev, max(4L, n_splines - 2L))

rhs <- switch(formula_id,
  te_xy_s_elev = sprintf("te(x, yc, k=c(%d,%d)) + s(elev, k=%d)", kx, ky, ke),
  te_xy_ti_elev = sprintf(
    "te(x, yc, k=c(%d,%d)) + s(elev, k=%d) + ti(x, elev, k=c(%d,%d)) + ti(yc, elev, k=c(%d,%d))",
    kx, ky, ke, kx, ke, ky, ke
  ),
  te_xyelev = sprintf("te(x, yc, elev, k=c(%d,%d,%d))", kx, ky, ke),
  s_xy_s_elev = sprintf("s(x, k=%d) + s(yc, k=%d) + s(elev, k=%d)", kx, ky, ke),
  stop("unknown formula_id: ", formula_id)
)
for (col in c("slope", "sinasp", "cosasp", "tmean", "tpi", "signed_dist_to_ridge")) {
  if (!has(col) || nuniq(df[[col]]) < 6L) next
  # tmean is the RH covariate. If it is (near-)identical to the response,
  # it is the target itself and must not enter the formula.
  if (col == "tmean" && "y" %in% names(df)) {
    ok <- is.finite(df$y) & is.finite(df[[col]])
    if (sum(ok) > 10L && abs(cor(df$y[ok], df[[col]][ok])) > 0.999) next
  }
  rhs <- paste(rhs, sprintf("+ s(%s, k=%d)", col, k_ok(df[[col]], max(4L, n_splines %/% 2L))))
}
if (has("clc")) {
  df$clc <- factor(df$clc)
  if (nlevels(df$clc) >= 2L) {
    rhs <- paste(rhs, "+ clc")
  }
}
# region is an unordered basin id, so a factor, not a smooth on the integer.
if (has("region")) {
  df$region <- factor(df$region)
  if (nlevels(df$region) >= 2L) {
    rhs <- paste(rhs, "+ region")
  }
}
# Enabled continuous core covariates as linear terms (already standardized in Python).
# Skip near-constant or elev-collinear columns; they trigger NaN edf in mgcv.
for (col in c("smooth_elev", "northness", "dist_to_coast",
              "basin_continuity", "elev_above_valley")) {
  if (!has(col) || nuniq(df[[col]]) < 6L) next
  ok <- is.finite(df$elev) & is.finite(df[[col]])
  if (sum(ok) > 20L && abs(cor(df$elev[ok], df[[col]][ok])) > 0.95) next
  rhs <- paste(rhs, "+", col)
}
# s(elev_bin) keeps the elevation effect smooth; ti(x, elev_bin) and
# ti(yc, elev_bin) let location override the bin. Continuous elev is
# already in the formula, so the bin is an additional stratified signal.
if (has("elev_bin")) {
  df$elev_bin <- as.numeric(df$elev_bin)
  if (nuniq(df$elev_bin) >= 4L) {
    kb <- k_ok(df$elev_bin, max(4L, n_splines %/% 2L))
    rhs <- paste(rhs, sprintf("+ s(elev_bin, k=%d)", kb))
    if (nuniq(df$x) >= 6L) {
      rhs <- paste(rhs, sprintf("+ ti(x, elev_bin, k=c(%d,%d))", k_ok(df$x, 6L), kb))
    }
    if (nuniq(df$yc) >= 6L) {
      rhs <- paste(rhs, sprintf("+ ti(yc, elev_bin, k=c(%d,%d))", k_ok(df$yc, 6L), kb))
    }
  }
}

fam <- switch(tolower(family_name),
  gaussian = gaussian(),
  binomial = binomial(),
  gamma = Gamma(link = "log"),
  gaussian()
)

form <- as.formula(paste("y ~", rhs))

# bam + discrete is the scalable path for tensor smooths at station density.
# Falls back to gam, then to a stripped formula, then to a pure spatial+elev
# model. A single degenerate timestamp must not abort the whole tune.
ctrl <- mgcv::gam.control(maxit = 40)

fit_try <- function(fml, discrete = TRUE) {
  tryCatch(
    mgcv::bam(fml, data = df, family = fam, method = "fREML",
              discrete = discrete, control = ctrl),
    error = function(e) {
      message("bam failed (", conditionMessage(e), "), trying gam")
      tryCatch(
        mgcv::gam(fml, data = df, family = fam, method = "REML", control = ctrl),
        error = function(e2) {
          message("gam failed (", conditionMessage(e2), ")")
          NULL
        }
      )
    }
  )
}

model <- fit_try(form, discrete = TRUE)
if (is.null(model)) {
  # Drop linear extras and ti terms; keep the core spatial + elev smooths.
  simple_rhs <- switch(formula_id,
    te_xy_s_elev = sprintf("te(x, yc, k=c(%d,%d)) + s(elev, k=%d)", kx, ky, ke),
    te_xy_ti_elev = sprintf("te(x, yc, k=c(%d,%d)) + s(elev, k=%d)", kx, ky, ke),
    te_xyelev = sprintf("te(x, yc, elev, k=c(%d,%d,%d))", kx, ky, ke),
    s_xy_s_elev = sprintf("s(x, k=%d) + s(yc, k=%d) + s(elev, k=%d)", kx, ky, ke),
    sprintf("s(x, k=%d) + s(yc, k=%d) + s(elev, k=%d)", kx, ky, ke)
  )
  message("retrying with simplified formula")
  model <- fit_try(as.formula(paste("y ~", simple_rhs)), discrete = FALSE)
}
if (is.null(model)) {
  message("retrying with pure s(x)+s(yc)+s(elev)")
  model <- fit_try(as.formula("y ~ s(x, k=6) + s(yc, k=6) + s(elev, k=6)"), discrete = FALSE)
}
if (is.null(model)) {
  stop("all mgcv attempts failed")
}

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(model, file.path(out_dir, "model.rds"))
writeLines(formula_id, file.path(out_dir, "formula.txt"))
sink(file.path(out_dir, "summary.txt"))
print(summary(model))
sink()
