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
  stop("unknown formula_id: ", formula_id)
)
for (col in c("slope", "sinasp", "cosasp")) {
  if (has(col) && nuniq(df[[col]]) >= 6L) {
    rhs <- paste(rhs, sprintf("+ s(%s, k=%d)", col, k_ok(df[[col]], max(4L, n_splines %/% 2L))))
  }
}
if (has("clc")) {
  df$clc <- factor(df$clc)
  if (nlevels(df$clc) >= 2L) {
    rhs <- paste(rhs, "+ clc")
  }
}

fam <- switch(tolower(family_name),
  gaussian = gaussian(),
  binomial = binomial(),
  gamma = Gamma(link = "log"),
  gaussian()
)

form <- as.formula(paste("y ~", rhs))
model <- mgcv::gam(form, data = df, family = fam, method = "REML")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(model, file.path(out_dir, "model.rds"))
writeLines(formula_id, file.path(out_dir, "formula.txt"))
sink(file.path(out_dir, "summary.txt"))
print(summary(model))
sink()
