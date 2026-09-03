#!/usr/bin/env Rscript
# Fit one mgcv GAM from a station parquet and write model + station fitted values.
# Called from gam_core.py. Columns required on input:
#   y, x, ycoord, elev [, slope, sinasp, cosasp, clc]
# Args: train.parquet out_dir formula_id family
# family: gaussian | binomial | Gamma

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: fit_gam.R train.parquet out_dir formula_id [family]")
}

train_path <- args[[1]]
out_dir <- args[[2]]
formula_id <- args[[3]]
family_name <- if (length(args) >= 4) args[[4]] else "gaussian"

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
clc <- has("clc")
nuniq <- function(v) length(unique(v[is.finite(v)]))
k_ok <- function(v, k = 10L) max(4L, min(as.integer(k), nuniq(v) - 1L))

kx <- k_ok(df$x, 10L)
ky <- k_ok(df$yc, 10L)
ke <- k_ok(df$elev, 8L)

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
    rhs <- paste(rhs, sprintf("+ s(%s, k=%d)", col, k_ok(df[[col]], 6L)))
  }
}
if (clc) {
  df$clc <- factor(df$clc)
  rhs <- paste(rhs, "+ clc")
}

fam <- switch(family_name,
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
