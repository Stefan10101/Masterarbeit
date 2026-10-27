#!/usr/bin/env Rscript
# Predict from a saved mgcv model.
# Args: model.rds query.parquet out.parquet

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: predict_gam.R model.rds query.parquet out.parquet")
}

suppressPackageStartupMessages({
  library(mgcv)
  library(arrow)
})

model <- readRDS(args[[1]])
q <- arrow::read_parquet(args[[2]])
names(q) <- tolower(names(q))
cat("PREDICT cols:", paste(names(q), collapse=","), "\n")
cat("PREDICT nrows:", nrow(q), "\n")
if ("y" %in% names(q)) {
  cat("WARNING: response column y present in query\n")
}
if ("ycoord" %in% names(q)) {
  q$yc <- q$ycoord
} else if ("y" %in% names(q)) {
  q$yc <- q$y
  cat("WARNING: yc taken from y (response)\n")
}
if ("clc" %in% names(q) && "clc" %in% names(model$model)) {
  q$clc <- factor(q$clc, levels = levels(model$model$clc))
} else if ("clc" %in% names(q)) {
  q$clc <- factor(q$clc)
}

pred <- as.numeric(predict(model, newdata = q, type = "response"))
cat("PREDICT len:", length(pred), " first3:", paste(head(pred, 3), collapse=","), "\n")
out <- data.frame(predicted = pred)
arrow::write_parquet(out, args[[3]])
