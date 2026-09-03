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
if ("ycoord" %in% names(q)) {
  q$yc <- q$ycoord
} else if ("y" %in% names(q)) {
  q$yc <- q$y
}
if ("clc" %in% names(q) && "clc" %in% names(model$model)) {
  q$clc <- factor(q$clc, levels = levels(model$model$clc))
} else if ("clc" %in% names(q)) {
  q$clc <- factor(q$clc)
}

pred <- as.numeric(predict(model, newdata = q, type = "response"))
out <- data.frame(predicted = pred)
arrow::write_parquet(out, args[[3]])
