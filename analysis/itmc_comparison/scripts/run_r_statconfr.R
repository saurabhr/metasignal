#!/usr/bin/env Rscript
# Compute meta-I measures (R, statConfR::estimateMetaI) on the shared dataset.
# Mirrors analysis/itmc_comparison/scripts/run_python_itmc.py so both sides
# consume the exact same trial data.
#
# Usage:
#   Rscript analysis/itmc_comparison/scripts/run_r_statconfr.R

suppressMessages(library(statConfR))

here <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
root <- normalizePath(file.path(here, ".."))

data <- read.csv(file.path(root, "data", "shared_trials.csv"))
data$stimulus <- factor(data$stimulus)
data$rating <- factor(data$rating, ordered = TRUE)
data$correct <- as.integer(data$stimulus == factor(data$response, levels = levels(data$stimulus)))

cat(sprintf("Loaded %d trials, %d participants\n", nrow(data), length(unique(data$participant))))

set.seed(42)
no_bc <- estimateMetaI(data, bias_reduction = FALSE)
write.csv(no_bc, file.path(root, "results", "r_no_bias_correction.csv"), row.names = FALSE)
cat(sprintf("Wrote %d rows -> r_no_bias_correction.csv\n", nrow(no_bc)))

set.seed(42)
bc <- estimateMetaI(data, bias_reduction = TRUE)
write.csv(bc, file.path(root, "results", "r_bias_corrected.csv"), row.names = FALSE)
cat(sprintf("Wrote %d rows -> r_bias_corrected.csv\n", nrow(bc)))
