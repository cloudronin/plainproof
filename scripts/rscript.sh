#!/bin/bash
# Run an R script with any local R (4.x) that has catIrt 0.5.1 installed: install.packages("catIrt") pins the version via
# the CRAN archive if needed. The sealed scripts/edm_v2_e0/r_env.sh is kept verbatim as the record of the original runtime.
exec Rscript "$@"
