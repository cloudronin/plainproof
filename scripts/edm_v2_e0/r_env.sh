#!/bin/bash
# Native R runtime from published-cat-audit-v1 (runtime-receipt-v2.json); no system R change.
N=/Users/vishnu/pipeline-store/reasoning-boundary/published-cat-audit-v1/native-r-v1
P=$N/expanded/R-fw.pkg/Payload/R.framework/Versions/4.6/Resources
export R_HOME=$P RHOME=$P DYLD_LIBRARY_PATH=$P/lib DYLD_FRAMEWORK_PATH=$N/expanded/R-fw.pkg/Payload R_LIBS_USER=$N/library TMPDIR=$N/tmp
exec "$P/bin/Rscript" "$@"
