# Install catIrt (CRAN binary) into the existing native R library for the SPRT reference.
# Owner approved this one download (24 Oct 2026). Writes a receipt; no system R change.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1) stop('receipt path required')
receipt <- args[[1]]
lib <- Sys.getenv('R_LIBS_USER'); if(!nzchar(lib)) stop('R_LIBS_USER unset')
.libPaths(c(lib,file.path(R.home(),'library')))
options(repos=c(CRAN='https://cloud.r-project.org'),timeout=60)
available <- available.packages(type='binary')
if(!'catIrt' %in% rownames(available)) stop('No catIrt binary on CRAN; stop and ask')
deps <- tools::package_dependencies('catIrt',db=available,which=c('Depends','Imports','LinkingTo'),recursive=TRUE)[[1]]
need <- setdiff(c(deps,'catIrt'),c('R',rownames(installed.packages(lib.loc=.libPaths()))))
cache <- file.path(dirname(lib),'package-archives'); dir.create(cache,showWarnings=FALSE)
got <- download.packages(need,destdir=cache,available=available,type='binary')
md5 <- unname(tools::md5sum(got[,2]))
if(any(md5!=available[got[,1],'MD5sum'])) stop('MD5 mismatch; no install')
install.packages(got[,2],lib=lib,repos=NULL,type='binary')
if(!requireNamespace('catIrt',quietly=TRUE)) stop('catIrt failed to load')
sha <- vapply(got[,2],function(p) system2('shasum',c('-a','256',shQuote(p)),stdout=TRUE),'')
writeLines(c(paste('catIrt',as.character(packageVersion('catIrt',lib.loc=lib))),
             paste('installed:',paste(got[,1],collapse=',')),
             paste('sha256:',sha),paste('date:',format(Sys.time(),tz='UTC',usetz=TRUE))),receipt)
cat(readLines(receipt),sep='\n')
