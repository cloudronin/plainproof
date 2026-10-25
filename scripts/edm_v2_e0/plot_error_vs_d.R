# Figure: realised decision error against d, one line per method, both outsourcing models, AI accuracy 0.95.
# Input: reports/edm-v2-e0/error-vs-d.csv (from report.py, which names the summary hash). Base R only.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=3) stop('csv, png, summary sha256 required')
x <- read.csv(args[[1]]); sha <- args[[3]]
methods <- c('SPRT','Fixed (10 forms)','Random','BOBCAT','SPRT native stop')
cols <- c('#1b6ca8','#7a7a7a','#b8860b','#b03a2e','#1b6ca8'); lty <- c(1,1,1,1,2); pch <- c(16,15,17,18,1)
png(args[[2]],width=1800,height=820,res=170)
par(mfrow=c(1,2),mar=c(4.2,4.4,2.6,0.8),oma=c(2.2,0,0,0))
for(m in c('uniform','uncertainty')){
  plot(NA,xlim=c(0,1),ylim=c(0,55),xlab='Share of follow-ups outsourced (d)',ylab='Realised decision error (%)',
       main=paste(m,'outsourcing, AI accuracy 0.95'),xaxt='n')
  axis(1,at=c(0,.25,.5,.75,1))
  abline(h=10,col='grey80',lty=3); abline(h=50,col='grey80',lty=3)
  for(i in seq_along(methods)){
    s <- x[x$model==m & x$method==methods[i],]; s <- s[order(s$d),]
    lines(s$d,s$error_pct,col=cols[i],lty=lty[i],lwd=2); points(s$d,s$error_pct,col=cols[i],pch=pch[i])
  }
  if(m=='uniform') legend('topleft',legend=methods,col=cols,lty=lty,pch=pch,lwd=2,bty='n',cex=.8)
}
mtext(paste0('Held-out, 8,100 learners per cell. Error among decided, non-borderline learners at the calibrated 0.90 rule (native stop: SPRT boundary). Dotted lines: 10% and chance (50%). summary.json sha256 ',substr(sha,1,12)),
      side=1,outer=TRUE,cex=.62,line=0.8)
invisible(dev.off())
cat('wrote',args[[2]],'\n')
