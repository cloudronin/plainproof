# Figure: raw-hit AUC against d, one line per outsourcing model, AI accuracy 0.95, 15 follow-ups. Base R only.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=3) stop('csv, png, summary sha256 required')
x <- read.csv(args[[1]]); sha <- args[[3]]
models <- c('uniform','uncertainty','giveup'); labels <- c('Uniform','Uncertainty-triggered (null by construction)','Give-up (exploratory)')
cols <- c('#1b6ca8','#7a7a7a','#b03a2e'); pch <- c(16,15,17)
png(args[[2]],width=1200,height=900,res=170)
par(mar=c(4.4,4.6,2.6,1),oma=c(2,0,0,0))
plot(NA,xlim=c(0,1),ylim=c(0.45,1.02),xlab='Share of follow-ups outsourced (d)',ylab='AUC, raw probe hits (honest vs outsourcing)',
     main='Planted probes, AI accuracy 0.95, 15 follow-ups',xaxt='n')
axis(1,at=c(0,.25,.5,.75,1)); abline(h=0.75,col='grey55',lty=2); abline(h=0.5,col='grey85',lty=3)
text(0.02,0.765,'pre-registered line 0.75 (uniform, d = 0.5)',adj=0,cex=.7,col='grey35')
for(i in seq_along(models)){ s <- x[x$model==models[i],]; s <- s[order(s$d),]
  lines(s$d,s$raw_hits_auc,col=cols[i],lwd=2); points(s$d,s$raw_hits_auc,col=cols[i],pch=pch[i]) }
legend('bottomright',legend=labels,col=cols,pch=pch,lwd=2,bty='n',cex=.75)
mtext(paste0('Held-out, 2,000 learners per cell; d = 0 is the honest reference (AUC 0.5). E0b summary.json sha256 ',substr(sha,1,12)),side=1,outer=TRUE,cex=.62,line=0.6)
invisible(dev.off()); cat('wrote',args[[2]],'\n')
