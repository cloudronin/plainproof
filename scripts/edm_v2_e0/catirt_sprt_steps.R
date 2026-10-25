# Run catIrt 0.5.1's SPRT classification on a fixed bank and fixed response strings; save every step.
# Usage: catirt_sprt_steps.R <out_dir>. Writes bank.csv, resp.csv, steps.csv, config.csv. No data read.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1) stop('out_dir required')
out <- args[[1]]; dir.create(out,recursive=TRUE,showWarnings=FALSE)
suppressMessages(library(catIrt))
if(as.character(packageVersion('catIrt'))!='0.5.1') stop('catIrt version changed')
set.seed(20261024)
n_items <- 40; n_sim <- 25
cut <- 0.0; delta <- 0.3; alpha <- 0.05; beta <- 0.05; n_max <- 30
a <- round(runif(n_items,0.6,2.0),4); b <- round(rnorm(n_items),4); g <- round(runif(n_items,0,0.25),4)
params <- cbind(a,b,g)
theta <- seq(-2,2,length.out=n_sim)
resp <- simIrt(theta=theta,params=params,mod='brm')$resp
catStart <- list(n.start=1,init.theta=cut,select='UW-FI',at='bounds',n.select=1,it.range=NULL,
                 score='step',range=c(-6,6),step.size=3,leave.after.MLE=FALSE)
catMiddle <- list(select='UW-FI',at='bounds',n.select=1,it.range=NULL,score='MLE',range=c(-6,6),expos='none')
catTerm <- list(term='class',score='MLE',n.min=1,n.max=n_max,
                c.term=list(method='SPRT',bounds=cut,categ=c(0,1),delta=delta,alpha=alpha,beta=beta))
res <- catIrt(params=params,mod='brm',resp=resp,catStart=catStart,catMiddle=catMiddle,catTerm=catTerm,progress=FALSE)
rows <- list()
for(i in seq_len(n_sim)){
  ind <- res$cat_indiv[[i]]
  L <- res$cat_length[i]; its <- ind$cat_it[1:L]; u <- ind$cat_resp[1:L]
  for(k in seq_along(its)){
    pk <- params[its[1:k],,drop=FALSE]; class(pk) <- c('brm','matrix')
    lr <- catIrt:::logLik.brm(u[1:k],cut+delta,pk)-catIrt:::logLik.brm(u[1:k],cut-delta,pk)
    rows[[length(rows)+1]] <- data.frame(sim=i-1,step=k,item=its[k]-1,resp=u[k],log_ratio=sprintf('%.17g',lr))
  }
}
steps <- do.call(rbind,rows)
term <- data.frame(sim=seq_len(n_sim)-1,categ=res$cat_categ,term=res$cat_term,
                   n=res$cat_length,theta_hat=sprintf('%.17g',res$cat_theta))
write.csv(data.frame(a=sprintf('%.17g',a),b=sprintf('%.17g',b),c=sprintf('%.17g',g)),file.path(out,'bank.csv'),row.names=FALSE)
write.csv(resp,file.path(out,'resp.csv'),row.names=FALSE)
write.csv(steps,file.path(out,'steps.csv'),row.names=FALSE)
write.csv(term,file.path(out,'terminal.csv'),row.names=FALSE)
write.csv(data.frame(cut=cut,delta=delta,alpha=alpha,beta=beta,n_max=n_max,catIrt=as.character(packageVersion('catIrt')),
                     R=R.version.string),file.path(out,'config.csv'),row.names=FALSE)
cat('sims',n_sim,'steps',nrow(steps),'\n'); print(table(term$term,useNA='ifany'))
