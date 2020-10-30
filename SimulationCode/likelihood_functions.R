# WRITE PARAMETER FILE, RUN COMMAND (use "./is_moments" and "paramfile"
write_param_file <- function(parameters) write(parameters, "paramfile", ncol=length(parameters))
run <- function(executable, data, nparticles, isss, paramfile) as.numeric(system(paste(executable,data,as.integer(nparticles),as.integer(isss),paramfile, T), intern=T))

# LOG LIKELIHOOD FOR ONE SITE
site_loglike <- function(site, params, isss=5000, is.iter=1, pop="growing"){
	write(params,"paramfile",ncol=length(params))
	target = paste(pop, "/gtree_file",site, sep="")
	return(log(as.numeric(system(paste("./is_moments",target,as.integer(is.iter),as.integer(isss),"paramfile", T), intern=T))))
}


## FULL LOG LIKELIHOOD (SUM OF SITES LIKELIHOODS)
full_loglike <- function(sites, params, isss=5000, pop="growing"){
  write(params, "paramfile", ncol=length(params)) 
  chunk_files <- list()
  for (i in seq_along(sites)){
    chunk_files[[i]] <- paste(pop, "/gtree_file", sites[[i]], sep="")
  }
  # Now compute the product of all the likelihoods for the chunk
  ll <- 0.0
  for (target in chunk_files){
    ll <- ll + log(as.numeric(system(paste("./is_moments",target,as.integer(1),as.integer(isss),"paramfile", T), intern=T)))
  }
  return(ll)
}




