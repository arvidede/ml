
sampleMultimodal <- function(k,mu=c(),sigma=c(), start=0, end=10, N=1000) {
  ### Generate Data ###
  
  if(!length(mu)) mu <- runif(k, start, end)
  if(!length(sigma)) sigma <- runif(k, 0, 1)
  
  multimodal <- c()
  
  for(i in 1:k) {
    multimodal <- c(multimodal,rnorm(N/k, mean=mu[i], sd=sigma[i]))
  }
  
  xGrid <- seq(start, end, length.out = length(multimodal))
  
  return(list(xGrid=xGrid, multimodal=multimodal, mu=mu, sigma=sigma))
}
