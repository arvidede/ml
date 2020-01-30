### Setup ###
initRandom <- function(X, k) {
  sampleSet <- X
  mu <- rep(0, k)
  sigma <- rep(0, k)
  for(i in 1:k) {
    samples <- sample(sampleSet, size=floor(length(X)/k), replace=FALSE)
    mu[i] <- mean(samples)
    sigma[i] <- sd(samples)
    sampleSet <- setdiff(sampleSet, samples)
  }
  return(list(mu = mu, sigma = sigma))
}

# Init with equal euclidian distance between centroids, assumes sorted data
initMinMax <- function(X, k) {
  max <- which.max(X)
  min <- which.min(X)
  # Tbc
}

initKMeans <- function(X, k) {
  clusters <- kmeans(X, k)
  return(list(mu=clusters$centers, sigma=clusters$withinss/length(X)))
}

# Evaluate P(Z|X,Theta) given assumed mu & sigma
E <- function(X, k, mu, sigma, mixComp) {
  b <- matrix(nrow = k, ncol = length(X))
  
  for(i in 1:k) {
    b[i,] <- mixComp[i] * dnorm(X, mu[i], sigma[i])
    if(is.infinite(b[i,]) || is.na(b[i,])) b[i,] <- rep(0, length(X))
  }
  
  dens <- colSums(b)
  responsibility <- matrix(nrow = k, ncol = length(X))
  
  for(i in 1:k) {
    responsibility[i,] <- b[i,] / dens
    mixComp[i] <- sum(responsibility[i,]) / length(X)
  }
  
  return(list(dens = dens, responsibility = responsibility , mixComp = mixComp))
}

# Infer mu & sigma given expected data
M <- function(X, k, responsibility) {
  mu <- rep(0,k)
  sigma <- rep(0,k)
  for(i in 1:k) {
    mu[i] <- sum(responsibility[i,] * X) / sum(responsibility[i,])
    sigma[i] <- sqrt(sum(responsibility[i,] * (X-mu[i])^2) / sum(responsibility[i,]))
  }

  # In case of convergence towards same mean, give it a push to prevent local suboptimum
  if(abs(sum(diff(mu))) <= 0.1) mu <- mu * runif(length(mu), 0, 2)
  
  return(list(mu=mu, sigma=sigma))
}

plotConvergence <- function(X, mu, sigma, k, dist) {
  # Plot convergence
  y <- rep(0, length(X))
  for(i in 1:k) {
    y <- y + dnorm(xGrid, mu[i], sigma[i])
  }
  
  plot(xGrid, y / k, lty=2, type='l')
  if(dist > 1) {
    Sys.sleep(0.5)
  } else {
    Sys.sleep(0.005)
  }
}

GMM <- function(X, k, iters=1000, eps=1e-6, animate=FALSE, init=0) {
  # Setup, assuming uniform prior for the mixture components
  if(init) {
    par <- initKMeans(X, k)
  } else {
    par <- initRandom(X, k)
  }
  
  mu <- par$mu
  sigma <- par$sigma
  dist <- 1
  t <- 1
  Q <- 0
  mixComp <- rep(1/k, k)
  
  hist(X, 50, probability = TRUE)
  
  while(dist > eps & t < iters) {
    # E-step
    expectation <- E(X, k, mu, sigma, mixComp)
    
    # M-step
    maximization <- M(X, k, expectation$responsibility)
    mixComp <- expectation$mixComp
    mu <- maximization$mu
    sigma <- maximization$sigma
    
    # Calculate loglik for convergence
    t <- t + 1
      
    Q[t] <- sum(log(expectation$dens))
    
    dist <- abs(Q[t] - Q[t-1])
    if(animate) plotConvergence(X, mu, sigma, k, dist)
    
  }
  return(list(mu=mu, sigma=sigma, Q=Q, iters=t, mixComp=mixComp))
}


# Run
require('../utils/Data.r')

k <- round(runif(1,0,10))
data <- sampleMultimodal(k=k)
multimodal <- data$multimodal
xGrid <- data$xGrid

gmm <- GMM(multimodal, k=k, iters = 1000, animate = TRUE)

hist(multimodal, 50, probability = TRUE)
# lines(density(multimodal))

y <- rep(0, length(multimodal))
for(i in 1:k) {
  y <- y + gmm$mixComp[i] * dnorm(xGrid, gmm$mu[i], gmm$sigma[i])
}

lines(xGrid, y, col='red')



