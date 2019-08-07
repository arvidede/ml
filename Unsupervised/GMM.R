
### Generate Data ###
start <- 0
end <- 10
k <- 2
mu <- c(3,7)
sigma <- c(0.4, 1)
N <- 1000
xGrid <- seq(start, end, length.out = N)
multimodal <- c()

for(i in 1:k) {
  multimodal <- c(multimodal,rnorm(N, mean=mu[i], sd=sigma[i]))
}

hist(multimodal, 40)


### Setup ###

initRandom <- function(X, k) {
  sampleSet <- X
  mu <- rep(0, k)
  sigma <- rep(0, k)
  for(i in 1:k) {
    samples <- sample(sampleSet, size=round(length(X)/k), replace=FALSE)
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

# Evaluate P(Z|X,Theta) given assumed mu & sigma
E <- function(X, k, mu, sigma, mixComp) {
  b <- c()
  for(i in 1:k) {
    b <- c(b, mixComp[i] * dnorm(X, mu[i], sigma[i]))
  }
  
  for(i in 1:k) {
    mixComp[i] <- sum(b[i]) / length(X)
  }
  
  return(list(dens = b, responsibility = b / sum(b), mixComp = mixComp))
}

# Infer mu & sigma given expected data
M <- function(X, k, responsibility) {
  for(i in 1:k) {
    mu[i] <- sum(responsibility[i] * X) / sum(responsibility[i])
    print(sum((X-mu[i])^2))
    sigma[i] <- sqrt(sum(responsibility[i] * (X-mu[i])^2) / sum(responsibility[i]))
  }
  return(list(mu=mu, sigma=sigma))
}

GMM <- function(X, k, iters=1000, eps=1e-6) {
  # Setup, assuming uniform prior for the mixture components
  par <- initRandom(X, k)
  mu <- par$mu
  sigma <- par$sigma
  dist <- 1
  t <- 1
  Q <- 0
  mixComp <- rep(1/k, k)
  
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
    # print(expectation)
    # print(maximization)
    Sys.sleep(5)
  }
  return(list(mu=mu, sigma=sigma, Q=Q))
}


# Run

GMM(multivariateGaussian, 2)

