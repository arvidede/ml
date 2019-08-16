#################################################
# The Kalman Filter                             #
# Bayesed on lectures by Jose Pena, LiU         # 
# Arvid Edenheim                                #
# arvid@edenheim.se                             #
# Last edited: 2019-08-16                       #
#################################################

# T = Timesteps
# A = Adaptation matrix to convert previous state to new state
# B = Adaptation matrix to convert action to new state
# C = Process covariance matrix
#     Represents the error of the prediction
# Q = Process noise
#     Covariance matrix for emission model
# R = Sensor noise
#     Covariance matrix for transition model
# u = Control variable matrix
# y = Observations

# Variables
# K = Kalman gain. Decides how much weight to put on the new measurement
#     given the error of the measurement. Large error => Smaller KG
# x = Latent state, weighted
# x_prediction = Predicted state, unweighted
# Sigma = Prediction error, weighted
# Sigma_prediction = Prediction error, unweighted

kalmanFilter <- function(T, A=rep(1,T), B=rep(1,T), C=rep(1,T), Q=rep(10,T), R=rep(1,T), u=rep(1,T), y) {

  # Initial setup
  K <- rep(0,T)
  x <- rep(0,T)
  x_prediction <- rep(0,T)
  Sigma <- rep(0,T)
  Sigma_prediction <- rep(0,T)
  
  # First iteration
  x[1] = x_prediction[1] <- y[1] # Guessing with first observation
  Sigma = Sigma_prediction[1] <- 1 # Or what?
  
  for(t in 2:T) {
    # Prediction
    x_prediction[t] <- A[t] * x[t-1] + B[t] * u[t]
    Sigma_prediction[t] <- A[t] * Sigma[t-1] * t(A[t]) + R[t]
    
    # Kalman gain
    K[t] <- Sigma_prediction[t] * t(C[t]) * solve(C[t] * Sigma_prediction[t] * t(C[t]) + Q[t])

    # Correction
    x[t] <- x_prediction[t] + K[t] * (y[t] - C[t] * x_prediction[t])
    Sigma[t] <- Sigma_prediction[t] * (1 - K[t] * C[t]) # Should probably be identity matrix for multidimensional support
  }
  return(list(x=x, Sigma=Sigma))
}


simulateData <- function(T, emissionSigma=1, transitionSigma=1, step=1) {
  x <- rep(0, T)
  y <- rep(0, T)
  init <- round(runif(1, 0, 10))
  for(t in 1:T) {
    # Transition
    x[t] <- if(t == 1) init else rnorm(1, mean = x[t-1] + step, transitionSigma)
    
    # Emission
    y[t] <- rnorm(1, mean = x[t], sd = emissionSigma)
    
  }
  return(list(states=x, observations=y))
}


######### Example #########


set.seed(111)

T <- 1000
data <- simulateData(T = T, emissionSigma = 3)
states <- data$states
observations <- data$observations

kalmanStates <- kalmanFilter(T=T, y=observations)

plot(states[1:50], type="l", xlab="t", ylab="State")
lines(observations[1:50], col="red")
lines(kalmanStates$x[1:50], col="blue")
legend("topleft", legend=c("States", "Observations", "Kalman approximation"), col=c("black", "red", "blue"), lty=c(1,1,1))
