set.seed(403)
n <- 10000
beta0 <- c(-3,2)
p <- length(beta0)
r <- 10
design <- "FC"

rcumsumr <- function(x){
  rev(cumsum(rev(as.vector(x))))
}

##Likelihood
Lik <- function(theta){
  bZ <- as.vector(Z%*%theta)
  L <- mean((bZ-log(rcumsumr(D/pi*exp(bZ))))[which(d == 1)])
  return(-L)
}

##gradient of Likelihood
gradLik <- function(theta){
  bZ <- as.vector(Z%*%theta)
  gradL <- apply((Z - (apply(D/pi*exp(bZ)*Z,2,rcumsumr))/(rcumsumr(D/pi*exp(bZ))))[which(d == 1),],2,mean)
  return(-gradL)
}

##stochastic gradient of Likelihood
sgradLik <- function(theta, j, N){ #N is useless
  bZ <- as.vector(Z[j:n,]%*%theta)
  gradL <- Z[j,] - (apply(as.matrix(D[j:n]/pi[j:n]*exp(bZ)*Z[j:n,]),2,sum))/(sum(D[j:n]/pi[j:n]*exp(bZ)))
  return(-gradL)
}

##stochastic gradent of Likelihood (MCMC)
sgradLik.MCMC <- function(theta, j, N){
  bZ <- as.vector(Z%*%theta)
  index <- rep(n, N+1)
  for(i in 2:length(index)){
    j.temp <- sample(n-j+1, 1)+j-1
    a <- min(exp(bZ[j.temp]-bZ[index[i-1]]),1)
    u <- rbinom(1, 1, a)
    index[i] <- j.temp*u + index[i-1]*(1-u)
  }
  index <- index[-1]
  gradL <- Z[j,] - apply(Z[index,],2,mean)
  return(-gradL)
}

##stochastic gradient Matrix of Likelihood
sgradMatrixLik <- function(theta){
  bZ <- as.vector(Z%*%theta)
  gradL <- (Z - (apply(D/pi*exp(bZ)*Z,2,rcumsumr))/(rcumsumr(D/pi*exp(bZ))))[which(d == 1),]
  return(-gradL)
}

##Gradient Decent
GD <- function(threshold, G, g, alpha, initial){
  converged <- F
  x <- initial
  y <- G(x)
  while(converged == F){
    x1 <- x - g(x)*alpha
    y1 <- G(x1)
    if(abs(y - y1) <= threshold){
      converged <- T
    }
    x <- x1
    y <- y1
  }
  return(x)
}

##Stochastic Gradient Decent
SGD <- function(K, m, N.k, g, gMatrix, alpha, initial){
  xTilde <- x <- initial
  gradMatrix <- gMatrix(xTilde)
  tempMatrix <- matrix(0,m,p)
  k <- 1
  while(k <= K){
    t <- 1
    while(t <= m){
      u <- sample(ne, 1)
      tempMatrix[t,] <- x
      x <- x - alpha*(g(x, which(d == 1)[u], N.k[k]) - gradMatrix[u,] + apply(gradMatrix, 2, mean))
      t <- t+1
    }
    xTilde <- x <- apply(tempMatrix, 2, mean)
    gradMatrix <- gMatrix(xTilde)
    k <- k+1
  }
  return(x)
}

result <- matrix(0,r,p)
unix.time(
  for(i in 1:r){
    Z <- matrix(rnorm(p*n,1,1),n)
    pop <- rexp(n,2*exp(Z%*%beta0))
    cen <- rexp(n,2*exp(Z%*%c(-3,1)))
    y <- pmin(pop,cen)
    d <- as.numeric(pop < cen)
    dat <- data.frame(y,d,Z)
    dat <- dat[order(y),]
    Z <- as.matrix(dat[,-c(1,2)])
    y <- dat$y
    d <- dat$d
    ne <- sum(dat$d)
    if(design == "EP"){
      pi <- d + (1 - d) * 0.7 * (1 / (1/exp(3*y-1.5) + 1))
      D <- rbinom(n,1,pi)
    }else if(design == "CC"){
      subSize <- 300
      D <- d
      D[sample(n,subSize)] <- 1
      D <- pmin(D,1)
      pi <- d + (1 - d) * subSize/n
    }else if(design == "FC"){
      pi <- rep(1,n)
      D <- rep(1,n)
    }
    
    #result[i,] <- optim(c(-2,1),Lik,gradLik,method = "BFGS")$par
    #result[i,] <- GD(1e-5, Lik, gradLik, 0.1, c(-2,1))
    result[i,] <- SGD(10, 100, 100*(1:10), sgradLik.MCMC, sgradMatrixLik, 0.1, c(-2,1))
    print(i)
  }
)

apply(result,2,mean)
apply(result,2,sd)