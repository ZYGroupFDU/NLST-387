import numpy as np
import random
n = 1000
beta0 = np.array([-3,2])
p = beta0.shape[0]
r = 10

def rcumsumr(x):
    return np.cumsum(x[::-1])[::-1]

##Likelihood
def Lik(theta):
    bZ = Z.dot(theta)
    L = np.mean((bZ - np.log(rcumsumr(np.exp(bZ))))[d == 1])
    return -L

##Stochastic Gradient of Likelihood (MCMC)
def SGradLik_MCMC(theta, j, N):
    bZ = Z.dot(theta)
    index = [n-1] * (N+1)
    for i in range(N):
        j_temp = random.sample(range(j-1,n), 1)[0]
        a = min(np.exp(bZ[j_temp]-bZ[index[i]]), 1)
        u = np.random.binomial(1, a, 1)[0]
        index[i+1] = (j_temp*u + index[i]*(1-u))
    index = index[1:]
    GradL = Z[j,:] - np.mean(Z[index,:], axis=0)
    return -GradL

##Stochastic Gradient Matrix of Likelihood
def SGradMatrixLik(theta):
    bZ = Z.dot(theta)
    GradL = (Z - (np.apply_along_axis(rcumsumr,0,(np.exp(bZ)*Z.T).T).T/rcumsumr(np.exp(bZ))).T)[d == 1,:]
    return -GradL

##Stochastic Gradient Decent
def SGD(K, m, N_k, g, gMatrix, alpha, initial):
    xTilde = x = initial
    GradMatrix = gMatrix(xTilde)
    tempMatrix = np.zeros((m,p))
    k = 0
    while k < K:
        t = 0
        while t < m:
            u = random.sample(range(ne), 1)[0]
            tempMatrix[t,] = x
            x = x - alpha*(g(x, np.where(d == 1)[0][u], N_k[k]) - GradMatrix[u,] + np.mean(GradMatrix, axis=0))
            t = t + 1
        xTilde = x = np.mean(tempMatrix, axis=0)
        GradMatrix = gMatrix(xTilde)
        k = k + 1
    return x

##Simulation
result = np.zeros((r,p))
for i in range(r):
    Z = np.random.normal(1, 1, n*p).reshape(n, p) + 1
    pop = np.random.exponential((2*np.exp(Z.dot(beta0)))**(-1), n)
    cen = np.random.exponential((2*np.exp(Z.dot(np.array([-3,1]))))**(-1), n)
    y = np.min(np.vstack((pop, cen)), axis=0)
    d = (pop < cen).astype(int)
    dat = np.column_stack((y,d,Z))
    dat = dat[np.argsort(y),:]
    y = dat[:,0]
    d = dat[:,1]
    Z = dat[:,2:4]
    ne = int(np.sum(d))
    result[i,:] = SGD(10, 100, 100*np.array(range(1,11)), SGradLik_MCMC, SGradMatrixLik, 0.1, np.array([-2,1]))
print(np.mean(result, axis=0))
print(np.std(result, axis=0))
