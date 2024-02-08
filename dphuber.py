"""
     M-estimator of Mean Vectors
"""

import numpy as np
from numpy import linalg as LA
import numpy.random as rgt
import math
from scipy.stats import norm
from scipy.optimize import minimize


class huberReg():
    

    def __init__(self, X, Y, intercept=True):
        '''
        Argumemnts
            X: n by d numpy array. n is the number of observations.
               Each row is an observation vector of dimension d.

            Y: n response variable
        '''
        self.n, self.d = X.shape

        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.concatenate([np.ones((self.n,1)), X], axis=1)
            self.X1 = np.concatenate([np.ones((self.n,1)), (X - self.mX)/self.sdX], axis=1)
        else:
            self.X, self.X1 = X, X/self.sdX
        self.Y = Y
        self.tau = (self.n/np.log(self.n))**0.5

    def stability(self, epsilon = 0.5, delta = None, K = None):
        '''
        Return noisy estimate of $\EE\|X - \mu\|_2^2$.
        '''
        if K == None:
            K = math.floor(np.sqrt(self.n)/2)

        if delta == None :
            delta = norm.cdf(-1 + epsilon/2) - np.exp(epsilon)*norm.cdf(-1 - epsilon/2)

        distances = [(LA.norm(self.X[2*i,:] - self.X[2*i + 1])**2)/2 for i in range(math.floor(self.n/2) - 1)]
        partition_size = len(distances) // K
        partitions = [distances[i:i+partition_size] for i in range(0, len(distances), partition_size)]
        medians = [np.median(partition) for partition in partitions]

        smallest_median = min(medians)
        largest_median = max(medians)

        smallest_bin = math.floor(np.log2(smallest_median))
        largest_bin = math.floor(np.log2(largest_median))

        count = np.zeros(largest_bin - smallest_bin + 1)

        for median in medians :
            bin_exponent = math.floor(np.log2(median))
            count[bin_exponent - smallest_bin] += 1

        prob = count/len(medians)
        t = 2*np.log(2/delta)/(epsilon*K) + 1/K

        for i in range(len(prob)):
            if prob[i] > 0:
                prob[i] += rgt.laplace(0, 2/(epsilon*K))
                if prob[i] < t:
                    prob[i] = 0

        return 2**(smallest_bin + np.argmax(prob))


    def robust_weight(self, x, method="Huber"):
        if method == "Huber":
            return np.where(x>1, 1/x, 1)
        if method == "Catoni":
            return np.nan_to_num(np.log(1 + x + 0.5*x**2)/x, nan=1)
        
    def robust_reg_weight_low(self, x, gamma=1):
        return np.where(x>gamma, gamma/x, 1)
    
    def robust_tau(self, c=0.5, dim='low'):
        if dim == 'low':
            return c*np.std(self.Y, ddof=1)*((self.n/np.log(self.n))**1/2)
        if dim == 'high':
            return c*c*np.std(self.Y, ddof=1)*((self.n/(np.log(self.n)*np.log(self.d)))**1/2)
        
    def huber_loss_score_function(self, x, tau):
    
        # Calculate the score function
        score = np.where(np.abs(x) <= tau, x, tau * np.sign(x))
        return score
    
    def noisyht(self, v, s, epsilon, delta, lambda_scale):
        d = len(v)
        S =set()
        for _ in range(s):
            w= np.random.laplace(0, lambda_scale*2*np.sqrt(3*s*np.log(1/delta))/epsilon, (d,))
            candidates = [(abs(v[j]) + w[j], j) for j in range(d) if j not in S]
            _, j_max = max(candidates, key=lambda x: x[0])
            S.add(j_max)

        v_S = np.zeros(d)
        for j in S:
            v_S[j] = v[j]
        noise = np.random.laplace(0, lambda_scale*2*np.sqrt(3*s*np.log(1/delta))/epsilon, (d,))
        for j in S:
            v_S[j] += noise[j]

        return v_S

        


    def f1(self, x, resSq, n, rhs):
        """
        Function to be zeroed.
        x: Candidate value for tau.
        resSq: Squared residuals.
        n: Number of observations.
        rhs: Right-hand side value, a predetermined constant.
        """
        return np.mean(np.minimum(resSq / x, np.ones(n))) - rhs

    def rootf1(self, resSq, n, rhs, low, up, tol=0.001, maxIte=500):
        """
        Bisection method to find the root of f1.
        resSq: Squared residuals.
        n: Number of observations.
        rhs: Right-hand side value.
        low, up: Initial lower and upper bounds for the root.
        tol: Tolerance for the root's accuracy.
        maxIte: Maximum number of iterations.
        """
        ite = 1
        while ite <= maxIte and up - low > tol:
            mid = 0.5 * (low + up)
            val = self.f1(mid, resSq, n, rhs)
            if val < 0:
                up = mid
            else:
                low = mid
            ite += 1

        return 0.5 * (low + up)

    def ada_huber_reg_lowdim(self, beta0, maxit, eta, epsilon=10**(-4), tau=None):

        if beta0 is None:
            beta0 = np.zeros(self.d)

        if tau == None:
            
            tau = self.robust_tau(dim='low')
            beta1 = beta0
            res = self.Y-self.X.dot(beta1)
            l2 =1
            count = 0

            while count < maxit and l2>epsilon:
                grad1 = -self.X.T.dot(self.huber_loss_score_function(res, tau=tau)*self.robust_reg_weight_low(np.sum(self.X**2, axis=1)**0.5))/self.n
                l2 = np.sqrt(np.sum(grad1 ** 2))
                beta0 = beta1
                beta1 += -eta*grad1
                res = self.Y-self.X.dot(beta1)
                
                #beta_seq[:, count+1] = beta1
                count += 1
            
        else:
            beta1 = beta0
            res = self.Y-self.X.dot(beta1)
            l2 =1
            count = 0

            while count < maxit and l2>epsilon:
                grad1 = self.X.T.dot(self.huber_loss_score_function(res, tau=tau)*self.robust_reg_weight_low(np.sum(self.X**2, axis=1)**0.5))/self.n
                l2 = np.sqrt(np.sum(grad1 ** 2))
                beta0 = beta1
                beta1 += -eta*grad1
                res = self.Y-self.X.dot(beta1)
                #beta_seq[:, count+1] = beta1
                count += 1


        return beta1, [res, tau, count]


    def noisy_huber_reg_lowdim(self, epsilon, T, delta, eta, beta0 = None, gamma=1, tau=None, 
                               standardize=True, adjust=True):

        if beta0 is None:
            beta0 = np.zeros(self.d+int(self.itcp))

        if T is None:
            T = int((np.log(self.n)))
        if standardize: X = self.X1
        else: X= self.X
        beta_seq = np.zeros([X.shape[1], T+1])
        beta_seq[:,0] = beta0

        if tau == None:
            
            tau = 1.345*np.median(np.abs(self.Y-np.median(self.Y)))
            #tau = self.robust_tau()
            beta1 = beta0
            res = self.Y-X.dot(beta1)
            count = 0
            rownorm = np.sum(X**2, axis=1)**0.5

            while count < T:
                grad1 = X.T.dot(self.huber_loss_score_function(res, tau=tau)*self.robust_reg_weight_low(rownorm))/self.n
                noise = rgt.multivariate_normal(np.zeros(X.shape[1]), np.identity(X.shape[1]))
                diff = eta*grad1 + 2*eta*T*np.sqrt(2*np.log(2*T/delta))*gamma*tau*noise/(epsilon*self.n)
                beta1 += diff
                res = self.Y-X.dot(beta1)
                beta_seq[:, count+1] = np.copy(beta1)
                count += 1
            
        else:
            beta1 = beta0
            res = self.Y-X.dot(beta1)
            count = 0
            rownorm = np.sum(X**2, axis=1)**0.5

            while count < T:
                grad1 = X.T.dot(self.huber_loss_score_function(res, tau=tau)*self.robust_reg_weight_low(rownorm))/self.n
                noise = rgt.multivariate_normal(np.zeros(X.shape[1]), np.identity(X.shape[1]))
                diff = eta*grad1 + 2*eta*T*np.sqrt(2*np.log(2*T/delta))*gamma*tau*noise/(epsilon*self.n)
                beta1 += diff
                res = self.Y-X.dot(beta1)
                beta_seq[:, count+1] = np.copy(beta1)
                count += 1

        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: 
                beta0[0] -= self.mX.dot(beta0[1:])
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])


        return beta1, [res, tau, count], beta_seq
    
    def noisy_huber_reg_lowdim_dev(self, epsilon, T, delta, eta, beta0 = None, gamma=1, tau=None, 
                               standardize=True, adjust=True):

        if beta0 is None:
            beta0 = np.zeros(self.d+int(self.itcp))

        if T is None:
            T = int((np.log(self.n)))
        if standardize: X = self.X1
        else: X= self.X
        beta_seq = np.zeros([X.shape[1], T+1])
        beta_seq[:,0] = beta0

        if tau == None:
            rhs = (1/self.n)*(self.d+np.log(self.n*self.d))
            tau = 1.345*np.median(np.abs(self.Y-np.median(self.Y)))
            #tau = self.robust_tau()
            beta1 = beta0
            res = self.Y-X.dot(beta1)
            count = 0
            rownorm = np.sum(X**2, axis=1)**0.5

            while count < T:
                grad1 = X.T.dot(self.huber_loss_score_function(res, tau=tau)*self.robust_reg_weight_low(rownorm))/self.n
                noise = rgt.multivariate_normal(np.zeros(X.shape[1]), np.identity(X.shape[1]))
                diff = eta*grad1 + 2*eta*T*np.sqrt(2*np.log(2*T/delta))*gamma*tau*noise/(epsilon*self.n)
                beta1 += diff
                res = self.Y-X.dot(beta1)
                resSq = res*res
                tau = self.rootf1(resSq=resSq, n=self.n, rhs=rhs, low=np.min(resSq), up=np.sum(resSq))
                beta_seq[:, count+1] = np.copy(beta1)
                count += 1
            
        else:
            beta1 = beta0
            res = self.Y-X.dot(beta1)
            count = 0
            rownorm = np.sum(X**2, axis=1)**0.5

            while count < T:
                grad1 = X.T.dot(self.huber_loss_score_function(res, tau=tau)*self.robust_reg_weight_low(rownorm))/self.n
                noise = rgt.multivariate_normal(np.zeros(X.shape[1]), np.identity(X.shape[1]))
                diff = eta*grad1 + 2*eta*T*np.sqrt(2*np.log(2*T/delta))*gamma*tau*noise/(epsilon*self.n)
                beta1 += diff
                res = self.Y-X.dot(beta1)
                beta_seq[:, count+1] = np.copy(beta1)
                count += 1

        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: 
                beta0[0] -= self.mX.dot(beta0[1:])
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])


        return beta1, [res, tau, count], beta_seq

    def noisy_huber_reg_highdim(self, beta0, epsilon, T, delta, eta, s, gamma=1, tau=None, standardize=True, adjust=True):

        if beta0 is None:
            beta0 = np.zeros(self.d+int(self.itcp))
        if T == None:
            T = int((np.log(self.n)))

        if standardize: X = self.X1
        else: X= self.X
        beta_seq = np.zeros([X.shape[1], T+1])
        beta_seq[:,0] = beta0


        if tau == None:
            
            tau = 1.345*np.median(np.abs(self.Y-np.median(self.Y)))
            beta1 = beta0
            res = self.Y-self.X.dot(beta1)
            count = 0

            while count < T:
                grad1 = X.T.dot(self.huber_loss_score_function(res, tau=tau)*self.robust_reg_weight_low(np.max(np.abs(X), axis=1)))/self.n
                diff = eta*grad1 
                beta1 += diff
                beta1 =  self.noisyht(beta1, s=s, epsilon=epsilon/T, delta=delta/T, lambda_scale=2*(eta/self.n)*gamma*tau)
                res = self.Y-X.dot(beta1)
                beta_seq[:, count+1] = beta1
                count += 1

        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: 
                beta0[0] -= self.mX.dot(beta0[1:])
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])
            
 

        return beta_seq[:,-1], [res, tau, count], beta_seq
    
    def noisy_ls_highdim(self, beta0, epsilon, T, delta, eta, s, standardize=True, adjust=True):

        if beta0 is None:
            beta0 = np.zeros(self.d+int(self.itcp))
        if T == None:
            T = int((np.log(self.n)))

        if standardize: X = self.X1
        else: X= self.X
        beta_seq = np.zeros([X.shape[1], T+1])
        beta_seq[:,0] = beta0
        res = self.Y-X.dot(beta0)
        count = 0
        beta1 = beta0

        while count < T:
            grad1 = X.T.dot(res)/self.n
            diff = eta*grad1 
            beta1 += diff
            beta1 =  self.noisyht(beta1, s=s, epsilon=epsilon/T, delta=delta/T, lambda_scale=4*(eta/self.n)*np.sqrt(2*np.log(self.n)/s))  
            res = self.Y-X.dot(beta1)              
            beta_seq[:, count+1] = beta1
            count += 1

        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: 
                beta0[0] -= self.mX.dot(beta0[1:])
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])
            
 

        return beta_seq[:,-1], res, beta_seq

        
