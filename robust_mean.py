"""
     M-estimator of Mean Vectors
"""

import numpy as np
from numpy import linalg as LA
import numpy.random as rgt
import math
from scipy.stats import norm

class m_est():
    methods = ["Catoni", "Huber"]

    def __init__(self, X):
        '''
        Argumemnts
            X: n by d numpy array. n is the number of observations.
               Each row is an observation vector of dimension d.
        '''
        self.n, self.d = X.shape
        self.X = X
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

    def trun_cov(self, res, w):
        cov = (res*w[:,None]).T.dot(res*w[:,None])/self.n
        return [LA.eigvalsh(cov)[-1], cov]

    def m_q(self, res, q = 4):
        mq = np.median(np.linalg.norm(res, axis = 1)**q)
        return mq

    def mean(self, tau=None, method="Huber", mu0=np.array([]), max_iter=500, tol=1e-10):

        if tau==None: tau = self.tau

        if method not in self.methods:
            raise ValueError("method must be either Catoni or Huber")

        if not mu0.any():
            mu0 = np.mean(self.X, axis=0)

        res = mu0 - self.X
        weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method)
        grad0 = res.T.dot(weight)/self.n
        diff_mu = -grad0
        mu1 = mu0 + diff_mu
        res = self.X - mu1
        weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method)

        r0, count = 1, 0
        while r0 > tol and count <= max_iter:
            grad1 = res.T.dot(weight)/self.n
            diff_grad = grad1 - grad0
            r0, r1 = diff_mu.dot(diff_mu), diff_grad.dot(diff_grad)
            r01 = diff_grad.dot(diff_mu)
            lr1, lr2 = r01/r1, r0/r01
            grad0, mu0 = grad1, mu1
            diff_mu = -min(lr1, lr2, 10)*grad1
            mu1 += diff_mu
            res = mu1 - self.X
            weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method)
            count += 1

        return mu1, [res, weight, tau, count]

    def Agnostic_mean(self):
        if self.d == 1:
            return self.X.mean(0)

        S = np.cov(self.X.T)
        W,V = LA.eigh(S)

        PV1 = np.matmul(V[:, :int(self.d/2)], V[:, :int(self.d/2)].T)
        ProjX = np.matmul(self.X, PV1)
        est1 = ProjX.mean(0)

        V2 = V[:, int(self.d/2):]
        X2 = np.matmul(self.X, V2)
        est2 = m_est(X2).Agnostic_mean()
        est2 = np.matmul(est2, V2.T)

        return est1 + est2

    def Minsker_mean(self, mu0=np.array([]), max_iter=500, tol=1e-10):
        if not mu0.any():
            mu0 = np.mean(self.X, axis = 0)

        alpha, p = 7/18, 0.1
        delta = 0.05
        psi = (1-alpha)*np.log((1-alpha)/(1-p)) + alpha*np.log(alpha/p)
        k = int(np.log(1/delta)/psi) + 1
        Z = np.zeros((k, self.d))
        m = int(self.n/k)
        for i in range(k):
            if i == k-1:
                Z[i, :] = self.X[m*i:, :].mean(0)
            else :
                Z[i, :] = self.X[m*i:m*(i+1), :].mean(0)

        mu1 = mu0
        r0, count = 1,0
        while r0 > tol and count <= max_iter:  #use Weiszfeld's algorithm
            weight = np.zeros(k)
            for i in range(k):
                weight[i] = (Z[i, :] - mu1).dot(Z[i, :] - mu1)**(-0.5)

            weight = weight/np.sum(weight, axis = 0)
            mu2 = np.matmul(Z.T, weight)
            r0 = (mu1 - mu2).dot(mu1 - mu2)**0.5
            mu1 = mu2
            count += 1

        return mu1, count


    def adaptive_huber(self, mu0= np.array([]), q = 4, max_iter=500, tol=1e-10):

        if len(mu0.tolist()) == 0 :
            mu0 = np.mean(self.X, axis=0)

        mu_seq = np.zeros([self.d, int(max_iter)+1])
        mu_seq[:,0] = mu0

        tau0, res = 0.5*(self.n/np.log(self.n))**(3/(2*q)), mu0 - self.X
        weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau0, method="Huber")
        grad0 = res.T.dot(weight)/self.n
        diff_mu = -grad0
        mu1 = mu0 + diff_mu
        res = self.X - mu1
        weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau0, method="Huber")
        tau = tau0

        r0, count = 1, 0
        while r0 > tol and count <= max_iter:
            grad1 = res.T.dot(weight)/self.n
            diff_grad = grad1 - grad0
            r0, r1 = diff_mu.dot(diff_mu), diff_grad.dot(diff_grad)
            r01 = diff_grad.dot(diff_mu)
            lr1, lr2 = r01/r1, r0/r01
            grad0, mu0 = grad1, mu1
            diff_mu = -min(lr1, lr2, 10)*grad1
            mu1 += diff_mu
            mu_seq[:, count+1] = mu1
            res = mu1 - self.X
            weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")
            tau = 0.5*tau0 * (self.m_q(res, weight, q)**(1/q))

            count += 1

        return mu1, [res, weight, tau, count], mu_seq[:, :count + 1]

    def adaptive_huber2(self, mu0= np.array([]), q = 4, max_iter=500, tol=1e-10, gamma = None):

        if len(mu0.tolist()) == 0 :
            mu0 = np.mean(self.X, axis=0)

        mu_seq = np.zeros([self.d, int(max_iter)+1])
        mu_seq[:,0] = mu0

        if gamma == None:
            gamma = 1/2
        tau0 = (self.n/np.log(self.n))**(gamma)
        res = mu0 - self.X
        tau = 0.2*tau0 * (self.m_q(res, q)**(1/q))
        weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")
        mu1 = mu0
        r0, count = 1, 0

        while r0 > tol and count <= max_iter:
            grad1 = res.T.dot(weight)/self.n
            diff_mu = -grad1
            r0 = diff_mu.dot(diff_mu)
            mu1 += diff_mu
            mu_seq[:, count+1] = mu1
            res = mu1 - self.X
            tau = 0.2*tau0 * (self.m_q(res, q)**(1/q))
            weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")
            count += 1

        return mu1, [res, weight, tau, count], mu_seq[:, :count + 1]

    def adaptive_hubercov(self, mu0=np.array([])):

        if not mu0.any():
            mu0, _, _ = self.adaptive_huber2()

        res = mu0 - self.X
        sigma = self.m_q(res, 4)**(1/2)
        xi = sigma*np.sqrt(self.n/np.log(self.d*self.n))
        weight = self.robust_weight(np.sum(res**2, axis=1)/xi, method="Huber")
        Sigma = (res*weight[:, None]).T.dot(res)/self.n

        return Sigma, xi

    def noisy_hubercov(self, epsilon = 0.5, mu0 = np.array([]), threshold = 0.2) :

        if not mu0.any() :
            mu0, _, _ = self.noisy_adaptive_huber(epsilon = epsilon) 

        res = mu0 - self.X
        sigma = self.stability(epsilon = epsilon/(2))
        xi = 10*(sigma)*np.sqrt(self.n/np.log(self.d*self.n))
        weight = self.robust_weight(np.sum(res**2, axis=1)/xi, method="Huber")
        Sigma = (res*weight[:, None]).T.dot(res)/self.n
        M = rgt.normal(size = (self.d,self.d))
        E = np.triu(M, k = 0) + np.triu(M, k = 1).T
        Sigma += 2*xi/(epsilon*self.n)*E

        eigenvalues, eigenvectors = LA.eigh(Sigma)
        truncated_eigenvalues = np.diag(np.maximum(eigenvalues, threshold))

        Sigma = np.dot(np.dot(eigenvectors, truncated_eigenvalues), eigenvectors.T)

        return Sigma , xi

    def noisy_adaptive_huber(self, mu0=np.array([]), epsilon = 0.5, T = None, eta = 1, m = None):

        if not mu0.any():
            mu0 = np.zeros(self.d)

        if T == None:
            T = int((eta**(-2))*(np.log(self.n)))

        mu_seq = np.zeros([self.d, T+1])
        mu_seq[:,0] = mu0

        if m == None:
            m = self.stability(epsilon = epsilon/(T+1))
            mu1 = mu0
            res = mu1 - self.X
            tau = 0.5*(m**1/2)*(epsilon*self.n/(np.sqrt(self.n*(self.d + np.log(self.n)))))**(1/2)
            weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")

            count = 0

            while count < T:
                grad1 = res.T.dot(weight)/self.n
                noise = rgt.multivariate_normal(np.zeros(self.d), np.identity(self.d))
                diff_mu = -grad1 + 2*(eta)*((T + 1)**0.5)*tau/(epsilon*self.n)*noise
                mu1 += diff_mu
                mu_seq[:, count+1] = mu1
                res = mu1 - self.X
                weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")
                count += 1

        else :
            mu1 = mu0
            res = mu1 - self.X
            tau = 0.5*(m**1/2)*(epsilon*self.n/(np.sqrt(self.n*(self.d + np.log(self.n)))))**(1/2)
            weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")

            count = 0

            while count < T:
                grad1 = res.T.dot(weight)/self.n
                noise = rgt.multivariate_normal(np.zeros(self.d), np.identity(self.d))
                diff_mu = -grad1 + 2*(eta)*((T)**0.5)*tau/(epsilon*self.n)*noise
                mu1 += diff_mu
                mu_seq[:, count+1] = mu1
                res = mu1 - self.X
                weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method="Huber")
                count += 1

        return mu1, [res, weight, tau, count], mu_seq

    def noisy_trun_mean(self, mu0=np.array([]), epsilon = 0.5,R = None, delta = None):

        if R == None:
            R = np.amax(np.absolute(self.X))

        if delta == None :
            delta = 1/2

        Y = np.where(self.X > R, R, self.X)
        Y = np.where(Y < -R, -R, Y)

        if not mu0.any():
            mu0 = np.mean(Y, axis = 0)

        noise = rgt.multivariate_normal(np.zeros(self.d), np.identity(self.d))
        mu = mu0 + noise*2*R*((self.d*np.log(1/delta))**0.5)/(self.n*epsilon)
        return mu


    def gd_mean(self, tau=None, method="Huber", mu0=np.array([]), lr=1, max_iter=500, tol=1e-10):

        if tau==None: tau = self.tau

        if method not in self.methods:
            raise ValueError("method must be either Catoni or Huber")

        if not mu0.any():
            mu0 = np.mean(self.X, axis=0)

        mu_seq = np.zeros([self.d, max_iter+1])
        mu_seq[:,0] = mu0

        res = mu0 - self.X
        weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method)

        r0, count = 1, 0
        while r0 > tol and count <= max_iter:
            grad = res.T.dot(weight)/self.n
            mu0 -= lr*grad
            mu_seq[:, count+1] = mu0
            r0, res = np.max(abs(grad)), mu0 - self.X
            weight = self.robust_weight(np.sum(res**2, axis=1)**0.5/tau, method)
            tau = self.tau * self.trun_cov(res, weight)[0]**0.5
            count += 1

        return mu0, [res, weight, tau, count], mu_seq[:,:count+1]


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

        
