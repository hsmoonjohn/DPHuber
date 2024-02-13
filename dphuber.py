"""
     M-estimator of Mean Vectors
"""

import numpy as np
from numpy import linalg as LA
import numpy.random as rgt
import math
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn import linear_model

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
    
    def huber_loss(self, residuals, tau):
            return np.where(np.abs(residuals) <= tau, 0.5 * residuals**2, tau * (np.abs(residuals) - 0.5 * tau))
    
    def soft_thresh(self, x, c):
        tmp = abs(x) - c
        return np.sign(x) * np.where(tmp <= 0, 0, tmp)

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

    def ordhuber(self, beta0, method='BFGS'):
        X = self.X
        
        def huber_loss(params, X, y, tau):
            predictions = np.dot(X, params)
            residuals = y - predictions
            loss = np.where(np.abs(residuals) <= tau,0.5 * residuals**2, tau * (np.abs(residuals) - 0.5 * tau))
            return np.sum(loss) / len(y)
        
        initial_params = beta0
        tau = 1.345*np.median(np.abs(self.Y-np.median(self.Y)))

        result = minimize(huber_loss, initial_params, args=(X,self.Y, tau), method=method)

        return result.x

    def huber_lasso(self, lambda_=None, tau=None, tol=1e-6):
        X=self.X
        #X = np.concatenate([np.ones((self.n,1)), X], axis=1)

        if tau == None:  
            tau = 1.345*np.median(np.abs(self.Y-np.median(self.Y)))

        if lambda_ == None:
            lambda_ = np.sqrt(np.log(self.d)/self.n)*tau


        def huber_loss(residuals, tau):
            return np.where(np.abs(residuals) <= tau, 0.5 * residuals**2, tau * (np.abs(residuals) - 0.5 * tau))
        
        def huber_lasso_objective(beta, X, y, tau, lambda_):

            residuals = y - np.dot(X, beta) 
            huber_loss_value = huber_loss(residuals, tau).sum()
            lasso_penalty = lambda_ * np.sum(np.abs(beta))  # Excluding the intercept from the L1 penalty
            
            return huber_loss_value + lasso_penalty
        
        n_features = X.shape[1]
        initial_beta = np.zeros(n_features)  # Including intercept
        result = minimize(huber_lasso_objective, initial_beta, args=(X, self.Y, tau, lambda_), method='L-BFGS-B', tol=tol, options={'disp': False})

        return result.x
    
    def l1huber(self, Lambda=np.array([]), tau=None, beta0=None, res=np.array([]), standardize=True,
                adjust=True, phi=0.1, tol=1e-5, max_iter=1e3, gamma=1.25):
        
        if standardize:
            X = self.X1
        else:
            X = self.X

        if not np.array(Lambda).any():
            lambda_max = np.max(np.abs(self.Y.dot(X)))/self.n
            lambda_min = 0.01 * lambda_max
            Lambda =  math.exp(0.7 * math.log(lambda_max) + 0.3 * math.log(lambda_min))

        if tau == None:
            clf = linear_model.Lasso(alpha=Lambda)
            clf.fit(X, self.Y)
            res0 = self.Y - clf.predict(X)  
            tau = 1.345*np.median(np.abs(res0-np.median(res0)))

        r0 = 1
        count = 0
        if beta0 is None:
            beta0 = clf.coef_

        res = self.Y-X.dot(beta0)
        phi0 = phi

        while (r0>tol) and count < max_iter:

            grad0 = X.T.dot(self.huber_loss_score_function(-res, tau=tau))/self.n
            loss_eval0 = np.mean(self.huber_loss(residuals=res, tau=tau))
            beta1 = beta0 -grad0/phi
            beta1[self.itcp:]= self.soft_thresh(beta1[self.itcp:], Lambda/phi)
            diff_beta = beta1 - beta0
            #r0 = diff_beta.dot(diff_beta)
            r0 = max(np.abs(diff_beta))
            res = self.Y-X.dot(beta1)
            loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5 * phi * r0
            loss_eval1 = np.mean(self.huber_loss(residuals=res, tau=tau))

            while loss_proxy < loss_eval1:
                phi *=  gamma
                beta1 = beta0 -grad0/phi
                beta1[self.itcp:]= self.soft_thresh(beta1[self.itcp:], Lambda/phi)
                diff_beta = beta1 - beta0
                #r0 = diff_beta.dot(diff_beta)
                r0 = max(np.abs(diff_beta))
                res = self.Y-X.dot(beta1)
                loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5 * phi * r0
                loss_eval1 = np.mean(self.huber_loss(residuals=res, tau=tau))

            beta0 = beta1
            phi = max(phi0, phi/gamma)
            count += 1

        if standardize and adjust:
            beta1[self.itcp:] = beta1[self.itcp:] / self.sdX
            if self.itcp: beta1[0] -= self.mX.dot(beta1[1:])

        return {'beta': beta1, 'res': res, 'niter': count, 'lambda': Lambda, 'tau':tau}
    
   
        


