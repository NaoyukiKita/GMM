import numpy as np
from numpy import abs, exp, log, mean, outer, sum
from numpy.linalg import det, inv
from numpy.random import default_rng

from util.stats import norm_pdf
from util.special import digamma

class EMGMM():
    def __init__(self, n_components=1, tol=1e-3, max_iter=100, rng=default_rng()):
        self.K = n_components
        self.tol = 1e-3
        self.max_iter = max_iter
        self.rng = rng

        return
    
    def fit(self, X: np.array):
        N, D = X.shape
        pi, mu, Sigma = self.init_params(D, N, self.K)

        self.losses = []
        for iter in range(self.max_iter):
            r = self.e_step(D, N, self.K, X, pi, mu, Sigma)
            pi, mu, Sigma = self.m_step(D, N, self.K, X, mu, r)

            loss = self.log_likelihood(D, N, self.K, X, pi, mu, Sigma)
            self.losses.append(loss)

            if iter >= 2 and abs(self.losses[-1] - self.losses[-2]) < self.tol:
                break
        
        if iter != self.max_iter:
            print(f"ConvergenceWarning: EMGMM failed to converge.")
        
        self.pi = pi
        self.mu = mu
        self.Sigma = Sigma

        return self
    
    def init_params(self, D, N, K):
        pi = self.rng.random(K) # ~ Uni(0, 1)
        pi /= sum(pi)
        mu = self.rng.random([K, D]) # ~ Uni(0, 1)

        Sigma = np.zeros([K, D, D])
        for k in range(K):
            Sigma[k] = np.identity(D)
        
        return pi, mu, Sigma
    
    def e_step(self, D, N, K, X, pi, mu, Sigma):
        r = np.zeros([N, K])
        for n in range(N):
            for k in range(K):
                r[n][k] = pi[k] * norm_pdf(D, X[n], mu[k], Sigma[k])
            r[n] /= sum(r[n])

        return r

    def m_step(self, D, N, K, X, mu, r):
        N_ = sum(r, axis=0)

        pi = N_ / N
        new_mu = sum(np.einsum("nk,nd,k->nkd", r, X, 1.0/N_), axis=0)

        kernels = np.zeros([N, K, D, D])
        for n in range(N):
            for k in range(K):
                diff = X[n]-mu[k]
                kernels[n][k] = outer(diff, diff)
        Sigma = sum(np.einsum("nk,nkij,k->nkij", r, kernels, 1.0/N_), axis=0)

        return pi, new_mu, Sigma

    def log_likelihood(self, D, N, K, X, pi, mu, Sigma):
        arr = np.zeros([N, K])
        for n in range(N):
            for k in range(K):
                arr[n][k] = pi[k] * norm_pdf(D, X[n], mu[k], Sigma[k])
        
        return mean(log(sum(arr, 1)))

class VBGMM():
    def __init__(self, n_components=1, tol=1e-3, max_iter=100, rng=default_rng()):
        self.K = n_components
        self.tol = 1e-3
        self.max_iter = max_iter
        self.rng = rng

        return

    def fit(self, X: np.array):
        N, D = X.shape
        alpha0, beta0, nu0, m0, W0 = self.seed_params(D, N)
        alpha, beta, nu, m, W = self.init_params(D, N, self.K, alpha0, beta0, nu0, m0, W0)

        self.losses = []
        for iter in range(self.max_iter):
            r = self.e_step(D, N, self.K, X, alpha, beta, nu, m, W, iter)

            alpha, beta, nu, m, W = self.m_step(D, N, self.K, X, alpha0, beta0, nu0, m0, W0, r)

            pi, mu, Sigma = self.observe(alpha, nu, m, W)

            loss = self.log_likelihood(D, N, self.K, X, pi, mu, Sigma)
            self.losses.append(loss)

            if iter >= 2 and abs(self.losses[-1] - self.losses[-2]) < self.tol:
                break
        
        if iter != self.max_iter:
            print(f"ConvergenceWarning: VBGMM failed to converge.")
        
        self.pi, self.mu, self.Sigma = self.observe(alpha, nu, m, W)

        return self

    def seed_params(self, D, N):
        alpha0 = 0.01
        beta0 = 0.01
        nu0 = float(D)
        m0 = np.zeros([D])
        W0 = np.identity(D)

        return alpha0, beta0, nu0, m0, W0

    def init_params(self, D, N, K, alpha0, beta0, nu0, m0, W0):
        alpha = np.full([K], alpha0)
        beta = np.full([K], beta0)
        nu = np.full([K], nu0)
        m = self.rng.random([K, D])
        W = self.rng.random([K, D, D])

        for k in range(K):
            # m[k] = m0
            W[k] = W0

        return alpha, beta, nu, m, W

    def e_step(self, D, N, K, X, alpha, beta, nu, m, W, iter):
        log_tilde_pi = digamma(alpha) - digamma(alpha.sum())

        log_tilde_Sigma = np.zeros([K])
        for k in range(K):
            log_tilde_Sigma[k] = D*log(2.0) + log(abs(det(W[k])))
            for i in range(D):
                log_tilde_Sigma[k] += digamma((nu[k] + 1 - i) / 2)
        
        log_rho = np.zeros([N, K])
        for n in range(N):
            for k in range(K):
                diff = X[n] - m[k]
                log_rho[n][k] += log_tilde_pi[k] + 0.5*log_tilde_Sigma[k]
                log_rho[n][k] -= 0.5*(nu[k] * diff.T @ W[k] @ diff + D / beta[k])
        
        rho = exp(log_rho)
        r = np.zeros([N, K])
        for n in range(N):
            r[n] = rho[n] / sum(rho[n])

        return r

    def m_step(self, D, N, K, X, alpha0, beta0, nu0, m0, W0, r):
        N_ = sum(r, axis=0)
        bar_x = sum(np.einsum("nk,nd,k->nkd", r, X, 1.0/N_), axis=0)

        s = np.zeros([N, K, D, D])
        for n in range(N):
            for k in range(K):
                diff = X[n] - bar_x[k]
                s[n][k] = (r[n][k] / N_[k]) * outer(diff, diff)
        S = sum(s, axis=0)

        alpha = np.full([K], alpha0) + N_
        beta = np.full([K], beta0) + N_
        nu = np.full([K], nu0) + N_

        m = np.zeros([K, D])
        W = np.zeros([K, D, D])
        for k in range(K):
            diff = bar_x[k] - m0
            m[k] = (beta0 * m0 + N_[k] * bar_x[k]) / beta[k]
            W[k] = inv(inv(W0) + N_[k]*S[k] + beta0*N_[k]/(beta0+N_[k]) * outer(diff, diff))

        return alpha, beta, nu, m, W

    def log_likelihood(self, D, N, K, X, pi, mu, Sigma):
        arr = np.zeros([N, K])
        for n in range(N):
            for k in range(K):
                arr[n][k] = pi[k] * norm_pdf(D, X[n], mu[k], Sigma[k])
        
        return mean(log(sum(arr, 1)))

    def observe(self, alpha, nu, m, W):
        pi = alpha / sum(alpha)
        mu = m
        Sigma = inv(np.einsum("k,kij->kij", nu, W))

        return pi, mu, Sigma
