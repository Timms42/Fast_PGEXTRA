"""
Decentralised and non-decentralised first-order algorithms, and error metrics
"""
import numpy as np
import time

def pgextra(x0, JA, B, pars, data):
    """
    :param x0: initial iterate
    :param JA: function handle for computing resolvent of tau*A. Called as JA(z, tau, data)
    :param B: function handle for computing forward operator. Called as B(z, data).
    :param pars: dictionary {W: mixing matrix, tau: step size, maxit: max. iterations}.
    :param met: error metric from ['res', 'rel', 'con']
    :param tol: stopping criterion, metric <= tol
    :param data: optional dictionary of problem data, if necessary to compute JA() and B().
    :return: x2, res, rel, dur
    """
    # Unpack parameters and construct matrices
    tau = pars['tau']
    W = pars['W']
    maxit = pars['maxit']
    n = W.shape[0]
    tW = 0.5 * (np.eye(n) + W)  # (I+W)/2
    K2 = 0.5 * (np.eye(n) - W)

    res = []
    rel = []
    cons = []

    # Initialise variables
    y0 = B(x0, data)
    u1 = x0 - tau*y0
    x1 = JA(u1, tau, data)
    q1 = x0 + x1 # q^k = \sum_{t=0}^k x^k # Note: residue() and relerror() apply K^2 to q^k, don't need to compute $\sum_{t=0}^k K x^k$ here

    # Main loop
    for k in range(1, int(maxit)+1):
        y1 = B(x1, data)
        u2 = u1 + W@x1 - tW@x0 - tau*(y1 - y0)
        x2 = JA(u2, tau, data)

        res.append(residue_pg(x2, x1, K2, tW))
        rel.append(relerror_pg(x2, u1, q1, B(x2, data), tau, tW, K2))
        cons.append(consensus_pg(x2, K2))

        q1 += x2
        y0 = y1.copy()
        u1 = u2.copy()
        x0 = x1.copy()
        x1 = x2.copy()

        print(f'PG-EXTRA: iteration {k:04} of {maxit}', end='\r')
    return x2, res, rel, cons

def nids(x0, JA, B, pars, data):
    """
    :param x0: initial iterate
    :param JA: function handle for computing resolvent of tau*A. Called as JA(z, tau, data)
    :param B: function handle for computing forward operator. Called as B(z, data).
    :param pars: dictionary {W: mixing matrix, tau: step size, beta: communication parameter, maxit: max. iterations}.
    :param data: dictionary of problem data, if necessary to compute JA() and B().
    :return: x2, res, rel, dur
    """
    # Unpack parameters and construct matrices
    tau, beta = pars['tau'], pars['beta']
    W = pars['W']
    maxit = pars['maxit']
    n = W.shape[0]
    tW = 0.5*(np.eye(n) + W)
    tildeW =  np.eye(n) - tau*0.5*beta*(np.eye(n) - W)  # when tau_1=...=tau_n and \beta = 1/max(tau), tildeW=tW
    K2 = 0.5 * (np.eye(n) - W)

    res = []
    rel = []
    cons = []

    # Initialise variables
    yzero = B(x0, data) # preserve original gradient
    y0 = B(x0, data)
    u1 = x0 - tau*y0
    x1 = JA(u1, tau, data)
    q1 = x0 + x1

    # Main loop
    for k in range(1, int(maxit)+1):
        y1 = B(x1, data)
        u2 = u1 - x1 + tildeW@(2*x1 - x0 - tau*(y1 - y0))
        x2 = JA(u2, tau, data)

        y2 = B(x2, data)        
        res.append(residue_nids(x2, x1, y2, y1, tau, tW, K2))
        rel.append(relerror_nids(x2, u2, q1-tau*(y1-yzero), y2, tau, beta, tW, K2))
        cons.append(consensus_nids(x2, K2))

        q1 += x2
        y0 = y1.copy()
        u1 = u2.copy()
        x0 = x1.copy()
        x1 = x2.copy()

        print(f'NIDS: iteration {k:04} of {maxit}', end='\r')
    return x2, res, rel, cons
 

def gf_pgextra(x0, JA, B, pars, data):
    r"""
    An accelerated PG-EXTRA algorithm derived from Generalised Fast Condat-Vu algorithm.
    :param z0: array, initial iterate
    :param JA: function handle for computing resolvent of tau*A. Called as JA(z, tau, data)
    :param B: function handle for computing forward operator. Called as B(z, data).
    :param pars: dictionary containing
        :param tau: step size
        :param alpha: function handle for acceleration parameter from Generalised Fast KM
        :param sigma: function handle, acceleration parameter
        :param eta: (float) averaged constant for parameter theta = (1-\eta) + \eta(alpha-1), \eta \in (0,1)
        :param W: mixing matrix
        :param maxit: max. iterations
    :param data: dictionary of problem data for computing JA() and B().
    :return: x2, res, rel, dur
    """
    # Unpack parameters and construct matrices
    tau, alpha, s, sigma, eta = pars['tau'], pars['alpha'], pars['s'], pars['sigma'], pars['eta']
    W = pars['W']
    maxit = pars['maxit']  
    n = W.shape[0]
    K2 = 0.5 * (np.eye(n) - W)
    tW = 0.5 * (np.eye(n) + W)

    res = []
    rel = []
    cons = []

    # Initialise variables
    x1 = x0.copy()
    v1 = np.zeros_like(x0)
    y0 = B(x0, data)
    u0 = x0 - tau*y0
    z0 = JA(u0, tau, data)
    q1 = np.zeros_like(x0)  # \sum_{t=1}^k v^t
    
    # Main loop
    for k in np.arange(1, maxit+1, dtype=float):
        alpha1 = alpha(k)
        sigma1 = sigma(k)
        theta1 = (1.-eta) + eta*(alpha1-1.)

        y1 = B(x1, data)
        u1 = u0 + x1 - x0 - v1 - tau*(y1 - y0)
        z1 = JA(u1, tau, data)
        x2 = (1.-s*theta1/(k+sigma1))*x1 + (1.-s)*(k+sigma1-alpha1)/(k+sigma1)*(x1-x0) + s*theta1/(k+sigma1)*z1 + s*(k+sigma1-alpha1)/(k+sigma1)*(z1-z0)
        v2 = (k+sigma1-alpha1)/(k+sigma1)*v1 - K2@(
            s*theta1/(k+sigma1)*x1 + s*(k+sigma1-alpha1)/(k+sigma1)*(x1-x0) - 2.*s*theta1/(k+sigma1)*z1 - 2*s*(k+sigma1-alpha1)/(k+sigma1)*(z1-z0)
        )

        res.append(residue_fpg(z1, x1, tW, K2))
        rel.append(relerror_fpg(z1, B(z1, data), u1, q1, tau, tW))
        cons.append(consensus_fpg(z1, K2))

        q1 += v2
        y0 = y1.copy()
        v1 = v2.copy()
        u0 = u1.copy()
        z0 = z1.copy()
        x0 = x1.copy()
        x1 = x2.copy()

        print(f'Gen. Fast PG-EXTRA: iteration {int(k):04} of {maxit}', end='\r')
    return z1, res, rel, cons


# Error metrics for PG-EXTRA, Fast PG-EXTRA, and NIDS
def residue_pg(x2, x1, K2, tW):
    r""" Distance between iterates, $||x^{k+1} - x^k}||^2_\tilde{W} + ||q^{k+1} - q^k}||^2$ where \tilde{W} = (I+W)/2
    :param x2: new output of prox()
    :param x2: previous output of prox()
    :param: K2: K**2 = (I-W)/2
    :param: tW: = (I+W)/2
    """
    return np.vdot(x2, K2@x2) + np.vdot(x2 - x1, tW@(x2-x1))

def relerror_pg(x2, u1, q1, y2, tau, tW, K2):
    """
    :param: x2: x^{k+1}
    :param: u1: u^{k}
    :param: q1: sum_{t=0}^k x^t
    :param: y2: B(x2)
    :param: tW: (I+W)/2
    :param: K2: K**2 = (I-W)/2
    """
    
    arg = tau*y2 + u1 - x2 + K2@q1
    return  np.vdot(arg, tW@arg)

def consensus_pg(x2, K2):
    """
    :param: K2: K**2 = (I-W)/2
    """
    return np.vdot(x2, K2@x2)

def residue_fpg(z1, x1, tW, K2):
    r""" Distance between iterates, $||\bar{x}^{k} - x^k||^2_\tilde{W} + ||\bar{x}^k||^2_{K^2}$ 
    np.vdot is the Frobenius inner product, which is the usual inner product on vectorised matrices
    :param: tW: = (I+W)/2
    :param: K2: = (I-W)/2 = I - tW
    """
    return np.vdot(z1 - x1, tW@(z1-x1)) + np.vdot(z1, K2@z1)

def relerror_fpg(z1, gradf_z1, u1, q1, tau, tW):
    r""" Optimality error, $||\tau \nabla f(\bar{x}^k) + \tau r^k + \tau Ky^k||^2_{(I+W)/2}$ where $r^k = (u^k-\bar{x}^k)/\tau$
    np.vdot is the Frobenius inner product, which is the usual inner product on vectorised matrices
    :param: gradf_z1: \nabla f(z1)
    :param: q1 = sum_{t=0}^k v^t
    :param: tW: = (I+W)/2
    """
    arg = tau*gradf_z1 + u1 - z1 + q1
    return np.vdot(arg,tW@arg)

def consensus_fpg(z1, K2):
    """
    :param: K2: K**2 = (I-W)/2
    """
    return np.vdot(z1, K2@z1)

def residue_nids(x2, x1, y2, y1, tau, tW, K2):
    r"""Same residue as PG-EXTRA
    Distance between iterates, $||x^{k+1} - x^k}||^2_\tilde{W} + ||q^{k+1} - q^k||^2$ where \tilde{W} = (I+W)/2
    :param: tW: = ((I+W)/2)^2
    :param: K2: = (I-W)/2
    """
    return np.vdot(x2-tau*(y2-y1), K2@x2) + np.vdot(x2 - x1, tW@(x2-x1))

def relerror_nids(x2, u2, q1, y2, tau, beta, tW, K2):
    r""" Same as PG-EXTRA
    :param: x2: x^{k+1}
    :param: u2: u^{k+1}
    :param: q1: sum_{t=0}^k x^t
    :param: y2: B(x2)
    :param: tW: (I+W)/2
    :param: K2: K**2 = (I-W)/2
    """
    
    arg = tau*y2 + u2 - x2 + tau*beta*K2@q1
    return  np.vdot(arg, tW@arg)

def consensus_nids(x2, K2):
    """ Same as PG-EXTRA
    :param: K2: K**2 = (I-W)/2
    """
    return np.vdot(x2, K2@x2)

# Condat-Vu and Fast Condat-Vu algorithms from Section 2
def condat_vu(x0, y0, prox_g, prox_h, grad_f, K, tau, sigma, rho, data, relerror, maxit):
    """The Condat-Vu splitting algorithm (not distributed)"""
    rel = []
    for k in range(maxit):
        xbar1 = prox_g(x0 - tau*K.T@y0 - tau*grad_f(x0, data), tau, data)
        arg = y0 + sigma*K@(2*xbar1 - x0)
        ybar1 = arg - sigma* prox_h(arg/sigma , 1./sigma, data)  # using Moreau's identity prox_{σ h*}(y) = y - σ prox{(1\σ)h}((1\σ)y)

        x1 = rho*xbar1 + (1-rho)*x0
        y1 = rho*ybar1 + (1-rho)*y0

        rel.append(relerror(x0, x1, y0, y1, K, tau, sigma))

        x0 = x1.copy()
        y0 = y1.copy()

        print(f'Condat-V\u0169: iteration {k:04} of {maxit}', end='\r')
    return x1, y1, rel

def fast_condat_vu(x0, y0, prox_g, prox_h, grad_f, K, tau, sigma, alpha, s, data, relerror, maxit):
    """The Fast Condat-Vu splitting algorithm (not distributed)"""
    rel = []
    # Initialisation
    x1, y1 = x0.copy(), y0.copy()
    xbar0 = prox_g(x0 - tau*K.T@y0 - tau*grad_f(x0, data), tau, data)
    arg = y0 + sigma*K@(2*xbar0 - x0)
    ybar0 = arg - sigma* prox_h(arg/sigma , 1./sigma, data)   # prox_{σ h*} = I - σ prox{(1\σ)h}

    for k in range(maxit):
        xbar1 = prox_g(x1 - tau*K.T@y1 - tau*grad_f(x1, data), tau, data)
        arg = y1 + sigma*K@(2*xbar1 - x1)
        ybar1 = arg - sigma* prox_h(arg/sigma , 1./sigma, data)

        x2 = (1-0.5*s*alpha/(k+alpha))*x1 + (1-s)*k/(k+alpha)*(x1-x0) + 0.5*s*alpha/(k+alpha)*xbar1 + s*k/(k+alpha)*(xbar1 - xbar0)
        y2 = (1-0.5*s*alpha/(k+alpha))*y1 + (1-s)*k/(k+alpha)*(y1-y0) + 0.5*s*alpha/(k+alpha)*ybar1 + s*k/(k+alpha)*(ybar1 - ybar0)

        rel.append(relerror(x1, xbar1, y1, ybar1, K, tau, sigma))

        x0 = x1.copy()
        x1 = x2.copy()
        y0 = y1.copy()
        y1 = y2.copy()
        xbar0 = xbar1.copy()
        ybar0 = ybar1.copy()

        print(f'Fast Condat-V\u0169: iteration {k:04} of {maxit}', end='\r')
    return x1, y1, rel


# Other algorithms and alternate formulations
# def nids_alt(x0, JA, B, W, tau: float, beta: float, residue, relerror, consensus, max_iters=1000, data=None):
#     """
#     Alternate expression of NIDS, from Li, Shi, Yan (2019) equation (12)
#     :param x0: initial iterate
#     :param JA: function handle for computing resolvent of tau*A. Called as JA(z, tau, data)
#     :param B: function handle for computing forward operator. Called as B(z, data).
#     :param tau: step size.
#     :param tau: parameter for communication
#     :param data: optional dictionary of problem data, if necessary to compute JA() and B().
#     :return: x2, res, rel, dur
#     """
#     # Initialise error
#     res = []
#     rel = []
#     cons = []

#     n = W.shape[0]
#     # tildeW = np.eye(n) - tau*0.5*beta*(np.eye(n) - W)  # I − \tau(β/2)(I − W)
#     tW = 0.5*beta*(np.eye(n) + W)   # beta*(I+W)/2)
#     K2 = 0.5 * (np.eye(n) - W)    # (W-I)/2

#     # Initialise variables
#     y0 = B(x0, data)
#     u1 = x0 - tau*y0
#     x1 = JA(u1, tau, data)
#     d1 = np.zeros_like(x1)

#     # Main loop
#     for k in range(1, int(max_iters)+1):
#         y1 = B(x1, data)
#         d2 = d1 + beta*tW@(2*x1 - u1 - tau*y1 - tau*d1)
#         u2 = x1 - tau*y1 - tau*d2
#         x2 = JA(u2, tau, data)

#         res.append(residue(u2, u1, x1, tau, beta, tW, K2))
#         rel.append(relerror(d2, u1, x1, y1, tW))
#         cons.append(consensus(x1, K2))

#         y0 = y1.copy()
#         u1 = u2.copy()
#         x0 = x1.copy()
#         x1 = x2.copy()

#         print(f'NIDS: iteration {k:04} of {max_iters}', end='\r')
#     return x2, res, rel, cons


# def fast_pgextra(x0, JA, B, W, tau, alpha, s, residue, relerror, consensus, max_iters=1000, data=None):
#     """
#     An accelerated PG-EXTRA algorithm derived from Fast Condat-Vu algorithm.
#     :param z0: array, initial iterate
#     :param JA: function handle for computing resolvent of tau*A. Called as JA(z, tau, data)
#     :param B: function handle for computing forward operator. Called as B(z, data).
#     :param tau: step size,
#     :param alpha: acceleration parameter from Fast KM
#     :param s: 
#     :param data: optional dictionary of problem data for computing JA() and B().
#     :return: x2, res, rel, dur
#     """
#     # Initialise error
#     res = []
#     rel = []
#     cons = []

#     n = W.shape[0]
#     K2 = 0.5 * (np.eye(n) - W)    # -(W-I)/2
#     tW = 0.5 * (np.eye(n) + W)

#     # Initialise variables
#     x1 = x0.copy()
#     v1 = np.zeros_like(x0)
#     y0 = B(x0, data)
#     u0 = x0 - tau*y0
#     z0 = JA(u0, tau, data)
#     q1 = np.zeros_like(x0)  # \sum_{t=1}^k v^t
    
#     # Main loop
#     for k in np.arange(1, max_iters+1, dtype=float):
#         y1 = B(x1, data)
#         u1 = u0 + x1 - x0 - v1 - tau*(y1 - y0)
#         z1 = JA(u1, tau, data)
#         x2 = (1.-0.5*s*alpha/(k+alpha))*x1 + (1.-s)*k/(k+alpha)*(x1-x0) + 0.5*s*alpha/(k+alpha)*z1 + s*k/(k+alpha)*(z1-z0)
#         v2 = k/(k+alpha)*v1 - K2 @ (0.5*s*alpha/(k+alpha)*x1 + s*k/(k+alpha)*(x1-x0) - s*alpha/(k+alpha)*z1 - 2*s*k/(k+alpha)*(z1 - z0))

#         res.append(residue(z1, x1, tW, K2))
#         rel.append(relerror(z1, B(z1, data), u1, q1, tW))
#         cons.append(consensus(z1, K2))

#         q1 += v2
#         y0 = y1.copy()
#         v1 = v2.copy()
#         u0 = u1.copy()
#         z0 = z1.copy()
#         x0 = x1.copy()
#         x1 = x2.copy()

#         print(f'Fast PG-EXTRA: iteration {int(k):04} of {max_iters}', end='\r')
#     return z1, res, rel, cons


# def accel_pgextra(x0, JA, B, W, tau, alpha, gamma, residue, relerror, consensus, max_iters=1000, data=None):
#     """
#     Based on Accelerated Condat-Vu from Driggs, Ehrhardt, Sch¨onlieb, and Tang (2024)
#     :param z0: array, initial iterate
#     :param JA: function handle for computing resolvent of tau*A. Called as JA(z, tau, data)
#     :param B: function handle for computing forward operator. Called as B(z, data).
#     :param tau: step size,
#     :param alpha: function handle for computing parameter alpha^k
#     :param gamma: function handle for computing parameter gamma^k
#     :param data: optional dictionary of problem data, if necessary to compute JA() and B().
#     :return: x2, res, rel, dur
#     """
#     # Initialise error
#     res = []
#     rel = []
#     cons = []

#     n = W.shape[0]
#     K = scipy.linalg.sqrtm(0.5 * (np.eye(n) - W))   # ((I-W)/2)^{1/2}
#     K2 = 0.5 * (np.eye(n) - W)    # (I-W)/2
#     tW = 0.5 * (np.eye(n) + W)

#     # Initialise variables
#     alpha0 = alpha(0)
#     gamma0 = gamma(0)
#     x1 = x0.copy()
#     v0 = np.zeros_like(x0)
#     u1 = alpha0*x0 + (1-alpha0)*v0
#     y1 = B(u1, data)
#     s1 = x0 - tau*y1 - tau*gamma0*K2@x0
#     x1 = JA(s1, tau, data)
#     v1 = alpha0*x1 + (1-alpha0)*v0

#     w0 = gamma0*(2*x1 + x1 - x0)    # Used to compute optimality error, taking theta^0 = 1
    
#     # Main loop
#     for k in range(1, int(max_iters)+1):
#         alpha1 = alpha(k)
#         gamma1 = gamma(k)
#         theta1 = gamma0/gamma1

#         w1 = w0 + gamma1*(x1 + theta1*(x1 - x0))    # Used to compute optimality error
#         u2 = alpha1*x1 + (1-alpha1)*v1
#         y2 = B(u2, data)
#         t2 = tau*gamma1*K2@(x1 - theta1*(x1-x0))
#         s2 = s1 + x1 - x0 - tau*(y2 - y1) - t2
#         x2 = JA(s2, tau, data)
#         v2 = alpha1*x2 + (1-alpha1)*v1
        
#         # res.append(residue(v2, v1, t2, tW, K2inv, tau))   # # Used with relerror_fpg() function
#         # rel.append(relerror(v2, x1, tW))
#         res.append(residue(w1, v1, v2, K, tW, alpha(k+1)))  # Used with residue_apg() function
#         rel.append(relerror(w0, v0, y1, s2, x2, K, K2, tW, tau, alpha1))
#         cons.append(consensus(x2, K2))
        
#         alpha0 = alpha1
#         gamma0 = gamma1
#         u1 = u2.copy()
#         y1 = y2.copy()
#         s1 = s2.copy()
#         x0 = x1.copy()
#         x1 = x2.copy()
#         v1 = v2.copy()
#         w0 = w1.copy()

#         print(f'Accel. PG-EXTRA: iteration {k:04} of {max_iters}', end='\r')
#     return v2, res, rel, cons

# def accel_pgextra_vary(x0, JA, B, W, tau, alpha, gamma, residue, relerror, consensus, max_iters=1000, data=None):
#     """
#     Based on Accelerated Condat-Vu from Driggs, Ehrhardt,e Sch¨onlieb, and Tang (2024), incorporating adaptive step sizes
#     :param z0: array, initial iterate
#     :param JA: function handle for computing resolvent of tau*A. Called as JA(z, tau, data)
#     :param B: function handle for computing forward operator. Called as B(z, data).
#     :param tau: function handle for computing step size tau^k = tau(alpha^k, gamma^k),
#     :param alpha: function handle for computing parameter alpha^k
#     :param gamma: function handle for computing parameter gamma^k
#     :param data: optional dictionary of problem data, if necessary to compute JA() and B().
#     :return: x2, res, rel, dur
#     """
#     # Initialise error
#     res = []
#     rel = []
#     cons = []

#     n = W.shape[0]
#     K = scipy.linalg.sqrtm(0.5 * (np.eye(n) - W))   # ((I-W)/2)^{1/2}
#     K2 = 0.5 * (np.eye(n) - W)    # (I-W)/2
#     tW = 0.5 * (np.eye(n) + W)

#     # Initialise variables
#     alpha0 = alpha(0)
#     gamma0 = gamma(0)
#     tau0 = tau(alpha0, gamma0)
#     x1 = x0.copy()
#     v0 = np.zeros_like(x0)
#     u1 = alpha0*x0 + (1-alpha0)*v0
#     w0 = gamma0*(2*x1 + x1 - x0)    # Taking theta^0 = 1
#     y1 = B(u1, data)
#     s1 = x0 - tau0*y1 - tau0*K2@w0
#     x1 = JA(s1, tau0, data)
#     v1 = alpha0*x1 + (1-alpha0)*v0
    
#     # Main loop
#     for k in range(1, int(max_iters)+1):
#         alpha1 = alpha(k)
#         gamma1 = gamma(k)
#         tau1 = tau(alpha1, gamma1)
#         theta1 = gamma(k-1)/gamma(k)

#         w1 = w0 + gamma1*(x1 + theta1*(x1 - x0))
#         u2 = alpha1*x1 + (1-alpha1)*v1
#         y2 = B(u2, data)
#         t2 = K2@(tau1*gamma1*(x1 - theta1*(x1-x0)) + (tau1-tau0)*w1)
#         s2 = s1 + x1 - x0 - tau1*(y2 - y1) - t2
#         x2 = JA(s2, tau1, data)
#         v2 = alpha1*x2 + (1-alpha1)*v1
        
#         # res.append(residue(v2, v1, t2, tW, K2inv, tau1))  # Used with relerror_fpg() function
#         # rel.append(relerror(v2, x1, tW))
#         res.append(residue(w1, v1, v2, K, tW, alpha(k+1)))  # Used with residue_apg() function
#         rel.append(relerror(w0, v0, y1, s2, x2, K, K2, tW, tau1, alpha1))
#         cons.append(consensus(x2, K2))
        
#         tau0 = tau1
#         u1 = u2.copy()
#         y1 = y2.copy()
#         s1 = s2.copy()
#         x0 = x1.copy()
#         x1 = x2.copy()
#         v1 = v2.copy()
#         w0 = w1.copy()

#         print(f'Accel. PG-EXTRA, varying tau: iteration {k:04} of {max_iters}', end='\r')
#     return v2, res, rel, cons