import numpy as np
import time

def pgextra(x0, JA, B, pars, met, tol, data):
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
    K2 = 0.5 * (np.eye(n) - W)    # (W-I)/2

    err = [np.inf]    # Initialise error, iterations, and time
    its = 0
    time_start = time.perf_counter()

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

        its += 1

        if met == 'res':
            err.append(residue_pg(x2, x1, K2, tW))
        elif met == 'rel':
            err.append(relerror_pg(x2, u1, q1, B(x2, data), tau, tW, K2))
        elif met == 'con':
            err.append(consensus_pg(x2, K2))
        else:
            raise ValueError('not valid "met", must be "res", "rel", or "con"')

        if err[-1] <= tol :
            break
        else: # Update variables, continue with loop
            q1 += x2
            y0 = y1.copy()
            u1 = u2.copy()
            x0 = x1.copy()
            x1 = x2.copy()

            print(f'PG-EXTRA: iteration {k:04} of {maxit}', end='\r')
    return its, err[-1], time.perf_counter()-time_start

def nids(x0, JA, B, pars, met, tol, data):
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
    tW = 0.5*(np.eye(n) + W)   # beta*(I+W)/2)
    tildeW =  np.eye(n) - tau*0.5*beta*(np.eye(n) - W)  # when tau_1=...=tau_n and \beta = 1/max(tau), tildeW=tW
    K2 = 0.5 * (np.eye(n) - W)    # (W-I)/2
    
    err = [np.inf]    # Initialise error, iterations, and time
    its = 0
    time_start = time.perf_counter()

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

        its += 1
        y2 = B(x2, data)
        if met == 'res':
            err.append(residue_nids(x2, x1, y2, y1, tau, tW, K2))
        elif met == 'rel':
            err.append(relerror_nids(x2, u2, q1-tau*(y1-yzero), y2, tau, beta, tW, K2))
        elif met == 'con':
            err.append(consensus_nids(x2, K2))
        else:
            raise ValueError('not valid "met", must be "res", "rel", or "con"')

        if err[-1] <= tol :
            break
        else: # Update variables, continue with loop
            q1 += x2
            y0 = y1.copy()
            u1 = u2.copy()
            x0 = x1.copy()
            x1 = x2.copy()

            print(f'NIDS: iteration {k:04} of {maxit}', end='\r')
    return its, err[-1], time.perf_counter()-time_start


def gf_pgextra(x0, JA, B, pars, met, tol, data):
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
    K2 = 0.5 * (np.eye(n) - W)    # (W-I)/2
    tW = 0.5 * (np.eye(n) + W)
    
    err = [np.inf]    # Initialise error, iterations, and time
    its = 0
    time_start = time.perf_counter()

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

        its += 1
                
        if met == 'res':
            err.append(residue_fpg(z1, x1, tW, K2))
        elif met == 'rel':
            err.append(relerror_fpg(z1, B(z1, data), u1, q1, tau, tW))        
        elif met == 'con':
            err.append(consensus_fpg(z1, K2))
        else:
            raise ValueError('not valid "met", must be "res", "rel", or "con"')

        if err[-1] <= tol :
            break
        else: # Update variables, continue with loop
            q1 += v2
            y0 = y1.copy()
            v1 = v2.copy()
            u0 = u1.copy()
            z0 = z1.copy()
            x0 = x1.copy()
            x1 = x2.copy()

            print(f'Gen. Fast PG-EXTRA: iteration {int(k):04} of {maxit}', end='\r')
    return its, err[-1], time.perf_counter()-time_start

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