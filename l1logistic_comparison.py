import numpy as np
import networkx as nx
from algorithms_comparison import pgextra, nids, gf_pgextra
import pickle
import pandas

# Resolvent - soft thresholding operator
def soft_threshold(x, s):
    return np.sign(x) * np.clip(np.abs(x) - s, a_min=0, a_max=None) # Using numpy array operations

def JA(z, step, data):
    r"""
    Soft-thresholding operator: sgn(z[i])*max(abs(z[i])-\tau \lambda[i]), 0)
    """
    n = z.shape[0]
    JAz = np.zeros(z.shape)
    for ii in range(n):
        zi = z[ii].flatten()
        lambda_i = data['lambda'][ii]
        JAz[ii,:] = soft_threshold(zi, step*lambda_i)
    
    return JAz

def B(z, data):
    r"""
    Evaulate operator B(z) = [\grad_x fi(x, c); \grad_c fi(x,c)]
    """
    n = z.shape[0]
    mdims = data['m']
    Bz = np.zeros(z.shape)
    for ii in range(n):
        zi = z[ii].flatten()
        Abi = data['A'][ii]*data['b'][ii] # broadcast bi over columns of Ai
        
        pi = 1./(1+np.exp(-Abi@zi))
        Bz[ii,:] = (-1./mdims)*(Abi.T@(1-pi))
        
    return Bz


def algorithm_testing(alg, resolvent, grad, xzero, pars, data, tols, num: int, nam: str, fname):
    """
    Run algorithm on specified problem instance
    :param alg: function handle for algorithm
    :param resolvent: function handle for resolvent computation
    :param grad: function handle for gradient computation
    :param xzero: starting point for algorithm
    :param pars: list of algorithm parameters
    :param data: problem dataset
    :param tols: dictionary of {metric: tolerance}
    :param num: index of the dataset in the dictionary are we running
    :param nam: which dataset in the dictionary are we running
    """
    # --- Run PGEXTRA ---
    results = {'dataset': nam, 'res_it': [0.], 'res_err': [0.], 'res_time': [0.], 'rel_it': [0.], 'rel_err': [0.], 'rel_time': [0.],
                'con_it': [0.], 'con_err': [0.], 'con_time': [0.]}
    for met in tols:
        tol = tols[met]
        results[f'{met}_it'][0], results[f'{met}_err'][0], results[f'{met}_time'][0] = alg(xzero, resolvent, grad, pars, met, tol, data)
    print()
    
    df = pandas.DataFrame(results) # Append PGEXTRA results to excel file
    with pandas.ExcelWriter(fname, engine='openpyxl', mode='a',if_sheet_exists='overlay') as writer:
        sheet_name = alg.__name__
        if alg.__name__ == 'gf_pgextra':
            sheet_name += f'_eta{pars['eta']}_alp{int(pars['alpha'](0))}'   # append \eta and \alpha_0 to name for excel sheet

        if num==0:
            df.to_excel(writer,sheet_name=sheet_name, header=True, index_label=num)
        else:
            df.to_excel(writer,sheet_name=sheet_name,startrow=num+1, header=False, index_label=num)


# --- Algorithm parameters - non-data specific ---
maxit = 10000
nagents = 10

eta_list = [0.1, 0.5, 0.8, 0.95, 1.0]
toler = {'res': 1e-8, 'rel': 1e-8, 'con': 1e-12}
filename = 'results_fpgextra_log.xlsx'

# Define parameter functions for Fast PG-EXTRA
alpha = 5.
def alpha_k(k):
    r"""
    Linear cooling heuristic: alpha(k) increasing linearly to 100*\alpha at maxit/2 iterations. 
    \alpha_max = 100*\alpha
    :param: k: iteration number
    """
    if k<= (maxit//2):
        ak = alpha + 99*alpha*k/(maxit//2)
    else:
        ak = 100*alpha
    return ak

def sigma_k(k):
    return alpha_k(k)

# Graph from PG-EXTRA
edges = [(0,1),(0,3),(0,4),(0,6),
         (1,6),
         (2,6),(2,8),
         (3,4),(3,6),(3,8),(3,9),
         (4,5),
         (5,8),(5,9),
         (6,7),(6,8),
         (7,9),
         (8,9)]
        
G = nx.Graph(edges)
# Metropolis constant edge weight matrix
W = np.zeros((nagents, nagents))
for i,j in edges:
    W[i,j] = 1./(max(G.degree[i], G.degree[j])+1.) # w_ij = 1/(max{d_i, d_j}+\eps) for i,j\in Edges
    W[j, i] = W[i,j]
for i in range(nagents):
    W[i,i] = 1. - np.sum(W[i,:])    # w_ii = 1 - \sum_{j\in Edges} w_ij

with open(r'dict_of_log_instances', 'rb') as f:
    instance_dict = pickle.load(f)

# Compare algorithms for each instance in set of datasets
for jj, nam in enumerate(instance_dict):
    print(f'\nInstance {nam}\n')
    rng_gen = np.random.default_rng(int(f'{2026}{jj}'))
    datadict = instance_dict[nam] # Import dataset
    A = datadict['A']
    
    p, mdim = A[0].shape[1], datadict['m']
    x0 = np.zeros((nagents, p)) # Initial iterate
    
    # Compute algorithm parameters
    L = (0.25/mdim)*np.max([np.linalg.svdvals(A[ii])[0]**2 for ii in range(nagents)])  # L = max_i L_i = max_i {(1/4m)|Mi^T @ Mi|}
    tau = 0.9 * (1 + np.min(np.linalg.eigvalsh(W))) / L     # tau_pg <= (1-||K||^2)/L
    thetainv = 2 - tau*L/(1 + np.min(np.linalg.eigvalsh(W)))
    s = 0.9* thetainv # 0 < s < (4\beta_M-1)/(2\beta_M)
    beta = 0.9/np.linalg.norm(tau*0.5*(np.eye(nagents)-W), ord=2)
    
     # --- Run PGEXTRA ---
    params = {'tau': tau,
                'W': W,
                'maxit': maxit}
    algorithm_testing(pgextra, JA, B, x0, params, datadict, toler, num=jj, nam=nam, fname=filename)

    # --- Run NIDS --- 
    params = {'tau': tau,
                'beta': beta,
                'W': W,
                'maxit': maxit}
    algorithm_testing(nids, JA, B, x0, params, datadict, toler, num=jj, nam=nam, fname=filename)
    
    # --- Run Generalised Fast PGEXTRA ---
    for eta in eta_list:
        params = {'tau': tau,
                        's': s, # 0 < s < (4\beta_M-1)/(2\beta_M)
                        'alpha': alpha_k,
                        'sigma': sigma_k,
                        'eta': eta,
                        'W': W,
                        'maxit': maxit}
        algorithm_testing(gf_pgextra, JA, B, x0, params, datadict, toler, num=jj, nam=nam, fname=filename)

    # --- Run Generalised Fast PGEXTRA with alpha==500 ---
    params = {'tau': tau,
                's': s,
                'alpha': lambda k: 500.,
                'sigma': lambda k: 500.,
                'eta': 1.0,
                'W': W,
                'maxit': maxit}
    algorithm_testing(gf_pgextra, JA, B, x0, params, datadict, toler, num=jj, fname=filename)