import numpy as np
import networkx as nx
from algorithms_comparison import pgextra, nids, gf_pgextra
import pickle
import pandas

# Resolvent - soft thresholding operator
def soft_threshold(z, s):
    return np.sign(z) * np.clip(np.abs(z) - s, a_min=0, a_max=None) # Using numpy array broadcasting

def JA(z, step, data):
    r"""
    Soft-thresholding operator for yi: sgn(z[i])*max(abs(z[i])-\tau \lambda[i]), 0)
    :param z: n*(2*n+\sum d_i)-length array, where z[i] = [x, y]. x is (2*n)-array of states, y is (\sum d_i)-array of bad data indicators
    """
    n = z.shape[0]  # no. of agents
    mi_csum = np.cumsum([0]+[len(data['m'][ii]) for ii in range(n)])   # cumulative no. of measurements for each agent
    p = z.shape[1] - mi_csum[-1]   # no. of state vars

    JAz = z.copy()
    for ii in range(n):
        zi = z[ii].flatten()
        xi, yii = zi[:p], zi[p+mi_csum[ii]:p+mi_csum[ii+1]] # agent local copy of states, agent-specific bad data indicators
        lambda_i = data['lambda'][ii]   # regularisation parameter

        JAz[ii,p+mi_csum[ii]:p+mi_csum[ii+1]] = soft_threshold(yii, step*lambda_i) # only threshold yii
    
    return JAz  # return other values as inputted

def B(z, data):
    r"""
    Evaulate gradient operator (B(z))_i = -[Hi^T, I]@(mi - Hi@xi - oi), 
    """
    n = z.shape[0]  # no. of agents
    mi_csum = np.cumsum([0]+[len(data['m'][ii]) for ii in range(n)])   # cumulative no. of measurements for each agent
    p = z.shape[1] - mi_csum[-1]   # no. of state vars
    Bz = np.zeros(z.shape)

    for ii in range(n):
        zi = z[ii].flatten()
        xi, yii = zi[:p], zi[p+mi_csum[ii]:p+mi_csum[ii+1]] # agent local copy of states, agent-specific bad data indicators
        Hi = data['H'][ii]  # Measurement matrix
        mi = data['m'][ii]  # Measurements

        Bz[ii,:p] = -Hi.T@(mi - Hi@xi - yii)
        Bz[ii,p+mi_csum[ii]:p+mi_csum[ii+1]] = -(mi - Hi@xi - yii)
        
    return Bz


def algorithm_testing(alg, resolvent, grad, xzero, pars, data, tols, num: int, fname):
    """
    Run algorithm on specified problem instance
    :param alg: function handle for algorithm
    :param resolvent: function handle for resolvent computation
    :param grad: function handle for gradient computation
    :param xzero: starting point for algorithm
    :param pars: list of algorithm parameters
    :param data: problem dataset
    :param tols: dictionary of {metric: tolerance}
    :param num: which dataset in the dictionary are we running
    """
    # --- Run PGEXTRA ---
    results = {'dataset': num, 'res_it': [0.], 'res_err': [0.], 'res_time': [0.], 'rel_it': [0.], 'rel_err': [0.], 'rel_time': [0.],
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
nagents = 4
dim = 14 # no. of state variables (here buses)

eta_list = [0.1, 0.5, 0.8, 0.95, 1.0]
toler = {'res': 1e-6, 'rel': 1e-6, 'con': 1e-12}  # Stopping criteria - error tolerances
filename = 'results_fpgextra_psse.xlsx'

# Define parameter functions for Fast PG-EXTRA
alpha = 5.
def alpha_k(k):
    # Linear cooling heuristic: alpha(k) increasing linearly to 100*\alpha at maxit/2 iterations.
    if k<= (maxit//2):
        ak = alpha + 99*alpha*k/(maxit//2)
    else:
        ak = 100*alpha
    return ak

def sigma_k(k): return alpha_k(k)

# Construct mixing matrix W, using Metropolis constant edge weight matrix
edges = [(0,1),
         (1,2),
         (2,3)]
        
G = nx.Graph(edges)
# Metropolis constant edge weight matrix
W = np.zeros((nagents, nagents))
for i,j in edges: 
    W[i,j] = 1./(max(G.degree[i], G.degree[j])+1.) # w_ij = 1/(max{d_i, d_j}+\eps) for i,j\in Edges, set \eps=1
    W[j, i] = W[i,j]
for i in range(nagents):
    W[i,i] = 1. - np.sum(W[i,:])    # w_ii = 1 - \sum_{j\in Edges} w_ij

with open(r'dict_of_psse_instances', 'rb') as f:
    instance_dict = pickle.load(f)

# Run algorithms on dataset for each error metric
for jj in instance_dict:
    print(f'Instance {jj}\n')
    rng_gen = np.random.default_rng(int(f'{2026}{jj}'))
    # Import dataset and set up parameters
    datadict = instance_dict[jj]
    measures = sum(len(datadict['m'][ii]) for ii in datadict['m'])
    x0 = np.zeros((nagents, dim+measures)) # state variables + 1 bad data variable per measurement

    # Compute algorithm parameters
    H = datadict['H']
    Hi_norms = np.array([np.linalg.norm(H[ii], ord=2)**2 for ii in H]) # norms of matrices Hi
    L = np.max([(Hi_norms + 1)*np.clip(Hi_norms, a_min=1, a_max=None)]) # max{(|Hi|**2+1)*max{|Hi|**2,1}}
    tau = 0.9 * (1 + np.min(np.linalg.eigvalsh(W))) / L  # tau <= (1-||K||^2)/L
    thetainv = 2. - tau*L/(1 + np.min(np.linalg.eigvalsh(W)))   # 1/averaged constant of Condat-Vu operator
    s = 0.9* thetainv # 0 < s < (4\beta_M-1)/(2\beta_M)
    beta = 0.9/np.linalg.norm(tau*0.5*(np.eye(nagents)-W), ord=2)

    # --- Run PGEXTRA ---
    params = {'tau': tau,
              'W': W,
              'maxit': maxit}
    algorithm_testing(pgextra, JA, B, x0, params, datadict, toler, num=jj, fname=filename)

    # --- Run NIDS --- 
    params = {'tau': tau,
              'beta': beta,
              'W': W,
              'maxit': maxit}
    algorithm_testing(nids, JA, B, x0, params, datadict, toler, num=jj, fname=filename)
    
    # --- Run Generalised Fast PGEXTRA ---
    for eta in eta_list:
        params = {'tau': tau,
                     's': s, # 0 < s < (4\beta_M-1)/(2\beta_M)
                     'alpha': alpha_k,
                     'sigma': sigma_k,
                     'eta': eta,
                     'W': W,
                     'maxit': maxit}
        algorithm_testing(gf_pgextra, JA, B, x0, params, datadict, toler, num=jj, fname=filename)

    # --- Run Generalised Fast PGEXTRA with alpha==500 ---
    params = {'tau': tau,
                's': s,
                'alpha': lambda k: 500.,
                'sigma': lambda k: 500.,
                'eta': 1.0,
                'W': W,
                'maxit': maxit}
    algorithm_testing(gf_pgextra, JA, B, x0, params, datadict, toler, num=jj, fname=filename)
    print()
print('Done!')