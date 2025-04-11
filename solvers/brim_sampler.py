import dimod
import torch
import numpy as np
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def sample_BRIM(problem: dimod.BQM, 
                dt: float, 
                tstop: float, 
                scale0: float, 
                scale1: float, 
                samples: int = 20,
                verbose: bool = True,
                device: str = device):
    N = problem.num_variables
    if problem.vartype == dimod.Vartype.SPIN:
        problem.change_vartype('BINARY')
    mat = torch.zeros(N, N, device=device).double()
    biases, (row_inds, col_inds, vals), const =\
        problem.to_numpy_vectors(sort_labels=True)
    biases = torch.tensor(biases, device=device)
    for i, j, v in zip(row_inds, col_inds, vals):
        mat[i, j] = v
        mat[j, i] = v
    x = torch.rand((samples, N), device=device).double()
    nsteps = int(np.ceil(tstop / dt))
    step = (scale1 - scale0) / (max(nsteps-1, 1)) / np.sqrt(dt)
    scale = scale0 / np.sqrt(dt)
    for _ in tqdm(range(nsteps), disable=not verbose):
        grad = x @ mat + biases + scale * torch.randn_like(x) 
        x.add_(-grad, alpha=dt).clip_(0, 1)
        scale += step
    x = (x >= 0.5).double()
    ene = 0.5 * (x @ mat @ x.T).diag() + (x @ biases).squeeze() + const
    return ene, x, tstop
