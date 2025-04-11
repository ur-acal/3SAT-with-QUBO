# import dimod
# import torch
# import numpy as np
# from tqdm import tqdm

# TODO
# def sample_BRIM(problem: dimod.BQM, 
#                 dt: float, 
#                 tstop: float, 
#                 scale0: float, 
#                 scale1: float, 
#                 samples: int = 20,
#                 verbose: bool = True,
#                 device: str = 'cuda:0'):
#     N = problem.num_variables
#     if problem.vartype == dimod.Vartype.SPIN:
#         problem.change_vartype('BINARY')
#     mat = torch.zeros(N, N).double()
#     biases, (row_inds, col_inds, vals), const =\
#         problem.to_numpy_vectors(sort_labels=True)
#     biases = torch.tensor(biases)
#     for i, j, v in zip(row_inds, col_inds, vals):
#         mat[i, j] = v
#         mat[j, i] = v
#     x = torch.rand((samples, N)).double() * 2 - 1
#     nsteps = int(np.ceil(tstop / dt))
#     step = (scale1 - scale0) / (np.max(nsteps-1, 1)) / np.sqrt(dt)
#     scale = scale0 / np.sqrt(dt)
#     for _ in tqdm(range(nsteps), disable=not verbose):
#         repvec = x.tile()
#         grad = mat * x. + biases + scale * torch.randn_like(x) 
#         x.add_(grad, alpha=dt).clip_(0, 1)
#         scale += step
#     ene = 0.5 * (x.T @ mat @ x).diag() + (biases.T @ x).squeeze() + const
#     return ene, x
