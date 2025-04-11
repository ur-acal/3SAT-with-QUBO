import simulated_bifurcation as sb
import dimod
import torch


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def sample_SBM(problem: dimod.BQM, **kwargs):
    N = problem.num_variables
    mat = torch.zeros(N, N, device=device).float()
    biases, (row_inds, col_inds, vals), const =\
        problem.to_numpy_vectors(sort_labels=True)
    biases = torch.tensor(biases, device=device).float()
    for i, j, v in zip(row_inds, col_inds, vals):
        mat[i, j] = v
        mat[j, i] = v
    binary_vector, binary_value = sb.minimize(mat, 
                                              biases, 
                                              torch.tensor([const], device=device), 
                                              domain='binary',
                                              agents=256,
                                              max_steps=100000,
                                              ballistic=False,
                                              device=device,
                                              use_window=False,
                                              best_only=False)
    return binary_value.cpu().numpy(), binary_vector.cpu().numpy(), -1