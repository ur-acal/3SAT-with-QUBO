import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler
from pysa.sa import Solver
import numpy as np


def sample_SA(problem: dimod.BQM, **kwargs):
    solver = SimulatedAnnealingSampler()
    result = solver.sample(problem,**kwargs)
    time = result.info['timing']
    time = time['preprocessing_ns'] + time['sampling_ns']\
        + time['postprocessing_ns']
    values, samples = [], []
    for s in result.samples():
        value = problem.energy(s)
        values.append(value)
        samples.append(np.array(list(s.values())))
    return values, samples, time / 1e9


def sample_PT(problem: dimod.BQM, **kwargs):
    kwargs = dict([(i, None if j == -1 else j) for (i, j) in kwargs.items()])
    N = problem.num_variables
    mat = np.zeros((N, N))
    kwargs['parallel'] = True
    biases, (row_inds, col_inds, vals), const =\
        problem.to_numpy_vectors(sort_labels=True)
    mat[row_inds, col_inds] = vals
    mat = mat + mat.T
    mat[np.arange(N), np.arange(N)] = biases
    solver = Solver(problem=mat, problem_type='qubo')    
    df = solver.metropolis_update(**kwargs)    
    return df['best_energy'], df['best_state'], df['runtime (us)'].mean() / 1e6


def sample_tabu(problem: dimod.BQM, **kwargs):
    solver = TabuSampler()
    result = solver.sample(problem, **kwargs)
    values, samples = [], []
    time = time['preprocessing_ns'] + time['sampling_ns']\
        + time['postprocessing_ns']
    for s in result.samples():
        value = problem.energy(s)
        values.append(value)
        samples.append(np.array(list(s.values())))
    return values, samples, time / 1e9