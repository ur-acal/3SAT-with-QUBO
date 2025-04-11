import utils
import nuesslein1
import nuesslein2
import chancellor
import choi
import os
import json
import numpy as np
from itertools import product
from typing import Iterable
import argparse as ap
import dimod
from tqdm import tqdm
from solvers import sample_SBM, sample_BRIM, sample_SA, sample_tabu, sample_PT
import pandas as pd

options = {
    'nuesslein1': nuesslein1.nuesslein1,
    'nuesslein2': nuesslein2.nuesslein2,
    'chancellor': chancellor.chancellor,
    'choi': choi.choi
}

solver_options = {
    'sa': sample_SA,
    'sbm': sample_SBM,
    'tabu': sample_tabu,
    'pt': sample_PT,
    'brim': sample_BRIM
}
def to_value(val: str):
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def main(problems: Iterable[str], 
         methods: Iterable[str], 
         solvers: Iterable[str], 
         log_solutions: bool = False):
    # formula = utils.create_formula(V, num_clauses=C, k=3)
    data = []
    solver_configs = {}
    for s in solvers:
        with open(f'configs/{s}.json') as infile:
            conf = json.loads(infile.read())
            
        conf = dict([(i, 
                    [j]
                      if not isinstance(j, Iterable) or isinstance(j, str) else j)
                     for i, j in conf.items()])
        solver_configs[s] = []
        for vals in product(*conf.values()):
            solver_configs[s].append(
                dict(zip(conf.keys(), vals)))
    for problem in tqdm(problems):
        V, C, formula = utils.read_formula(problem)
        for method in methods:
            method_func = options[method]
            formulation = method_func(formula, V)
            formulation.fillQ()
            bqm = dimod.BQM.from_qubo(formulation.Q)
            for solver in solvers:
                for config in solver_configs[solver]:
                    _, solutions, time = solver_options[solver](
                        bqm, 
                        **config)
                    
                    for sol in solutions:
                        assignment = [int(sol[i]) for i in range(V)]
                        clause_sat = utils.check_solution(formula, assignment)
                        data.append((problem, method, solver, len(formula)- clause_sat, time, config.get('num_sweeps', 0)))
            
        df = pd.DataFrame(
            data=data,
            columns=[
                'problem',
                'reduction',
                'solver',
                'unsat_clauses',
                'time',
                'sweeps'
            ] + (['sol'] if log_solutions else [])
        )
        df.to_csv('result_n_gt_50.csv')
    print("nuesslein_{n+m}: ", nuesslein2.nuesslein2(formula, V).solve())
    print("nuesslein_{2n+m}: ", nuesslein1.nuesslein1(formula, V).solve())
    print("chancellor_{n+m}: ", chancellor.chancellor(formula, V).solve())


parser = ap.ArgumentParser()
parser.add_argument('cnf', help='Path to DIMACS-formatted CNF file ' +
                    '(DARPA comment tomfoolery permitted)', nargs='+')
parser.add_argument('--method', 
                    help='Reduction method', 
                    choices=options.keys(),
                    default=options.keys(),
                    nargs='+')
parser.add_argument('--solver', 
                    help='Solver method', 
                    choices=solver_options.keys(),
                    default=['sa'],
                    nargs='+')
parser.add_argument('--log-solutions', 
                    dest='log',
                    help='Save solutions to records', 
                    default=False,
                    action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.cnf, args.method, args.solver, args.log)