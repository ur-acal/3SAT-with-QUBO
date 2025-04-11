# from dwave_qbsolv import QBSolv
import numpy as np
from neal import SimulatedAnnealingSampler



# this function downloads a random k-SAT formula
def create_formula(num_vars, num_clauses, k):

    formula = []
    while len(formula) < num_clauses:
        vars = np.random.choice(range(1,num_vars+1), size=k, replace=False)
        signs = np.random.choice([-1,+1], size=k, replace=True)
        formula.append(vars * signs)

    return formula

def read_formula(path):
    with open(path) as infile:
        formula = []
        for line in infile:
            line = line.strip()
            if line == '' or line[0] == 'c':
                continue
            if line[0] == 'p':
                n, m = map(int, line.split()[2:])
                continue
            if line[0] == '%':
                break
            args = line.split()[:-1]
            formula.append(np.array(list(map(int, args))))
        return n, m, formula

# this function solves a given QUBO-Matrix Q with Qbsolv
def solve_with_qbsolv(Q):
    response = SimulatedAnnealingSampler().sample_qubo(Q, num_sweeps=20000,num_repeats=1000)
    return response.samples()[0]


# this function calculates the value of a solution for a given QUBO-Matrix Q
def getValue(Q, solution):
    ones = [x for x in solution.keys() if solution[x] == 1]
    value = 0
    for x in ones:
        for y in ones:
            if (x,y) in Q.keys():
                value += Q[(x,y)]
    return value


# this function prints the first n row/columns of a QUBO-Matrix Q
def printQUBO(Q, n):
    for row in range(n):
        for column in range(n):
            if row > column:
                print("      ", end = '')
                continue
            printing = ""
            if (row,column) in Q.keys() and Q[(row,column)] != 0:
                printing = str(Q[(row,column)])
            printing += "_____"
            printing = printing[:5]
            printing += " "
            print(printing, end = '')
        print("")


# this function checks, whether a given assignment satisfies a given SAT-formula
def check_solution(formula, assignment):
    n = 0
    for c in formula:
        for l in c:
            if l < 0 and int(assignment[abs(l)-1]) == 0:
                n += 1
                break
            elif l > 0 and int(assignment[abs(l)-1]) == 1:
                n += 1
                break
    return n


def get_n_couplings(Q):
    n = 0
    for k in Q.keys():
        if Q[k] != 0:
            n += 1
    return n
