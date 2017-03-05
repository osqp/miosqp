import osqp
# import osqppurepy as osqp

# Read max_iter_problems from files
import pickle
from os import listdir
# from os.path import splitext


# Solve problem with another solver
import mathprogbasepy as mpbpy

# Lead maxiter problems
# list_dir = listdir('./max_iter_examples')
prob_file = '76'

# Load one problem
with open('./max_iter_examples/%s.pickle' % prob_file, 'rb') as f:
    problem = pickle.load(f)

problem['settings']['verbose'] = True
problem['settings']['rho'] = 0.01
# problem['settings']['max_iter'] = 10**10   # With the same rho it converges in 163k iters


# Solve with OSQP
model = osqp.OSQP()
model.setup(problem['P'], problem['q'], problem['A'],
            problem['l'], problem['u'], **problem['settings'])
model.solve()


# Create QP problem and solve it with CPLEX
# prob = mpbpy.QuadprogProblem(problem['P'], problem['q'], problem['A'],
#                              problem['l'], problem['u'])
# resCPLEX = prob.solve(solver=mpbpy.CPLEX)
