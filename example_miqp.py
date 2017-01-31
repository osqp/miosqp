"""
Solve MIQP using mathprogbasepy
"""
import scipy as sp
import scipy.sparse as spspa
import numpy as np
import mathprogbasepy as mpbpy


# Random Example
n = 30
m = 50
# Generate random Matrices
Pt = sp.randn(n, n)
P = spspa.csc_matrix(np.dot(Pt.T, Pt))
q = sp.randn(n)
A = spspa.csc_matrix(sp.randn(m, n))
u = 3 + sp.randn(m)
# l = u
l = -3 + sp.randn(m)
i_idx = np.array([0, 3, 7])  # index of integer variables

# Create MIQP problem
prob = mpbpy.QuadprogProblem(P, q, A, l, u, i_idx)
resGUROBI = prob.solve(solver=mpbpy.GUROBI)
resCPLEX = prob.solve(solver=mpbpy.CPLEX)
