"""
Solve MIQP using mathprogbasepy
"""
from __future__ import division
from __future__ import print_function
from builtins import range
import scipy as sp
import scipy.sparse as spa
import numpy as np
import numpy.linalg as la
import mathprogbasepy as mpbpy
import miosqp

# Reload miosqp module
try:
    reload  # Python 2.7
except NameError:
    from importlib import reload  # Python 3.4+

reload(miosqp)

if __name__ == "__main__":


    # Nasty problem
    n = 100
    m = 500
    np.random.seed(7)

    # Random Example
    # n = 50
    # m = 80
    # np.random.seed(7)

    # Other example
    # n = 20
    # m = 30
    # np.random.seed(7)



    # 10 times slower than gurobi
    # i_idx = np.random.choice(np.arange(1,n+1), (int(n/2)), replace=False)
    i_idx = np.random.choice(np.arange(0,n), (int(n/3)), replace=False)


    # Generate random Matrices
    Pt = sp.randn(n, n)
    P = spa.csc_matrix(np.dot(Pt.T, Pt))
    q = sp.randn(n)
    A = spa.csc_matrix(sp.randn(m, n))
    u = 3 + sp.randn(m)
    # l = u
    l = -3 + sp.randn(m)
    # i_idx = np.array([0, 2, 3, 7])  # index of integer variables



    # Enforce integer variables to be binary => {0, 1}
    I_int = spa.identity(n).tocsc()
    I_int = I_int[i_idx, :]
    l_int = np.empty((n,))
    l_int.fill(0.)
    l_int = l_int[i_idx]
    u_int = np.empty((n,))
    u_int.fill(1)
    u_int = u_int[i_idx]
    A = spa.vstack([A, I_int]).tocsc()      # Extend problem constraints matrix A
    l = np.append(l, l_int)         # Extend problem constraints
    u = np.append(u, u_int)         # Extend problem constraints






    # Create MIQP problem
    prob = mpbpy.QuadprogProblem(P, q, A, l, u, i_idx)
    resGUROBI = prob.solve(solver=mpbpy.GUROBI)
    # resCPLEX = prob.solve(solver=mpbpy.CPLEX)


    # Try miOSQP

    # Define problem settings
    miosqp_settings = {'eps_int_feas': 1e-03,   # integer feasibility tolerance
                       'max_iter_bb': 10000,     # maximum number of iterations
                       'tree_explor_rule': 1,   # tree exploration rule
                                                #   [0] depth first
                                                #   [1] two-phase: depth first  until first incumbent and then  best bound
                       'branching_rule': 0,     # branching rule
                                                #   [0] max fractional part
                       'verbose': True}

    osqp_settings = {'eps_abs': 1e-04,
                     'eps_rel': 1e-04,
                     'eps_inf': 1e-04,
                     'rho': 0.005,
                     'sigma': 0.01,
                     'alpha': 1.5,
                     'polish': False,
                     'max_iter': 2000,
                     'verbose': False}

    work = miosqp.miosqp_solve(P, q, A, l, u, i_idx,
                               miosqp_settings, osqp_settings,
                            #    resGUROBI.x
                               )



    print("\n\n\nDifference solutions miOSQP and GUROBI               %.4e" % la.norm(resGUROBI.x - work.x))
    # print("\n\n\nDifference solutions miOSQP and CPLEX               %.4e" % la.norm(resCPLEX.x - work.x))


    print("miOSQP")
    print("------")
    print("Objective value       %.4e" % work.upper_glob)
    print("Elapsed time          %.4e\n" % work.run_time)

    print("GUROBI")
    print("------")
    print("Objective value       %.4e" % resGUROBI.obj_val)
    print("Elapsed time          %.4e" % resGUROBI.cputime)

    # print("CPLEX")
    # print("------")
    # print("Objective value       %.4e" % resCPLEX.objval)
    # print("Elapsed time          %.4e" % resCPLEX.cputime)
