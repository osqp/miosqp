"""
Solve MIQP using mathprogbasepy
"""
from __future__ import division
from __future__ import print_function
from builtins import range
import scipy as sp
import scipy.sparse as spspa
import numpy as np
import numpy.linalg as la
import mathprogbasepy as mpbpy
import random
import miosqp
# reload(miosqp)

if __name__ == "__main__":
    # Random Example
    n = 20
    m = 50

    # Choose random list of integer elements within x components
    random.seed(3)
    i_idx = np.array(random.sample(list(range(1,n)), int(n/2)))

    # np.random.seed(3)  # Working with few iters
    # np.random.seed(4)  # Working with few iters
    # np.random.seed(5)

    # Generate random Matrices
    Pt = sp.randn(n, n)
    P = spspa.csc_matrix(np.dot(Pt.T, Pt))
    q = sp.randn(n)
    A = spspa.csc_matrix(sp.randn(m, n))
    u = 3 + sp.randn(m)
    # l = u
    l = -3 + sp.randn(m)
    # i_idx = np.array([0, 2, 3, 7])  # index of integer variables

    # Create MIQP problem
    prob = mpbpy.QuadprogProblem(P, q, A, l, u, i_idx)
    resGUROBI = prob.solve(solver=mpbpy.GUROBI)
    # resCPLEX = prob.solve(solver=mpbpy.CPLEX)


    # Try miOSQP
    work = miosqp.miosqp_solve(P, q, A, l, u, i_idx)



    print("\n\n\nDifference solutions miOSQP and GUROBI               %.4e" % la.norm(resGUROBI.x - work.x))
    # print("\n\n\nDifference solutions miOSQP and CPLEX               %.4e" % la.norm(resCPLEX.x - work.x))


    print("miOSQP")
    print("------")
    print("Objective value       %.4e" % work.upper_glob)
    print("Elapsed time          %.4e\n" % work.run_time)

    print("GUROBI")
    print("------")
    print("Objective value       %.4e" % resGUROBI.objval)
    print("Elapsed time          %.4e" % resGUROBI.cputime)

    # print("CPLEX")
    # print("------")
    # print("Objective value       %.4e" % resCPLEX.objval)
    # print("Elapsed time          %.4e" % resCPLEX.cputime)
