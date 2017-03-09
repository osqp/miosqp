"""
Hybrid vehicle example from http://web.stanford.edu/~boyd/papers/miqp_admm.html

Bartolomeo Stellato, University of Oxford, 2017
"""
import examples.vehicle_matrices as mat
import mathprogbasepy as mpbpy
import matplotlib.pylab as plt
import numpy as np

import miosqp

def run_example():
    """
    Generate and solve vehicle example
    """

    T = 72       # Horizon length
    n_x = 1     # Number of states
    n_u = 4     # Number of inputs

    # Generate problem matrices
    P, q, A, l, u, r, i_idx = mat.generate_example(T)

    # Solve with gurobi
    # Create MIQP problem
    prob = mpbpy.QuadprogProblem(P, q, A, l, u, i_idx)

    '''
    Solve with GUROBI (Time limit 20 sec)
    '''
    resGUROBI = prob.solve(solver=mpbpy.GUROBI, TimeLimit=20)

    '''
    Solve with CPLEX (Time limit 20 sec)
    '''
    # resCPLEX = prob.solve(solver=mpbpy.CPLEX, timelimit=20)

    '''
    Solve with MIOSQP
    '''
    # Define problem settings
    miosqp_settings = {'eps_int_feas': 1e-03,   # integer feasibility tolerance
                       'max_iter_bb': 1000,     # maximum number of iterations
                       'tree_explor_rule': 1,   # tree exploration rule
                                                #   [0] depth first
                                                #   [1] two-phase: depth first  until first incumbent and then  best bound
                       'branching_rule': 0,     # branching rule
                                                #   [0] max fractional part
                       'verbose': True}

    osqp_settings = {'eps_abs': 1e-03,
                     'eps_rel': 1e-03,
                     'eps_inf': 1e-03,
                     'rho': 0.001,
                     'sigma': 0.01,
                     'alpha': 1.5,
                     'polish': False,
                     'max_iter': 2000,
                     'verbose': False}

    work = miosqp.miosqp_solve(P, q, A, l, u, i_idx,
                               miosqp_settings, osqp_settings,
                            #    resGUROBI.x
                               )




    # import ipdb; ipdb.set_trace()

    # Get results
    # x = resCPLEX.x
    x = work.x
    z = x[2:n_u*T:n_u]
    s = x[3:n_u*T:n_u]
    P_eng = x[1:n_u*T:n_u]
    P_batt = x[0:n_u*T:n_u]
    E = x[n_u*T:]

    import ipdb; ipdb.set_trace()

    # Plot results
    t = np.arange(0, T, 1)
    fig, ax = plt.subplots(5, 1)
    ax[0].step(t, E)
    ax[0].set_ylabel('E')
    ax[1].step(t, P_batt)
    ax[1].set_ylabel('P_batt')
    ax[2].step(t, P_eng)
    ax[2].set_ylabel('P_eng')
    ax[3].step(t, z)
    ax[3].set_ylabel('z')
    ax[4].step(t, s)
    ax[4].set_ylabel('s')

    plt.show(block=False)


    # Print
    print("Optimal cost: %.4e" % (work.upper_glob + r))

    import ipdb; ipdb.set_trace()
