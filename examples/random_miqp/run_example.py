"""
Solve MIQP using mathprogbasepy
"""
from __future__ import division
from __future__ import print_function
from builtins import range
import scipy as sp
import scipy.sparse as spa
import numpy as np
import pandas as pd


import mathprogbasepy as mpbpy
import miosqp

# Import progress bar
from tqdm import tqdm

# Reload miosqp module
try:
    reload  # Python 2.7
except NameError:
    from importlib import reload  # Python 3.4+

reload(miosqp)


def solve(n_vec, m_vec, p_vec, repeat, dns_level, seed, solver='gurobi'):
    """
    Solve random optimization problems
    """

    print("Solving random problems with solver %s\n" % solver)

    # Define statistics to record
    std_solve_time = np.zeros(len(n_vec))
    avg_solve_time = np.zeros(len(n_vec))
    min_solve_time = np.zeros(len(n_vec))
    max_solve_time = np.zeros(len(n_vec))

    n_prob = len(n_vec)

    # Store also OSQP time
    if solver == 'miosqp':
        # Add OSQP solve times statistics
        avg_osqp_solve_time = np.zeros(len(n_vec))

    # reset random seed
    np.random.seed(seed)

    for i in range(n_prob):

        # Get dimensions
        n = n_vec[i]
        m = m_vec[i]
        p = p_vec[i]

        print("problem n = %i, m = %i, p = %i" % (n, m, p))

        # Define vector of cpu times
        solve_time_temp = np.zeros(repeat)

        # Store also OSQP time
        if solver == 'miosqp':
            osqp_solve_time_temp = np.zeros(repeat)

        for j in tqdm(range(repeat)):
        #  for j in range(repeat):

            # Generate random vector of indeces
            i_idx = np.random.choice(np.arange(0, n), p, replace=False)

            # Generate random Matrices
            Pt = spa.random(n, n, density=dns_level)
            P = spa.csc_matrix(np.dot(Pt, Pt.T))
            q = sp.randn(n)
            A = spa.random(m, n, density=dns_level)
            u = 2 + sp.rand(m)
            l = -2 + sp.rand(m)

            # Enforce [0, 1] bounds on variables
            i_l = np.zeros(p)
            i_u = np.ones(p)
            #  A, l, u = miosqp.add_bounds(i_idx, 0., 1., A, l, u)

            if solver == 'gurobi':
                # Solve with gurobi
                prob = mpbpy.QuadprogProblem(P, q, A, l, u, i_idx, i_l, i_u)
                res_gurobi = prob.solve(solver=mpbpy.GUROBI,
                                        verbose=False, Threads=1)
                if res_gurobi.status != 'optimal':
                    import ipdb
                    ipdb.set_trace()
                solve_time_temp[j] = 1e3 * res_gurobi.cputime

            elif solver == 'miosqp':
                # Define problem settings
                miosqp_settings = {
                                   # integer feasibility tolerance
                                   'eps_int_feas': 1e-03,
                                   # maximum number of iterations
                                   'max_iter_bb': 1000,
                                   # tree exploration rule
                                   #   [0] depth first
                                   #   [1] two-phase: depth first until first incumbent and then  best bound
                                   'tree_explor_rule': 1,
                                   # branching rule
                                   #   [0] max fractional part
                                   'branching_rule': 0,
                                   'verbose': False,
                                   'print_interval': 1}

                osqp_settings = {'eps_abs': 1e-03,
                                 'eps_rel': 1e-03,
                                 'eps_prim_inf': 1e-04,
                                 'verbose': False}

                model = miosqp.MIOSQP()
                model.setup(P, q, A, l, u, i_idx, i_l, i_u,
                            miosqp_settings,
                            osqp_settings)
                res_miosqp = model.solve()

                # DEBUG (check if solutions match)
                #  prob = mpbpy.QuadprogProblem(P, q, A, l, u, i_idx, i_l, i_u)
                #  res_gurobi = prob.solve(solver=mpbpy.GUROBI, verbose=False)
                #  if (np.linalg.norm(res_gurobi.x - res_miosqp.x) /
                #          np.linalg.norm(res_gurobi.x)) > 1e-02:
                #     import ipdb; ipdb.set_trace()
#
                #  import ipdb; ipdb.set_trace()

                if res_miosqp.status != miosqp.MI_SOLVED:
                    import ipdb
                    ipdb.set_trace()
                
                # Solution time 
                solve_time_temp[j] = 1e3 * res_miosqp.run_time

                # Store OSQP time in percentage
                if solver == 'miosqp':
                    osqp_solve_time_temp[j] = \
                        100 * (res_miosqp.osqp_solve_time / res_miosqp.run_time)

        # Get time statistics
        std_solve_time[i] = np.std(solve_time_temp)
        avg_solve_time[i] = np.mean(solve_time_temp)
        max_solve_time[i] = np.max(solve_time_temp)
        min_solve_time[i] = np.min(solve_time_temp)

        # Store also OSQP time
        if solver == 'miosqp':
            avg_osqp_solve_time[i] = np.mean(osqp_solve_time_temp)

    # Create pandas dataframe for the results
    df_dict = {'n': n_vec,
               'm': m_vec,
               'p': p_vec,
               't_min': min_solve_time,
               't_max': max_solve_time,
               't_avg': avg_solve_time,
               't_std': std_solve_time}

    # Store also OSQP time
    if solver == 'miosqp':
        df_dict.update({'t_osqp_avg': avg_osqp_solve_time})

    timings = pd.DataFrame(df_dict)

    return timings


def run_example():

    # General settings
    n_repeat = 10               # Number of repetitions for each problem
    problem_set = 1             # Problem sets 1 (q << n) or 2 (q = n)
    density_level = 0.7         # density level for sparse matrices
    random_seed = 0             # set random seed to make results reproducible

    if problem_set == 1:
        n_arr = np.array([10, 10,  50, 50,  100, 100, 150, 150])
        m_arr = np.array([5,  100, 25, 200, 50,  200, 100, 300])
        p_arr = np.array([2,  2,   5,  10,  2,   15,  5,   20])

    # Other problems n = q
    elif problem_set == 2:
        n_arr = np.array([2, 4, 8, 12, 20, 26, 30, 36])
        m_arr = 5 * n_arr
        p_arr = (0.5 * n_arr).astype(int)

    timings_gurobi = solve(n_arr, m_arr, p_arr, n_repeat,
                           density_level, random_seed, solver='gurobi')

    timings_miosqp = solve(n_arr, m_arr, p_arr, n_repeat,
                           density_level, random_seed, solver='miosqp')

    print("Comparison table")
    df_dict = {'n': n_arr,
               'm': m_arr,
               'p': p_arr,
               't_miosqp_avg': timings_miosqp['t_avg'],
               't_miosqp_std': timings_miosqp['t_std'],
               't_miosqp_max': timings_miosqp['t_max'],
               't_miosqp_osqp_avg': timings_miosqp['t_osqp_avg'],
               't_gurobi_avg': timings_gurobi['t_avg'],
               't_gurobi_std': timings_gurobi['t_std'],
               't_gurobi_max': timings_gurobi['t_max']}
    comparison_table = pd.DataFrame(df_dict)
    cols = ['n', 'm', 'p', 't_miosqp_avg', 't_miosqp_std',
            't_miosqp_max', 't_miosqp_osqp_avg',
            't_gurobi_avg', 't_gurobi_std',
            't_gurobi_max']
    comparison_table = comparison_table[cols]  # Sort table columns
    comparison_table.to_csv('results/random_miqp.csv', index=False)
    print(comparison_table)

    # Converting results to latex table and storing them to a file
    formatter = lambda x: '%1.2f' % x
    latex_table = comparison_table.to_latex(header=False, index=False,
                                            float_format=formatter)
    f = open('results/random_miqp.tex', 'w')
    f.write(latex_table)
    f.close()
