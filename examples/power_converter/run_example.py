"""
Simulate inverter model with ADP formulation from the paper
"High-Speed Finite Control Set Model Predictive Control for Power Electronics", B. Stellato, T. Geyer and P. Goulart

"""

# Import numpy
import numpy as np
import pandas as pd

# Power converter model files
from .power_converter import Model

# Import plotting library
import matplotlib.pylab as plt
colors = {'b': '#1f77b4',
          'g': '#2ca02c',
          'o': '#ff7f0e'}


def run_example():
    '''
    Simulation parameters
    '''
    Ts = 25.0e-06            # Sampling time
    freq = 50.               # Switching frequency
    torque = 1.              # Desired torque
    t0 = 0.0                 # Initial time
    init_periods = 1         # Number of periods to settle before simulation
    sim_periods = 2          # Numer of simulated periods
    flag_steady_trans = 0    # Flag Steady State (0) or Transients (1)

    '''
    ADP Parameters
    '''
    gamma = 0.95                # Forgetting factor
    N_adp = np.arange(1, 6)     # Horizon length
    #  N_adp = np.array([2])
    delta = 5.5                 # Switching frequency penalty
    N_tail = 50

    # Switching filter parameters
    k1 = 0.8e03
    k2 = 0.8e03
    fsw_des = 300

    '''
    Setup model
    '''
    model = Model()

    # Set model parameters
    model.set_params(Ts, freq, k1, k2, torque)

    # Set simulation time
    model.set_time(t0, init_periods, sim_periods)

    # Set initial conditions
    model.set_initial_conditions()

    '''
    Allocate output statistics
    '''
    #  THD_adp = []
    #  fsw_adp = []
    #  Te_adp = []
    #  Times_adp = []

    '''
    Run simulations
    '''

    # Generate extended adp model
    model.gen_dynamical_system(fsw_des, delta)

    # Generate tail cost
    model.gen_tail_cost(N_tail, gamma, name='delta_550.mat')

    gurobi_std_time = np.zeros(len(N_adp))
    gurobi_avg_time = np.zeros(len(N_adp))
    gurobi_min_time = np.zeros(len(N_adp))
    gurobi_max_time = np.zeros(len(N_adp))

    miosqp_std_time = np.zeros(len(N_adp))
    miosqp_avg_time = np.zeros(len(N_adp))
    miosqp_min_time = np.zeros(len(N_adp))
    miosqp_max_time = np.zeros(len(N_adp))
    miosqp_avg_osqp_iter = np.zeros(len(N_adp))
    miosqp_osqp_avg_time = np.zeros(len(N_adp))

    # Simulate model
    for i in range(len(N_adp)):

        stats_gurobi = model.simulate_cl(N_adp[i], flag_steady_trans, 
                                         solver='gurobi')

        gurobi_std_time[i] = stats_gurobi.std_solve_time
        gurobi_avg_time[i] = stats_gurobi.avg_solve_time
        gurobi_min_time[i] = stats_gurobi.min_solve_time
        gurobi_max_time[i] = stats_gurobi.max_solve_time

        # Make plots for horizon 3
        if N_adp[i] == 3:
            plot_flag = 1
        else:
            plot_flag = 0

        stats_miosqp = model.simulate_cl(N_adp[i], flag_steady_trans, 
                                         solver='miosqp', plot=plot_flag)

        miosqp_std_time[i] = stats_miosqp.std_solve_time
        miosqp_avg_time[i] = stats_miosqp.avg_solve_time
        miosqp_min_time[i] = stats_miosqp.min_solve_time
        miosqp_max_time[i] = stats_miosqp.max_solve_time
        miosqp_avg_osqp_iter[i] = stats_miosqp.miosqp_avg_osqp_iter
        miosqp_osqp_avg_time[i] = stats_miosqp.miosqp_osqp_avg_time

    # Create pandas dataframe
    timings = pd.DataFrame({'T': N_adp,
                            'grb_avg': gurobi_avg_time,
                            'grb_std': gurobi_std_time,
                            'grb_min': gurobi_min_time,
                            'grb_max': gurobi_max_time,
                            'miosqp_avg': miosqp_avg_time,
                            'miosqp_std': miosqp_std_time,
                            'miosqp_min': miosqp_min_time,
                            'miosqp_max': miosqp_max_time,
                            'miosqp_osqp_avg_time': miosqp_osqp_avg_time,
                            'miosqp_avg_osqp_iter': miosqp_avg_osqp_iter})

    print("Results")
    print(timings)

    timings.to_csv('results/power_converter_timings.csv')

    # Create error plots with fill_between
    plt.figure()
    ax = plt.gca()
    plt.semilogy(N_adp, gurobi_avg_time, color=colors['o'],
                 label='GUROBI')
    plt.semilogy(N_adp, miosqp_avg_time, color=colors['b'],
                 label='miOSQP')
    plt.xticks(N_adp)
    ax.set_xlabel(r'Horizon length $T$')
    ax.set_ylabel(r'Time [s]')
    ax.legend(loc='upper left')
    plt.grid(True, which="both")
    plt.tight_layout()
    # plt.show()
    plt.savefig('results/power_converter_timings.pdf')

