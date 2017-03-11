"""
Simulate inverter model with ADP formulation from the paper
"High-Speed Finite Control Set Model Predictive Control for Power Electronics", B. Stellato, T. Geyer and P. Goulart

"""

# Import numpy
import numpy as np
import pandas as pd

# Power converter model files
from power_converter import Model


'''
Simulation parameters
'''
Ts = 25.0e-06            # Sampling time
freq = 50.               # Switching frequency
torque = 1.              # Desired torque
t0 = 0.0                 # Initial time
init_periods = 1         # Number of integer period to settle before simulation
sim_periods = 2          # Numer of simulated periods
flag_steady_trans = 0    # Flag Steady State (0) or Transients (1)


'''
ADP Parameters
'''
gamma = 0.95                # Forgetting factor
N_adp = np.arange(1, 7)     # Horizon length
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
THD_adp = []
fsw_adp = []
Te_adp = []
Times_adp = []


'''
Run simulations
'''

# Generate extended adp model
model.gen_dynamical_system(fsw_des, delta)

# Generate tail cost
model.gen_tail_cost(N_tail, gamma, name='delta_550.mat')

# Simulate model
gurobi_avg_time = np.zeros(len(N_adp))
gurobi_min_time = np.zeros(len(N_adp))
gurobi_max_time = np.zeros(len(N_adp))

miosqp_avg_time = np.zeros(len(N_adp))
miosqp_min_time = np.zeros(len(N_adp))
miosqp_max_time = np.zeros(len(N_adp))


for i in range(len(N_adp)):

    stats_gurobi = model.simulate_cl(N_adp[i], flag_steady_trans, solver='gurobi')

    gurobi_avg_time[i] = stats_gurobi.avg_solve_time
    gurobi_min_time[i] = stats_gurobi.min_solve_time
    gurobi_max_time[i] = stats_gurobi.max_solve_time

    stats_miosqp = model.simulate_cl(N_adp[i], flag_steady_trans, solver='miosqp')

    miosqp_avg_time[i] = stats_miosqp.avg_solve_time
    miosqp_min_time[i] = stats_miosqp.min_solve_time
    miosqp_max_time[i] = stats_miosqp.max_solve_time


# Create pandas dataframe
timings = pd.DataFrame({ 'grb_avg' : 1e03 * gurobi_avg_time,
                         'grb_min' : 1e03 * gurobi_min_time,
                         'grb_max' : 1e03 * gurobi_max_time,
                         'miosqp_avg' : 1e03 * miosqp_avg_time,
                         'miosqp_min' : 1e03 * miosqp_min_time,
                         'miosqp_max' : 1e03 * miosqp_max_time},
                         index=N_adp)

print("Results")
print(timings)
