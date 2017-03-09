"""
Simulate inverter model with ADP formulation from the paper
"High-Speed Finite Control Set Model Predictive Control for Power Electronics", B. Stellato, T. Geyer and P. Goulart

"""
import numpy as np

# Power converter model files
from power_converter import Model
from tail_cost import TailCost


'''
Simulation parameters
'''
Ts = 25.0e-06            # Sampling time
freq = 50.               # Switching frequency
torque = 1.              # Desired torque
t0 = 0.0                 # Initial time
init_periods = 2         # Number of integer period to settle before simulation
sim_periods = 3          # Numer of simulated periods
flag_steady_trans = 0    # Flag Steady State (0) or Transients (1)


'''
ADP Parameters
'''
gamma = 0.95          # Forgetting factor
N_adp = 1              # Horizon length
delta = 4             # Switching frequency penalty
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

#TODO: Add for loop for different N_adp, fsw_des, delta

# Generate extended adp model
model.gen_dynamical_system(fsw_des, delta)

# Generate tail cost
model.gen_tail_cost(N_tail, gamma, name='tailBackupN1Delta4.00.mat')

# Simulate model
model.simulate_cl(N_adp, flag_steady_trans)
