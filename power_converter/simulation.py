"""
Simulate inverter model with ADP formulation from the paper
"High-Speed Finite Control Set Model Predictive Control for Power Electronics", B. Stellato, T. Geyer and P. Goulart

"""
import scipy.io as sio
import numpy as np

# Power converter model files
import model

'''
Simulation parameters
'''
Ts = 25.0e-06         # Sampling time
freq = 50.             # Switching frequency
torque = 1.            # Desired torque
init_periods = 4       # Number of integer period to settle before simulation
sim_periods = 20       # Numer of simulated periods
flag_steady_trans = 0    # Flag Steady State (0) or Transients (1)


'''
ADP Parameters
'''
gamma = 0.95          # Forgetting factor
N_adp = 1              # Horizon length
delta = 4             # Switching frequency penalty

# Switching filter parameters
k1 = 0.8e03
k2 = 0.8e03
fsw_des = 300


'''
Obtain ADP tail
'''
# Load samples mean and variance

# Compute ADP tail by solving an SDP

# Load tail instead
tail = sio.loadmat('tailBackupN1Delta4.00.mat')
P0 = tail['P0']
q0 = tail['q0']
r0 = tail['r0']


'''
Generate model parameters
'''
params = model.get_params(Ts, freq, k1, k2)


# Get total number of simulation steps
T_final = (init_periods + sim_periods) * params['Nstpp']


'''
Setup initial conditions
'''
t0 = 0.0  # Initial time

# Compute steady state values with T
x0 = model.get_initial_states(params, torque, 1)
x_null_torque = model.get_initial_states( params, 0., 1. )

# Previous input
uprev = np.array([0., 0., 0.])


r'''
Current at the torque step

Te = 0 => $i_{s\beta} = i_{s\alpha} \frac{\phi_{r\beta}}{\phi_{r\alpha}}$

'''
# cur_step_torque = -x_null_torque[:2]  # Compute steady states with desired torque
cur_step_torque = np.array([-x_null_torque[0], -x_null_torque[0]*x0[3]/x0[2]])

'''
Get initial conditions
'''
x0init = np.concatenate((x0, x0[:2], uprev, np.array([1., 1., 1.])))
init_conditions = {'x0': x0,
                   'cur_step_torque': cur_step_torque,
                   't0': t0}


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
adp_model = model.gen_adp_model(params, fsw_des, delta)

#
