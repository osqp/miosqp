"""
Hybrid vehicle example from http://web.stanford.edu/~boyd/papers/miqp_admm.html


Bartolomeo Stellato, University of Oxford, 2017
"""
import numpy as np
# import scipy as sp
import scipy.linalg as sla
import scipy.sparse as spa

def generate_P_des(T_horizon):
    """
    Generate desired power P_des trend
    """

    # The desired power is supposed to be piecewise linear. Each element has
    # length l[i] and angle a[i]
    a = np.array([ 0.5, -0.5,  0.2, -0.7,  0.6, -0.2,  0.7, -0.5,  0.8, -0.4]) / 10.
    l = np.array([40., 20., 40., 40., 20., 40., 30., 40., 30., 60.])

    # Get required power
    Preq = np.arange(a[0],(a[0]*l[0]) + a[0], a[0])

    for i in range(1, len(l)):
        Preq = np.append(Preq, np.arange(Preq[-1]+a[i], Preq[-1]+ a[i]*l[i] + a[i], a[i]))

    # Slice Preq to match the step length (tau = 4, 5 steps each 20.)
    Preq = Preq[0:len(Preq):5]

    # Slice up to get the desired horizon
    Preq = Preq[:T_horizon]

    return Preq


def generate_example(T):
    """
    Main function to generate data

    State x: E
    Input u: [P_batt, P_eng, z, s]

    """

    # Define parameters
    tau = 4.    # length of the time interval

    # Constraints on electric charge
    E_max = 40.  # Maximum charge
    E_0 = 40.    # Initial charge
    x0 = E_0      # Initial state

    # Constraints on power
    P_max = 1.

    # Previous binary input
    z_prev = 0.  # z_{-1}

    # Define quadratic cost
    alpha = 1.   # quadratic term
    beta = 1.   # linear term
    gamma = 1  # constant term
    delta = 1.  # cost of turning on the switches
    eta = 0.1    # Penalty on last stage

    n_u = 4                  # Size of inputs
    n_x = 1                  # Size of state
    n = n_x * T + n_u * T    # Number of variables

    '''
    Generate desired power
    '''
    P_des = generate_P_des(T)

    '''
    Generate constraints A, l, u
    '''
    A = spa.csc_matrix((0, n))
    l = np.empty((0))
    u = np.empty((0))

    # Dynamics E_{t+1} = E_t - tau P_batt
    A_dyn = 1
    B_dyn = np.array([-tau, 0., 0., 0.])
    I_temp = np.eye(T)
    A_temp_u = np.kron(I_temp, -B_dyn)
    A_temp_x = np.eye(T * n_x) + np.kron(np.eye(T, k=-1), -A_dyn)
    A_temp = np.hstack((A_temp_u, A_temp_x))
    l_temp = np.append(A_dyn * x0, np.zeros((T - 1) * n_x))
    u_temp = l_temp

    A = spa.vstack((A, spa.csc_matrix(A_temp)), 'csc')
    l = np.append(l, l_temp)
    u = np.append(u, u_temp)

    # 0 <= P_eng <= P_max * z
    A_P_eng = np.array([[0., 1., -P_max, 0.], [0., -1., 0., 0.]])
    A_temp = np.kron(np.eye(T), A_P_eng)
    A_temp = np.hstack((A_temp, np.zeros((A_P_eng.shape[0] * T, n_x * T))))
    u_temp = np.zeros(A_P_eng.shape[0] * T)
    l_temp = -np.inf * np.ones(A_P_eng.shape[0] * T)

    A = spa.vstack((A, spa.csc_matrix(A_temp)), 'csc')
    l = np.append(l, l_temp)
    u = np.append(u, u_temp)


    # 0 <= E <= E_max
    A_E = np.array([[1.]])
    A_temp = np.kron(np.eye(T), A_E)
    A_temp = np.hstack((np.zeros((T, n_u * T)), A_temp))
    l_temp = np.zeros(T)
    u_temp = E_max * np.ones(T)

    A = spa.vstack((A, spa.csc_matrix(A_temp)), 'csc')
    l = np.append(l, l_temp)
    u = np.append(u, u_temp)


    # P_des <= P_batt + P_eng
    A_P_des = np.array([[1., 1., 0., 0.]])
    A_temp = np.kron(np.eye(T), A_P_des)
    A_temp = np.hstack((A_temp, np.zeros((A_P_des.shape[0] * T, n_x * T))))
    l_temp = P_des
    u_temp = np.inf * np.ones(A_P_des.shape[0] * T)

    A = spa.vstack((A, spa.csc_matrix(A_temp)), 'csc')
    l = np.append(l, l_temp)
    u = np.append(u, u_temp)

    # z \in {0, 1}
    A_z = np.array([[0., 0., 1., 0.]])
    A_temp = np.kron(np.eye(T), A_z)
    A_temp = np.hstack((A_temp, np.zeros((A_z.shape[0] * T, n_x * T))))
    l_temp = np.zeros(A_z.shape[0] * T)
    u_temp = np.ones(A_z.shape[0] * T)

    A = spa.vstack((A, spa.csc_matrix(A_temp)), 'csc')
    l = np.append(l, l_temp)
    u = np.append(u, u_temp)

    # Switch costs s_t >= z_t - z_{t-1}
    A_s_t = np.array([[0., 0., 1., -1.]])
    A_s_tprev = np.array([[0., 0., -1., 0.]])
    A_temp = np.kron(np.eye(T), A_s_t) + np.kron(np.eye(T, k=-1), A_s_tprev)
    A_temp = np.hstack((A_temp, np.zeros((T, n_x * T))))
    u_temp = np.append(z_prev, np.zeros(T - 1))
    l_temp = -np.inf * np.ones(T)

    A = spa.vstack((A, spa.csc_matrix(A_temp)), 'csc')
    l = np.append(l, l_temp)
    u = np.append(u, u_temp)

    # s_t >= 0
    A_s_t = np.array([[0., 0., 0., 1.]])
    A_temp = np.kron(np.eye(T), A_s_t)
    A_temp = np.hstack((A_temp, np.zeros((A_s_t.shape[0] * T, n_x * T))))
    l_temp = np.zeros(A_s_t.shape[0] * T)
    u_temp = np.inf * np.ones(A_s_t.shape[0] * T)

    A = spa.vstack((A, spa.csc_matrix(A_temp)), 'csc')
    l = np.append(l, l_temp)
    u = np.append(u, u_temp)

    '''
    Generate cost function
    '''
    P = spa.csc_matrix((n, n))
    q = np.zeros(n)
    r = 0.

    # eta * (E_T - E_max)
    P_temp = np.zeros((n, n))
    P_temp[-1, -1] = eta
    q_temp = np.zeros(n)
    q_temp[-1] = -2 * eta * E_max

    P += spa.csc_matrix(P_temp)
    q += q_temp


    # delta s_t
    q_delta = np.array([0., 0., 0., delta])
    q_temp = np.tile(q_delta, T)
    q_temp = np.append(q_temp, np.zeros(n_x * T))

    q += q_temp

    # f(P_eng, z) = alpha * P_eng^2 + beta * P_eng + gamma * z
    P_temp = np.diag(np.array([0., alpha, 0., 0.]))
    P_temp = np.kron(np.eye(T), P_temp)
    P_temp = sla.block_diag(P_temp, np.zeros((n_x * T, n_x * T)))

    q_temp = np.array([0., beta, gamma, 0.])
    q_temp = np.tile(q_temp, T)
    q_temp = np.append(q_temp, np.zeros(n_x * T))

    r_temp = eta * (E_max ** 2)

    P += spa.csc_matrix(P_temp)
    q += q_temp
    r += r_temp


    # Adjust P so that the objective is 0.5 * x' * P * x
    P = 2.0 * P

    '''
    Generate vector of indeces for integer variables
    '''
    i_idx_bin = np.array([0, 0, 1, 0])
    i_idx_bin = np.tile(i_idx_bin, T)
    i_idx_bin = np.append(i_idx_bin, np.zeros(n_x * T))
    i_idx = np.flatnonzero(i_idx_bin)


    # import ipdb; ipdb.set_trace()

    return P, q, A, l, u, r, i_idx
