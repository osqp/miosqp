import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import math


def get_params(Ts, freq, k1, k2):
    """
    Compute model parameters given

    Args:
        Ts (double): Sampling time
        freq (double): Frequency
        k1 (double): filter constant
        k2 (double): filter constant
    """

    # Resistance
    Rs = 0.0108  # Stator
    Rr = 0.0091  # Rotor

    # Inductances
    Xls = 0.1493  # Stator
    Xlr = 0.1104  # Rotor
    Xm = 2.3489   # Mutual

    # Rotor speed
    omegar = 0.9911 # Nominal speed

    # Voltages
    Vdc = 1.930

    # Torque constant to fix [pu] units
    kT = 1.2361



    '''
    Define base model units
    '''
    omegab = 2 * math.pi * freq
    Tspu = Ts * omegab
    Nstpp = 1./freq/Ts  # Number of steps per period


    '''
    Compute intermediate parameters
    '''

    # Inductances
    Xs = Xls + Xm
    Xr = Xlr + Xm
    D = 0.6266   # D = Xs * Xr - Xm ^ 2

    # Time constants
    taus = (Xr * D) / (Rs * (Xr**2) + Rr * (Xm ** 2))
    taur = Xr / Rr


    '''
    Define Park transformation and its inverse
    '''
    P = 2./3. * np.array([[1.0, -1./2., -1./2.],
                          [0.0, np.sqrt(3.)/2., -np.sqrt(3.)/2.]])

    invP = np.array([ [1., 0.],
                      [-1./2., np.sqrt(3.)/2.],
                      [-1./2., -np.sqrt(3.)/2.]])


    '''
    Return parameters as dictionary
    '''
    params = {'Rs': Rs,
              'Rr': Rr,
              'Xm': Xm,
              'Xs': Xs,
              'Xr': Xr,
              'D': D,
              'taus': taus,
              'taur': taur,
              'omegar': omegar,
              'omegab': omegab,
              'Vdc': Vdc,
              'kT': kT,
              'Tspu': Tspu,
              'Nstpp': Nstpp,
              'P': P,
              'invP': invP,
              'freq': freq,
              'Ts': Ts,
              'Acur': Acur,
              'k1': k1,
              'k2': k2
              }

    return params






def get_initial_states(params, T, psiS_mag):
    """
    Given the torque and stator flux magnitude, compute
    the stator and rotor flux vectors in alpha/beta and the slip
    frequency. The stator flux vector is aligned with the
    alpha-axis.


    Args:
        params (dict): parameters
        T (double): torque reference [pu]
        psiS_mag (double): reference of the stator flux magnitude [pu]


    Returns:
        x0 (array): initial state [i_s, phi_r]
    """


    # Get parameters
    Rs = params['Rs']
    Rr = params['Rr']
    Xss = params['Xs']
    Xrr = params['Xr']
    Xm = params['Xm']
    D = params['D']
    kT = params['kT']  # Torque constant to correct [pu]


    # Stator flux components
    psiSa = psiS_mag
    psiSb = 0

    # PsiR alpha and beta components
    psiRb = -T/psiS_mag * D / Xm / kT
    dis = np.sqrt((Xm**2)*(psiSa**2) - 4. * (Xss**2) * (psiRb**2))
    psiRa1 = (Xm*psiSa + dis)/(2. * Xss)
    psiRa2 = (Xm*psiSa - dis)/(2*Xss)

    psiRa = psiRa1  # make sure that this is always the correct choice!

    # Slip frequency
    ws = -Rr * Xss / D * psiRb / psiRa

    # Build stator and rotor flux vectors
    PsiS = np.array([psiSa,  psiSb])
    PsiR = np.array([psiRa, psiRb])

    # Initialize the transformation matrix M
    M = 1./ D * np.array([[Xrr, 0, -Xm, 0.],
                          [0, Xrr, 0, -Xm]])


    # Stator currents in alpha/beta
    Is = M.dot(np.append(PsiS, PsiR))

    # Initial state
    x0 = np.append(Is, PsiR)

    return x0




def gen_adp_model(params, fsw_des, delta):
    """
    Generate extended ADP model


    Args:
    params (dict): model parameters
    fsw_des (double): desired switching frequency
    delta (double): relative weighting cost function


    Returns:
    model (dict): extended adp model

    """

    # Get parameters
    taus = params['taus']
    taur = params['taur']
    D = params['D']
    omegar = params['omegar']
    Vdc = params['Vdc']
    Xm = params['Xm']
    Xr = params['Xr']
    P = params['P']
    Tspu = params['Tspu']
    k1 = params['k1']
    k2 = params['k2']
    Ts = params['Ts']


    '''
    Generate individual system matrices
    '''
    # Physical system matrices
    F = np.array([[-1./taus, 0., Xm/(taur * D), omegar*Xm/D],
                  [0., -1./taus, -omegar*Xm/D, Xm/(taur * D)],
                  [Xm/taur, 0., -1./taur, -omegar],
                  [0., Xm/taur, omegar, -1./taur]])

    G = Xr / D * Vdc / 2. * np.array([[1., 0], [0., 1], [0., 0.], [0., 0.]]) * P


    # Discretize physical system
    A_phys = sla.expm(F * Tspu)
    B_phys = -(nla.inv(F) * (np.eye(A_phys.shape[0]) - A_phys) * G)


    # Concatenate oscillating states
    A_osc = np.array([[np.cos(Tspu), -np.sin(Tspu)],
                     [np.sin(Tspu), np.cos(Tspu)]])
    B_osc = np.zeros((2, 3))

    # Concatenate previous input as a state
    A_prev = np.zeros((3, 3))
    B_prev = np.eye(3)


    # Concatenate filter states
    a1 = 1. - 1./k1
    a2 = 1. - 1./k2

    A_sw = np.array([[a1, 0.],
                     [(1. - a1), a2]])
    # NB 1: Please note the 1 / 12 division to average over all the physical switches
    # NB 2: Please note the 1/fsw_des division to normalize switching frequency
    B_sw = 1./ fsw_des * 1./12. * (1 - a1) / Ts * \
        np.array([[1., 1., 1.], [0., 0., 0.]])



    # Concatenate switching frequency state
    A_fsw = np.array([[1.]])

    '''
    Generate complete system
    '''
    A = sla.block_diag(A_phys, A_osc, A_prev, A_sw, A_fsw)
    B = np.bmat([[B_phys, np.zeros((4, 3))],
                 [B_osc, np.zeros((2, 3))],
                 [B_prev, np.zeros((3, 3))],
                 [np.zeros((2, 3)), B_sw],
                 [np.zeros((1, 6))]])
    C = np.array([[1., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.,],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -delta, delta]])


    '''
    Extract previous input from data
    '''
    W = np.hstack((np.zeros((3, 6)), np.eye(3), np.zeros((3, 3))))

    '''
    Matrix to extract original input and auxiliary one from state
    '''
    G = np.hstack((np.eye(3), np.zeros((3, 3))))
    T = np.hstack((np.zeros((3, 3)), np.eye(3)))


    '''
    Generate matrix of possible input combinations
    '''
    M = np.zeros((3, 27))
    M[0, :] = np.hstack((-np.ones((1, 0)), np.zeros((1, 9)), np.ones((1, 9))))
    M[1, 0] =








    #
