import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import math
from ..power_converter import utils 

# Import progress bar
from tqdm import tqdm

# Import mathprogbasepy
import mathprogbasepy as mpbpy

# import miosqp solver
import miosqp


# Internal functions and objects
from .tail_cost import TailCost
from .quadratic_program import MIQP


class Statistics(object):
    def __init__(self, fsw, thd, max_solve_time, min_solve_time,
                 avg_solve_time, std_solve_time):
        self.fsw = fsw
        self.thd = thd
        self.max_solve_time = max_solve_time
        self.min_solve_time = min_solve_time
        self.avg_solve_time = avg_solve_time
        self.std_solve_time = std_solve_time


class SimulationResults(object):
    """
    Simulation results signals
    """

    def __init__(self, X, U, Y_phase, Y_star_phase, T_e, T_e_des, solve_times):
        self.X = X
        self.U = U
        self.Y_phase = Y_phase
        self.Y_star_phase = Y_star_phase
        self.T_e = T_e
        self.T_e_des = T_e_des
        self.solve_times = solve_times


class DynamicalSystem(object):
    """
    Power converter dynamical system
    """

    def __init__(self, params, fsw_des, delta):
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
        taus = params.taus
        taur = params.taur
        D = params.D
        omegar = params.omegar
        Vdc = params.Vdc
        Xm = params.Xm
        Xr = params.Xr
        P = params.P
        Tspu = params.Tspu
        k1 = params.k1
        k2 = params.k2
        Ts = params.Ts

        '''
        Generate individual system matrices
        '''
        # Physical system matrices
        F = np.array([[-1. / taus, 0., Xm / (taur * D), omegar * Xm / D],
                      [0., -1. / taus, -omegar * Xm / D, Xm / (taur * D)],
                      [Xm / taur, 0., -1. / taur, -omegar],
                      [0., Xm / taur, omegar, -1. / taur]])

        G = Xr / D * Vdc / 2. * \
            np.array([[1., 0], [0., 1], [0., 0.], [0., 0.]]).dot(P)

        # Discretize physical system
        A_phys = sla.expm(F * Tspu)
        B_phys = -(nla.inv(F).dot(np.eye(A_phys.shape[0]) - A_phys).dot(G))

        # Concatenate oscillating states
        A_osc = np.array([[np.cos(Tspu), -np.sin(Tspu)],
                          [np.sin(Tspu), np.cos(Tspu)]])
        B_osc = np.zeros((2, 3))

        # Concatenate previous input as a state
        A_prev = np.zeros((3, 3))
        B_prev = np.eye(3)

        # Concatenate filter states
        a1 = 1. - 1. / k1
        a2 = 1. - 1. / k2

        A_sw = np.array([[a1, 0.],
                         [(1. - a1), a2]])
        # NB 1: Please note the 1 / 12 division to average over all the physical switches
        # NB 2: Please note the 1/fsw_des division to normalize switching
        # frequency
        B_sw = 1. / fsw_des * 1. / 12. * (1 - a1) / Ts * \
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
                      [0., 1., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., ],
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
        M[0, :] = np.hstack(
            (-np.ones((1, 9)), np.zeros((1, 9)), np.ones((1, 9))))
        M[1, :] = np.tile(np.hstack((-np.ones((1, 3)), np.zeros((1, 3)),
                                     np.ones((1, 3)))), (1, 3))
        M[2, :] = np.tile(np.array([-1, 0, 1]), (1, 9))

        '''
        Define system attributes
        '''
        self.A = A
        self.B = B
        self.C = C
        self.W = W
        self.G = G
        self.T = T
        self.M = M
        self.fsw_des = fsw_des
        self.delta = delta


class InitialConditions(object):
    """
    Power converter initial conditions

    Attributes:
        x0 (array): initial state for simulations
        cur_step_torque (array): currents when there is a torque step
    """

    def __init__(self, params):
        """
        Setup initial conditions
        """
        torque = params.torque

        # Compute steady state values with T
        x0 = self.get_initial_states(params, torque, 1)
        x_null_torque = self.get_initial_states(params, 0., 1.)

        # Previous input
        uprev = np.array([0., 0., 0.])

        r'''
        Current at the torque step

        Te = 0 =>
                $i_{s\beta} = i_{s\alpha} \frac{\phi_{r\beta}}{\phi_{r\alpha}}$

        '''
        cur_step_torque = np.array([-x_null_torque[0],
                                    -x_null_torque[0] * x0[3] / x0[2]])

        self.x0 = np.concatenate((x0, x0[:2], uprev, np.array([1., 1., 1.])))
        self.cur_step_torque = cur_step_torque

    def get_initial_states(self, params, T, psiS_mag):
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
        Rs = params.Rs
        Rr = params.Rr
        Xss = params.Xs
        Xrr = params.Xr
        Xm = params.Xm
        D = params.D
        kT = params.kT  # Torque constant to correct [pu]

        # Stator flux components
        psiSa = psiS_mag
        psiSb = 0

        # PsiR alpha and beta components
        psiRb = -T / psiS_mag * D / Xm / kT
        dis = np.sqrt((Xm**2) * (psiSa**2) - 4. * (Xss**2) * (psiRb**2))
        psiRa1 = (Xm * psiSa + dis) / (2. * Xss)
        psiRa2 = (Xm * psiSa - dis) / (2 * Xss)

        psiRa = psiRa1  # make sure that this is always the correct choice!

        # Slip frequency
        ws = -Rr * Xss / D * psiRb / psiRa

        # Build stator and rotor flux vectors
        PsiS = np.array([psiSa,  psiSb])
        PsiR = np.array([psiRa, psiRb])

        # Initialize the transformation matrix M
        M = 1. / D * np.array([[Xrr, 0, -Xm, 0.],
                               [0, Xrr, 0, -Xm]])

        # Stator currents in alpha/beta
        Is = M.dot(np.append(PsiS, PsiR))

        # Initial state
        x0 = np.append(Is, PsiR)

        return x0


class Params(object):
    """
    Power converter model parameters
    """

    def __init__(self, Ts, freq, k1, k2, torque):
        """
        Compute model parameters given

        Args:
            Ts (double): Sampling time
            freq (double): Frequency
            k1 (double): filter constant
            k2 (double): filter constant
            torque (double): motor torque
        """

        # Resistance
        Rs = 0.0108  # Stator
        Rr = 0.0091  # Rotor

        # Inductances
        Xls = 0.1493  # Stator
        Xlr = 0.1104  # Rotor
        Xm = 2.3489   # Mutual

        # Rotor speed
        omegar = 0.9911  # Nominal speed

        # Voltages
        Vdc = 1.930

        # Torque constant to fix [pu] units
        kT = 1.2361

        '''
        Define base model units
        '''
        omegab = 2 * math.pi * freq
        Tspu = Ts * omegab
        Nstpp = int(1. / freq / Ts)  # Number of steps per period

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
        P = 2. / 3. * np.array([[1.0, -1. / 2., -1. / 2.],
                                [0.0, np.sqrt(3.) / 2., -np.sqrt(3.) / 2.]])

        invP = np.array([[1., 0.],
                         [-1. / 2., np.sqrt(3.) / 2.],
                         [-1. / 2., -np.sqrt(3.) / 2.]])

        '''
        Store model parameters
        '''
        self.Rs = Rs
        self.Rr = Rr
        self.Xm = Xm
        self.Xs = Xs
        self.Xr = Xr
        self.D = D
        self.taus = taus
        self.taur = taur
        self.omegar = omegar
        self.omegab = omegab
        self.Vdc = Vdc
        self.kT = kT
        self.Tspu = Tspu
        self.Nstpp = Nstpp
        self.P = P
        self.invP = invP
        self.freq = freq
        self.Ts = Ts
        self.k1 = k1
        self.k2 = k2
        self.torque = torque


class Time(object):
    """
    Power converter time structure
    """

    def __init__(self, t0, Ts, init_periods, sim_periods, Nstpp):
        """
        Setup time object
        """
        self.init_periods = init_periods
        self.sim_periods = sim_periods
        self.Nstpp = Nstpp
        self.T_final = (init_periods + sim_periods) * Nstpp
        self.T_timing = sim_periods * Nstpp
        self.Ts = Ts
        self.t0 = t0
        self.t = np.linspace(t0, Ts * self.T_final, self.T_final + 1)


class Model(object):
    """
    Power converter model
    """

    def __init__(self):
        self.params = None
        self.dyn_system = None
        self.tail_cost = None
        self.init_conditions = None
        self.time = None
        self.qp_matrices = None
        self.solver = None

    def set_params(self, Ts, freq, k1, k2, torque):
        self.params = Params(Ts, freq, k1, k2, torque)

    def set_time(self, t0, init_periods, sim_periods):
        """
        Set simulation time structure
        """
        self.time = Time(t0, self.params.Ts, init_periods, sim_periods,
                         self.params.Nstpp)

    def set_initial_conditions(self):
        """
        Set power converter initial conditions
        """
        self.init_conditions = InitialConditions(self.params)

    def gen_dynamical_system(self, fsw_des, delta):
        """
        Generate dynamical system given parameters and
        """
        self.dyn_system = DynamicalSystem(self.params, fsw_des, delta)

    def gen_tail_cost(self, N_tail, gamma, name=None):
        '''
        Compute or load tail cost
        '''
        self.tail_cost = TailCost(self.dyn_system, gamma)

        if name is not None:
            self.tail_cost.load(name)
        else:
            self.tail_cost.compute(self.dyn_system, N_tail)

    def compute_mpc_input(self, x0, u_prev, solver='gurobi'):
        """
        Compute MPC input at initial state x0 with specified solver
        """
        qp = self.qp_matrices

        N = qp.N

        # Update objective
        q = 2. * (qp.q_x.dot(x0) + qp.q_u)

        # Update bounds
        SA_tildex0 = qp.SA_tilde.dot(x0)
        qp.u[:6 * N] = SA_tildex0
        # qp.l[:6 * N] = -SA_tildex0

        if solver == 'gurobi':
            # Solve problem
            prob = mpbpy.QuadprogProblem(qp.P, q, qp.A, qp.l, qp.u, qp.i_idx,
                                         qp.i_l, qp.i_u, x0=u_prev)
            res_gurobi = prob.solve(solver=mpbpy.GUROBI, verbose=False,
                                    Threads=1)
            u = res_gurobi.x
            obj_val = res_gurobi.obj_val
            solve_time = res_gurobi.cputime

        elif solver == 'miosqp':

            if self.solver is None:
                # Define problem settings
                miosqp_settings = {'eps_int_feas': 1e-02,   # integer feasibility tolerance
                                   'max_iter_bb': 2000,     # maximum number of iterations
                                   'tree_explor_rule': 1,   # tree exploration rule
                                                            #   [0] depth first
                                                            #   [1] two-phase: depth first  until first incumbent and then  best bound
                                   'branching_rule': 0,     # branching rule
                                                            #   [0] max fractional part
                                   'verbose': False,
                                   'print_interval': 1}

                osqp_settings = {'eps_abs': 1e-03,
                                 'eps_rel': 1e-03,
                                 'eps_prim_inf': 1e-04,
                                 #  'rho': 0.001,
                                 #  'rho': 0.1,
                                 'verbose': False}
                self.solver = miosqp.MIOSQP()
                self.solver.setup(qp.P, q, qp.A, qp.l,
                                  qp.u, qp.i_idx, qp.i_l, qp.i_u,
                                  miosqp_settings,
                                  osqp_settings)
            else:
                self.solver.update_vectors(q, qp.l, qp.u)

            self.solver.set_x0(u_prev)
            res_miosqp = self.solver.solve()

            # import ipdb; ipdb.set_trace()

            # DEBUG Check if gurobi gives same solution
            # N.B. They do not match when the norm of the
            #      difference of the objective functions
            #      is below the tolerance
            #
            # prob = mpbpy.QuadprogProblem(qp.P, q, qp.A, qp.l, qp.u, qp.i_idx)
            # res_gurobi = prob.solve(solver=mpbpy.GUROBI, verbose=False, x0=u_prev)
            # if np.linalg.norm(res_miosqp.x - res_gurobi.x)> 1e-02:
            #     print("Norm of difference of solution = %.4e" % \
            #           np.linalg.norm(res_miosqp.x - res_gurobi.x))
                # import ipdb; ipdb.set_trace()


            if res_miosqp.status != miosqp.MI_SOLVED:
                import ipdb; ipdb.set_trace()
            u = res_miosqp.x
            obj_val = res_miosqp.upper_glob
            solve_time = res_miosqp.run_time
            osqp_solve_time = 100 * res_miosqp.osqp_solve_time / res_miosqp.run_time

        # Get first input
        u0 = u[:6]

        if solver == 'miosqp':
            return u0, obj_val, solve_time, u, \
                    osqp_solve_time, \
                    res_miosqp.osqp_iter_avg
        else:
            return u0, obj_val, solve_time, u, 0, 0

    def simulate_one_step(self, x, u):
        """
        Simulate power converter for one step
        """
        xnew = self.dyn_system.A.dot(x) + self.dyn_system.B.dot(u)
        ynew = self.dyn_system.C.dot(x)

        return xnew, ynew

    def compute_signals(self, X):
        """
        Compute signals for plotting
        """

        T_final = self.time.T_final

        # Phase currents
        Y_phase = np.zeros((3, T_final))
        for i in range(T_final):
            Y_phase[:, i] = self.params.invP.dot(X[0:2, i])

        # Referente currents
        Y_star_phase = np.zeros((3, T_final))
        for i in range(T_final):
            Y_star_phase[:, i] = self.params.invP.dot(X[4:6, i])

        # Compute torque
        T_e = np.zeros(T_final)
        for i in range(T_final):
            T_e[i] = (self.params.Xm / self.params.Xr) * \
                (X[2, i] * X[1, i] - X[3, i] * X[0, i])

        T_e *= self.params.kT  # Torque constant normalization

        # Desired torque
        T_e_des = self.params.torque * np.ones(T_final)

        return Y_phase, Y_star_phase, T_e, T_e_des

    def get_statistics(self, results):
        """
        Get statistics of the results
        """

        # Get results
        U = results.U
        Y_phase = results.Y_phase

        # Get switching frequency
        init_periods = self.time.init_periods
        sim_periods = self.time.sim_periods
        Nstpp = self.params.Nstpp
        T_final = self.time.T_final
        N_sw = np.zeros(12)  # Number of changes per semiconductor device

        for i in range(init_periods * Nstpp, T_final):
            # Compute ON transitions for each stage of the simulation
            N_sw += utils.compute_on_transitions(U[:3, i], U[:3, i-1])

        freq_sw = N_sw / (1. / self.params.freq * sim_periods)
        fsw = np.mean(freq_sw)  # Compute average between 12 switches

        # Get THD
        t = self.time.t
        t_init = init_periods * Nstpp
        freq = self.params.freq

        thd = utils.get_thd(Y_phase[:, t_init:].T, t[t_init + 1:], freq)

        # Get solve times statustics
        max_solve_time = np.max(results.solve_times)
        min_solve_time = np.min(results.solve_times)
        avg_solve_time = np.mean(results.solve_times)
        std_solve_time = np.std(results.solve_times)

        return Statistics(fsw, thd,
                          max_solve_time, min_solve_time,
                          avg_solve_time, std_solve_time)

    def simulate_cl(self, N, steady_trans, solver='gurobi', plot=False):
        """
        Perform closed loop simulation
        """

        print("Simulating closed loop N = %i with solver %s" %
              (N, solver))

        # Reset solver
        self.solver = None

        if solver == 'miosqp':
            # If miosqp, set avg numer of iterations to 0
            miosqp_avg_osqp_iter = 0
            miosqp_osqp_avg_time = 0

        # Rename some variables for notation ease
        nx = self.dyn_system.A.shape[0]
        nu = self.dyn_system.B.shape[1]
        ny = self.dyn_system.C.shape[0]
        T_final = self.time.T_final
        T_timing = self.time.T_timing

        # Compute QP matrices
        self.qp_matrices = MIQP(self.dyn_system, N, self.tail_cost)

        # Preallocate vectors of results
        X = np.zeros((nx, T_final + 1))
        U = np.zeros((nu, T_final))
        Y = np.zeros((ny, T_final))
        solve_times = np.zeros(T_timing)  # Computing times
        obj_vals = np.zeros(T_final)     # Objective values

        # Set initial statte
        X[:, 0] = self.init_conditions.x0

        # Temporary previous MIQP solution
        u_prev = np.zeros(nu * N)

        # Run loop
        for i in tqdm(range(T_final)):

            # Compute mpc inputs
            U[:, i], obj_vals[i], time_temp, u_prev, osqp_time, osqp_iter = \
                self.compute_mpc_input(X[:, i], u_prev, solver=solver)

            # Store time if after the init periods
            if i >= self.time.init_periods * self.time.Nstpp:
                solve_times[i - self.time.init_periods * self.time.Nstpp] = \
                        time_temp

            # Simulate one step
            X[:, i + 1], Y[:, i] = self.simulate_one_step(X[:, i], U[:, i])

            # Shift u_prev
            u_prev = np.append(u_prev[nu:], u_prev[-nu:])

            if solver == 'miosqp':
                # Append average number of osqp iterations
                miosqp_avg_osqp_iter += osqp_iter
                miosqp_osqp_avg_time += osqp_time

        if solver == 'miosqp':
            # Divide total number of average OSQP iterations 
            # and solve time by time steps
            miosqp_avg_osqp_iter /= T_final
            miosqp_osqp_avg_time /= T_final

        # Compute additional signals for plotting
        Y_phase, Y_star_phase, T_e, T_e_des = self.compute_signals(X)

        # Create simulation results
        results = SimulationResults(X, U, Y_phase, Y_star_phase, T_e, T_e_des,
                                    solve_times)

        if plot:
            # Plot results
            utils.plot(results, self.time)

        # Get statistics
        stats = self.get_statistics(results)

        if solver == 'miosqp':
            stats.miosqp_avg_osqp_iter = miosqp_avg_osqp_iter
            stats.miosqp_osqp_avg_time = miosqp_osqp_avg_time

        return stats
