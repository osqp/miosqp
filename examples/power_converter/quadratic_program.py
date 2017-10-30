import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import scipy.sparse as spa


class MIQP(object):
    """
    Mixed-Integer Quadratic Program matrices and vectors
    """
    def __init__(self, dyn_system, N, tail_cost):
        """
        Generate matrices in for QP solvers
        """

        # Get internal variables
        nx = dyn_system.A.shape[0]
        nu = dyn_system.B.shape[1]

        # Get matrices
        A = dyn_system.A
        B = dyn_system.B
        C = dyn_system.C
        W = dyn_system.W
        G = dyn_system.G
        T = dyn_system.T

        # Get tail cost
        gamma = tail_cost.gamma
        P0 = tail_cost.P0
        q0 = tail_cost.q0
        r0 = tail_cost.r0

        # A_tilde matrix
        A_tilde = np.eye(nx)
        for i in range(1, N + 1):
            A_tilde = np.vstack((A_tilde, nla.matrix_power(A, i)))
        A_tilde_end = nla.matrix_power(A, N)

        # B_tilde matrix
        B_tilde = np.zeros((nx, nu * N))
        for i in range(1, N + 1):
            temp = nla.matrix_power(A, i - 1) * B
            for j in range(2, N + 1):
                if j <= i:
                    temp = np.hstack((temp, nla.matrix_power(A, i - j) * B))
                else:
                    temp = np.hstack((temp, np.zeros((nx, nu))))

            B_tilde = np.vstack((B_tilde, temp))

        B_tilde_end = B_tilde[-nx:, :]

        '''
        Define cost function
        '''
        # H matrices to define stage and input costs
        diagHx = C.T.dot(C)
        Hx = diagHx
        for i in range(2, N + 1):
            Hx = sla.block_diag(Hx, (gamma**(i - 1)) * diagHx)

        Hx = sla.block_diag(Hx, np.zeros((nx, nx)))

        # Quadratic form matrices
        qp_P = spa.csc_matrix(2. * (B_tilde.T.dot(Hx).dot(B_tilde) +
                              (gamma**N) *
                              B_tilde_end.T.dot(P0).dot(B_tilde_end)))
        qp_q_x = B_tilde.T.dot(Hx.T).dot(A_tilde) + \
            (gamma ** N) * (B_tilde_end).T.dot(P0).dot(A_tilde_end)
        qp_q_u = (gamma ** N) * B_tilde_end.T.dot(q0)

        # Constant part of the cost function
        qp_const_P = A_tilde.T.dot(Hx).dot(A_tilde) + \
            (gamma ** N) * A_tilde_end.T.dot(P0).dot(A_tilde_end)
        qp_const_q = ((gamma ** N) * q0.T.dot(A_tilde_end))
        qp_const_r = (gamma ** N) * r0

        '''
        Define constraints
        '''

        # Matrices required for constraint satisfaction S, R, T
        S1 = np.hstack((np.kron(np.eye(N), W), np.zeros((3 * N, nx))))
        S2 = np.hstack((np.kron(np.eye(N), -W), np.zeros((3 * N, nx))))
        S = np.vstack((S1, S2))

        R = np.vstack((np.kron(np.eye(N), G - T), np.kron(np.eye(N), -G - T)))
        F = np.kron(np.eye(N), T)

        # Linear constraints
        qp_A = spa.csc_matrix(np.vstack((R - S.dot(B_tilde), F)))

        # upper bound
        qp_u = np.append(np.zeros(6 * N), np.ones(3 * N))

        # lower bound
        qp_l = np.append(-np.inf * np.ones(6 * N), -np.ones(3 * N))

        # Constrain bounds to be within -1 and 1
        #  u_sw_idx = np.append(np.ones(3), np.zeros(3))
        #  u_sw_idx = np.tile(u_sw_idx, N)
        #  u_sw_idx = np.flatnonzero(u_sw_idx)
        #  qp_A, qp_l, qp_u = self.add_bounds(u_sw_idx, -1., 1.,
        #                                     qp_A, qp_l, qp_u)
       
        # SA_tilde needed to update bounds
        qp_SA_tilde = S.dot(A_tilde)

        # Index of integer variables
        i_idx = np.arange(nu * N)

        # Bounds on integer variables
        #  i_idx = u_sw_idx
        i_l = -1. * np.ones(nu * N)
        i_u = 1. * np.ones(nu * N)

        '''
        Define problem matrices
        '''

        # Inequality matrix
        self.P = qp_P
        self.A = qp_A
        self.q_x = np.asarray(qp_q_x)
        self.q_u = np.asarray(qp_q_u).flatten()
        self.u = qp_u
        self.l = qp_l
        self.SA_tilde = qp_SA_tilde
        self.const_P = qp_const_P
        self.const_q = qp_const_q
        self.const_r = qp_const_r
        self.N = N
        self.i_idx = i_idx
        self.i_l = i_l
        self.i_u = i_u

    #  def add_bounds(self, i_idx, l_new, u_new, A, l, u):
    #      """
    #      Add new bounds on the variables
    #
    #          l_new <= x_i <= u_new for i in i_idx
    #
    #      It is done by adding rows to the contraints
    #
    #          l <= A x <= u
    #      """
    #
    #      n = A.shape[1]
    #
    #      # Enforce integer variables to be binary => {0, 1}
    #      I_int = spa.identity(n).tocsc()
    #      I_int = I_int[i_idx, :]
    #      l_int = np.empty((n,))
    #      l_int.fill(l_new)
    #      l_int = l_int[i_idx]
    #      u_int = np.empty((n,))
    #      u_int.fill(u_new)
    #      u_int = u_int[i_idx]
    #      A = spa.vstack([A, I_int]).tocsc()      # Extend problem constraints matrix A
    #      l = np.append(l, l_int)         # Extend problem constraints
    #      u = np.append(u, u_int)         # Extend problem constraints
    #
    #      return A, l, u
