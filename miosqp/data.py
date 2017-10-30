import scipy.sparse as spa
import numpy as np


def add_bounds(i_idx, l_new, u_new, A, l, u):
    """
    Add new bounds on the variables

        l_new <= x_i <= u_new for i in i_idx

    It is done by adding rows to the contraints

        l <= A x <= u
    """

    n = A.shape[1]

    I_int = spa.identity(n).tocsc()
    I_int = I_int[i_idx, :]
    l_int = l_new
    u_int = u_new
    #  l_int = np.empty((n,))
    #  l_int.fill(l_new)
    #  l_int = l_int[i_idx]
    #  u_int = np.empty((n,))
    #  u_int.fill(u_new)
    #  u_int = u_int[i_idx]
    # Extend problem constraints matrix A
    A = spa.vstack([A, I_int]).tocsc()
    l = np.append(l, l_int)         # Extend problem constraints
    u = np.append(u, u_int)         # Extend problem constraints

    return A, l, u


class Data(object):
    """
    Data for the relaxed qp problem in the form

        min    1/2 x' P x + q' x
        s.t.   l <= A x <= u

        where l = [l_orig]   and u = [u_orig] \\in R^{m + len(i_idx)}
                  [  i_l ]           [ +i_u ]
        and A = [A_orig] \\in R^{m + len(i_idx) \\times n}
                [  I   ]
        are the newly introduced constraints to deal with integer variables

    Attributes
    ----------
    n: int
        number of variables
    n_idx: int
        number of integer variables
    i_u: numpy array
        Upper bound on integer variables
    i_l: numpy array
        Lower bound on integer variables
    m: int
        number of constraints in original MIQP problem
    P: scipy sparse matrix
        cost function matrix
    q: numpy array
        linear part of the cost
    A: scipy sparse matrix
        extended constraints matrix
    l: numpy array
        extended array of lower bounds
    u: numpy array
        extended array of upper bounds

    Methods
    -------
    compute_obj_val
        compute objective value
    """

    def __init__(self, P, q, A, l, u, i_idx, i_l, i_u):
        # MIQP problem dimensions
        self.n = A.shape[1]
        self.m = A.shape[0]
        self.n_int = len(i_idx)   # Number of integer variables

        # Extend problem with new constraints
        # to accomodate integral constraints
        self.A, self.l, self.u = add_bounds(i_idx, i_l, i_u, A, l, u)

        # Define problem cost function
        self.P = P.tocsc()
        self.q = q

        # Define index of integer variables
        self.i_idx = i_idx

        # Define bounds on integer variables
        self.i_l = i_l
        self.i_u = i_u

    def compute_obj_val(self, x):
        """
        Compute objective value at x
        """
        return .5 * np.dot(x, self.P.dot(x)) + np.dot(self.q, x)

    def update_vectors(self, q=None, l=None, u=None):
        """
        Update prblem vectors
        """

        # Update cost
        if q is not None:
            if len(q) != self.n:
                raise ValueError('Wrong q dimension!')
            self.q = q

        # Update lower bound
        if l is not None:
            if len(l) != self.m:
                raise ValueError('Wrong l dimension!')
            self.l[:self.m] = l

        # Update upper bound
        if u is not None:
            if len(u) != self.m:
                raise ValueError('Wrong u dimension!')
            self.u[:self.m] = u
