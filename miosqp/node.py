import numpy as np
import osqp


class Node(object):
    """
    Branch-and-bound node class

    Attributes
    ----------
    lower: double
        node's lower bound
    data: Data structure
        problem data
    depth: int
        depth in the tree
    frac_idx: array of int
        index within the i_idx vector of the elements of x that are still fractional
    l: numpy array
        vector of lower bounds in relaxed QP problem
    u: numpy array
        vector of upper bounds in relaxed QP problem
    x: numpy array
        node's relaxed solution. At the beginning it is the warm-starting value x0
    y: numpy array
        node's relaxed solution. At the beginning it is the warm-starting value y0
    status: int
        qp solver return status
    num_iter: int
        number of OSQP ADMM iterations
    osqp_solve_time: int
        time to solve QP problem
    solver: solver
        QP solver object instance
    nextvar_idx: int
        index of next variable within solution x
    constr_idx: int
        index of constraint to change for branching on next variable
    """

    def __init__(self, data, l, u, solver, depth=0, lower=None,
                 x0=None, y0=None):
        """
        Initialize node class
        """

        # Assign data structure
        self.data = data

        # Set l and u for relaxed QP problem
        self.l = l
        self.u = u

        # Assign solver
        self.solver = solver

        # Set depth
        self.depth = depth

        # Set bound
        if lower is None:
            self.lower = -np.inf
        else:
            self.lower = lower

        # Frac_idx, intin
        self.frac_idx = None
        self.intinf = None

        # Number of integer infeasible variables
        self.intinf = None

        # Number of OSQP ADMM iterations
        self.num_iter = 0

        # Time to solve the QP
        self.osqp_solve_time = 0

        # Warm-start variables which are also the relaxed solutions
        if x0 is None:
            x0 = np.zeros(self.data.n)
        if y0 is None:
            y0 = np.zeros(self.data.m + self.data.n_int)
        self.x = x0
        self.y = y0

        # Set QP solver return status
        self.status = osqp.constant('OSQP_UNSOLVED')

        # Add next variable elements
        self.nextvar_idx = None   # Index of next variable within solution x
        # Index of constraint to change for branching
        # on next variable
        self.constr_idx = None

    def solve(self):
        """
        Find lower bound of the relaxed problem corresponding to this node
        """

        # Update l and u in the solver instance
        self.solver.update(l=self.l, u=self.u)

        # Warm start solver with currently stored solution
        self.solver.warm_start(x=self.x, y=self.y)

        # Solve current problem
        results = self.solver.solve()

        # Store solver status
        self.status = results.info.status_val

        # DEBUG: Problems that hit max_iter are infeasible
        # if self.status == osqp.constant('OSQP_MAX_ITER_REACHED'):
        #     self.status = osqp.constant('OSQP_PRIMAL_INFEASIBLE')

        # Store number of iterations
        self.num_iter = results.info.iter

        # Store solve time
        self.osqp_solve_time = results.info.run_time

        # Store solver solution
        self.x = results.x
        self.y = results.y

        # Enforce integer variables to be exactly within the bounds
        if self.status == osqp.constant('OSQP_SOLVED') or \
                self.status == osqp.constant('OSQP_MAX_ITER_REACHED'):
            #  import ipdb; ipdb.set_trace()
            n_int = self.data.n_int
            i_idx = self.data.i_idx
            self.x[i_idx] = \
                np.minimum(np.maximum(self.x[i_idx],
                                      self.l[-n_int:]),
                           self.u[-n_int:])
            #  if any(self.x[i_idx] < self.l[-n_int:]):
            #      import ipdb; ipdb.set_trace()
            #  if any(self.x[i_idx] > self.u[-n_int:]):
            #      import ipdb; ipdb.set_trace()

            # Update objective value of relaxed problem (lower bound)
            self.lower = self.data.compute_obj_val(self.x)

        #  # Get lower bound (objective value of relaxed problem)
        #  self.lower = results.info.obj_val

