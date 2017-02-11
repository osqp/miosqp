"""
Solve Mixed-Integer QPs using Branch and Bound and OSQP Solver


Written by Bartolomeo Stellato, February 2017, University of Oxford
"""

from __future__ import print_function, division
import osqp  # Import OSQP Solver
import numpy as np
# import numpy.linalg as la
import scipy.sparse as spa
from time import time


import ipdb

# Solver statuses
MI_UNSOLVED = 10
MI_SOLVED = 11
MI_INFEASIBLE = 9
MI_UNBOUNDED = 8
MI_MAX_ITER_FEASIBLE = 12
MI_MAX_ITER_UNSOLVED = 12


# Nodes statuses
MI_NODE_PENDING = 100
MI_NODE_SOLVED = 101
MI_NODE_FATHOMED = 99

class Data(object):
    """
    Data for the relaxed qp problem in the form

        min    1/2 x' P x + q' x
        s.t.   l <= A x <= u

        where l = [l_orig]   and u = [u_orig] \\in R^{m + len(i_idx)}
                  [ -inf ]           [ +inf ]
        and A = [A_orig] \\in R^{m + len(i_idx) \\times n}
                [  I   ]
        are the newly introduced constraints to deal with integer variables

    Attributes
    ----------
    n: int
        number of variables
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

    def __init__(self, P, q, A, l, u, i_idx):
        # MIQP problem dimensions
        self.n = A.shape[1]
        self.m = A.shape[0]
        self.n_int = len(i_idx)   # Number of integer variables

        #
        # Extend problem with new constraints to accomodate integral constraints
        #
        I_int = spa.identity(self.n).tocsc()
        I_int = I_int[i_idx, :]     # Extend constraints matrix A with only the rows of
                                    # the identity relative to the integer variables

        # Extend the bounds only for the variables which are integer
        l_int = np.empty((self.n,))
        l_int.fill(-np.inf)
        l_int = l_int[i_idx]
        u_int = np.empty((self.n,))
        u_int.fill(np.inf)
        u_int = u_int[i_idx]

        self.A = spa.vstack([A, I_int]).tocsc()      # Extend problem constraints matrix A
        self.l = np.append(l, l_int)         # Extend problem constraints
        self.u = np.append(u, u_int)         # Extend problem constraints

        #
        # Define problem cost function
        #
        self.P = P.tocsc()
        self.q = q

        # Define index of integer variables
        self.i_idx = i_idx

    def compute_obj_val(self, x):
        """
        Compute objective value at x
        """
        return .5 * np.dot(x, self.P.dot(x)) + np.dot(self.q, x)

class Node:
    """
    Branch-and-bound node class

    Attributes
    ----------
    lower: double
        node's lower bound
    upper: double
        node's upper bound
    n: int
        depth in the tree
    l: numpy array
        vector of lower bounds in relaxed QP problem
    u: numpy array
        vector of upper bounds in relaxed QP problem
    x: numpy array
        node's integer (possibly not feasible) MIQP solution
    x_relaxed: numpy array
        node's relaxed solution. At the beginning it is the warm-starting value x0
    y_relaxed: numpy array
        node's relaxed solution. At the beginning it is the warm-starting value y0
    status: int
        node's status
    qp_status: int
        qp solver return status
    nextvar: int
        index of next variable to split upon (position within i_idx vector)
    left: Node
        left child
    right: Node
        right child
    work: Workspace
        MIQP solver workspace
    """

    def __init__(self, l, u, n, work,
                 x0=None, y0=None):
        """
        Initialize node class
        """

        # Set bounds
        self.lower = -np.inf
        self.upper = np.inf

        # Set depth
        self.n = n

        # Set l and u for relaxed QP problem
        self.l = l
        self.u = u

        # Assign solver Workspace
        self.work = work

        # Integer MIQP solution
        self.x = None

        # Warm-start variables which are also the relaxed solutions
        if x0 is None:
            x0 = np.zeros(self.work.data.n)
        if y0 is None:
            y0 = np.zeros(self.work.data.m + self.work.data.n_int)
        self.x_relaxed = x0
        self.y_relaxed = y0

        # Set node status
        self.status = MI_NODE_PENDING

        # Set QP solver return status
        self.qp_status = self.work.solver.constant('OSQP_UNSOLVED')

        # Next variable to split on
        self.nextvar = None

        # Predefine left and right children nodes
        self.left = None
        self.right = None

    def nodes(self):
        """
        Returns a list of all non fathomed nodes at or below this point
        """
        nodes_list = []
        if self.status != MI_NODE_FATHOMED:
            nodes_list += [self]
        if self.left is not None and self.left.status != MI_NODE_FATHOMED:
            nodes_list += self.left.nodes()
        if self.right is not None and self.right.status != MI_NODE_FATHOMED:
            nodes_list += self.right.nodes()

        return nodes_list

    def add_left(self):
        """
        Add node to the left-hand side of the tree and solve it
        """
        # Get next variable constraint idx
        constr_idx = self.work.data.m + self.nextvar

        # Get next variable idx
        nextvar_idx = self.work.data.i_idx[self.nextvar]

        # Get updated QP interval bounds
        l_left = np.copy(self.l)
        u_left = np.copy(self.u)

        # Add constraint if it make the feasible region smaller (min())
        # x <= floor(x_relaxed)
        u_left[constr_idx] = min(u_left[constr_idx],
                                 np.floor(self.x_relaxed[nextvar_idx]))

        # Constrain it with lower bound
        u_left[constr_idx] = max(u_left[constr_idx], l_left[constr_idx])

        # print("Branch left: x[%i] <= %.4f\n" % (nextvar_idx, u_left[constr_idx]))

        if self.left is None:
            # Create node with new bounds
            self.left = Node(l_left, u_left, self.n + 1, self.work,
                             self.x_relaxed, self.y_relaxed)
            # Get bounds from that node
            self.left.get_bounds()

        return self.left


    def add_right(self):
        """
        Add node to the right-hand side of the tree and solve it
        """
        # Get next variable constraint idx
        constr_idx = self.work.data.m + self.nextvar

        # Get next variable idx
        nextvar_idx = self.work.data.i_idx[self.nextvar]

        # Get updated QP interval bounds
        l_right = np.copy(self.l)
        u_right = np.copy(self.u)

        # Add constraint if it make the feasible region smaller (max())
        # ceil(x_relaxed) <= x
        l_right[constr_idx] = max(l_right[constr_idx],
                                 np.ceil(self.x_relaxed[nextvar_idx]))

        # Constrain it with upper bound
        l_right[constr_idx] = min(l_right[constr_idx], u_right[constr_idx])

        # print("Branch right: %.4f <= x[%i] \n" % (u_right[constr_idx], nextvar_idx))

        if self.right is None:
            # Create node with new bounds
            self.right = Node(l_right, u_right, self.n + 1, self.work,
                              self.x_relaxed, self.y_relaxed)
            # Get bounds from that node
            self.right.get_bounds()

        return self.right

    def is_within_bounds(self, x):
        """
        Check if current solution is within bounds
        """

        # Check if it satisfies current l and u bounds
        z = self.work.data.A.dot(x)
        if any(z < self.l) | any(z > self.u):
            return False

        # If we got here, it is integer feasible
        return True


    def is_int_feas(self, x):
        """
        Check if current solution is integer feasible
        """

        # Check if integer variables are feasible up to tolerance
        x_int = x[self.work.data.i_idx]
        if any(abs(x_int - np.round(x_int))) > self.work.settings['eps_int_feas']:
            return False

        # If we got here, it is integer feasible
        return True

    def get_integer_solution(self, x):
        """
        Round obtained solution to integer feasibility
        """
        x_int = np.copy(x)
        x_int[self.work.data.i_idx] = np.round(x[self.work.data.i_idx])
        return x_int

    def pick_nextvar(self, x, branching_rule):
        """
        Pick next variable to branch upon
        """

        # Part of solution vector that is supposed to be integer
        x_int = x[self.work.data.i_idx]

        if branching_rule == 0:

            # Get vector of fractional parts
            fract_part = abs(x_int - np.round(x_int))

            # Get index of max fractional part
            next_idx = np.argmax(fract_part)

            # Get index of next variable as position in the i_idx vector
            self.nextvar = next_idx
        else:
            raise ValueError('No variable selection rule recognized!')

    def get_bounds(self):
        """
        Find upper and lower bounds for the relaxed problem corresponding to this node
        """
        # Update l and u in the solver instance
        self.work.solver.update(l=self.l, u=self.u)

        # Warm start solver with currently stored solution
        self.work.solver.warm_start(x=self.x_relaxed, y=self.y_relaxed)

        # Solve current problem
        results = self.work.solver.solve()

        # Store solver status
        self.qp_status = results.info.status_val

        # Check if maximum number of iterations reached
        if (self.qp_status == \
            self.work.solver.constant('OSQP_MAX_ITER_REACHED')):
            print("ERROR: Max Iter Reached!")
            ipdb.set_trace()

        # Check if infeasible or unbounded -> Node becomes fathomed
        if (self.qp_status == \
            self.work.solver.constant('OSQP_INFEASIBLE')) | \
            self.qp_status == \
            self.work.solver.constant('OSQP_UNBOUNDED'):
            self.status = MI_NODE_FATHOMED
            return
        else:
            self.status = MI_NODE_SOLVED
            self.x_relaxed = results.x
            self.y_relaxed = results.y

        # Check if integer feasible solution withing l and u bounds
        if (self.is_int_feas(self.x_relaxed) and \
            self.is_within_bounds(self.x_relaxed)):
            #   -> update bounds, status and variable x
            self.lower = results.info.obj_val
            self.upper = results.info.obj_val
            self.status = MI_NODE_FATHOMED
            self.x = self.x_relaxed
            return

        # Get lower bound (objective value of relaxed problem)
        self.lower = results.info.obj_val

        # Compute upper bound (get integer solution)
        self.x = self.get_integer_solution(self.x_relaxed)

        #   - check feasibility
        if self.is_within_bounds(self.x):
            #   - if rounded is feasible --> compute new upper bound
            self.upper = self.work.data.compute_obj_val(self.x)
        else:
            self.x = None

        # Pick next variable to choose
        self.pick_nextvar(self.x_relaxed,
                          self.work.settings['branching_rule'])


class Workspace(object):
    """
    Workspace class

    Attributes
    ----------
    data: class
        miqp problem data
    settings: dictionary
        branch and bound settings
    qp_settings: dictionary
        branch and bound settings
    solver: class
        QP solver class
    root: class
        root node of the tree
    leaves: list
        leaves in the tree
    non_fathomed_leaves: list
        leaves that can still be branched

    Other internal variables
    ------------------------
    iter_num: int
        number of iterations
    explored_nodes: int
        number of nodes explored
    run_time: double
        runtime of the algorithm
    upper_glob: double
        global upper bound
    lower_glober: double
        global lower bound
    status: int
        MIQP solver status
    x: numpy array
        current best solution
    """
    def __init__(self, data, settings, qp_settings=None):
        self.data = data
        self.settings = settings

        # Setup OSQP solver instance
        self.solver = osqp.OSQP()
        if qp_settings is None:
            qp_settings = {}
        self.solver.setup(self.data.P, self.data.q, self.data.A,
                          self.data.l, self.data.u, **qp_settings)


        # Define root node
        self.root = Node(self.data.l, self.data.u, 0, self)

        # Define leaves at the beginning (only root)
        self.leaves = [self.root]
        self.not_fathomed_leaves = [self.root]

        # Define other internal variables
        self.iter_num = 1
        self.upper_glob = np.inf
        self.lower_glob = -np.inf
        self.status = MI_UNSOLVED
        self.x = np.empty(self.data.n)


    def can_continue(self):
        """
        Check if the solver can continue
        """
        check = self.upper_glob - self.lower_glob > \
            self.settings['eps_bb_abs']
        check &= (self.upper_glob - self.lower_glob)/abs(self.lower_glob) > \
                 self.settings['eps_bb_rel']
        check &= self.iter_num < self.settings['max_iter_bb']

        # Get branchable leaves. Check if there is at least one to continue
        self.not_fathomed_leaves = [leaf for leaf in self.leaves \
                                    if leaf.status != MI_NODE_FATHOMED]
        check &= any(self.not_fathomed_leaves)

        return check


    def choose_leaf(self, tree_explor_rule):
        """
        Choose next leaf to branch from the ones that can still be expanded
        depending on branch_rule
        """
        if tree_explor_rule == 0:
            # Depth first: Choose leaf with highest depth
            leaf = np.argmax([leaf.n for leaf in self.leaves])

            # Old stuff
            # # Choose leaf with lowest lower bound between leaves which
            # # can be expanded
            # min_lower = min([leaf.lower for leaf in \
            #                  self.not_fathomed_leaves])
            # for x in self.leaves:
            #     if x.lower == min_lower:
            #         leaf = x
        else:
            raise ValueError('Tree exploring strategy not recognized')
        return leaf


    def branch(self, leaf):
        """
        Branch
            - Expand leaf within leaves list in branch and bound tree.
            - Solve the relaxed problems in the right and left children
            - Obtain new global upper and global lower bounds
        """
        left = leaf.add_left()
        right = leaf.add_right()
        self.leaves.remove(leaf)
        self.leaves += [left, right]

        # Update lower and upper bound
        self.lower_glob = min([x.lower for x in self.leaves])
        # TODO: Can't the lower_glob update be only a min between current
        #       lower_glob and the new ones for the leaves?
        #       (just like the upper bound?)

        # Update upper bound
        upper_bounds = np.array([self.upper_glob, left.upper, right.upper])
        upper_idx = np.argmin(upper_bounds)
        self.upper_glob = upper_bounds[upper_idx]

        # if uppwer bound improved -> Store node solution x
        if upper_idx == 1:
            self.x = left.x # Update solution
        elif upper_idx == 2:
            self.x = right.x # Update solution


    def bound(self):
        """
        Bound
            - prune tree nodes if their lower value is greater than the current
        upper bound
        """
        for node in self.root.nodes():
            if node.lower > self.upper_glob:
                node.status = MI_NODE_FATHOMED

    def init_root(self):
        """
        Initialize root node and get bounds
        """
        # Get bounds from root node
        self.root.get_bounds()
        if self.root.status == MI_NODE_FATHOMED:
            # Root node infeasible or unbounded
            if self.root.qp_status == self.solver.constant('OSQP_INFEASIBLE'):
                self.status = MI_INFEASIBLE
                return
            if self.root.qp_status == self.solver.constant('OSQP_UNBOUNDED'):
                self.status = MI_UNBOUNDED
                return

        # Update global bounds
        self.upper_glob = self.root.upper
        self.lower_glob = self.root.lower

        # Update global solution if integer feasible
        if self.root.x is not None:
            self.x = self.root.x

    def get_return_status(self):
        """
        Get return status for MIQP solver
        """
        if self.iter_num < self.settings['max_iter_bb']:  # Finished

            if self.upper_glob != np.inf:
                self.status = MI_SOLVED
            else:
                if self.upper_glob >= 0:  # +inf
                    self.status = MI_INFEASIBLE
                else:                     # -inf
                    self.status = MI_UNBOUNDED

        else:  # Hit maximum number of iterations
            if self.upper_glob != np.inf:
                self.status = MI_MAX_ITER_FEASIBLE
            else:
                if self.upper_glob >= 0:  # +inf
                    self.status = MI_MAX_ITER_UNSOLVED
                else:                     # -inf
                    self.status = MI_UNBOUNDED


    def print_headline(self):
        """
        Print headline
        """
        print("     Nodes      |     Current Node     |   Objective Bounds   |     Work")
        print("  Expr Unexplr  |  Obj  Depth  IntInf  |  Lower  Upper   Gap  |  Iter  Time")


    def print_progress(self, leaf):
        """
        Print progress at each iteration
        """
        print("iter %.3d   lower bound: %.5f, upper bound %.5f" %
              (self.iter_num, self.lower_glob, self.upper_glob))

    def solve(self):
        """
        Solve MIQP problem. This is the actual branch-and-bound algorithm
        """

        # Print header
        self.print_headline()

        # Initialize root node and get bounds
        self.init_root()

        # Loop tree until the cost function gap has disappeared
        while any(self.leaves):

            # 1) Choose leaf
            #   -> Use tree exploration rule
            leaf = self.choose_leaf(self.settings['tree_explor_rule'])

            # 2) Branch leaf
            #   -> Solve children
            #   -> Update lower and upper bounds
            self.branch(leaf)

            # 3) Bound
            #   -> prune nodes with lower bound above upper bound
            self.bound()

            # Print progress
            self.print_progress(leaf)

            # Update iteration number
            self.iter_num += 1

            # Stop
            # pdb.set_trace()


        print("Done!")
        # Get final status
        self.get_return_status()

        return


def miosqp_solve(P, q, A, l, u, i_idx):
    """
    Solve MIQP problem using MIOSQP solver
    """

    # Start timer
    start = time()


    # Define problem settings
    settings = {'eps_bb_abs': 1e-03,           # absolute convergence tolerance
                'eps_bb_rel': 1e-03,           # relative convergence tolerance
                'eps_int_feas': 1e-03,         # integer feasibility tolerance
                'max_iter_bb': 1000,           # maximum number of iterations
                'tree_explor_rule': 0,         # tree exploration rule
                                               #   [0] depth first
                'branching_rule': 0}           # branching rule
                                               #   [0] max fractional part

    qp_settings = {'eps_abs': 1e-04,
                   'eps_rel': 1e-04,
                   'eps_inf': 1e-03,
                   'rho': 0.1,
                   'sigma': 0.01,
                   'polishing': False,
                   'max_iter': 5000,
                   'verbose': False}

    # Create data class instance
    data = Data(P, q, A, l, u, i_idx)

    # Create Workspace
    work = Workspace(data, settings, qp_settings)

    # Solve problem
    work.solve()

    # Stop
    stop = time()
    work.run_time = stop - start
    print("Elapsed time: %.4es" % work.run_time)
    return work
