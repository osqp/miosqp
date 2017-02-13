"""
Solve Mixed-Integer QPs using Branch and Bound and OSQP Solver


Written by Bartolomeo Stellato, February 2017, University of Oxford
"""

from __future__ import print_function, division
# import osqp  # Import OSQP Solver
import osqppurepy as osqp # Import OSQP Solver implementation in Pure Python
import numpy as np
# import numpy.linalg as la
import scipy.sparse as spa
from time import time


import pdb

# Solver statuses
MI_UNSOLVED = 10
MI_SOLVED = 11
MI_INFEASIBLE = 9
MI_UNBOUNDED = 8
MI_MAX_ITER_FEASIBLE = 12
MI_MAX_ITER_UNSOLVED = 13

# Printing interval constant
PRINT_INTERVAL = 1

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
        node's relaxed solution. At the beginning it is the warm-starting value x0
    y: numpy array
        node's relaxed solution. At the beginning it is the warm-starting value y0
    status: int
        qp solver return status
    solver: solver
        QP solver object instance
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

        # Set bounds
        if lower==None:
            self.lower = -np.inf
        else:
            self.lower = lower

        # Warm-start variables which are also the relaxed solutions
        if x0 is None:
            x0 = np.zeros(self.data.n)
        if y0 is None:
            y0 = np.zeros(self.data.m + self.data.n_int)
        self.x = x0
        self.y = y0

        # Set QP solver return status
        self.status = self.solver.constant('OSQP_UNSOLVED')

        # Add next variable elements
        self.nextvar_idx = None   # Index of next variable within solution x
        self.constr_idx = None    # Index of constraint to change for branching
                                  #     on next variable


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

        # Check if maximum number of iterations reached
        if (self.status == \
            self.solver.constant('OSQP_MAX_ITER_REACHED')):
            print("ERROR: Max Iter Reached!")
            pdb.set_trace()

        # Store solver solution
        self.x = results.x
        self.y = results.y

        # Get lower bound (objective value of relaxed problem)
        self.lower = results.info.obj_val

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

        # Define other internal variables
        self.iter_num = 1
        self.upper_glob = np.inf
        self.lower_glob = -np.inf
        self.status = MI_UNSOLVED
        self.x = np.empty(self.data.n)

        # Define root node
        self.root = Node(self.data, self.data.l, self.data.u, self.solver)

        # Define leaves at the beginning (only root)
        self.leaves = [self.root]

        # Define runtime
        self.run_time = 0

    def can_continue(self):
        """
        Check if the solver can continue
        """

        # Check if there are any leaves left in the list
        check = any(self.leaves)

        # Check if the number of iterations is within the limit
        check &= self.iter_num < self.settings['max_iter_bb']

        return check


    def choose_leaf(self, tree_explor_rule):
        """
        Choose next leaf to branch from the ones that can still be expanded
        depending on branch_rule
        """
        if tree_explor_rule == 0:
            # Depth first: Choose leaf with highest depth
            leaf_idx = np.argmax([leaf.depth for leaf in self.leaves])
            leaf = self.leaves[leaf_idx]
        else:
            raise ValueError('Tree exploring strategy not recognized')
        # Remove leaf from the list of leaves
        self.leaves.remove(leaf)

        return leaf

    def add_left(self, leaf):
        """
        Add left node from the current leaf
        """
        # Get updated QP interval bounds
        l_left = np.copy(leaf.l)
        u_left = np.copy(leaf.u)

        # Add constraint if it make the feasible region smaller (min())
        # x <= floor(x_relaxed)
        u_left[leaf.constr_idx] = min(u_left[leaf.constr_idx],
                                      np.floor(leaf.x[leaf.nextvar_idx]))

        # Constrain it with lower bound
        u_left[leaf.constr_idx] = max(u_left[leaf.constr_idx],
                                      l_left[leaf.constr_idx])

        # print("Branch left: x[%i] <= %.4f\n" % (leaf.nextvar_idx, u_left[leaf.constr_idx]))

        # Create new leaf
        new_leaf = Node(self.data, l_left, u_left, self.solver,
                        depth=leaf.depth + 1, lower=leaf.lower,
                        x0=leaf.x, y0=leaf.y)

        # Add leaf to the leaves list
        self.leaves.append(new_leaf)

    def add_right(self, leaf):
        """
        Add right node from the current leaf
        """
        # Get updated QP interval bounds
        l_right = np.copy(leaf.l)
        u_right = np.copy(leaf.u)

        # Add constraint if it make the feasible region smaller (max())
        # ceil(x_relaxed) <= x
        l_right[leaf.constr_idx] = max(l_right[leaf.constr_idx],
                                       np.ceil(leaf.x[leaf.nextvar_idx]))

        # Constrain it with upper bound
        l_right[leaf.constr_idx] = min(l_right[leaf.constr_idx],
                                       u_right[leaf.constr_idx])

        # print("Branch right: %.4f <= x[%i] \n" % (u_right[leaf.constr_idx], leaf.nextvar_idx))

        # Create new leaf
        new_leaf = Node(self.data, l_right, u_right, self.solver,
                        depth=leaf.depth + 1, lower=leaf.lower,
                        x0=leaf.x, y0=leaf.y)

        # Add leaf to the leaves list
        self.leaves.append(new_leaf)

    def pick_nextvar(self, leaf):
        """
        Pick next variable to branch upon
        """

        # Part of solution that is supposed to be integer but is fractional
        x_frac = leaf.x[self.data.i_idx]

        if self.settings['branching_rule'] == 0:

            # Get vector of fractional parts
            fract_part = abs(x_frac - np.round(x_frac))

            # Get index of max fractional part
            next_idx = np.argmax(fract_part)

            # Get index of next variable as position in the i_idx vector
            nextvar = next_idx
        else:
            raise ValueError('No variable selection rule recognized!')

        # Get next variable constraint index
        leaf.constr_idx = self.data.m + nextvar

        # Get next variable idx
        leaf.nextvar_idx = self.data.i_idx[nextvar]

    def is_within_bounds(self, x, leaf):
        """
        Check if solution x is within bounds of leaf
        """

        # Check if it satisfies current l and u bounds
        z = self.data.A.dot(x)
        if any(z < leaf.l) | any(z > leaf.u):
            return False

        # If we got here, it is integer feasible
        return True

    def is_int_feas(self, x):
        """
        Check if current solution is integer feasible
        """

        # Check if integer variables are feasible up to tolerance
        x_int = x[self.data.i_idx]
        if any(abs(x_int - np.round(x_int)) > self.settings['eps_int_feas']):
            return False

        # If we got here, it is integer feasible
        return True

    def get_integer_solution(self, x):
        """
        Round obtained solution to integer feasibility
        """
        x_int = np.copy(x)
        x_int[self.data.i_idx] = np.round(x[self.data.i_idx])
        return x_int

    def prune(self):
        """
        Prune all leaves whose lower bound is greater than current upper bound
        """
        for leaf in self.leaves:
            if leaf.lower > self.upper_glob:
                self.leaves.remove(leaf)

    def bound_and_branch(self, leaf):
        """
        Analize result from leaf solution and bound
        """

        # 1) If infeasible or unbounded, then return (prune)
        if leaf.status == self.solver.constant('OSQP_INFEASIBLE') or \
            leaf.status == self.solver.constant('OSQP_UNBOUNDED'):
            return

        # 2) If lower bound is greater than upper bound, then return (prune)
        if leaf.lower > self.upper_glob:
            return

        # 3) If integer feasible, then
        #   - update best solution
        #   - update best upper bound
        #   - prune all leaves with lower bound greater than best upper bound
        if (self.is_int_feas(leaf.x) and \
            self.is_within_bounds(leaf.x, leaf)):
            # Update best solution so far
            self.x = leaf.x
            # Update bounds
            self.lower_glob = leaf.lower
            self.upper_glob = leaf.lower
            # Prune all nodes
            self.prune()
            return

        # 4) If fractional, get integer solution using heuristic.
        #    If integer solution is within bounds (feasible), then:
        #    - compute objective value at integer x
        #    If objective value improves the upper bound
        #       - update best upper bound
        #       - prune all leaves with lower bound greater than current one
        x_int = self.get_integer_solution(leaf.x)
        if self.is_within_bounds(x_int, leaf):
            obj_int = self.data.compute_obj_val(x_int)
            if obj_int < self.upper_glob:
                self.upper_glob = obj_int
                self.prune()


        # 5) If we got here, branch current leaf producing two children
        self.branch(leaf)

        # 6) Update lower bound with minimum between lower bounds
        self.lower_glob = min([leaf.lower for leaf in self.leaves])

    def branch(self, leaf):
        """
        Branch current leaf according to branching_rule
        """

        # Branch obtaining two children and add to leaves list

        # Pick next variable to branch upon
        self.pick_nextvar(leaf)

        # Add left node to the leaves list
        self.add_left(leaf)

        # Add right node to the leaves list
        self.add_right(leaf)

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
        print("     Nodes      |           Current Node        |             Objective Bounds           ")
        print("Explr\tUnexplr\t|      Obj\tDepth\tIntInf  |    Lower\t   Upper\t    Gap  ")


    def print_progress(self, leaf):
        """
        Print progress at each iteration
        """
        if self.upper_glob == np.inf:
            gap = "    ---"
        else:
            gap = "%8.2f%%" % ((self.upper_glob - self.lower_glob)/abs(self.upper_glob)*100)

        print("%4d\t%4d\t  %10.2e\t%4d\t%5d\t  %10.2e\t%10.2e\t%s" %
              (self.iter_num, len(self.leaves), leaf.lower,  leaf.depth, 1000, self.lower_glob, self.upper_glob, gap))

    def solve(self):
        """
        Solve MIQP problem. This is the actual branch-and-bound algorithm
        """

        # Print header
        self.print_headline()

        # Loop tree until there are active leaves
        while self.can_continue():

            # 1) Choose leaf
            leaf = self.choose_leaf(self.settings['tree_explor_rule'])

            # 2) Solve relaxed problem in leaf
            leaf.solve()

            # 3) Bound and Branch
            self.bound_and_branch(leaf)

            if (self.iter_num % PRINT_INTERVAL == 0):
                # Print progress
                self.print_progress(leaf)

            # Delete leaf object
            del(leaf)

            # Update iteration number
            self.iter_num += 1

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
                   'eps_inf': 1e-04,
                   'rho': 0.1,
                   'sigma': 0.01,
                   'polishing': False,
                   'max_iter': 2500,
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
