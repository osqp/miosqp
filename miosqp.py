"""
Solve Mixed-Integer QPs using Branch and Bound and OSQP Solver


Written by Bartolomeo Stellato, February 2017, University of Oxford
"""

from __future__ import print_function, division
from builtins import str
from builtins import object
import osqp  # Import OSQP Solver
# import osqppurepy as osqp # Import OSQP Solver implementation in Pure Python
import numpy as np
# import numpy.linalg as la
import scipy.sparse as spa
from time import time


# Dump max_iter_problems to files
import pickle
from os import listdir
from os.path import splitext

# Plotting
import matplotlib.pylab as plt



# Solver statuses
MI_UNSOLVED = 'Unolved'
MI_SOLVED = 'Solved'
MI_INFEASIBLE = 'Infeasible'
MI_UNBOUNDED = 'Unbounded'
MI_MAX_ITER_FEASIBLE = 'Max-iter feasible'
MI_MAX_ITER_UNSOLVED = 'Max-iter unsolved'


def add_bounds(i_idx, l_new, u_new, A, l, u):
    """
    Add new bounds on the variables

        l_new <= x_i <= u_new for i in i_idx

    It is done by adding rows to the contraints

        l <= A x <= u
    """

    n = A.shape[1]

    # Enforce integer variables to be binary => {0, 1}
    I_int = spa.identity(n).tocsc()
    I_int = I_int[i_idx, :]
    l_int = np.empty((n,))
    l_int.fill(l_new)
    l_int = l_int[i_idx]
    u_int = np.empty((n,))
    u_int.fill(u_new)
    u_int = u_int[i_idx]
    A = spa.vstack([A, I_int]).tocsc()      # Extend problem constraints matrix A
    l = np.append(l, l_int)         # Extend problem constraints
    u = np.append(u, u_int)         # Extend problem constraints

    return A, l, u




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
        self.A, self.l, self.u = add_bounds(i_idx, -np.inf, np.inf, A, l, u)

        #
        # I_int = spa.identity(self.n).tocsc()
        # I_int = I_int[i_idx, :]     # Extend constraints matrix A with only the rows of
        #                             # the identity relative to the integer variables
        #
        # # Extend the bounds only for the variables which are integer
        # l_int = np.empty((self.n,))
        # l_int.fill(-np.inf)
        # l_int = l_int[i_idx]
        # u_int = np.empty((self.n,))
        # u_int.fill(np.inf)
        # u_int = u_int[i_idx]
        #
        # self.A = spa.vstack([A, I_int]).tocsc()      # Extend problem constraints matrix A
        # self.l = np.append(l, l_int)         # Extend problem constraints
        # self.u = np.append(u, u_int)         # Extend problem constraints

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
    ©: int
        number of fractional elements which are supposed to be integer
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
        if lower==None:
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

        # DEBUG: Problems that hit max_iter are infeasible
        # if self.status == self.solver.constant('OSQP_MAX_ITER_REACHED'):
            # self.status = self.solver.constant('OSQP_INFEASIBLE')

        # Store number of iterations
        self.num_iter = results.info.iter

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
    leaves: list
        leaves in the tree

    Other internal variables
    ------------------------
    iter_num: int
        number of iterations
    osqp_iter: int
        number of osqp admm iteration
    run_time: double
        runtime of the algorithm
    upper_glob: double
        global upper bound
    lower_glob: double
        global lower bound
    status: int
        MIQP solver status
    x: numpy array
        current best solution
    """
    def __init__(self, data, settings, qp_settings=None, x0=None):
        self.data = data
        self.settings = settings

        # Setup OSQP solver instance
        self.solver = osqp.OSQP()
        self.qp_settings = qp_settings
        if self.qp_settings is None:
            self.qp_settings = {}
        self.solver.setup(self.data.P, self.data.q, self.data.A,
                          self.data.l, self.data.u, **qp_settings)

        # Define other internal variables
        self.iter_num = 1
        self.osqp_iter = 0
        self.lower_glob = -np.inf
        self.status = MI_UNSOLVED


        # Define root node
        root = Node(self.data, self.data.l, self.data.u, self.solver)

        # Define leaves at the beginning (only root)
        self.leaves = [root]

        # Add initial solution and objective value
        if x0 is not None:
            if self.is_within_bounds(x0, root) and self.is_int_feas(x0, root):
                self.x = x0
                self.upper_glob = self.data.compute_obj_val(x0)
            else:
                self.upper_glob = np.inf
                self.x = np.empty(self.data.n)
        else:
            self.upper_glob = np.inf
            self.x = np.empty(self.data.n)

        # Define runtime
        self.run_time = 0


    def can_continue(self):
        """
        Check if the solver can continue
        """

        # Check if there are any leaves left in the list
        if not any(self.leaves):
            return False

        # Check if the number of iterations is within the limit
        if not self.iter_num < self.settings['max_iter_bb']:
            return False

        return True


    def choose_leaf(self, tree_explor_rule):
        """
        Choose next leaf to branch from the ones that can still be expanded
        depending on branch_rule
        """
        if tree_explor_rule == 0:
            # Depth first: Choose leaf with highest depth
            leaf_idx = np.argmax([leaf.depth for leaf in self.leaves])
        elif tree_explor_rule == 1:
            # Two-phase method.
            #   - First perform depth first
            #   - Then choose leaves with best bound
            if np.isinf(self.upper_glob):
                # First phase
                leaf_idx = np.argmax([leaf.depth for leaf in self.leaves])
            else:
                # Second phase
                leaf_idx = np.argmax([leaf.lower for leaf in self.leaves])
        else:
            raise ValueError('Tree exploring strategy not recognized')

        # Get leaf
        leaf = self.leaves[leaf_idx]

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
        # Part of solution that is supposed to be integer
        x_frac = leaf.x[self.data.i_idx[leaf.frac_idx]]

        if self.settings['branching_rule'] == 0:

            # Get vector of fractional parts
            fract_part = abs(x_frac - np.round(x_frac))

            # Get index of max fractional part
            next_idx = np.argmax(fract_part)

            # Get index of next variable as position in the i_idx vector
            nextvar = leaf.frac_idx[next_idx]

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
        if any(z < leaf.l - self.qp_settings['eps_abs']) | \
            any(z > leaf.u + self.qp_settings['eps_abs']):
            return False

        # If we got here, it is integer feasible
        return True

    def is_int_feas(self, x, leaf):
        """
        Check if current solution is integer feasible
        """

        # Part of solution that is supposed to be integer
        x_int = x[self.data.i_idx]

        # Part of the solution that is still fractional
        int_feas_false = abs(x_int - np.round(x_int)) >\
            self.settings['eps_int_feas']
        leaf.frac_idx = np.where(int_feas_false)[0].tolist()  # Index of frac parts

        # Store number of fractional elements (integer infeasible)
        leaf.intinf = np.sum(int_feas_false)
        if leaf.intinf > 0:
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

        # Update total number of OSQP ADMM iteration
        self.osqp_iter += leaf.num_iter

        # 1) If infeasible or unbounded, then return (prune)
        if leaf.status == self.solver.constant('OSQP_INFEASIBLE') or \
            leaf.status == self.solver.constant('OSQP_UNBOUNDED'):
            # ipdb.set_trace()
            return

        # 2) If lower bound is greater than upper bound, then return (prune)
        if leaf.lower > self.upper_glob:
            return

        # 3) If integer feasible, then
        #   - update best solution
        #   - update best upper bound
        #   - prune all leaves with lower bound greater than best upper bound
        if (self.is_int_feas(leaf.x, leaf)):
            # Update best solution so far
            self.x = leaf.x
            # Update upper bound
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
                self.x = x_int
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
        print("     Nodes      |           Current Node        |             Objective Bounds             |   Cur Node")
        print("Explr\tUnexplr\t|      Obj\tDepth\tIntInf  |    Lower\t   Upper\t    Gap    |     Iter")


    def print_progress(self, leaf):
        """
        Print progress at each iteration
        """
        if self.upper_glob == np.inf:
            gap = "    --- "
        else:
            gap = "%8.2f%%" % ((self.upper_glob - self.lower_glob)/abs(self.lower_glob)*100)

        if leaf.status == self.solver.constant('OSQP_INFEASIBLE') or \
            leaf.status == self.solver.constant('OSQP_UNBOUNDED'):
            obj = np.inf
        else:
            obj = leaf.lower

        if leaf.intinf is None:
            intinf = "  ---"
        else:
            intinf = "%5d" % leaf.intinf

        print("%4d\t%4d\t  %10.2e\t%4d\t%s\t  %10.2e\t%10.2e\t%s\t%5d" %
              (self.iter_num, len(self.leaves), obj,  leaf.depth, intinf, self.lower_glob, self.upper_glob, gap, leaf.num_iter), end='')

        if leaf.status == self.solver.constant('OSQP_MAX_ITER_REACHED'):
            print("!")
        else:
            print("")



    def print_footer(self):
        """
        Print final statistics
        """

        print("\n")
        print("Status: %s" % self.status)
        if self.status == MI_SOLVED:
            print("Objective bound: %6.3e" % self.upper_glob)
        print("Total number of OSQP iterations: %d" % self.osqp_iter)


    def solve(self):
        """
        Solve MIQP problem. This is the actual branch-and-bound algorithm
        """

        # Store bounds behavior for plotting
        # lowers = []
        # uppers = []

        if self.settings['verbose']:
            # Print header
            self.print_headline()

        # Loop tree until there are active leaves
        while self.can_continue():

            # 1) Choose leaf
            leaf = self.choose_leaf(self.settings['tree_explor_rule'])

            # 2) Solve relaxed problem in leaf
            leaf.solve()

            # Check if maximum number of iterations reached
            # if (leaf.status == self.solver.constant('OSQP_MAX_ITER_REACHED')):
                # print("ERROR: Max Iter Reached!")
                # # Dump file to 'max_iter_examples'folder
                # problem = {'P': self.data.P,
                #            'q': self.data.q,
                #            'A': self.data.A,
                #            'l': leaf.l,
                #            'u': leaf.u,
                #            'i_idx': self.data.i_idx,
                #            'settings': self.qp_settings}
                # # Get new filename
                # list_dir = listdir('./max_iter_examples')
                # last_name = int(splitext(list_dir[-1])[0])
                # with open('max_iter_examples/%s.pickle' % str(last_name + 1), 'wb') as f:
                #     pickle.dump(problem, f)
                # import pdb; pdb.set_trace()

            # 3) Bound and Branch
            self.bound_and_branch(leaf)

            if (self.iter_num % (self.settings['print_interval']) == 0) and \
                self.settings['verbose']:
                # Print progress
                self.print_progress(leaf)

            # Delete leaf object
            del(leaf)


            # Update iteration number
            self.iter_num += 1

            # Store bounds for plotting
            # lowers.append(self.lower_glob)
            # uppers.append(self.upper_glob)

        # Get final status
        self.get_return_status()

        if self.settings['verbose']:
            # Print footer
            self.print_footer()

        # Print bounds
        # plt.figure(1)
        # plt.cla()
        # plt.plot(uppers)
        # plt.plot(lowers)
        # plt.legend(('upper bound', 'lower bound'))
        # plt.xlabel('iteration')
        # plt.ylabel('bounds')
        # plt.title('Global lower and upper bounds')
        # plt.grid()
        # plt.show(block=False)

        # ipdb.set_trace()

        return


def miosqp_solve(P, q, A, l, u, i_idx, settings, qp_settings, x0=None):
    """
    Solve MIQP problem using MIOSQP solver
    """

    # Start timer
    start = time()


    # Create data class instance
    data = Data(P, q, A, l, u, i_idx)

    # Create Workspace
    work = Workspace(data, settings, qp_settings, x0)

    # Solve problem
    work.solve()

    # Stop
    stop = time()
    work.run_time = stop - start
    if settings['verbose']:
        print("Elapsed time: %.4es" % work.run_time)
    return work
