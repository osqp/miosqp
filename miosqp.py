"""
Solve Mixed-Integer QPs using Branch and Bound and OSQP Solver


Written by Bartolomeo Stellato, February 2017, University of Oxford
"""

from __future__ import print_function
import osqp  # Import OSQP Solver
import numpy as np
# import numpy.linalg as la
import scipy.sparse as spa

# Solver statuses
MI_UNSOLVED = 10
MI_SOLVED = 11
MI_INFEASIBLE_OR_UNBOUNDED = 9

# Nodes statuses
MI_NODE_PENDING = 100
MI_NODE_SOLVED = 101
MI_NODE_FATHOMED = 99

class Data(object):
    """
    Data for the relaxed qp problem
    """
    def __init__(self, P, q, A, l, u, i_idx):
        # Get problem dimensions
        self.n = A.shape[1]
        self.m = A.shape[0]

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

        self.A = spa.vstack([A, I_int])      # Extend problem constraints matrix A
        self.l = l.append(l_int)             # Extend problem constraints
        self.u = u.append(u_int)             # Extend problem constraints

        #
        # Define problem cost function
        #
        self.P = P
        self.q = q

        # Define index of integer variables
        self.i_idx = i_idx

class Info(object):
    """
    Branch and bound solver information
    """
    def __init__(self):
        # Initialize branch and bound iterations
        self.iter_num = 0
        self.upper_glob = np.inf
        self.lower_glob = -np.inf
        self.obj_val = np.inf
        self.status = MI_UNSOLVED

class Solution(object):
    """
    Define best solution found so far
    """
    def __init__(self):
        self.x = None
        self.y = None

class Node:
    """
    Branch-and-bound node class
    """
    def __init__(self, l, u):
        """
        Initialize node class
        """
        # Set bounds
        self.lower = -np.inf
        self.upper = np.inf

        # Set l and u for relaxed QP problem
        self.l = l
        self.u = u

        # Set parent node (TODO: Need parent?)
        # self.parent = parent

        # Set node status
        self.status = MI_NODE_PENDING

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
            nodes_list += self.left.nodes(self)
        if self.right is not None and self.right.status != MI_NODE_FATHOMED:
            nodes_list += self.right.nodes(self)

        return nodes_list

    def get_bounds(self):
        """
        Find upper and lower bounds for the relaxed problem corresponding to this node
        """

        # Update lower and upper bounds is OSQP to solve the current problem

        # Solve current problem

        # Check if infeasible or unbounded -> Node becomes fathomed

        # Check if integral feasible solution
        #       -> Node becomes fathomed
        #       -> Objective value is updated
        #       -> New solution is stored

        # Compute lower bound (objective value of relaxed problem)

        # Compute upper bound (round solution, check feasibility, compute obj value)
        # if rounde is infeasible --> infinite upper bound



class Workspace(object):
    """
    Workspace class

    Attributes
    ----------
    data: class
        miqp problem data
    settings: dictionary
        branch and bound settings
    osqp: class
        osqp solver class
    root: class
        root node of the tree
    leaves: list
        leaves in the tree
    info: class
        information on the algorithm progress
    """
    def __init__(self, data, settings):
        self.data = data
        self.settings = settings
        self.info = Info()           # Initialize information
        self.solution = Solution()   # Initialize problem solution

        # Define root node
        self.root = Node(self.data.l, self.data.u)
        self.leaves = [self.root]  # At the initialization there is only the root node


    def can_continue(self):
        """
        Check if the solver can continue
        """
        check = self.info.upper_glob - self.info.lower_glob > self.settings.eps_bb_abs
        check &= (self.info.upper_glob - self.info.lower_glob)/abs(self.info.lower_glob) > \
                 self.settings.eps_bb_rel
        check &= self.info.iter_num < self.settings.max_iter_bb
        return check


    def choose_leaf(self, tree_explor_rule):
        """
        Choose next leaf to branch from the ones that can still be expanded
        depending on branch_rule
        """
        if tree_explor_rule == 0:
            # Choose leaf with lowest lower bound between leaves which
            # can be expanded
            min_lower = min([leaf.lower for leaf in self.leaves \
                            if leaf.status != MI_NODE_FATHOMED])
            for x in self.leaves:
                if x.lower == min_lower:
                    leaf = x
        else:
            raise ValueError('Tree exploring strategy not recognized')
        return leaf


    def branch_leaf(self, leaf):
        """
        Branch: Expand leaf within leaves list in branch and bound tree. Then solve
        the problems in the right and left children obtaining their respective
        lower and upper bounds
        """
        left = leaf.add_left()
        right = leaf.add_right()
        self.leaves.remove(leaf)
        self.leaves += [left, right]

        # Update lower and upper bound
        self.info.lower_glob = min([x.lower for x in self.leaves])
        # TODO: Can't the lower_glob update be only a min between current
        #       lower_glob and the new ones for the leaves?
        #       (just like the upper bound?)

        # Update upper bound
        self.info.upper_glob = min(self.info.upper_glob, left.upper, right.upper)
        # if uppwer bound improved -> Store node solution x


    def bound(self):
        """
        Bound: prune tree nodes if their lower value is greater than the current
        upper bound
        """
        for node in self.root.nodes():
            if node.lower > self.info.upper_glob:
                node.status = MI_NODE_FATHOMED

    def solve(self):
        """
        Solve MIQP problem. This is the actual branch-and-bound algorithm
        """

        # Get bounds from root node
        self.root.get_bounds()
        if self.root.status == MI_NODE_FATHOMED:
            # Root node infeasible or unbounded
            self.info.status = MI_INFEASIBLE_OR_UNBOUNDED
            return
        self.info.upper_glob = self.root.lower
        self.info.lower_glob = self.root.upper


        # Loop tree until the cost function gap has disappeared
        while self.can_continue():

            # 1) Choose leaf
            #   -> Use tree exploration rule
            leaf = self.choose_leaf(self.settings.tree_explor_rule)

            # 2) Branch leaf
            #   -> Solve children
            #   -> Update lower and upper bounds
            self.branch_leaf(leaf)

            # 3) Bound
            #   -> prune nodes with lower bound above upper bound
            self.bound()

            # Update iteration number
            self.info.iter_num += 1

            # Print progress
            print("iter %.3d   lower bound: %.5f, upper bound %.5f",
                  self.info.iter_num, self.info.lower_glob, self.info.upper_glob)

        return


def miosqp_solve(P, q, A, l, u, i_idx):
    """
    Solve MIQP problem using MIOSQP solver
    """


    # Define problem settings
    settings = {'eps_bb_abs': 1e-03,           # absolute convergence tolerance
                'eps_bb_rel': 1e-03,           # relative convergence tolerance
                'eps_int_feas': 1e-03,         # integer feasibility tolerance
                'max_iter_bb': 1000,           # maximum number of iterations
                'tree_explor_rule': 0,         # tree exploration rule
                                               #   [0] lowest lower bound
                'branching_rule': 0,           # branching rule
                                               #   [0] max fractional part
                'variable_selection_rule': 0}  # select next variable to split upon


    # Create data class instance
    data = Data(P, q, A, l, u, i_idx)

    # Create Workspace
    work = Workspace(data, settings)

    # Solve problem
    work.solve()
