from __future__ import print_function

import numpy as np

# Import osqp solver
import osqp

# Miosqp files
from miosqp.node import Node
from miosqp.constants import MI_UNSOLVED, \
                             MI_PRIMAL_INFEASIBLE, \
                             MI_DUAL_INFEASIBLE, \
                             MI_SOLVED, \
                             MI_MAX_ITER_FEASIBLE, \
                             MI_MAX_ITER_UNSOLVED


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
        number of osqp admm iterations
    osqp_iter_avg: int
        average number of osqp admm iterations
    osqp_solve_time: float
        total time required to solve OSQP problems
    run_time: double
        runtime of the algorithm
    first_run: int
        has the problem been solved already?
    upper_glob: double
        global upper bound
    lower_glob: double
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
        self.qp_settings = qp_settings
        if self.qp_settings is None:
            self.qp_settings = {}
        self.solver.setup(self.data.P, self.data.q, self.data.A,
                          self.data.l, self.data.u, **qp_settings)

        # Define other internal variables
        self.first_run = 1
        self.iter_num = 1
        self.osqp_solve_time = 0.
        self.osqp_iter = 0
        self.osqp_iter_avg = 0
        self.lower_glob = -np.inf
        self.status = MI_UNSOLVED

        # Define root node
        root = Node(self.data, self.data.l, self.data.u, self.solver)

        # Define leaves at the beginning (only root)
        self.leaves = [root]

        # Set initial solution and objective value
        self.upper_glob = np.inf
        self.x = np.empty(self.data.n)

        # Define timings
        self.setup_time = 0.
        self.solve_time = 0.
        self.run_time = 0.

    def set_x0(self, x0):
        """
        Set initial condition
        """

        # Suppose that algorithm hasn't been run yet.
        # Tree has only one leaf (root)
        root = self.leaves[0]

        # Add initial solution and objective value
        if self.satisfies_lin_constraints(x0, root.l, root.u) and \
                self.is_int_feas(x0, root):
            self.x = x0
            self.upper_glob = self.data.compute_obj_val(x0)
        else:
            print('Invalid initial solution!\n')
            self.upper_glob = np.inf
            self.x = np.empty(self.data.n)

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

        # x <= floor(x_relaxed)
        u_left[leaf.constr_idx] = \
            np.floor(leaf.x[leaf.nextvar_idx])

        # DEBUG:
        if any(l_left > u_left):
            import ipdb; ipdb.set_trace()

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

        # ceil(x_relaxed) <= x
        l_right[leaf.constr_idx] = \
            np.ceil(leaf.x[leaf.nextvar_idx])

        # DEBUG:
        if any(l_right > u_right):
            import ipdb; ipdb.set_trace()

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

    def satisfies_lin_constraints(self, x, l, u):
        """
        Check if solution x is within linear constraints
        """
        # Check if it satisfies current l and u bounds
        z = self.data.A.dot(x)
        if any(z < l - self.qp_settings['eps_abs']) | \
                any(z > u + self.qp_settings['eps_abs']):
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
        # Index of frac parts
        leaf.frac_idx = np.where(int_feas_false)[0].tolist()
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

        # Update total time to solve OSQP problems
        self.osqp_solve_time += leaf.osqp_solve_time

        # 1) If infeasible or unbounded, then return (prune)
        if leaf.status == osqp.constant('OSQP_PRIMAL_INFEASIBLE') or \
                leaf.status == osqp.constant('OSQP_DUAL_INFEASIBLE'):
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
        #    If integer solution satisfies linear constraints, then:
        #    - compute objective value at integer x
        #    If objective value improves the upper bound
        #       - update best upper bound
        #       - prune all leaves with lower bound greater than current one
        x_int = self.get_integer_solution(leaf.x)
        if self.satisfies_lin_constraints(x_int,
                                          self.data.l, self.data.u):
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
                    self.status = MI_PRIMAL_INFEASIBLE
                else:                     # -inf
                    self.status = MI_DUAL_INFEASIBLE

        else:  # Hit maximum number of iterations
            if self.upper_glob != np.inf:
                self.status = MI_MAX_ITER_FEASIBLE
            else:
                if self.upper_glob >= 0:  # +inf
                    self.status = MI_MAX_ITER_UNSOLVED
                else:                     # -inf
                    self.status = MI_DUAL_INFEASIBLE

    def get_return_solution(self):
        """
        Get exact mixed-integer solution
        """

        if self.status == MI_SOLVED or \
                self.status == MI_MAX_ITER_FEASIBLE:
            # Part of solution that is supposed to be integer
            self.x[self.data.i_idx] = \
                np.round(self.x[self.data.i_idx])

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
            gap = "%8.2f%%" % \
                ((self.upper_glob - self.lower_glob)/abs(self.lower_glob)*100)

        if leaf.status == osqp.constant('OSQP_PRIMAL_INFEASIBLE') or \
                leaf.status == osqp.constant('OSQP_DUAL_INFEASIBLE'):
            obj = np.inf
        else:
            obj = leaf.lower

        if leaf.intinf is None:
            intinf = "  ---"
        else:
            intinf = "%5d" % leaf.intinf

        print("%4d\t%4d\t  %10.2e\t%4d\t%s\t  %10.2e\t%10.2e\t%s\t%5d" %
              (self.iter_num, len(self.leaves), obj,
               leaf.depth, intinf, self.lower_glob,
               self.upper_glob, gap, leaf.num_iter), end='')

        if leaf.status == osqp.constant('OSQP_MAX_ITER_REACHED'):
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
