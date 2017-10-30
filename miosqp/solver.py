"""
Solve Mixed-Integer QPs using Branch and Bound and OSQP Solver


Written by Bartolomeo Stellato, February 2017, University of Oxford
"""

from __future__ import print_function, division
from builtins import object
# import osqppurepy as osqp # Import OSQP Solver implementation in Pure Python
import numpy as np
# import numpy.linalg as la
from time import time


# Import data
from miosqp.data import Data
from miosqp.workspace import Workspace
from miosqp.results import Results
from miosqp.node import Node
from miosqp.constants import MI_UNSOLVED

# Dump max_iter_problems to files
# import pickle
# from os import listdir
# from os.path import splitext

# Plotting
# import matplotlib.pylab as plt


class MIOSQP(object):
    """
    MIOSQP Solver class
    """
    def __init__(self):
        self.data = None
        self.work = None

    def setup(self, P, q, A, l, u, i_idx, i_l, i_u,
              settings,
              qp_settings):
        """
        Setup MIQP problem using MIOSQP solver
        """
        # Start timer
        start = time()

        # Assign integer variables bounds if they are none
        if i_l is None:
            i_l = -np.inf * np.ones(len(i_idx))
        if i_u is None:
            i_u = np.inf * np.ones(len(i_idx))

        # Create data class instance
        data = Data(P, q, A, l, u, i_idx, i_l, i_u)

        # Create Workspace
        self.work = Workspace(data, settings, qp_settings)

        stop = time()

        self.work.setup_time = stop - start

    def solve(self):
        """
        Solve MIQP problem with branch-and-bound algorithm
        """

        # Start timer
        start = time()

        # Store self.work in work so that name is shorter
        work = self.work

        # Store bounds behavior for plotting
        # lowers = []
        # uppers = []

        if work.settings['verbose']:
            # Print header
            work.print_headline()

        # Loop tree until there are active leaves
        while work.can_continue():

            # 1) Choose leaf
            leaf = work.choose_leaf(work.settings['tree_explor_rule'])

            # 2) Solve relaxed problem in leaf
            leaf.solve()

            # Check if maximum number of iterations reached
            # if (leaf.status == work.solver.constant('OSQP_MAX_ITER_REACHED')):
                # print("ERROR: Max Iter Reached!")
                # # Dump file to 'max_iter_examples'folder
                # problem = {'P': work.data.P,
                #            'q': work.data.q,
                #            'A': work.data.A,
                #            'l': leaf.l,
                #            'u': leaf.u,
                #            'i_idx': work.data.i_idx,
                #            'settings': work.qp_settings}
                # # Get new filename
                # list_dir = listdir('./max_iter_examples')
                # last_name = int(splitext(list_dir[-1])[0])
                # with open('max_iter_examples/%s.pickle' % str(last_name + 1), 'wb') as f:
                #     pickle.dump(problem, f)
                # import pdb; pdb.set_trace()

            # 3) Bound and Branch
            work.bound_and_branch(leaf)

            if (work.iter_num % (work.settings['print_interval']) == 0) and \
                    work.settings['verbose']:
                # Print progress
                work.print_progress(leaf)

            # Delete leaf object
            del(leaf)

            # Update iteration number
            work.iter_num += 1

            # Store bounds for plotting
            # lowers.append(work.lower_glob)
            # uppers.append(work.upper_glob)

        # Update average number of OSQP iterations
        work.osqp_iter_avg = work.osqp_iter / work.iter_num

        # Get final status
        work.get_return_status()

        # Get exact mixed-integer solution
        work.get_return_solution()

        if work.settings['verbose']:
            # Print footer
            work.print_footer()

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

        # Stop timer
        stop = time()

        self.work.solve_time = stop - start

        if self.work.first_run:
            self.work.first_run = 0
            self.work.run_time = self.work.setup_time + \
                self.work.solve_time
        else:
            self.work.run_time = self.work.solve_time

        if self.work.settings['verbose']:
            print("Elapsed time: %.4es" % self.work.run_time)

        # Return results
        return Results(self.work.x, self.work.upper_glob,
                       self.work.run_time, self.work.status,
                       self.work.osqp_solve_time, self.work.osqp_iter_avg)

    def update_vectors(self, q=None, l=None, u=None):
        """
        Update problem vectors without running setup again
        """
        work = self.work

        # Update data
        work.data.update_vectors(q, l, u)

        if q is not None:
            # Update cost in OSQP solver
            work.solver.update(q=q)

        # Create root node
        root = Node(work.data, work.data.l, work.data.u, work.solver)
        work.leaves = [root]

        # Reset statistics
        work.iter_num = 1
        work.osqp_solve_time = 0.
        work.osqp_iter = 0
        work.osqp_iter_avg = 0
        work.lower_glob = -np.inf
        work.status = MI_UNSOLVED

        # Timings (setup_time is the same)
        work.solve_time = 0.
        work.run_time = 0.

        # Reset solution
        work.x = np.empty(work.data.n)
        work.upper_glob = np.inf

    def set_x0(self, x0):
        """
        Set initial solution x0
        """

        self.work.set_x0(x0)
