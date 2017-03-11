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


# Import data
from miosqp.data import Data
from miosqp.workspace import Workspace
from miosqp.results import Results

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

    def setup(self, P, q, A, l, u, i_idx, settings,
             qp_settings, x0=None):
        """
        Setup MIQP problem using MIOSQP solver
        """
        # Start timer
        start = time()

        # Create data class instance
        self.data = Data(P, q, A, l, u, i_idx)

        # Create Workspace
        self.work = Workspace(self.data, settings, qp_settings, x0)

        stop = time()

        self.work.setup_time = stop - start


    def solve(self):
        """
        Solve MIQP problem using MIOSQP solver
        """

        # Start timer
        start = time()

        # Solve problem
        self.work.solve()

        # Stop
        stop = time()

        self.work.solve_time = stop - start

        if self.work.first_run == 1:
            self.work.first_run = 0
            self.work.run_time = self.work.setup_time + \
                self.work.solve_time
        else:
            self.work.run_time =  self.work.solve_time


        if self.work.settings['verbose']:
            print("Elapsed time: %.4es" % self.work.run_time)

        # Return results
        return Results(self.work.x, self.work.upper_glob,
                       self.work.run_time, self.work.status,
                       self.work.osqp_solve_time)


    def update_vectors(self, q=None, l=None, u=None):
        """
        Update problem vectors without running setup again
        """

        print("Hello!")
