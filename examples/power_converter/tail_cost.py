"""
Compute or load tail cost of
"""
import scipy.io as sio
import numpy as np

class TailCost(object):
    def __init__(self, dyn_system, gamma):
        C = dyn_system.C
        self.P0 = C.T.dot(C)
        self.q0 = np.zeros(C.shape[1])
        self.r0 = 0.
        self.gamma = gamma

    def load(self, name):
        tail_mat = sio.loadmat('examples/power_converter/tail_backups/'+name)
        self.P0 = tail_mat['P0']
        self.q0 = tail_mat['q0']
        self.r0 = tail_mat['r0']

    def compute(self, dyn_system, N_tail):
        """
        Compute tail cost by solving an SDP
        """

        # Load samples mean and variance
        # TODO: Complete


        # Compute ADP tail by solving an SDP
        # TODO: Complete
