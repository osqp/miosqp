"""
Solve Mixed-Integer QPs using Branch and Bound and OSQP Solver
"""

from __future__ import print
import osqp  # Import OSQP Solver
import numpy as np


def choose_leaf(leaves, u_glob, branch_rule):
    """
    Choose next leaf to branch depending on rule
    """
    if branch_rule == 0:
        leaf = argminl(leaves, u_glob)
    else:
        rause ValueError('Branching rule not recognized')


def branch_leaf(leaf, leaves):
    """
    Expand leaf within leaves list in branch and bound tree
    """
    left = l.add_left()
    right = l.add_right()
    leaves.remove(l)
    leaves += [left, right]
    return left, right


def miosqp_solve(P, q, A, l, u, i_idx):
    """
    Solve MIQP problem using MIOSQP solver
    """

    # Branch and bound parameters
    eps_bb = 1e-03          # tolerance for difference between upper and lower bound
    eps_int_feas = 1e-03    # tolerance for integer feasibility
    branch_rule = 0         # branching rule [0] lowest lower bound

    # Extend problem with new constraints
    # Extend matrix A and bounds l and u

    # Initialize branch and bound iterations
    iter = 0
    u_glob = root.solve()
    l_glob = -np.inf
    leaves = [top]
    # masses = []
    # massesind = []
    
    # Loop tree until the gap has disappeared
    while u_glob - l_glob > eps_bb:

        # Choose leaf to branch depending on branch_rule
        l = choose_leaf(leaves, u_glob, branch_rule)

        # Expand best leaf
        left, right = branch_leaf(l, leaves)

        # Update lower and upper bound
        l_glob = min([x.lower for x in leaves])
        u_glob = min(u_glob, left.upper, right.upper)

        # Update iteration number
        iter += 1

        # Print progress
        print("iter %.3d   lower bound: %.5f, upper bound %.5f" % (iter,
            l_glob, u_glob))


