#!/usr/bin/env python
"""Branch and bound code for a 30 variable cardinality problem.

Written by Jacob Mattingley, March 2007, for EE364b Convex Optimization II,
Stanford University, Professor Stephen Boyd.

This file solves a (random) instance of the problem:
    minimize    card(x)
    subject to  Ax <= b
using branch and bound, where x is the optimization variable and is in
{0,1}^30, A is in R^(100x30) and b is in R^(100x1).
"""

from __future__ import division
from pylab import *
from cvxopt import setseed, normal, uniform, lapack, solvers
from cvxopt.base import matrix
from cvxopt.modeling import variable, op, max, sum
from ipdb import set_trace

inf = float('inf')


def sign(x):
    """Returns the sign of an object by comparing with <= 0, with 0
    arbitrarily assigned a sign of -1."""
    if x <= 0:
        return -1
    else:
        return 1


def rtz(x, tol=1e-3):
    """Rounds within-tol objects or entries of a matrix to exactly zero."""
    if isinstance(x, matrix):
        return matrix([rtz(y, tol) for y in x], x.size)
    elif abs(x) <= tol:
        return 0
    else:
        return x


def argmin(x, avoid=[]):
    """Returns the index of the minimum item in x with index not in the set
    avoid."""
    for i in range(len(x)):
        if i in avoid:
            x[i] = inf
    for i in range(len(x)):
        if x[i] == min(x):
            return i


def card(x):
    """Returns the cardinality (number of non-zero entries) of x."""
    return len([1 for y in rtz(x) if y != 0])


def funcfeas(x, tol=1e-3):
    """Determines if x is in the feasible set Ax <= b (with tolerance)."""
    y = A * x
    for i in range(len(y)):
        if y[i] - b[i] > tol:
            return False

    return True


class node:
    """node([n[, zeros[, ones[, parent]]]]) -> node object

    Create node of a branch-and-bound tree with size n.  Known-zero or
    known-one elements should be set using zeros and ones. Specify the
    optional parent if the node is not at the top of the tree.
    """

    def __init__(self, n, zeros=[], ones=[], parent=None):
        if (set(zeros) & set(ones)):
            raise Exception, 'cannot have item fixed to zero and fixed to one'

        zeros.sort()
        ones.sort()

        self.zeros = zeros
        self.ones = ones
        self.parent = parent
        self.left = None
        self.right = None
        self.n = n
        self.alive = True
        self.lower = inf
        self.upper = inf
        self.prunedmass = 0

    def solve(self):
        """Find upper and lower bounds of the sub-problem belonging to
        this node."""

        # Use cvxopt to solve the problem.
        x = variable(self.n)
        z = variable(self.n)
        constr = [z <= 1, z >= 0, x >= min_mat *
                  z, x <= max_mat * z, A * x <= b]
        for i in self.ones:
            constr.append(z[i] == 1)
        for i in self.zeros:
            constr.append(z[i] == 0)
        o = op(sum(z), constr)
        o.solve()

        if x.value is None:
            # We couldn't find a solution.
            self.raw = None
            self.rounded = None
            self.lower = inf
            self.upper = inf
            return inf

        # Use heuristic to choose which variable we should split on next. Don't
        # choose a variable we have already set.
        self.picknext = argmin(
            list(rtz(abs(z.value - 0.5))), self.zeros + self.ones)

        self.lower = sum(z.value)

        if funcfeas(x.value):
            self.upper = card(x.value)
        else:
            self.upper = inf

        return self.upper

    def mass(self, fullonly=True):
        """Find the number of nodes which could live below this node."""
        p = self.n - len(self.zeros) - len(self.ones)
        return 2**p

    def addleft(self):
        """Add a node to the left-hand side of the tree and solve."""
        if self.left is None:
            self.left = node(self.n, self.zeros +
                             [self.picknext], self.ones, self)

        self.left.solve()
        return self.left

    def addright(self):
        """Add a node to the right-hand side of the tree and solve."""
        if self.right is None:
            self.right = node(self.n, self.zeros,
                              self.ones + [self.picknext], self)

        self.right.solve()
        return self.right

    def potential(self):
        """Returns the number of nodes that could still be added below this node."""
        return self.mass() - self.taken()

    def taken(self):
        """Returns the number of nodes that live below this node."""
        t = 0
        if self.left is not None:
            t += 1
            if self.left.alive:
                t += self.left.taken()
            else:
                t += self.left.mass()

        if self.right is not None:
            t += 1
            if self.right.alive:
                t += self.right.taken()
            else:
                t += self.right.mass()

        return t

    def __repr__(self):
        return '<node: mass=%d, zeros=%s, ones=%s>' % (self.mass(),
                                                       str(self.zeros),
                                                       str(self.ones))

    def nodes(self, all=False):
        """Returns a list of all nodes that live at, or below, this point.

        If all is False, return nodes only if they are stil alive.
        """

        coll = []
        if self.alive or all:
            coll += [self]
        if self.left is not None and (self.left.alive or all):
            coll += self.left.nodes(all)
        if self.right is not None and (self.right.alive or all):
            coll += self.right.nodes(all)

        return coll

    def prune(self, upper):
        """Sets alive to False for any nodes in the tree with their lower less
        than upper.
        """

        # Note that we can use ceil(lower) instead of lower, because if we know
        # that cardinality is > 18.3, say, we know it must be 19 or more.
        if ceil(self.lower) > upper:
            p = self.parent
            while p is not None:
                p.prunedmass += self.mass() + 1
                p = p.parent
            self.alive = False
        else:
            if self.left is not None and self.left.alive:
                self.left.prune(upper)
            if self.right is not None and self.right.alive:
                self.right.prune(upper)


def argminl(nodes, Uglob):
    """Returns the node with lowest lower bound, from a list of nodes.

    Only considers nodes which can still be expanded.
    """
    m = min([x.lower for x in nodes if x.potential() > 0])
    for x in nodes:
        if x.lower == m:
            return x


if __name__ == "__main__":
    solvers.options['show_progress'] = False

    # Randomly generate data.
    setseed(3)
    m, n = 100, 30
    A = normal(m, n)
    x0 = normal(n, 1)
    b = A * x0 + 2 * uniform(m, 1)

    # Find lower and upper bounds on each element of x. If we have that for a
    # particular x_i, x_i > 0 or x_i < 0, we cannot hope to reduce the cardinality
    # by setting that x_i to 0, so exclude index i from consideration by adding it
    # to defones.
    defones = []
    mins = [0] * n
    maxes = [0] * n
    x = variable(n)
    for i in range(n):
        op(x[i], A * x <= b).solve()
        if x.value[i] <= 0:
            x.value[i] *= 1.0001
        else:
            x.value[i] *= 1 / 1.0001
        mins[i] = x.value[i]

        op(-x[i], A * x <= b).solve()
        if x.value[i] <= 0:
            x.value[i] *= 1 / 1.0001
        else:
            x.value[i] *= 1.0001
        maxes[i] = x.value[i]
        if mins[i] > 0:
            defones.append(i)
        if maxes[i] < 0:
            defones.append(i)

    # Later it is helpful to have these bounds stored in matrices.
    min_mat = matrix(0, (n, n), tc='d')
    min_mat[range(0, n**2, n + 1)] = matrix(mins)

    max_mat = matrix(0, (n, n), tc='d')
    max_mat[range(0, n**2, n + 1)] = matrix(maxes)

    # Various data structures for later plotting.
    uppers = []
    lowers = []
    masses = []

    # Create the top node in the tree.
    top = node(n, ones=defones)
    top.stillplotted = False

    Uglob = top.solve()
    Lglob = -inf
    # nodes = [top]
    leaves = [top]
    masses = []
    leavenums = []
    massesind = []
    oldline = None
    iter = 0

    # Expand the tree until the gap has disappeared.
    while Uglob - Lglob > 1:
        iter += 1
        # Expand the leaf with lowest lower bound.
        l = argminl(leaves, Uglob)

        # Add left and right branches and solve them
        left = l.addleft()
        right = l.addright()

        leaves.remove(l)
        leaves += [left, right]

        Lglob = min([x.lower for x in leaves])
        Uglob = min(Uglob, left.upper, right.upper)

        set_trace()
        
        lowers.append(Lglob)
        uppers.append(Uglob)

        for x in top.nodes():
            # Prune anything except the currently optimal solution. Doesn't
            # actually affect the progress of the algorithm.
            if x.lower > Uglob and x.upper != Uglob:
                x.alive = False

        print "iter %3d.  lower bound: %.5f, upper bound: %.0f" % (iter, Lglob, Uglob)

        massesind.append(len(lowers))
        masses.append(top.potential() / top.mass())

        leavenums.append(len([1 for x in leaves if x.alive]))

    print "done."

    figure(1)
    cla()
    plot(uppers)
    plot(lowers)
    plot(ceil(lowers), 'g--')
    legend(('upper bound', 'lower bound',
            'ceiling (lower bound)'), loc='lower right')
    axis((0, 123, 12, 21.2))
    xlabel('iteration')
    ylabel('cardinality')
    title('Global lower and upper bounds')

    figure(2)
    cla()
    semilogy(massesind, masses)
    xlabel('iteration')
    ylabel('portion of non-pruned Boolean values')
    title('Portion of non-pruned sparsity patterns')
    axis((0, 123, 0.01, 1.1))

    figure(3)
    cla()
    plot(leavenums)
    xlabel('iteration')
    ylabel('number of leaves on tree')
    title('Number of active leaves on tree')
    axis((0, 123, 0, 60))

    show()
