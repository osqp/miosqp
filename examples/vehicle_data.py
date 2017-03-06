# -*- coding: utf-8 -*-
"""
MATLAB code referred from: http://www.stanford.edu/~boyd/papers/matlab/miqp_admm/vehicle_data.m

Data file for hybrid vehicle example.

Fuel use is given by F(p) = p+ gamma*p^2 (for p>=0)
We assume that the path is piecewise linear and
the slope and length of each piece is stored in
a, and l, respectively.e
@author: Vihangkuamr V Naik, Mar 02, IMT Lucca, Italy
"""
# Python 2 support
from __future__ import division
from __future__ import print_function
from builtins import range


import scipy as sp
import scipy.sparse as spa
import scipy.linalg as spla


import numpy as np
import numpy.linalg as la
import mathprogbasepy as mpbpy

import miosqp




#a = np.matrix('0.5, -0.5,  0.2, -0.7,  0.6, -0.2,  0.7, -0.5,  0.8, -0.4') / 10
a = np.array([ 0.5, -0.5,  0.2, -0.7,  0.6, -0.2,  0.7, -0.5,  0.8, -0.4]) / 10.


#l=[40 20 40 40 20 40 30 40 30 60];
l = np.array([40., 20., 40., 40., 20., 40., 30., 40., 30., 60.])


# import matplotlib.pylab as plt
# x = np.linspace(0, 10)
# y = np.zeros(len(x))
# for i in range(len(x)):
#     y[i] = np.max(a.dot(x[i]) + l)
#
# plt.figure()
# plt.plot(x, y)
# plt.show(block=False)





#Preq=(a(1): a(1): a(1) * l(1))';
#Preq = np.arange(a.item(0),(a.item(0)*l.item(0))+a.item(0),a.item(0))
Preq = np.arange(a[0],(a[0]*l[0]) + a[0], a[0])

#for i=2: length(l)
#    Preq =[Preq; (Preq(end) + a(i): a(i): Preq(end) + a(i) * l(i))'];
#end
for i in range(1, l.size):
    Preq = np.append(Preq, np.arange(Preq[-1]+a[i], Preq[-1]+ a[i]*l[i] + a[i], a[i]))

# Preq = Preq.reshape((-1, 1))# <--- THIS IS THE TRICK TO DO TRANSPOSE OF AN ARRAY


# Model
#P_des = Preq(1: 5: end);
# P_des = Preq[0:Preq.size:5]
P_des = Preq[0:len(Preq):60]

#T = length(P_des);
T = len(P_des)

#tau = 4;
tau = 4.

#P_eng_max = 1;
P_eng_max = 1.

#alpha = 1;
alpha = 1.

#beta = 1;
beta = 1.

#gamma = 1;
gamma = 1.

#delta = 1;
delta = 1.

#eta = 1;
eta = 1.

#E_max = 40;
E_max = 40.

#E_0 = E_max;
E_0 = E_max

#f_tilde = @(P_eng, eng_on) alpha * square(P_eng) + beta * P_eng + gamma * eng_on;



#% Build matrices; vector is [P_eng(T); P_batt(T); E_batt(T); turn_on(T-1), eng_on(T)];
#% Dynamics
#A = [zeros(T - 1, T), ...
#     toeplitz([0, zeros(1, T - 2)], [0, 1, zeros(1, T - 2)]), ...
#     toeplitz([-1, zeros(1, T - 2)], [-1, 1, zeros(1, T - 2)]) / tau, ...
#     zeros(T - 1, T - 1), ...
#     zeros(T - 1, T)];
A = np.hstack((np.zeros((T-1, T)),
               spla.toeplitz(np.hstack(( [[0]],  np.zeros((1, T-2)) )),
                             np.hstack(( [[0]], np.ones((1, 1)), np.zeros((1, T-2)) ))),
               spla.toeplitz(np.hstack(( [[-1]],  np.zeros((1, T-2)) )),
                             np.hstack(( [[-1]], [[1]], np.zeros((1, T-2)) ))) / tau,
               np.zeros((T-1, T-1)),
               np.zeros((T-1, T)) ))

#b = zeros(T - 1, 1);
b = np.zeros((T-1, 1))

#% Initial dynamics
#A = [A; [zeros(1, T), tau, zeros(1, T - 1), 1, zeros(1, T - 1), zeros(1, T - 1), zeros(1, T)]];
#A_temp = np.hstack((np.zeros((1,T)), tau, np.zeros((1,T-1)), np.ones((1,1)), np.zeros((1,T-1)), np.zeros((1,T-1)), np.zeros((1,T))))
A = np.vstack((A, np.hstack((np.zeros((1, T)), [[tau]], np.zeros((1, T-1)),
                             [[1]], np.zeros((1, T-1)),
                             np.zeros((1, T-1)), np.zeros((1, T)))) ))

#b = [b; E_0];
b = np.append(b, E_0)



#% Power balance
#G = [eye(T), eye(T), zeros(T, 3 * T - 1)];
G = np.hstack((np.eye(T), np.eye(T), np.zeros((T, 3*T-1)) ))

#h = P_des;
h = P_des


#% Battery limits
#G = [G; [zeros(T, 2 * T), eye(T), zeros(T, 2 * T - 1)]];
#G_temp = np.hstack((np.zeros((T, 2*T)), np.eye(T), np.zeros((T, 2*T - 1))))
G = np.vstack((G, np.hstack((np.zeros((T, 2*T)), np.eye(T),
                             np.zeros((T, 2*T-1)))) ))

#h = [h; zeros(T, 1)];
h = np.append(h, np.zeros(T))

#G = [G; [zeros(T, 2 * T), -eye(T), zeros(T, 2 * T - 1)]];
#G_temp = np.hstack((np.zeros((T, 2*T)), -np.eye(T), np.zeros((T, 2*T - 1))))
G = np.vstack((G, np.hstack((np.zeros((T, 2*T)), -np.eye(T),
                             np.zeros((T, 2*T-1)))) ))

#h = [h; -E_max * ones(T, 1)];
h = np.append(h, -E_max * np.ones(T))



#% P_eng limits
#G = [G; [eye(T), zeros(T, 4 * T - 1)]];
#G_temp = np.hstack((np.eye(T), np.zeros((T, 4*T - 1))))
G = np.vstack((G, np.hstack((np.eye(T), np.zeros((T, 4*T-1)))) ))

#h = [h; zeros(T, 1)];
h = np.append(h, np.zeros(T))

#G = [G; [-eye(T), zeros(T, 3 * T - 1), P_eng_max * eye(T)]];
#G_temp = np.hstack((-np.eye(T), np.zeros((T, 3*T - 1)), P_eng_max*np.eye(T)))
G = np.vstack((G, np.hstack((-np.eye(T), np.zeros((T, 3*T-1)),
                             P_eng_max*np.eye(T))) ))

#h = [h; zeros(T, 1)];
h = np.append(h, np.zeros(T))



#% Turn_on
#G = [G; [zeros(T, 3 * T), eye(T, T -1 ), zeros(T)]];
#G_temp = np.hstack((np.zeros((T, 3*T)), np.eye(T, T-1 ), np.zeros((T,T))))
G = np.vstack((G, np.hstack((np.zeros((T, 3*T)), np.eye(T, T-1 ),
                             np.zeros((T, T)))) ))

#h = [h; zeros(T, 1)];
h = np.append(h, np.zeros(T))

#G = [G; [zeros(T - 1, 3 * T), ...
#     eye(T - 1, T - 1) ...
#     -toeplitz([-1, zeros(1, T - 2)], [-1, 1, zeros(1, T - 2)])]];
G = np.vstack((G, np.hstack((np.zeros((T-1, 3*T)),
                             np.eye(T-1, T-1),
                             -spla.toeplitz(np.hstack(( -np.ones((1, 1)),  np.zeros((1, T-2))  )),
                             np.hstack(( -np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, T-2)) ))),
                           ))
             ))

#h = [h; zeros(T - 1, 1)];
h = np.append(h, np.zeros(T-1))



#% Fuel cost
#Phalf = [ ...
#          sqrt(alpha) * eye(T), zeros(T, 4 * T - 1)
#          zeros(1, 3 * T - 1), sqrt(eta), zeros(1, 2 * T - 1)
#        ];
#Phalf_temp1 = np.hstack((np.sqrt(alpha)*np.eye(T), np.zeros((T, 4*T-1))))
#Phalf_temp2 = np.hstack((np.zeros((1, 3*T-1)), np.sqrt(eta), np.zeros((1, 2*T-1))))
Phalf = np.vstack((np.hstack((np.sqrt(alpha)*np.eye(T), np.zeros((T, 4*T-1)))),
                   np.hstack((np.zeros((1, 3*T-1)), [[np.sqrt(eta)]], np.zeros((1, 2*T-1)))) ))

#P = 2 * (Phalf' * Phalf); % Objective will be (1/2)x^TPx, not x^TPx
P = 2 * Phalf.T.dot(Phalf)

#q = [beta * ones(T, 1); zeros(4 * T - 1, 1)];
q = np.append(beta*np.ones(T), np.zeros(4*T-1))

#q = q + [zeros(3 * T - 1, 1); -2 * eta * E_max; zeros(2 * T - 1, 1)];
q =  q + np.hstack((np.zeros(3*T-1), [-2 * eta * E_max], np.zeros(2*T-1) ))

#q = q + [zeros(4 * T - 1, 1); gamma * ones(T, 1)];
q = q + np.hstack((np.zeros(4*T-1), gamma * np.ones(T)))

#r = eta * E_max^2;
# r = eta * (E_max ** 2)



#% turn-on cost
#q = q + delta * [zeros(3 * T, 1); ones(T - 1, 1); zeros(T, 1)];
q = q + delta * np.hstack((np.zeros(3*T), np.ones(T-1), np.zeros(T)))



#% put problem in standard form with slack variables
#l = length(h);
l = len(h) #h.size

#k = length(P);
k = np.size(P, 0)

#P = [P, zeros(k, l); zeros(l, k), zeros(l, l)];
#P_temp1 = np.hstack((P, np.zeros((k,l))))
#P_temp2 = np.hstack((np.zeros((l,k)), np.zeros((l,l))))
P = np.vstack((np.hstack((P, np.zeros((k, l)))),
               np.hstack((np.zeros((l, k)), np.zeros((l, l)))) ))

#q = [q; zeros(l, 1)];
q = np.hstack((q, np.zeros(l)))

#A = [A, zeros(size(A, 1), l); G, -eye(l)];
#A_temp1 = np.hstack((A, np.zeros((np.size(A,0), l))))
#A_temp2 = np.hstack((G, -np.eye(l)))# Incomplete
A = np.vstack((np.hstack((A, np.zeros((np.size(A, 0), l)))),
               np.hstack((G, -np.eye(l))) ))

#b = [b; h];
b = np.append(b, h)



#% permute vectors to the standard form
#l1 = T;
l1 = T

#l4 = l;
l4 = l

#l5 = 4 * T - 1;
l5 = 4*T - 1

#n = l1 + l4 + l5;
n = l1 + l4 + l5

#Perm = [ ...
#          zeros(l5, l1), zeros(l5, l4), eye(l5),
#          eye(l1),       zeros(l1, l4), zeros(l1, l5),
#          zeros(l4, l1), eye(l4),       zeros(l4, l5),
#       ];
Perm_temp1 = np.hstack((np.zeros((l5, l1)), np.zeros((l5, l4)), np.eye(l5)))
Perm_temp2 = np.hstack((np.eye(l1), np.zeros((l1, l4)), np.zeros((l1, l5))))
Perm_temp3 = np.hstack((np.zeros((l4, l1)), np.eye(l4), np.zeros((l4, l5))))
Perm = np.vstack((Perm_temp1, Perm_temp2, Perm_temp3))

#A = A * Perm;
A = np.matmul(A, Perm)

#P = Perm' * P * Perm;
P = Perm.T.dot(P.dot(Perm)) # or A.transpose().dot(b).dot(A)

#q = Perm' * q;
q = np.matmul(Perm.T, q)



# Make it compatible with mathprogbasepy

# Sparsify matrices
P = spa.csc_matrix(P)
A = spa.csc_matrix(A)

# Get indeces
i_idx = np.arange(T)

# Get bounds

# Add bounds on binary variables
A, l, u = miosqp.add_bounds(i_idx, 0., 1., A, l, u)

# # Enforce integer variables to be binary => {0, 1}
# I_int = spa.identity(n).tocsc()
# I_int = I_int[i_idx, :]
# l_int = np.empty((n,))
# l_int.fill(0.)
# l_int = l_int[i_idx]
# u_int = np.empty((n,))
# u_int.fill(1)
# u_int = u_int[i_idx]
# A = spa.vstack([A, I_int]).tocsc()      # Extend problem constraints matrix A
# l = np.append(l, l_int)         # Extend problem constraints
# u = np.append(u, u_int)         # Extend problem constraints





# Solve with gurobi
mpbpy.QuadprogProblem(P, q, A, l, u, i_idx);


# import matplotlib.pylab as plt
# plt.figure()
# plt.plot(P_des)
# plt.show(block=False)
