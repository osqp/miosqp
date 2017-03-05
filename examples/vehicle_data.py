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

import scipy as sp
import scipy.sparse as spspa
import numpy as np
import numpy.linalg as la
import mathprogbasepy as mpbpy
import scipy.linalg as lasp
from decimal import Decimal

#a = np.matrix('0.5, -0.5,  0.2, -0.7,  0.6, -0.2,  0.7, -0.5,  0.8, -0.4') / 10
a = np.array([[ 0.5, -0.5,  0.2, -0.7,  0.6, -0.2,  0.7, -0.5,  0.8, -0.4]]) / 10

#l=[40 20 40 40 20 40 30 40 30 60];
l = np.array([[40, 20, 40, 40, 20, 40, 30, 40, 30, 60]])



#Preq=(a(1): a(1): a(1) * l(1))';
#Preq = np.arange(a.item(0),(a.item(0)*l.item(0))+a.item(0),a.item(0))
Preq = np.arange(a[0, 0],(a[0, 0]*l[0, 0])+a[0, 0],a[0, 0])

#for i=2: length(l)
#    Preq =[Preq; (Preq(end) + a(i): a(i): Preq(end) + a(i) * l(i))'];
#end
for i in xrange(1, l.size):
    Preq = np.hstack(( Preq, np.arange(Preq[-1]+a[0, i], Preq[-1]+ a[0, i]*l[0, i] + a[0, i], a[0, i]) ))

Preq = Preq.reshape((-1, 1))# <--- THIS IS THE TRICK TO DO TRANSPOSE OF AN ARRAY

                   
                   
#% Model
#P_des = Preq(1: 5: end);
P_des = Preq[0:Preq.size:5]

#T = length(P_des);
T = P_des.size

#tau = 4;
tau = 4 * np.ones((1, 1))

#P_eng_max = 1;
P_eng_max = 1

#alpha = 1;
alpha = 1

#beta = 1;
beta = 1

#gamma = 1;
gamma = 1

#delta = 1;
delta = 1

#eta = 1;
eta = 1 * np.ones((1, 1))

#E_max = 40;
E_max = 40

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
               lasp.toeplitz(np.hstack(( np.zeros((1, 1)),  np.zeros((1, T-2)) )),
                             np.hstack(( np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, T-2)) ))), 
               lasp.toeplitz(np.hstack(( -np.ones((1, 1)),  np.zeros((1, T-2)) )),
                             np.hstack(( -np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, T-2)) ))) / tau, 
               np.zeros((T-1, T-1)),
               np.zeros((T-1, T)) ))
               
#b = zeros(T - 1, 1);
b = np.zeros((T-1, 1))

#% Initial dynamics
#A = [A; [zeros(1, T), tau, zeros(1, T - 1), 1, zeros(1, T - 1), zeros(1, T - 1), zeros(1, T)]];
#A_temp = np.hstack((np.zeros((1,T)), tau, np.zeros((1,T-1)), np.ones((1,1)), np.zeros((1,T-1)), np.zeros((1,T-1)), np.zeros((1,T))))
A = np.vstack((A, np.hstack((np.zeros((1, T)), tau, np.zeros((1, T-1)), np.ones((1, 1)), np.zeros((1, T-1)), np.zeros((1, T-1)), np.zeros((1, T)))) ))

#b = [b; E_0];
b = np.vstack((b, E_0))



#% Power balance
#G = [eye(T), eye(T), zeros(T, 3 * T - 1)];
G = np.hstack((np.eye(T), np.eye(T), np.zeros((T, 3*T-1)) ))

#h = P_des;
h = P_des



#% Battery limits
#G = [G; [zeros(T, 2 * T), eye(T), zeros(T, 2 * T - 1)]];
#G_temp = np.hstack((np.zeros((T, 2*T)), np.eye(T), np.zeros((T, 2*T - 1))))
G = np.vstack((G, np.hstack((np.zeros((T, 2*T)), np.eye(T), np.zeros((T, 2*T-1)))) ))

#h = [h; zeros(T, 1)];
h = np.vstack((h, np.zeros((T, 1)) ))

#G = [G; [zeros(T, 2 * T), -eye(T), zeros(T, 2 * T - 1)]];
#G_temp = np.hstack((np.zeros((T, 2*T)), -np.eye(T), np.zeros((T, 2*T - 1))))
G = np.vstack((G, np.hstack((np.zeros((T, 2*T)), -np.eye(T), np.zeros((T, 2*T-1)))) ))

#h = [h; -E_max * ones(T, 1)];
h = np.vstack(( h, -E_max * np.ones((T, 1)) ))



#% P_eng limits
#G = [G; [eye(T), zeros(T, 4 * T - 1)]];
#G_temp = np.hstack((np.eye(T), np.zeros((T, 4*T - 1))))
G = np.vstack((G, np.hstack((np.eye(T), np.zeros((T, 4*T-1)))) ))

#h = [h; zeros(T, 1)];
h = np.vstack((h, np.zeros((T, 1)) ))

#G = [G; [-eye(T), zeros(T, 3 * T - 1), P_eng_max * eye(T)]];
#G_temp = np.hstack((-np.eye(T), np.zeros((T, 3*T - 1)), P_eng_max*np.eye(T)))
G = np.vstack((G, np.hstack((-np.eye(T), np.zeros((T, 3*T-1)), P_eng_max*np.eye(T))) ))

#h = [h; zeros(T, 1)];
h = np.vstack((h, np.zeros((T, 1)) ))



#% Turn_on
#G = [G; [zeros(T, 3 * T), eye(T, T -1 ), zeros(T)]];
#G_temp = np.hstack((np.zeros((T, 3*T)), np.eye(T, T-1 ), np.zeros((T,T))))
G = np.vstack((G, np.hstack((np.zeros((T, 3*T)), np.eye(T, T-1 ), np.zeros((T, T)))) ))

#h = [h; zeros(T, 1)];
h = np.vstack((h, np.zeros((T, 1)) ))

#G = [G; [zeros(T - 1, 3 * T), ...
#     eye(T - 1, T - 1) ...
#     -toeplitz([-1, zeros(1, T - 2)], [-1, 1, zeros(1, T - 2)])]];
G = np.vstack((G, np.hstack((np.zeros((T-1, 3*T)),
                             np.eye(T-1, T-1),
                             -lasp.toeplitz(np.hstack(( -np.ones((1, 1)),  np.zeros((1, T-2))  )),
                             np.hstack(( -np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, T-2)) ))),                 
                           ))
             ))

#h = [h; zeros(T - 1, 1)];
h = np.vstack((h, np.zeros((T-1, 1)) ))



#% Fuel cost
#Phalf = [ ...
#          sqrt(alpha) * eye(T), zeros(T, 4 * T - 1)
#          zeros(1, 3 * T - 1), sqrt(eta), zeros(1, 2 * T - 1)
#        ];
#Phalf_temp1 = np.hstack((np.sqrt(alpha)*np.eye(T), np.zeros((T, 4*T-1))))
#Phalf_temp2 = np.hstack((np.zeros((1, 3*T-1)), np.sqrt(eta), np.zeros((1, 2*T-1))))
Phalf = np.vstack((np.hstack((np.sqrt(alpha)*np.eye(T), np.zeros((T, 4*T-1)))),
                   np.hstack((np.zeros((1, 3*T-1)), np.sqrt(eta), np.zeros((1, 2*T-1)))) ))

#P = 2 * (Phalf' * Phalf); % Objective will be (1/2)x^TPx, not x^TPx
P = 2 * np.matmul(Phalf.transpose(), Phalf)

#q = [beta * ones(T, 1); zeros(4 * T - 1, 1)];
q = np.vstack((beta*np.ones((T, 1)), np.zeros((4*T-1, 1)) ))

#q = q + [zeros(3 * T - 1, 1); -2 * eta * E_max; zeros(2 * T - 1, 1)];
q =  q + np.vstack((np.zeros((3*T-1, 1)), -2 * eta * E_max, np.zeros((2*T-1, 1)) ))

#q = q + [zeros(4 * T - 1, 1); gamma * ones(T, 1)];
q =  q + np.vstack((np.zeros((4*T-1, 1)), gamma * np.ones((T, 1)) ))

#r = eta * E_max^2;
r = eta * (E_max**2)



#% turn-on cost
#q = q + delta * [zeros(3 * T, 1); ones(T - 1, 1); zeros(T, 1)];
q = q + delta * np.vstack((np.zeros((3*T, 1)), np.ones((T-1, 1)), np.zeros((T, 1)) ))



#% put problem in standard form with slack variables
#l = length(h);
l = np.size(h, 0) #h.size

#k = length(P);
k = np.size(P, 0)

#P = [P, zeros(k, l); zeros(l, k), zeros(l, l)];
#P_temp1 = np.hstack((P, np.zeros((k,l)))) 
#P_temp2 = np.hstack((np.zeros((l,k)), np.zeros((l,l))))    
P = np.vstack((np.hstack((P, np.zeros((k, l)))),
               np.hstack((np.zeros((l, k)), np.zeros((l, l)))) )) 

#q = [q; zeros(l, 1)];
q = np.vstack((q, np.zeros((l, 1)) ))
      
#A = [A, zeros(size(A, 1), l); G, -eye(l)];
#A_temp1 = np.hstack((A, np.zeros((np.size(A,0), l))))
#A_temp2 = np.hstack((G, -np.eye(l)))# Incomplete
A = np.vstack((np.hstack((A, np.zeros((np.size(A, 0), l)))),
               np.hstack((G, -np.eye(l))) )) #incomplete
 
#b = [b; h];
b = np.vstack((b, h))



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
P = reduce(np.dot, [Perm.transpose(), P, Perm]) # or A.transpose().dot(b).dot(A)

#q = Perm' * q;
q = np.matmul(Perm.transpose(), q)

x = np.ones((862,1)) 
print reduce(np.dot, [0.5*x.transpose(), P, x]) + np.matmul(q.transpose(), x)
Ax = np.matmul(A, x)
print Ax
