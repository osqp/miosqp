"""
Hybrid vehicle example from http://web.stanford.edu/~boyd/papers/miqp_admm.html

Bartolomeo Stellato, University of Oxford, 2017
"""
import examples.vehicle_matrices as mat
import mathprogbasepy as mpbpy
import matplotlib.pylab as plt

def run_example():
    """
    Generate and solve vehicle example
    """

    T = 72      # Horizon length
    n_x = 1     # Number of states
    n_u = 4     # Number of inputs

    # Generate problem matrices
    P, q, A, l, u, i_idx = mat.generate_example(T)

    # Solve with gurobi
    # Create MIQP problem
    prob = mpbpy.QuadprogProblem(P, q, A, l, u, i_idx)
    resGUROBI = prob.solve(solver=mpbpy.GUROBI)



    # Get results
    x = resGUROBI.x
    z = x[2:n_u*T:n_u]
    P_eng = x[1:n_u*T:n_u]
    P_batt = x[0:n_u*T:n_u]
    E = x[n_u*T:-1]


    # Plot results
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(z)
    ax[0, 0].set_ylabel('z')
    ax[1, 0].plot(E)
    ax[1, 0].set_ylabel('E')
    ax[0, 1].plot(P_eng)
    ax[0, 1].set_ylabel('P_eng')
    ax[1, 1].plot(P_batt)
    ax[1, 1].set_ylabel('P_batt')
    plt.show(block=False)




    # z  = x_cvx(1: T);
    # P_eng   = x_cvx(end - 4 * T + 2: end - 3 * T + 1);
    # P_batt  = x_cvx(end - 3 * T + 2: end - 2 * T + 1);
    # E_batt  = x_cvx(end - 2 * T + 2: end - T + 1);

    # % Plotting the results
    # figure
    # subplot(411)
    # plot(0: T, [E_0; E_batt])
    # axis([0, 72, 0, E_max])
    # subplot(412)
    # plot(0: T, [P_batt; 0]);
    # axis([0, 72, -2, 2])
    # subplot(413)
    # plot(0: T, [P_eng; 0]);
    # axis([0, 72, 0, 1])
    # subplot(414)
    # plot(0: T, [eng_on; 0]);
    # axis([0, 72, 0, 1])



    import ipdb; ipdb.set_trace()
