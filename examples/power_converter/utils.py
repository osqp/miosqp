"""
Utilities for power converter simulation
"""
import matplotlib.pylab as plt
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np
import numpy.fft as fft

colors = { 'b': '#1f77b4',
           'g': '#2ca02c',
           'o': '#ff7f0e'}

def plot(results, time):
    """
    Plot simulation results
    """

    t = time.t
    t_init = t[0] + time.init_periods * time.Nstpp * time.Ts
    t_end = t_init + time.Nstpp * time.Ts
    Y_phase = results.Y_phase
    Y_star_phase = results.Y_star_phase
    U = results.U

    x_ticks = np.arange(t_init, t_end + 1e-08, 0.0025)
    x_ticks_labels = ['0', '', '5', '', '10', '', '15', '', '20']


    # Plot currents
    plt.figure()
    plt.plot(t[:-1], Y_phase[0,:], '-', color=colors['o'])
    plt.plot(t[:-1], Y_star_phase[0,:], '--', color=colors['o'])
    plt.plot(t[:-1], Y_phase[1,:], '-', color=colors['g'])
    plt.plot(t[:-1], Y_star_phase[1,:], '--', color=colors['g'])
    plt.plot(t[:-1], Y_phase[2,:], '-', color=colors['b'])
    plt.plot(t[:-1], Y_star_phase[2,:], '--', color=colors['b'])
    axes = plt.gca()
    axes.set_xlim([t_init, t_end])
    axes.set_ylim([-1.25, 1.25])
    plt.xticks(x_ticks, x_ticks_labels)
    plt.grid()
    axes.set_xlabel('Time [ms]')
    plt.tight_layout()
    plt.savefig('results/power_converter_currents.pdf')
    plt.show(block=False)


    # Plot inputs
    fig, ax = plt.subplots(3, 1)
    ax[0].step(t[:-1], U[0, :], color=colors['o'])
    ax[0].set_xlim([t_init, t_end])
    ax[0].set_ylim([-1.25, 1.25])
    ax[0].grid(True)
    ax[0].set_xticks(x_ticks)
    ax[0].set_xticklabels(x_ticks_labels)
    ax[1].step(t[:-1], U[1, :], color=colors['g'])
    ax[1].set_ylim([-1.25, 1.25])
    ax[1].set_xlim([t_init, t_end])
    ax[1].set_xticks(x_ticks)
    ax[1].set_xticklabels(x_ticks_labels)
    ax[1].grid(True)
    ax[2].step(t[:-1], U[2, :], color=colors['b'])
    ax[2].set_ylim([-1.25, 1.25])
    ax[2].set_xlim([t_init, t_end])
    ax[2].set_xlabel('Time [ms]')
    ax[2].grid(True)
    ax[2].set_xticks(x_ticks)
    ax[2].set_xticklabels(x_ticks_labels)
    plt.tight_layout()
    plt.savefig('results/power_converter_inputs.pdf')
    plt.show(block=False)



def compute_on_transitions(u, u_prev):
    """
    Compute 12 switches ON transitions between u_prev and u in form {-1, 0 1}
    """

    on_transitions = np.zeros(12)

    for i in range(3):
        if u_prev[i] == 0 and u[i] == 1:
            on_transitions[i * 4 + 0] = 1
        elif u_prev[i] == 1 and u[i] == 0:
            on_transitions[i * 4 + 2] = 1
        elif u_prev[i] == 0 and u[i] == -1:
            on_transitions[i * 4 + 3] = 1
        elif u_prev[i] == -1 and u[i] == 0:
            on_transitions[i * 4 + 1] = 1

    return on_transitions


def get_thd(y, t, freq):
    """
    Get total harmonic distortion for a three phase quantity

    Args:
        y (array): signal in three phases [pu peak].
                   N.B. It is a matrix having a phase for each column
        t (array): time axis [s]
        freq (double): fundamental frequenct

    Returns:
        thd (double): total harmonic distortion [%]
    """
    thd = np.zeros(3)

    if y.shape[1] != 3:
        raise ValueError('Wrong dimensions of y. y must be 3 x N where N is the number of samples')

    # Compute THD of each phase individually
    for i in range(3):
        thd[i] = get_phase_thd(y[:, i], t, freq)

    # Take average THD
    return np.mean(thd)


def get_phase_thd(y, t, freq):

    # Get accurate DFT over time window which is multiple of the fundamental
    # period
    m = get_dft(y, t, freq)


    # Remove fundamental frequency
    i_fund = np.argmax(np.absolute(m))
    I = np.arange(np.maximum(0, i_fund - 1), i_fund + 2)  # remove neighbouring freqs
    m_ripple = m
    m_ripple[I] = 0.


    # THD up to Nyquist frequency [percent]
    Hi = np.power(np.abs(m_ripple), 2)
    thd = 100 * np.sqrt(np.sum(Hi))
    # relate the peak (not rms) value to the nominal value

    # comments about the THD:
    # Here, we assume that the rated peak base value is one (rms is 0.7071).
    # So we are relating here peak harmonics to a base peak value of 1. No
    # scaling factors are needed and no sqrt(2) or 0.5 is needed.
    # The torque equation is corrected by 1/cosphi to make sure that T=1 is
    # achieved when the peak current is one.
    # we use another function to compute the torque THD

    return thd





def get_dft(signal, time, freq):
    """
    Perform DFT over multiple of fundamental period

    Args:
        sigmal (array): signal of which DFT is to be computed
        time (array): time axis of the signal
        freq (double): fundamental frequency
                        N.B. It is used to compute the time window which is a multiple of the fundamental period thus ensuring that the frequency spectrum is 'sharp' and not smeared out.
    """


    Ts = np.mean(np.diff(time))     # Sampling interval
    fs = 1./Ts                      # Sampling frequency

    # Ensure that DFT is computed over multiple of fundamental period
    T_fund = 1./freq  # Fundamental time period
    n_samples = T_fund / Ts  # Number of samples in fundamental period

    n_period = np.floor(len(signal) / n_samples)  # Number of periods in time axis
    n_end = n_period * n_samples

    if n_period > 0:
        x = signal[:int(n_end)]
    else:
        x = signal
        raise ValueError("DFT: signal is too short; less than one fundamental period!")


    # Perform DFT on x and return components upto fs/2 in vector f
    N = len(x)
    Ns = len(signal)
    fh = fs/N

    # Scale amplitude to make it independent of length of fft
    m = fft.fft(x)/N

    # Remove negative frequencies
    if N % 2:
        # N is odd
        m = m[:int((N+1)/2)]
    else:
        # N is even
        m = m[:int(N/2) + 1]

    # Double amplitudes to reflect loss of negative freq components
    m[1:] = m[1:] * 2.

    # Ensure that m and f have same length as signal
    if Ns % 2:
        f = np.arange(int((Ns + 1)/2) + 1) * fh
    else:
        f = np.arange(int(Ns/2) + 1) * fh

    m = np.append(m, np.zeros(len(f) - len(m)))

    return m

    #
