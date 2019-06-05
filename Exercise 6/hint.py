import numpy as np
import scipy.signal as sig

def poisson(r, T):
    """This function generates a spike train by drawing spikes from an
    exponential ISI distribution"""

    # estimate the number of spikes
    N_est = int(np.ceil( T * r))
    # make sure to fill the full time T
    dims = int(np.ceil( N_est + 4.*np.sqrt(N_est)))
    # draw spikes
    S = np.cumsum(np.random.exponential(scale=1./r, size=dims ), axis=0)
    # drop spikes that lie beyond T
    S = S[0:(S>T).argmax()]

    return S

def power_spectrum(spike_train, T, dt=1e-4):
    # Prepare the spike train array for the discretized version
    Tn = int(round((T+4*np.sqrt(T))/dt))
    Sn = np.zeros((Tn))

    # Discretize spike times
    # idx contains the indices of the time bins of the spikes in S
    idx = np.digitize(spike_train, np.arange(-dt/2.,T+dt,dt))     # Unbiased bins

    Sn[idx] = 1.

    # Estimate Power spectrum of spike trains and plot
    # make stable agains different dt-values (always consider same spike windows)
    kwargs = dict(nperseg=int(round(1./dt)))
    freq, power = sig.welch(Sn, fs=1./dt, **kwargs)

    return freq, power


