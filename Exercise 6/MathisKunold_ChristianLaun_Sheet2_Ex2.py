"""
Sheet 6 Nr.2 by Christian Laun and Mathis Kunold
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

import sys

sys.path.insert(0, '/content/mynest/lib/python3.7/site-packages')
# nest simulation
from pylab import *
import pyNN.nest as sim

from scipy.optimize import curve_fit
import scipy.stats as st

histkwargs = dict(alpha=0.5, bins=50, density=False)


def poisson(r, T, ref):
    """This function generates a spike train by drawing spikes from an
    exponential ISI distribution"""

    # estimate the number of spikes
    N_est = int(np.ceil( T * r))
    # make sure to fill the full time T
    dims = int(np.ceil( N_est + 4.*np.sqrt(N_est)))
    # draw spikes
    S = np.cumsum(np.random.exponential(scale=1./r, size=dims ) + ref, axis=0)
    # drop spikes that lie beyond T
    S = S[0:(S>T).argmax()]

    return S


def theo_distr(t, ref, tau):
    return np.where(t < ref, 0, np.exp((ref-t)/tau)/tau)


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


T = 200
rate = 50
ref = 0.01
x = np.linspace(0, 0.1, 1000)
K = 30

spiketrain = poisson(rate, T, ref)

fig, axes = plt.subplots(nrows=3, figsize=(20, 10))

#axes[0].vlines(spiketrain, 0, 1)
axes[0].hist(np.diff(spiketrain), range=(0, 0.1), label='ISIs', bins=50, density=True)
axes[0].plot(x, theo_distr(x, ref, 1/rate), label='theoretical distribution')
axes[0].set_title('Histogram over ISIs of single spike train and theoretical curve')
axes[0].legend()

#b)

sum_spiketrain = []
for i in range(K):
    sum_spiketrain = np.append(sum_spiketrain, poisson(rate, T, ref))
sum_spiketrain = np.sort(sum_spiketrain)

axes[1].hist(np.diff(sum_spiketrain), range=(0, 0.005), label='sum over 30 trains', **histkwargs)

# average rate of process R:
# actual rate = 1/tau with tau = ref+1/r
# -> rate = 1/(ref+1/r)

newrate = K/(ref+1/rate)
replacement_train = poisson(newrate, T, 0)

axes[1].hist(np.diff(replacement_train), range=(0, 0.005), label='replacement poisson process', **histkwargs)
axes[1].set_title('Histogram over ISIs')
axes[1].legend()


def intervallcounter(spiketrain):
    dt = 0.01
    t = 0
    i = 0
    j = 0
    intervallcount = []
    while True:
        if t > T:
            break
        t += dt
        while i < len(spiketrain) and spiketrain[i] < t:
            i += 1
            j += 1
        intervallcount += [j]
        j = 0
    return intervallcount


histkwargs['bins'] = 22
histkwargs['range'] = (-0.5, 21.5)
histkwargs['density'] = True
axes[2].hist(intervallcounter(sum_spiketrain), label='sum of trains', **histkwargs)
axes[2].hist(intervallcounter(replacement_train),label='replacement poisson process', **histkwargs)


x = np.arange(0, 22)
mu = newrate*0.01
axes[2].plot(x, st.poisson.pmf(x, mu),label='theoretical expectation', ls='none', marker='x')

axes[2].set_title('Number of spikes within a 10ms-window')
axes[2].legend()
plt.show()
plt.savefig('2ab.png')

fig, axes = plt.subplots(nrows=3, figsize=(20, 10))
fig.suptitle('Power spectra')

titles = ['Single train', 'Sum of trains', 'Replacement poisson process']


for i, train in enumerate([spiketrain, sum_spiketrain, replacement_train]):
    freq, power = power_spectrum(train, T)
    axes[i].plot(freq, power)
    axes[i].set_title(titles[i])

axes[2].set_xlabel('freq')
axes[2].set_ylabel('power')
plt.show()
plt.savefig('2c.png')
