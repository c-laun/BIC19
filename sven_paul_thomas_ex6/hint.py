import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.stats import poisson as poiss

def poisson(r, T, tau):
    """This function generates a spike train by drawing spikes from an
    exponential ISI distribution"""

    # estimate the number of spikes
    N_est = int(np.ceil( T * r))
    # make sure to fill the full time T
    dims = int(np.ceil( N_est + 4.*np.sqrt(N_est)))
    # draw spikes
    isi = np.random.exponential(scale=1./r, size=dims ) + np.full(dims,tau)
    S = np.cumsum(isi , axis=0)
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

def exp(x,b,tau):
    return 1/b*np.exp(-(x-tau)/b)

if __name__=="__main__":
    #a)
    r = 50
    T = 200
    tau = 1e-2
    s = poisson(r,T,tau)
    isi = np.diff(s)
    plt.hist(isi)
    x = np.arange(0.8/r,10/r,0.05/r)
    plt.plot(x,T*exp(x,1/r,tau),color="red")
    plt.savefig("poisson_tau.png")
    plt.xlabel("Poisson ISIs with absolute refractory time")
    plt.clf()

    #b)
    K = 30
    spikes1 = []
    for i in range(K):
        spikes1.extend(poisson(r,T,tau))

    spikes1.sort()

    isi = np.diff(poisson(r,T,tau))
    nu = 1/np.mean(isi)

    spikes2 = poisson(nu*K,T,0)

    bins = np.arange(0,10/nu/K,0.5/nu/K)

    plt.hist(np.diff(spikes1), bins=bins, histtype="step", label="Pooled")
    plt.hist(np.diff(spikes2), bins=bins, histtype="step", label="Adapted")
    plt.legend()
    plt.xlabel("Pooled and adapted (regular poisson with increased firing rate) ISIs")
    plt.savefig("poisson_pooled_isi.png")
    plt.clf()

    bins = np.arange(0,T,0.1)

    hist1 = np.histogram(spikes1,bins=bins)[0]
    hist2 = np.histogram(spikes2,bins=bins)[0]

    x = np.arange(50,max(hist1)+5,1)

    plt.hist(hist1, bins=x, histtype="step", label="Pooled")
    plt.hist(hist2, bins=x, histtype="step", label="Adapted")
    plt.plot(x,len(hist1)*poiss.pmf(x,0.1*nu*K), label="Theory")

    plt.xlabel("Pooled and adapted (regular poisson with increased firing rate) spikes per 10ms")
    plt.legend()
    plt.savefig("poisson_pooled_10ms.png")
    plt.clf()

    #c)

    ps_R = power_spectrum(poisson(r,T,tau),T)
    ps_comb = power_spectrum(spikes1,T)
    ps_Knu = power_spectrum(spikes2,T)

    plt.plot(ps_R[0],ps_R[1],label="R")
    plt.plot(ps_comb[0],ps_comb[1],label="Pooled")
    plt.plot(ps_Knu[0],ps_Knu[1],label="Adapted")
    plt.xlabel("Power spectrum")
    plt.legend()
    plt.savefig("power_spectrum.png")
