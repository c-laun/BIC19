import matplotlib.pyplot as plt
import numpy.random as nr
import numpy as np

def spike_train_bins(length=100, frequency=10):
    """
    Creates a spike train through a series of Bernoulli experiments.
    input:  Frequency of the spike train in Hz, defaults to 10 Hz.
            Length of the spike train in s, defaults to 100 s.
    output: The spike train.
    """

    # Set timesteps so that the spike probability is 10%.
    dt     = 0.1/frequency
    spikes = []
    for i in range(int(length/dt)):
        if nr.rand() < 0.1:
            spikes.append(i*dt)

    return spikes

def spike_train_isi(length=100, frequency=10):
    """
    Creates a spike train by drawing ISIs from an exponential distribution.
    input:  Frequency of the spike train in Hz, defaults to 10 Hz.
            Length of the spike train in s, defaults to 100 s.
    output: The spike train.
    """

    isi = [nr.exponential(1/frequency)]
    while sum(isi) < 100:
        isi.append(nr.exponential(1/frequency))

    spikes = [sum(isi[:i]) for i in range(1,len(isi))]

    return spikes

def spike_train_uniform(length=100, frequency=10):
    """
    Creates a spike train by drawing a poisson distributed number of spikes 
    from a uniform distribution.
    input:  Frequency of the spike train in Hz, defaults to 10 Hz.
            Length of the spike train in s, defaults to 100 s.
    output: The spike train.
    """

    n      = nr.poisson(frequency*length)
    spikes = nr.uniform(0,length,n)

    return spikes

def plot_isi_dist(spikes, length=100, frequency=10, bins=10, s='', title=''):
    """
    Plots the histogram of the ISIs for given a spike train and the expected
    distribution given the frequencey and length of the train.
    input:  Spike train, and length and frequency for theoretical distribution.
            The number of bins for the histogram.
            String s to be added to filename.
            Title of the figure.
    output: None, saves a figure called isi_dist.pdf.
    """
    spikes = np.sort(spikes)
    isi = [spikes[i+1]-spikes[i] for i in range(len(spikes)-1)]

    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6,5))

    axs.hist(isi,np.linspace(0,5/frequency,bins))

    # Normalization factor to plot the exponential distribution on the same
    # scale as the histogram.
    norm = length*frequency*(1-np.exp(-5/bins))
    x = np.linspace(0,5/frequency,100)
    y = np.exp(-frequency*(x-2.5/bins/frequency))*norm
    axs.plot(x, y, label='Theoretical Distribution', lw=2.5)
    axs.legend()

    axs.set_title(r'Distribution of ISIs and $p_{ISI}(s)$ for '+title)

    fig.savefig('isi_dist'+s+'.pdf')

def poisson(l, k):
    return np.power(l,k)*np.exp(-l)/np.math.factorial(k)

def plot_spikes_dist(spikes, length=100, frequency=10, s='', title=''):
    """
    Plots the histogram of the number of spikes within 300ms intervals for
    a given spike train and the expected distribution given the frequencey and
    length of the train.
    input:  Spike train, and length and frequency for theoretical distribution.
            The number of bins for the histogram.
            String s to be added to filename.
            Title of the figure.
    output: None, saves a figure called spikes_dist.pdf.
    """
    n = [len(list(filter(lambda x: 0.3*i <= x < 0.3*(i+1),spikes))) for i in
            range(int(length/0.3))]

    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6,5))

    axs.hist(n,np.linspace(-0.5,max(n)-0.5,max(n)+1))

    x = np.linspace(0,max(n),max(n)+1)
    y = [length/0.3*poisson(frequency*0.3, a) for a in x]
    axs.plot(x, y, lw=0.5, marker='o', color='black', label='Theoretical Distribution')
    axs.legend()

    axs.set_title(r'Distribution of $N$ and $p_{300ms}(N)$ for '+title)

    fig.savefig('spikes_dist'+s+'.pdf')


if __name__ == '__main__':
    plot_isi_dist(spike_train_bins(length=20,frequency=100), s='_bin',
        title='Bernoulli based Spike Train', length=20,frequency=100)
    plot_isi_dist(spike_train_isi(), s='_isi',
        title='ISIs based Spike Train')
    plot_isi_dist(spike_train_uniform(), s='_uni',
        title=r'Spike Train from $N$ Uniform Samples')

    plot_spikes_dist(spike_train_bins(), s='_bin',
        title='Bernoulli based Spike Train')
    plot_spikes_dist(spike_train_isi(), s='_isi',
        title='ISIs based Spike Train')
    plot_spikes_dist(spike_train_uniform(), s='_uni',
        title=r'Spike Train from $N$ Uniform Samples')
