import numpy as np
import pylab as pl

pl.style.use('ggplot')


# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # C L A S S E S   A N D   F U N C T I O N S # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

class GibbsSamplerBM(object):
    def __init__(self, W, b):
        self.W = np.array(W)  # weights
        self.b = np.array(b)  # biases
        assert self.W.shape[0] == self.W.shape[0] == self.b.shape[0], "Shapes do not match!"
        assert np.allclose(self.W, self.W.T), "Weight matrix is not symmetric!"
        self.K = self.b.shape[0]  # num neurons
        self.z = np.zeros(self.K)  # current network state
        self.z[:] = np.random.randint(0, 2, self.K)  # random inititialization of z
        self.advance = self.advance_gibbs  # define which advance-method to use
        # (TODO: REGISTER YOUR GIBBS SAMPLING METHOD HERE.)

    def faulty_dummy_advance(self):  # Faulty dummy advance method
        idx = np.arange(self.K)  # index of all neurons
        for k in idx:  # Step through all neurons
            self.z[k] = np.random.randint(0, 2, 1)  # Assign some (faulty) coin toss value
        return self.z  # Return result

    def advance_gibbs(self):
        for k in range(self.K):
            uk = self.b[k] + np.sum(self.W[k]*self.z)
            if 1/(1+np.exp(-uk)) > np.random.rand():
                self.z[k] = 1
            else:
                self.z[k] = 0
        return self.z

    def calc_p_theo(self):  # Analytically calculate the correct distribution
        E = lambda z: - (0.5 * np.dot(np.dot(z, self.W), z) + np.dot(z, self.b))  # Energy fct.
        s = [2] * self.K  # shape of the state space
        P = np.zeros(s)  # This array has size 2^K -- so it would crash you memory for large K!
        for z in np.ndindex(*s):
            P[z] = np.exp(-E(z))
        P /= P.sum()  # This is the big sum for the partition function.
        return P


D_KL = lambda P, Q: np.log(
    (P / Q) ** P).sum()  # This is a very short form for calculating the Kullback-Leiber divergence.

# # # # # # # # # # # # # # # #
# # # P A R A M E T E R S # # #
# # # # # # # # # # # # # # # #

SEED = 0xDEADBEEF  # Random seed for numpy's RNG

T = 10000  # Total simulation time (= number of samples to generate)
W = np.load("W.npy")  # Load the weight matrix
b = np.load("b.npy")  # Load the bias vector

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # P R E P A R A T O R Y   F U N C T I O N S # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

np.random.seed(SEED)  # Apply random seed
net = GibbsSamplerBM(W, b)  # Generate network

K = net.K  # Number of binary random variables
Ptheo = net.calc_p_theo()  # theoretically calculated probabilities from energy E(x)

# # # # # # # # # # # # # # # # # # # # #
# # # C O L L E C T   S A M P L E S # # #
# # # # # # # # # # # # # # # # # # # # #

Z = np.zeros((T, K))  # This is where the samples will be stored.

# # # # # # # # # # # # # # # # # # # #
# # # R U N   S I M U L A T I O N # # #
# # # # # # # # # # # # # # # # # # # #

for t in range(T):  # For all time steps
    Z[t] = net.advance()  # generate a sample and store it.

# # # # # # # # # # # # # # # # # #
# # #   E V A L U A T I O N   # # #
# # # # # # # # # # # # # # # # # #


# Calculate the Kullback Leibler divergence D_KL between the sampled and the target distribution

T_DKL = 10 ** np.arange(1, np.log10(T) + 0.1).astype(
    int)  # Define some time points when to evaluate the Kullback-Leiber divergence
s = [2] * K  # shape of the state space
Pnet = np.zeros([len(T_DKL)] + s)  # array to store network sampled probabilities
for i, t_dkl in enumerate(T_DKL):  # For each time point of interest...
    for z in Z[:t_dkl]:  # For all samples z(t)
        z = (i,) + tuple(z.astype(int))  # convert to appropriate index format
        Pnet[z] += 1  # count the occurrence of z
    Pnet[i] /= t_dkl  # devide by total number of samples to get the probabilities.

D = np.zeros(len(T_DKL))  # array to store the D_KL values
for i, p in enumerate(Pnet):
    D[i] = D_KL(p, Ptheo)  # Calculate the D_KL

# # # # # # # # # # # # # # # #
# # #   P L O T T I N G   # # #
# # # # # # # # # # # # # # # #

# figure setup
fig = pl.figure(figsize=(12.0, 3.25))
rect = 0.07, 0.17, 0.50, 0.78
ax_bar = fig.add_axes(rect, ylabel="Probability P(z)")
rect = 0.69, 0.17, 0.28, 0.78
ax_dkl = fig.add_axes(rect, ylabel="D_KL (P_net || P_theo)")

# bar plot of probs
ax = ax_bar
xlabels = [str(x).replace(',', '').replace(' ', '')[1:-1] for x in np.ndindex(*([2] * K))]
kwargs = dict(align='center', fc='lightblue', ec='darkblue', zorder=1)
ax.bar(-0.20 + np.arange(2 ** K), Pnet[-1].flatten(), width=0.35, label="P_net(z)", **kwargs)
kwargs = dict(align='center', fc='darkgrey', ec='grey', zorder=0)
ax.bar(+0.20 + np.arange(2 ** K), Ptheo.flatten(), width=0.35, label="P_theo(z)", **kwargs)
ax.set_xlim(-0.5, 2 ** K - 0.5)
ax.set_ylim(0., 1.1 * ax.yaxis.get_data_interval()[1])
ax.set_xticks(np.arange(2 ** K))
ax.set_xticklabels(xlabels, rotation='vertical', family='monospace')
leg = ax.legend(loc="upper right")

# D_KL plot
ax = ax_dkl
ax.plot(T_DKL, D, 'r', lw=1.)
ax.plot(T_DKL, D, 'o', ms=5., mec='r', mfc='r')
ax.set_xlim(0.75 * T_DKL[0], 1.33 * T_DKL[-1])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Time [steps]", labelpad=2)

# pl.savefig('3a.png')
pl.show()


pl.figure()
pl.imshow(Z[:100].T)
pl.title('First 100 samples of Z')
pl.savefig('3b.png')
pl.show()






