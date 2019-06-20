"""
Sheet 7, Exercise 2 by Mathis Kunold and Christian Laun
"""
import numpy as np
import matplotlib.pyplot as plt

s = np.load('hopfield_pattern.npy', allow_pickle=True, encoding='bytes')
# s = np.load('hopfield_small_pattern.npy', allow_pickle=True, encoding='bytes')

K = s.shape[1]
d = int(np.sqrt(K))

W = 1/K*np.sum(s[:, None, :]*s[:, :, None], axis=0)
W -= np.identity(K)*W  # make sure diagonal elements are zero


def energy(z, W):
    return -1/2*(z@W@z)


z = np.random.binomial(1, 0.5, size=K)*2-1

print(f'The first pattern has energy {energy(s[0], W)}')
print(f'The second pattern has energy {energy(s[1], W)}')
print(f'A random pattern has energy {energy(z, W)}')


# Update policy: z_k^t+1 = sgn(sum_j w_kj*z_j)

for m in [0, 1]:
    z = np.random.binomial(1, 0.5, size=K)*2-1
    ind = np.random.choice(K, size=int(0.05*K), replace=False)
    z[ind] = s[m, ind]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    fig.suptitle(f'Initialized and compared with $s^{{{m+1}}}$')
    ax[0, 0].imshow(z.reshape((d, d)))
    ax[0, 0].set_title(f'initial network state')
    ax[0, 0].axis('off')

    overlap = [z@s[m]]
    e = [energy(z, W)]
    for steps in range(2):
        order = np.random.choice(K, size=K, replace=False)
        for i, o in enumerate(order):
            z[o] = np.sign(np.sum(W[o]*z))
            if i % 100 == 0:
                overlap += [z@s[m]]
                e += [energy(z, W)]

    ax[1, 0].plot(overlap)
    ax[1, 0].set_title(f'overlap')

    ax[1, 1].plot(e)
    ax[1, 1].set_title(f'energy')

    ax[0, 1].imshow(z.reshape((d, d)), cmap='hot')
    ax[0, 1].axis('off')
    ax[0, 1].set_title(f'final network state')
    plt.savefig(f'pattern_{m+1}')
plt.show()



