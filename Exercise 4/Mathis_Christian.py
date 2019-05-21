import numpy as np
import matplotlib.pyplot as plt


####################################################
################ Exercise 2 ########################
####################################################

voltage = [np.load('voltage_4.2_weak.npy')]
voltage += [np.load('voltage_4.2_medium.npy')]
voltage += [np.load('voltage_4.2_strong.npy')]

names = ['weak', 'medium', 'strong']

length = voltage[0].shape[0]
x = np.linspace(0, length/10, length)

probeidx = [200*i for i in range(1, 11)]

# for i in range(3):
#     plt.figure(figsize=(20, 4))
#     plt.vlines([20, 40, 60, 80, 100], -70, -65)
#     plt.plot(x, voltage[i])
#     plt.ylabel('U [mV]')
#     plt.xlabel('t [ms]')
#     plt.show()

for i in range(3):
    psp = np.zeros(200)
    for j in probeidx:
        psp += voltage[i][j:j+200] #add all psps together
    psp = psp/len(probeidx)  # normalize
    plt.plot(np.linspace(0, 20, 200), psp, label=names[i])

plt.title('PSPs for different background stimuli')
plt.ylabel('U [mV]')
plt.xlabel('t [ms]')
plt.legend()
plt.show()


####################################################
################ Exercise 3 ########################
####################################################























