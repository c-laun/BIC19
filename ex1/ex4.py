import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaussian(x, h,  mean, std):
    return h*np.exp(-(x-mean)**2/std**2)


samples = np.random.normal(loc=1.0, scale=0.5, size=1000)

print(f'The mean of our sample is {np.mean(samples)}'
      f', the standard deviation is {np.std(samples)}')

N, bins, patches = plt.hist(samples, label='histogram')

popt, pcov = curve_fit(gaussian, (bins[1:]+bins[:-1])/2, N, p0=[300, 1, 0.5])

x = np.linspace(-1, 4, 100)
plt.plot(x, gaussian(x, *popt), label='fit')

plt.legend()
plt.title('Histogram and fitted gaussian')
plt.show()

print(f'The curve-fitting resulted in a mean of {popt[1]} \n'
      f'and a standard deviation of {popt[2]} \n'
      f'The std is higher than the actual std because the binning in \n'
      f'the histogram resulted in a loss of positional information.')

