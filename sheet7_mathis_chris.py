import matplotlib.pyplot as plt
import numpy as np

V, X = np.load('wta_weights_and_inputs.npy', encoding='latin1', allow_pickle=True)

#Forward Euler integrator from our solution of sheet2:
#x0 is the initial vector
#n is the number of steps
#dt is the lenght of one timestep, therefore total time simulated is: t_sim = n * dt
#F is defined as dx/dt = F(x,t) dim(input F) = dim(output F) = dim(x0)
#this will return an array with n + 1 vectors, x[0] beeing x0, x[i] beeing the i-th step
def ForwardEuler(x0,n,dt,F):
    x = np.zeros(shape=np.append(n + 1,x0.shape))
    x[0] = x0
    
    for i in range(n):
        x[i + 1] = x[i] + dt * F(x[i],i * dt)
    return x

#network time constant
tau = 0.02
#non-linearity
def nonlinF(x):
    return np.exp(0.25 * (x - 2.0))

#v is the inner product of input and weight vector v[k] = V[k] * X
#12 values for k for each angle and intensity
v = np.zeros(shape=(36,5,12))
for i in range(36):
    for j in range(5):
        for k in range(12):
            v[i,j,k] = np.dot(V[k],X[i,j])



#A - no inhi, no exhitation
#differential equation
def z_dot_a(z,t):
    z_dot = np.zeros(shape=(36,5,12))
    for i in range(36):
        for j in range(5):
            for k in range(12):
                z_dot[i,j,k] = 1/tau * (-z[i,j,k] + nonlinF(v[i,j,k]))
    return z_dot

#assuming the original state to be all 0
z0 = np.zeros(shape=(36,5,12))

z_a = ForwardEuler(z0,300,0.001,z_dot_a)


#wta_plot_image.py copied and pasted

"""The function in this file can be used to plot num_x*num_y images into one
plot using pyplot suplots. E.g. plot_images(4, 3, results, 10.0)."""

import matplotlib.pyplot as plt

def plot_images(num_y, num_x, results, vmax, vmin=0.0):
    """Plots images of the steady-state activation for each unit.
    Plots receptive field curves for each neuron at max. contrast""" 
    f, ax = plt.subplots(num_y, num_x)
    for res, axes in zip(results, ax.flat):
        im = axes.imshow(res, vmin=vmin, vmax=vmax)
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im, cax=cbar_ax)

    return f, ax
	

#The array put out by the euler integrator does not have the right shape for plotting
#Reshaping so that the final value, so that each plot will represent one unit, phi is on the x axis, I on the y axis
def plot_shape(z):
	z_plot = np.zeros(shape=(12,5,36))
	for k in range(12):
		for j in range(5):
			for i in range(36):
				z_plot[k,j,i] = z[300,i,j,k]
	return z_plot

plot_images(4,3,plot_shape(z_a),100)
plt.show()



#B - self inhibition, exitation
#weights for connections and differential equation
Wee = 0.5
Wei = 0.25
def z_dot_b(z,t):
    z_dot = np.zeros(shape=(36,5,12))
    for i in range(36):
        for j in range(5):
            for k in range(12):
                z_dot[i,j,k] = 1/tau * (-z[i,j,k] + nonlinF(Wee * z[i,j,k] + v[i,j,k] - Wei * nonlinF(z[i,j,k])))
    return z_dot
	
z_b = ForwardEuler(z0,300,0.001,z_dot_b)
plot_images(4,3,plot_shape(z_b),20)
plt.show()


#C - mutual inhibition, exitation
#weights for connections and differential equation
def z_dot_c(z,t):
    z_dot = np.zeros(shape=(36,5,12))
    for i in range(36):
        for j in range(5):
            for k in range(12):
                z_dot[i,j,k] = 1/tau * (-z[i,j,k] + nonlinF(Wee * z[i,j,k] + v[i,j,k] - Wei * nonlinF(sum(z[i,j,:]))))
    return z_dot
	
z_c = ForwardEuler(z0,300,0.001,z_dot_c)
plot_images(4,3,plot_shape(z_c),8)
plt.show()