#BIC. sheet2
#Christian Laun, Mathis Kunold
#Exercise2

#getting libraries
import matplotlib.pyplot as plt
import numpy as np

#defining Forward Euler
#x0 is the initial vector
#n is the number of steps
#dt is the lenght of one timestep, therefore total time simulated is: t_sim = n * dt
#F is defined as dx/dt = F(x,t) dim(input F) = dim(output F) = dim(x0)
#this will return an array with n + 1 vectors, x[0] beeing x0, x[i] beeing the i-th step
def ForwardEuler(x0,n,dt,F):
	x = np.zeros(shape=(n + 1,len(x0)))
	x[0] = x0
	
	for i in range(n):
		x[i + 1] = x[i] + dt * F(x[i],i * dt)
		
	return x
	


#A)
#the ODE is tau du/dt = -u + I(t) with I(t) = Theta(t - 100)
# => F = du/dt = 1/tau (-u + Theta(t - 100))
#defining that with the constant tau = 10:
def F_A(u,t):
	if t < 100:
		F = -0.1 * u
	else:
		F = 0.1 * (-u + 1)
	
	return F

#u0 = 0, since the algorithm will work with the dimension of the vector x0, we need to put u0 in an array
uA0 = np.array([0])
#simulate a total time of t = 200 for step widhts 30,20,10,5,0.1
uA_30 = ForwardEuler(uA0,7,30,F_A) #here we run into problem since 200/30 is not an integer, closest spot is 210
uA_20 = ForwardEuler(uA0,10,20,F_A) #for dt=20 and the rest we can use t_sim/dt for the number of steps n
uA_10 = ForwardEuler(uA0,20,10,F_A)
uA_5 = ForwardEuler(uA0,40,5,F_A)
uA_0p1 = ForwardEuler(uA0,2000,0.1,F_A)

#linspaces from 0 to t_sim with n + 1 steps for the initial value and all simulated steps
plt.plot(np.linspace(0,210,num=8),uA_30,label="dt = 30")
plt.plot(np.linspace(0,200,num=11),uA_20,label="dt = 20")
plt.plot(np.linspace(0,200,num=21),uA_10,label="dt = 10")
plt.plot(np.linspace(0,200,num=41),uA_5,label="dt = 5")
plt.plot(np.linspace(0,200,num=2001),uA_0p1,label="dt = 0.1")

plt.title("A) 1D Forward Euler with different step sizes")
plt.xlabel("t")
plt.ylabel("u")
plt.legend(loc="best")
plt.show()



#B)
#the ODE is d^2/t^2 x = -x, the harmonic oscillator
#solve by introducing y == d/dt x => d/dt y = -x
#the vector X = (x,y) with given initial conditions can be obtained with 2D forward euler
def F_B(X,t):
	F = np.array([X[1],-X[0]])
	return F

#x0 = 1, y0 = 0
XB0 = np.array([1,0])

print("Computing a million steps, this takes a little while") #at least it does on my toaster of a computer

#simulate a total time of t_sim = 10 with step sizes dt 1,0.1,10^(-5)
XB_1 = ForwardEuler(XB0,10,1,F_B)
XB_0p1 = ForwardEuler(XB0,100,0.1,F_B)
XB_0p00001 = ForwardEuler(XB0,1000000,0.00001,F_B)

plt.plot(np.linspace(0,10,num=11),XB_1[:,0],label="dt = 1")
plt.plot(np.linspace(0,10,num=101),XB_0p1[:,0],label="dt = 0.1")
plt.plot(np.linspace(0,10,num=1000001),XB_0p00001[:,0],label="dt = 10^(-5)")

plt.title("B) Harmonic oscillator with different step sizes")
plt.xlabel("t")
plt.ylabel("x")
plt.legend(loc="best")
plt.show()



#C)
#units will be the following: nF for C, mV for U, ms for t, nA for I, muS for g
#defining alpha and beta as vector 3D vector functions of u
#vector values correspond to n, m, h in that order
def A(u):
	A = np.zeros(shape=3)
	A[0] = (-0.55 - 0.01 * u) / (np.exp(-5.5 - 0.1 * u) - 1.0)
	A[1] = (-4.0 - 0.1 * u) / (np.exp(-4.0 - 0.1 * u) - 1.0)
	A[2] = 0.07 * np.exp(-(u + 65)/20.0)
	return A
	
def B(u):
	B = np.zeros(shape=3)
	B[0] = 0.125 * np.exp(-(u + 65)/80.0)
	B[1] = 4.0 * np.exp(-(u + 65)/18.0)
	B[2] = 1.0 / (np.exp(-3.5 - 0.1 * u) + 1.0)
	return B

#now the inverse tau and the initial values of the parameters
def inv_tau(u):
	return (A(u) + B(u))
	
def param_0(u):
	return (A(u) / (A(u) + B(u)))
	
#more parameters, the potential and the conductence
#vector values correspond to Na, K, leak in that order
E = np.array([50.0,-77.0,-54.0])
g = np.array([120.0,36.0,0.3])

#the external stimulus is a step current of 7.5nA that lasts 50ms
def I_ext(t):
	I = 0.0
	if t in range(20,70):
		I = 7.5
	return I

#the 4D vector of ODEs has the form d/dt (n,m,h,u) = F_C
#the capacity is 1, so the factor 1/C is not needed
def F_C(X,t):
	F = np.zeros(shape=4)
	F[0] = inv_tau(X[3])[0] * (param_0(X[3])[0] - X[0])
	F[1] = inv_tau(X[3])[1] * (param_0(X[3])[1] - X[1])
	F[2] = inv_tau(X[3])[2] * (param_0(X[3])[2] - X[2])
	F[3] = g[2] * (E[2] - X[3]) + g[0] * X[1]**3 * X[2] * (E[0] - X[3]) + g[1] * X[0]**4 * (E[1] - X[3]) + I_ext(t)
	return F

#the initial conditions are
XC_0 = np.array([0.3,0.1,0.6,-65.0])

#with all of that we can simulate the cell with Forward Euler using a step size of 0.01ms for 100ms
XC = ForwardEuler(XC_0,10000,0.01,F_C)

plt.plot(np.linspace(0,100,num=10001),XC[:,0],label="n")
plt.plot(np.linspace(0,100,num=10001),XC[:,1],label="m")
plt.plot(np.linspace(0,100,num=10001),XC[:,2],label="h")
plt.legend(loc="best")
plt.xlabel("t [ms]")
plt.ylabel("x")
plt.title("C) Simulation of a Hodgkin-Huxley neuron for 100ms - gating variables")
plt.show()

plt.plot(np.linspace(0,100,num=10001),XC[:,3])
plt.title("C) Simulation of a Hodgkin-Huxley neuron for 100ms - membrane potetial")
plt.xlabel("t [ms]")
plt.ylabel("U [mV]")
plt.show()



#D) and E)
#the last to problems are variations of of C) where the external stimulus is different
def I_D(t):
	return t * 15.0/400.0
	
def I_E(t):
	I = 0.0
	if t in range (20,50):
		I = -3.0
	return I

#ODEs are pasted from C) the new stimuli instead of the original one
def F_D(X,t):
	F = np.zeros(shape=4)
	F[0] = inv_tau(X[3])[0] * (param_0(X[3])[0] - X[0])
	F[1] = inv_tau(X[3])[1] * (param_0(X[3])[1] - X[1])
	F[2] = inv_tau(X[3])[2] * (param_0(X[3])[2] - X[2])
	F[3] = g[2] * (E[2] - X[3]) + g[0] * X[1]**3 * X[2] * (E[0] - X[3]) + g[1] * X[0]**4 * (E[1] - X[3]) + I_D(t)
	return F

def F_E(X,t):
	F = np.zeros(shape=4)
	F[0] = inv_tau(X[3])[0] * (param_0(X[3])[0] - X[0])
	F[1] = inv_tau(X[3])[1] * (param_0(X[3])[1] - X[1])
	F[2] = inv_tau(X[3])[2] * (param_0(X[3])[2] - X[2])
	F[3] = g[2] * (E[2] - X[3]) + g[0] * X[1]**3 * X[2] * (E[0] - X[3]) + g[1] * X[0]**4 * (E[1] - X[3]) + I_E(t)
	return F

XD = ForwardEuler(XC_0,40000,0.01,F_D) #this one is also longer than the rest
XE = ForwardEuler(XC_0,10000,0.01,F_E)

plt.plot(np.linspace(0,400,num=40001),XD[:,3])
plt.title("D) Lack of a rheobase - membrane potential")
plt.xlabel("t [ms]")
plt.ylabel("U [mV]")
plt.show()

plt.plot(np.linspace(0,400,num=40001),XD[:,0],label="n")
plt.plot(np.linspace(0,400,num=40001),XD[:,1],label="m")
plt.plot(np.linspace(0,400,num=40001),XD[:,2],label="h")
plt.legend(loc="best")
plt.xlabel("t [ms]")
plt.ylabel("x")
plt.title("D) Lack of a rheobase - gating variables")
plt.show()



plt.plot(np.linspace(0,100,num=10001),XE[:,3])
plt.title("E) Post-inhibitory rebound spike - membrane potential")
plt.xlabel("t [ms]")
plt.ylabel("U [mV]")
plt.show()

plt.plot(np.linspace(0,100,num=10001),XE[:,0],label="n")
plt.plot(np.linspace(0,100,num=10001),XE[:,1],label="m")
plt.plot(np.linspace(0,100,num=10001),XE[:,2],label="h")
plt.legend(loc="best")
plt.xlabel("t [ms]")
plt.ylabel("x")
plt.title("E) Post-inhibitory rebound spike - gating variables")
plt.show()
