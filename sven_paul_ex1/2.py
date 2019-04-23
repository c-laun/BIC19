import matplotlib.pyplot as plt
import numpy as np

EL = -70.
CM = 100.
GL = 10.
I0 = 1
STEP = 0.01
START = 0
STOP = 10
T0 = 5
TAU = 1

def integrate(f, start=START, stop=STOP, step=STEP, u0 = EL):
    u = [EL]
    for t in np.arange(start, stop, step):
        u.append(u[-1]+f(u[-1],t)/CM)
    return u[1:]

def base(u,t):
    return -GL*(u-EL)

def a(u,t):
    return base(u,t)+I0*(CM/GL)*d(t-5)

def b(u,t):
    return base(u,t)+10*I0*h(t-5)

def c(u,t):
    return base(u,t)+10*I0*h(t-5)*CM/TAU/GL*np.exp(-(t-T0)/TAU)

def d(t):
    if t==0:
        return 1/STEP
    return 0

def h(t):
    if t>= 0:
        return 1
    else:
        return 0


def main():
    x = np.arange(START, STOP, STEP)
    plt.xlabel("Time")
    plt.ylabel("Voltage [mV]")
    plt.plot(x, integrate(a, START, STOP, STEP))
    plt.savefig("a.png")
    plt.clf()
    plt.plot(x, integrate(b))
    plt.savefig("b.png")
    plt.clf()
    plt.plot(x, integrate(c))
    plt.savefig("c.png")
    plt.clf()
    #plt.show()

if __name__ == "__main__":
    main()
