import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '/content/mynest/lib64/python2.7/site-packages')
import pyNN.nest as sim
from quantities import uS
from scipy.optimize import curve_fit


params = { 'cm': 1., # nF
         'vrest': -70., # mV
         'ereve':  20., # mV
         'erevi': -80., # mV
         'w_e': .005, # uS
         'w_i': .05, # uS
         'taum': 20., # ms
         'tausyne': 2., # ms
         'tausyni': 5., # ms
         'taur': 2., # ms
         'v_thresh': 21. # mV
     }

ratee = 1*10000. # Hz
ratei = 1*3788.

tsim = 500.
dt = .1
t = np.arange(0,tsim+dt,dt)

def simulate_cond(params, ratee, ratei):

    sim.setup(timestep=dt, min_delay=dt, max_delay=dt)

    # Hint: The CUBA neuron is called IF_curr_exp

    ifcellscond = sim.create(sim.IF_cond_exp, {'cm'         : params['cm'],
                                                'tau_m'      : params['taum'],
                                                'tau_syn_E'  : params['tausyne'],
                                                'tau_syn_I'  : params['tausyni'],
                                                'e_rev_E'    : params['ereve'],
                                                'e_rev_I'    : params['erevi'],
                                                'v_rest'     : params['vrest'],
                                                'v_thresh'   : params['v_thresh'],
                                                'tau_refrac' : params['taur']})

    ifcellscond.initialize(v=-70.)

    spikesource3E = sim.create(sim.SpikeSourcePoisson,
                               {'duration':tsim, 'start':0., 'rate':ratee})
    spikesource3I = sim.create(sim.SpikeSourcePoisson,
                               {'duration':tsim, 'start':0., 'rate':ratei})


    sim.connect(spikesource3E, ifcellscond, weight=params['w_e'],
                receptor_type='excitatory', delay=0.1)
    sim.connect(spikesource3I, ifcellscond, weight=params['w_i'],
                receptor_type='inhibitory', delay=0.1)


    ifcellscond.record(['v','gsyn_exc', 'gsyn_inh'])

    sim.run(tsim)

    vcond = np.array(ifcellscond.get_v().segments[0].analogsignals[0]).T[0]

    g = ifcellscond.get_data()
    g_exc = np.array(g.segments[0].analogsignals[0].T[0])
    g_inh = np.array(g.segments[0].analogsignals[1].T[0])

    sim.end()

    return vcond, g_exc, g_inh

def simulate_curr(params, ratee, ratei, taum):

    sim.setup(timestep=dt, min_delay=dt, max_delay=dt)

    # Hint: The CUBA neuron is called IF_curr_exp

    ifcellscond = sim.create(sim.IF_curr_exp, {'cm'         : params['cm'],
                                                'tau_m'      : taum,
                                                'tau_syn_E'  : params['tausyne'],
                                                'tau_syn_I'  : params['tausyni'],
                                                'v_rest'     : params['vrest'],
                                                'v_thresh'   : params['v_thresh'],
                                                'tau_refrac' : params['taur']})

    ifcellscond.initialize(v=-70.)

    spikesource3E = sim.create(sim.SpikeSourcePoisson,
                               {'duration':tsim, 'start':0., 'rate':ratee})
    spikesource3I = sim.create(sim.SpikeSourcePoisson,
                               {'duration':tsim, 'start':0., 'rate':ratei})

    we = params['w_e']*(params['ereve']-params['vrest'])
    wi = params['w_i']*(params['erevi']-params['vrest'])

    print(we,wi)

    sim.connect(spikesource3E, ifcellscond, weight=we,
                receptor_type='excitatory', delay=0.1)
    sim.connect(spikesource3I, ifcellscond, weight=wi,
                receptor_type='inhibitory', delay=0.1)


    ifcellscond.record(['v'])

    sim.run(tsim)

    vcond = np.array(ifcellscond.get_v().segments[0].analogsignals[0]).T[0]

    sim.end()

    return vcond

def lin(x, a, b):
    return a*x+b

def calc_nu_i():

    rates = range(3500,4500,10)
    vs = []

    for i in rates:

        ratei = i
        vcond, g_exc, g_inh = simulate_cond(params, ratee, ratei)

        vs.append(np.mean(vcond))


    popt,pcov = curve_fit(lin, rates, vs)

    return (-70-popt[1])/popt[0]

#print(calc_nu_i())
#3788
vcond, g_exc, g_inh = simulate_cond(params, ratee, ratei)

cm = params['cm']

t_eff = cm/(np.mean(g_exc)+np.mean(g_inh)+params['cm']/params['taum'])

vcurr = simulate_curr(params, ratee, ratei, t_eff)

print(t_eff)


x = np.arange(0,len(vcond)*dt,dt)

plt.plot(x,vcond,label="Conductance")
plt.plot(x,vcurr,label="Current")
plt.legend()
plt.savefig("trace_0.png")
#plt.show()
plt.clf()

bins = np.arange(-74,-64,1)
plt.hist(vcond,label="Conductance", histtype='step',bins=bins)
plt.hist(vcurr,label="Current", histtype='step',bins=bins)
plt.legend()
plt.savefig("hist_0.png")
#plt.show()
