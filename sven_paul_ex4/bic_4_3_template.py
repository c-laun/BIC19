# -*- coding: utf-8 -*-

## install nest 2.14.0
#!apt-get install -y build-essential cmake libltdl7-dev libreadline6-dev \
#libncurses5-dev libgsl0-dev python-all-dev python-numpy python-scipy \
#python-matplotlib ipython openmpi-bin libopenmpi-dev python-nose
#
#!wget https://github.com/nest/nest-simulator/releases/download/v2.14.0/nest-2.14.0.tar.gz
#!tar -xf nest-3.14.0.tar.gz
#!mkdir nest-simulator-2.14.0-build
#
#!cd nest-simulator-2.14.0-build && cmake -DCMAKE_INSTALL_PREFIX:PATH=/content/mynest ../nest-2.14.0 && make && make install
#
## install pynn 0.9.2 with dependencies
#!pip install pyNN==0.9.2
#!pip install lazyarray
#!pip install neo

# load nest into path
import sys
sys.path.insert(0, '/content/mynest/lib/python2.7/site-packages')

#nest simulation
from pylab import *
import pyNN.nest as sim

#known neuron parameters
cm = 1. # nF
ereve = 20. # mV
erevi = -80. # mV

# unknown neuron paramters (random values insertet here)
vrest = -70. # mV
wconde = .0035# uS
wcondi = .048# uS
taumcond = 9.9# ms
tausyne = 2.9# ms
tausyni = 5.2# ms

tsim = 500.
dt = .1

sim.setup(timestep=dt, min_delay=dt, max_delay=dt)

ifcellscond = [sim.create(sim.IF_cond_exp, {'cm' : cm,           'tau_m': taumcond,
                                            'tau_syn_E':tausyne, 'tau_syn_I': tausyni,
                                            'e_rev_E': ereve,    'e_rev_I'   : erevi,
                                            'v_rest' : vrest,    'v_thresh':0,
                                            'tau_refrac' : 2.0}) for i in range(1)]

for ifcell in ifcellscond:
    ifcell.initialize(v=vrest)

####### simulate #######

spkte = array([50., 100., 103., 106., 109., 112., 115., 118., 121., 124., 127.]) + 150.
spkti = spkte + 150.

# Subtract dt to guarantee exact onset of PSPs
spkte -= dt
spkti -= dt
pulse = sim.DCSource(amplitude=0.2, start=50., stop=150.)
spikesource1E = sim.create(sim.SpikeSourceArray, {'spike_times':spkte})
spikesource1I = sim.create(sim.SpikeSourceArray, {'spike_times':spkti})

pulse.inject_into(ifcellscond[0])
sim.connect(spikesource1E, ifcellscond[0], weight=wconde, receptor_type='excitatory', delay=0.1)
sim.connect(spikesource1I, ifcellscond[0], weight=wcondi, receptor_type='inhibitory', delay=0.1)

for ifcell in ifcellscond:
    ifcell.record_v()

sim.run(tsim)

t = arange(0,tsim+dt,dt)
v = [array(ifcell.get_v().segments[0].analogsignals[0]).T[0] for ifcell in ifcellscond]

sim.end()

target = np.load("data_4.3.npy", encoding="latin1")
#print(target)

####### plot #######

f, ax = subplots(1,1,figsize=(12, 6), dpi=80)

for x in target[1]:
    ax.axvline(x, color="red")
for x in target[2]:
    ax.axvline(x, color="blue")
vnoised = v[0]# + normal(size=len(v[0]))*.1
ax.plot(t,target[0],color='green',label='target')
ax.plot(t,vnoised,color='black',label='u')
ax.set_xticks(range(0,401,100))
ax.set_xlabel(r'$t$ [ms]')
ax.set_ylabel(r'$u$ [mV]')
plt.savefig("3.png")

show()

