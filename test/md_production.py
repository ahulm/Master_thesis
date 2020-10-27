#!/usr/bin/env python
#Imports
import sys
import os
import numpy as np
import random
import time
import subprocess

from InterfaceMD import *
#from ABF import *
#from eABF import *
#from metaD import * 

from adaptive_biasing import * 

bohr2angs = 0.52917721092e0
################# Imput Section ####################
# MD
seed            = np.random.randint(2147483647) # upper boundary is simply largest signed int value 
nsteps          = 10000000   # number of MD steps
dt              = 5.0e0      # fs
target_temp     = 300.0      # K
mass            = 10.0
friction        = 1.0e-3
coords          = [60.0,0.0]
potential       = '1'

# ABF
ats     = [[6,3000,28000,100,100]]#,2.0,100000]]

# metadynamics
ats = [[6,1000.0,28000.0,100.0,0.01,2.0,20,2000]]

# US
#ats = [[1,0.0,1000.0]]

#################### Pre-Loop ####################
start_loop = time.perf_counter()
step_count = 0

the_md = MD(mass_in=mass,coords_in=coords,potential=potential,dt_in=dt,target_temp_in=target_temp,seed_in=seed)
the_md.calc_init()
#the_md.confine_coordinate(ats)
the_bias = ABM(the_md, ats, method = 'ABF', output_freq = 10000, random_seed = seed)
the_bias.ABF()
#eABF(themd,ats,sigma=sigma,tau=tau)
#metaD(the_md, ats,variance=variance, height=height,time_int=tau,WT=WT,dT=dT)
the_md.calc_etvp()

print ("%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f" %(the_md.step*the_md.dt*it2fs,the_md.coords[0],the_md.coords[1],the_md.epot,the_md.ekin,the_md.epot+the_md.ekin,the_md.temp))

#################### the MD loop ####################
while step_count < nsteps:
    start_loop = time.perf_counter()
    the_md.step += 1
    step_count  += 1
	
    the_md.propagate(langevin=True, friction=friction)
    the_md.calc()
   # the_md.confine_coordinate(ats)
    the_bias.ABF()
  #  eABF(the_md, ats, sigma=sigma, tau=tau)
  #  metaD(the_md, ats, variance=variance, height=height,time_int=tau,WT=WT,dT=dT)
    the_md.up_momenta(langevin=True, friction=friction)
    the_md.calc_etvp()

    #if step_count%5000 == 0:
    #    subprocess.call(['sh', './copy.sh'])
    #    os.rename('abf_traj_copy.dat', 'abf_traj_%5d.dat' % (step_count))

    print ("%11.2f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f" %(the_md.step*the_md.dt*it2fs,the_md.coords[0],the_md.coords[1],the_md.epot,the_md.ekin,the_md.epot+the_md.ekin,the_md.temp))
        
