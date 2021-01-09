#!/usr/bin/env python

import sys
import os
import numpy as np
import random
import time

# Constants

it2fs = 1.0327503e0
au2k  = 315775.04e0
au2bar = 2.9421912e8

kB      = 1.380648e-23      # J / K
H_in_J  = 4.359744e-18
H_in_kJmol= 2625.499639
kB_a    = kB / H_in_J       # Hartree / K

class MD:
  '''Class for 2D MD with three test potentials 
  '''

  def __init__(self,mass_in=1,coords_in=[1,1],potential='1',dt_in=0.1e0,target_temp_in=298.15e0,seed_in=4911):
    # Init Vars

    self.step     = 0
    self.coords   = np.array(coords_in) 
    self.natoms   = 1
    self.dt_fs    = dt_in
    self.dt       = dt_in / it2fs
    self.target_temp  = target_temp_in
    self.forces   = np.zeros(2*self.natoms)
    self.momenta  = np.zeros(2*self.natoms)
    self.potential = potential

    # Random Number Generator
    
    if type(seed_in) is int:
      random.seed(seed_in)
      print("THE RANDOM NUMBER SEED WAS: %i" % (seed_in))
    else:
      try:
        random.setstate(seed_in)
      except:
        print("\tThe provided seed was neither an int nor a state of random")
        exit(1)


    # Mass
    
    self.mass   = mass_in
    self.conf_forces = np.zeros(2*self.natoms)
    self.masses = np.zeros(2*self.natoms)
    self.masses[0] = self.mass
    self.masses[1] = self.mass

    # Ekin and print
    
    self.epot  = 0.e0
    self.ekin  = 0.e0
    self.temp  = 0.e0
    self.vol   = 0.e0
    
  # -----------------------------------------------------------------------------------------------------
  def calc_init(self,init_momenta_in="random",momenta_in=np.array([]), init_temp=298.15e0):
    '''Initial calculation of energy, forces, momenta

    Args:
       init_momenta_in         (string,random,zero/random/read),
       momena_in               (list,np.array([]))
       init_temp               (float, 298.15)

    Returns:
       -
    '''
    (self.epot,self.forces) = self.calc_energy_forces_MD(self.potential)

    # Init momenta random
    self.momenta = np.zeros(2*self.natoms)
    self.momenta[0] = random.gauss(0.0,1.0) * np.sqrt(init_temp*self.mass)
    self.momenta[1] = random.gauss(0.0,1.0) * np.sqrt(init_temp*self.mass)

    TTT = (np.power(self.momenta, 2)/self.masses).sum()/2.0
    self.momenta *= np.sqrt(init_temp/(TTT*au2k))

  # -----------------------------------------------------------------------------------------------------
  def calc(self,only_energy=False):
    '''Calculation of energy, forces

    Args:
       only_energy (bool, False): Only energies are calculated

    Returns:
       energy (float): energy,
       forces (ndarray): forces
    '''
    self.epot = 0.e0
    self.forces = np.zeros(2*self.natoms)
    (self.epot,self.forces) = self.calc_energy_forces_MD(self.potential)


  # -----------------------------------------------------------------------------------------------------
  def calc_energy_forces_MD(self, potential):
    
    x = self.coords[0]
    y = self.coords[1]
    
    d = 40.0 
    e = 20.0 

    if potential == '1':
      
      a = 8.0e-6 / H_in_kJmol 
      b = 0.5    / H_in_kJmol 
      d = 80.0   
      e = 160.0   

      s1 = (x-d)*(x-d)
      s2 = (x-e)*(x-e)

      self.epot = a * s1*s2 + b * y*y
    
      self.forces[0] = 2.0 * a * ((x-d) * s2 + s1 * (x-e))
      self.forces[1] = 2.0 * b * y
    
    elif potential == '2':
      
      a = 1.4e-6 / H_in_kJmol
      b = 0.5    / H_in_kJmol
      c = 5.0e-5 / H_in_kJmol

      s1 = (x-d)*(x-d)
      s2 = (x+d)*(x+d)

      self.epot = a * s1*s2 + y*y * (b + c * s1*s2)
    
      self.forces[0] = 2.0 * a * ((x-d) * s2 + s1 * (x+d)) 
      self.forces[0] += 2.0 * c * y*y * ((x-d) * s2 + s1 * (x+d))
      self.forces[1] = 2.0 * y * (b + c * s1*s2) 
      
    elif potential == '3':
      
      a = 0.005 
      b = 0.040 
    
      exp_1 = np.exp((-a*(x-d)*(x-d)) + (-b*(y-e)*(y-e)))
      exp_2 = np.exp((-a*(x+d)*(x+d)) + (-b*(y+e)*(y+e)))

      self.epot = - np.log(exp_1 + exp_2) / H_in_kJmol

      self.forces[0] = -((-2.0*a*(x-d)*exp_1 - 2.0*a*(x+d)*exp_2) / (exp_1 + exp_2)) / H_in_kJmol
      self.forces[1] = -((-2.0*b*(y-e)*exp_1 - 2.0*b*(y+e)*exp_2) / (exp_1 + exp_2)) / H_in_kJmol
    
    else:
      print("\n\tInvalid Potential!")
      sys.exit(1)

    return (self.epot, self.forces)
   

  # -----------------------------------------------------------------------------------------------------
  def calc_etvp(self):
    '''Calculation of kinetic energy, total energy, volume, and pressure
  
    Args:
       -
  
    Returns:
       -
    '''
    # Ekin
    self.ekin = (np.power(self.momenta, 2)/self.masses).sum()
    self.ekin /= 2.0

    # T
    self.temp  = self.ekin/kB_a


  # -----------------------------------------------------------------------------------------------------
  def propagate(self, langevin=False, friction=1.0e-3):
    '''Propagate momenta/coords with Velocity Verlet

    Args:
       langevin                (bool, False)
       friction                (float, 10^-3 1/fs)
    Returns:
       -
    '''
    if langevin==True:
      prefac    = 2.0 / (2.0 + friction*self.dt_fs)
      rand_push = np.sqrt(self.target_temp*friction*self.dt_fs*kB_a/2.0e0)
      self.rand_gauss= np.zeros(shape=(len(self.momenta),), dtype=np.double)
      self.rand_gauss[0]   = random.gauss(0, 1)
      self.rand_gauss[1] = random.gauss(0, 1)
        
      self.momenta += np.sqrt(self.masses) * rand_push * self.rand_gauss
      self.momenta -= 0.5e0 * self.dt * self.forces
      self.coords  += prefac * self.dt * self.momenta / self.masses

    else:
      self.momenta -= 0.5e0 * self.dt * self.forces
      self.coords  += self.dt * self.momenta / self.masses


  # -----------------------------------------------------------------------------------------------------
  def up_momenta(self, langevin=False, friction=1.0e-3):
    '''Update momenta with Velocity Verlet
  
    Args:
       -
   
    Returns:
       -
    '''
    if langevin==True:
      prefac = (2.0e0 - friction*self.dt_fs)/(2.0e0 + friction*self.dt_fs)
      rand_push = np.sqrt(self.target_temp*friction*self.dt_fs*kB_a/2.0e0)
      self.momenta *= prefac
      self.momenta += np.sqrt(self.masses) * rand_push * self.rand_gauss
      self.momenta -= 0.5e0 * self.dt * self.forces
    else:
      self.momenta -= 0.5e0 * self.dt * self.forces


  # -----------------------------------------------------------------------------------------------------
  def confine_coordinate(self, ats):
    '''Confine coordinate

    Args:
      ats       [[dim, x0, k],[...]]
    Returns:
      -
    '''
    conf_energy = 0.0e0
    self.conf_forces = np.zeros(2)
    for i in range(len(ats)):

        coord = self.coords[ats[i][0]] 
        coord0 = ats[i][1] 
        k = ats[i][2] / H_in_kJmol
        
        self.epot += 0.5 * k * np.power(coord-coord0,2.e0)
        gradient = np.array([1.0,0.0]) if ats[i][0] == 1 else np.array([0.0,1.0])
        self.conf_forces += k * (coord-coord0) * gradient
    
    self.forces += self.conf_forces
    self.epot += conf_energy


