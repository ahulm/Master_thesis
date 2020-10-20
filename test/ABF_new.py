import sys
import numpy as np

from eABF import *
from ABF_helpers import *

bohr2angs = 0.52917721092e0
kB      = 1.380648e-23      # J / K
H_in_J  = 4.359744e-18
kB_a    = kB / H_in_J       # Hartree / K

class ABF():
    '''Adaptive Biasing force method 
    '''	

    def __init__(self, ats, T=300):
        
        self.minx = np.array([])	
        self.maxx = np.array([])
        self.dx = np.array([])
        self.xi = np.array([])
        self.ramp_count = np.array([])   	
        self.delta_xi = [np.zeros(2*self.natoms) for i in range(len(ats))]
    
        # loop over reaction coordinates
        for i in range(len(ats)):	
    	
            self.minx = np.append(minx, ats[i][1])    
            self.maxx = np.append(maxx, ats[i][2])
            self.dx   = np.append(dx, ats[i][3])  
            self.ramp_count = np.append(ramp_count,ats[i][4])
    
            if ats[i][0] == 1:
                self.xi = np.append(xi, self.coords[0]*bohr2angs)
                self.delta_xi[i] += np.array([1,0])
            
            elif ats[i][0] == 2:
                self.xi = np.append(xi, self.coords[1]*bohr2angs)
                self.delta_xi[i] += np.array([0,1])
    
        # number of bins
        self.nbins = int(np.prod(np.floor(np.abs(maxx-minx)/dx))) 
    
        # initialize variables for ABF
        self.bin_list = np.array(np.zeros(nbins), dtype=np.int32)
        self.biases = np.array(np.zeros((2,nbins)), dtype=np.float64)
        self.traj = np.array([xi],dtype=np.float64)
   
    def ABF(self, the_md)
        
        for i in range(len(self.xi)):
            
            if ats[i][0] == 1:
                
                self.xi = np.append(xi, self.coords[0]*bohr2angs)
                self.delta_xi[i] += np.array([1,0])
            
            elif ats[i][0] == 2:
                self.xi = np.append(xi, self.coords[1]*bohr2angs)
                self.delta_xi[i] += np.array([0,1])
        for i in range(len(self.xi)):
            
            if (self.xi <= self.maxx).all() and (self.xi >= self.minx).all():
        
                if len(self.xi) == 1:
                    bink = int(np.floor(abs(self.xi[0]-self.minx[0])/self.dx[0]))
                else:
                    bin0 = int(np.floor(abs(self.xi[0]-self.minx[0])/self.dx[0]))
                    bin1 = int(np.floor(abs(self.xi[1]-self.minx[1])/self.dx[1]))
                    bink = bin0 + int(np.floor(np.abs(self.maxx[0]-selfminx[0])/self.dx[0]))*bin1

                self.bin_list[bink] += 1		
        
                Rk = 1.0 if self.bin_list[bink] > self.ramp_count[i] else self.bin_list[bink]/self.ramp_count[i]	
                self.delta_xi_n = np.linalg.norm(self.delta_xi[i])
                v_i = self.delta_xi[i]/(self.delta_xi_n*self.delta_xi_n)
                F_xi = np.dot(the_md.forces, v_i) 	
                self.biases[i][bink] += F_xi 
                the_md.forces -= Rk * (self.biases[i][bink]/self.bin_list[bink]) * self.delta_xi[i]

    if self.step%10000 == 0:
        write_output(self, ats, minx, maxx, dx)
	
