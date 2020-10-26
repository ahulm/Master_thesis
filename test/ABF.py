import sys
import numpy as np

from eABF import *
from ABF_helpers import *

bohr2angs   = 0.52917721092e0
kB          = 1.380648e-23      # J / K
H_in_J      = 4.359744e-18
kB_a        = kB / H_in_J       # Hartree / K

def ABF(self, ats):
    '''Adaptive Biasing force method 
    '''			
    # get reaction coordinate
    (xi, delta_xi) = get_coord(self, ats)
    
    if self.step == 0: 
        self.minx       = np.array([1/item[2] for item in ats])	
        self.maxx       = np.array([1/item[1] for item in ats])
        self.dx         = np.array([1/item[3] for item in ats])
        self.ramp_count = np.array([item[4] for item in ats])   	
        self.traj       = np.array([xi])
         
        self.nbins      = int(np.prod(np.floor(np.abs((1/self.maxx)-(1/self.minx))/(1/self.dx)))) 
        self.bin_list   = np.array(np.zeros(self.nbins), dtype=np.int32)
        self.biases     = np.array(np.zeros((len(ats),self.nbins)), dtype=np.float64)
        self.grad_mean  = np.array(np.zeros(self.nbins), dtype=np.float64) 
        
        self.grid = np.array([1.0/(ats[0][1]+i*ats[0][3]) for i in range(self.nbins)])

    else:
        self.traj = np.append(self.traj, [xi], axis = 0)
    
    # main ABF routine
    if (xi <= self.maxx).all() and (xi >= self.minx).all():
        
        # get current bin for 1D or 2D reaction coordinate
        if len(ats) == 1:
            bink = int(np.floor(abs(1/xi[0]-ats[0][1])/ats[0][3]))
        else:
            bin0 = int(np.floor(abs(xi[0]-self.minx[0])/self.dx[0]))
            bin1 = int(np.floor(abs(xi[1]-self.minx[1])/self.dx[1]))
            bink = bin0 + int(np.floor(np.abs(self.maxx[0]-self.minx[0])/self.dx[0]))*bin1
        
        self.bin_list[bink] += 1		
        
        for i in range(len(ats)):
        	
            # R(N,k)
            Rk = 1.0 if self.bin_list[bink] > self.ramp_count[i] else self.bin_list[bink]/self.ramp_count[i]	
            
            # inverse gradient v_i
            delta_xi_n = np.linalg.norm(delta_xi[i])
            v_i = delta_xi[i]/(delta_xi_n*delta_xi_n)
            self.grad_mean[bink] += delta_xi_n
            
            # apply biase force
            self.biases[i][bink] += np.dot(self.forces, v_i) + 2*kB_a*100*xi[i] 
            self.forces -= Rk * (self.biases[i][bink]/self.bin_list[bink]) * delta_xi[i]
     
    write_traj(self)
    
    # Output result
    if self.step%1000 == 0:
        write_output(self, ats)
	
