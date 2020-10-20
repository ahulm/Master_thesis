import sys
import numpy as np

from eABF import *
from ABF_helpers import *

bohr2angs = 0.52917721092e0
kB      = 1.380648e-23      # J / K
H_in_J  = 4.359744e-18
kB_a    = kB / H_in_J       # Hartree / K

def metaD(self, ats, variance=1.0, height=0.1, time_int=20, WT=True, dT=20):
    '''Metadynamics and Well-Tempered Metadynamics
    '''			
    # get reaction coordinate
    (xi, delta_xi) = get_coord(self, ats)
    
    if self.step == 0: 
        
        self.minx       = np.array([item[1] for item in ats])	
        self.maxx       = np.array([item[2] for item in ats])
        self.dx         = np.array([variance/5 for item in ats])
        
        self.height     = height/H_in_kJmol/time_int

        self.nbins      = int(np.prod(np.floor(np.abs(self.maxx-self.minx)/self.dx))) 
        self.grid       = np.array([self.minx+i*self.dx+self.dx/2 for i in range(self.nbins)])
        self.bias_pot   = np.array(np.zeros((len(ats),self.nbins)), dtype=np.float64)
        self.bias_force = np.array(np.zeros((len(ats),self.nbins)), dtype=np.float64)
        
        self.traj       = np.array([xi])
    
    else:
        self.traj = np.append(self.traj, [xi], axis = 0)
    
    # main metaD routine
    if (xi <= self.maxx).all() and (xi >= self.minx).all():
        
        # get current bin for 1D or 2D reaction coordinate
        if len(ats) == 1:
            bink = int(np.floor(abs(xi[0]-self.minx[0])/self.dx[0]))
        else:
            bin0 = int(np.floor(abs(self.xi[0]-self.minx[0])/self.dx[0]))
            bin1 = int(np.floor(abs(xi[1]-minx[1])/dx[1]))
            bink = bin0 + int(np.floor(np.abs(self.maxx[0]-self.minx[0])/self.dx[0]))*bin1

        for i in range(len(ats)):
            
            # update bias every time_int's step
            if self.step%time_int == 0:
                
                dx = self.grid - xi[i]
                bias_factor = time_int * self.height * np.exp(-0.5*np.power(dx ,2.0)/variance)
                
                if WT == True:
                    bias_factor *= np.exp(-self.bias_pot[i][bink]/(kB_a*dT))
                
                self.bias_pot   += bias_factor.T
                self.bias_force -= bias_factor.T * dx.T/variance
            
            # add bias force to system
            self.forces += self.bias_force[i][bink] * delta_xi[i]
    
    # Output result
    if self.step%10000 == 0:
        out = open(f"out_metaD.txt", "w")
        out.write("%6s\t%14s\t%14s\t%14s\n" % ("Bin", "Xi", "Bias Pot", "Bias Force"))
        for i in range(len(self.bias_pot[0])):
            out.write("%6d\t%14.6f\t%14.6f\t%14.6f\n" % (i, self.grid[i], self.bias_pot[0][i], self.bias_force[0][i]))
        out.close()

	
