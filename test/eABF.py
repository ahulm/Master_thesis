from ABF_helpers import *

fs_to_au = 41.341374575751    # a.u. per fs
H_to_u   = 2.921264912428e-8  # Hartree to mass unit
bohr2angs = 0.52917721092e0

def eABF(self, ats, sigma = 2.0, tau = 10000.0):
    '''extended Adaptive Biasing force (eABF) method 
    
    implemented reaction coordinates: bend angle or torsion angle,
    type of reaction coordinate chosen by number of given atoms
    
    Args:
      ats		(array, [[[index of atom1,index of atom2,...],minx,maxx,bin_width,ramp_count][...]]
      sigma		(double, standard devation between fictitious particle and collective variable in Bohr or Degree)
      tau		(double, oscillation periode of fictitious particle in fs)
    
    returns: 
      - 
    '''			
    # get reaction coordinate 
    (xi, delta_xi) = get_coord(self, ats)

    # initialize eABF variables during first step
    if self.step == 0:
        
        self.minx       = np.array([item[1] for item in ats])    
        self.maxx       = np.array([item[2] for item in ats])
        self.dx         = np.array([item[3] for item in ats])  
        self.ramp_count = np.array([item[4] for item in ats])
        
        self.nbins      = int(np.prod(np.floor(np.abs(self.maxx-self.minx)/self.dx))) 
        self.bin_list   = np.array(np.zeros(self.nbins), dtype=np.int32)
        self.biases     = np.array(np.zeros((len(ats), self.nbins)), dtype=np.float64)
        
        tau             = tau*fs_to_au
        self.k          = (kB_a*T) / (sigma*sigma) 
        self.mass       = kB_a * H_to_u * self.target_temp * (tau/(2*np.pi*sigma)) * (tau/(2*np.pi*sigma))
        
        print("  Spring constant for extended variable:\t%14.6f Hartree/radiant^2" % (self.k))
        print("  fictitious mass for extended variable:\t%14.6f u" % (self.mass))
        
        for i in range(len(xi)):
            extend_system(self, i, xi)
        
        self.traj       = np.array([xi])
        self.etraj      = np.array([self.ext_coords])
    
    else:

    	self.traj       = np.append(self.traj, [xi], axis = 0)
    	self.etraj      = np.append(self.etraj, [self.ext_coords], axis = 0)		
    
    # velocity verlet for extended variable
    propagate_extended(self, langevin = True)	
   
    # main eABF routine
    if (self.ext_coords <= self.maxx).all() and (self.ext_coords >= self.minx).all():
    	
    	# get currend lambda-bin for 1D or 2D reaction coordinate 
    	if len(ats) == 1:
    		bink = int(np.floor(abs(self.ext_coords[0]-self.minx[0])/self.dx[0]))
    	
    	elif len(ats) == 2:
    		bin0 = int(np.floor(abs(self.ext_coords[0]-self.minx[0])/self.dx[0]))
    		bin1 = int(np.floor(abs(self.ext_coords[1]-self.minx[1])/self.dx[1]))
    		bink = bin0  + int(np.floor(np.abs(self.maxx[0]-self.minx[0])/self.dx[0]))*bin1
    	
    	self.bin_list[bink] += 1		
    	
    	for i in range(len(ats)):
            
            # harmonic coupling of exteded coordinate to reaction coordinate 
            dxi                 = self.ext_coords[i] - xi[i]
            self.ext_forces[i]  = self.k * dxi			
            self.forces        -= self.k * dxi * delta_xi[i] 
           
            # ramp function R(N,k)
            Rk = 1.0 if self.bin_list[bink] > self.ramp_count[i] else self.bin_list[bink]/self.ramp_count[i]	
            
            # apply average force on extended coordinate
            self.biases[i][bink] += self.k * dxi
            self.ext_forces      -= Rk * self.biases[i][bink]/self.bin_list[bink]
    
    else:

        for i in range(len(ats)):
            
            # outside of bins only harmonic coupling without bias
            dxi                = self.ext_coords[i] - xi[i]
            self.ext_forces[i] = self.k * dxi
            self.forces       -= self.k * dxi * delta_xi[i]
            
    # velocity verlet for extended variable
    up_momenta_extended(self, langevin = True)	
   
    # output section
    write_traj(self, extended = True)
    if self.step%10000 == 0:
    	write_output(self, ats)

	
