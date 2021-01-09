import sys
import time
import random
import numpy as np
import scipy.integrate as spi
from scipy import interpolate

# energy units
kB          = 1.380648e-23      # J / K
H_to_kJmol  = 2625.499639       #
H_to_J      = 4.359744e-18      #
kB_a        = kB / H_to_J       # Hartree / K
H2au        = 2.921264912428e-8 # Hartree to aromic mass unit
au2k        = 315775.04e0

# time units
it2fs       = 1.0327503e0       # fs per iteration
fs2au       = 41.341374575751   # a.u. per fs

class ABM:
    '''Class for adaptive biasing methods for 1D or 2D collective variables (CV)

    Available methods:
        ABF             Adaptive biasing force method, CV's have to be orthogonal to constraints and to each other!
                        1D: On-the-fly free energy estimate, 
                        2D: Postprocessing step (FEM integration) necessary 

                        N_full:     Sampels per bin when full bias is applied, if N_bin < N_full: Bias = 1/N_bin * F_bias

                        ats = [[CV1, minx, maxx, dx],[CV2, ...]]

        metaD           Metadynamics or Well-Tempered metadynamics for 1D or 2D CV's
                        On-the-fly free energy estimate: dF= -(T+deltaT)/deltaT * V_bias

                        variance:   Gaussian variance [Bohr]
                        height:     Gaussian height [kJ/mol]
                        update_int: Intervall for deposition of gaussians [steps]
                        deltaT:     for Well-Tempered-metaD: deltaT -> 0            ordinary MD
                                                             500 < deltaT < 5000    WT-metaD
                                                             deltaT -> inf          standard metaD

                        ats = [[CV1, minx, maxx, dx, variance],[CV2,...]

        eABF            extended adaptive biasing force method for 1D or 2D CV's
                        Unbiased force estimate obtained from CZAR estimator (Lesage, JPCB, 2016)

                        N_full:     Samples per bin where full bias is applied
                        sigma:      standard deviation between CV and fictitious particle [Bohr]
                                    both are connected by spring force with force constant k=1/(beta*sigma^2) [Hartree]
                        mass:       mass of fictitious particle [a.u.]

                        ats = [[CV1, minx, maxx, dx, sigma, mass],[CV2, ...]]

        meta-eABF       Bias of extended coordinate by (WT) metaD + eABF (WTM-eABF or meta-eABF)
                        Unbiased force estimate obtained from CZAR estimator (Lesage, JPCB, 2016)

                        ats = [[CV1, minx, maxx, dx, sigma, tau, variance],[CV2,...]]

    Init parameters:
        MD:             MD object from InterfaceMD
        method:         'ABF', 'metaD', 'eABF' or 'meta-eABF'
        ats:            input parameters that have to be initialized beforehand
        output_frec:    Number of steps between outputs (default: 1000)
        friction:       use same friction coefficient for Langevin dynamics of extended system and physical system (default: 1.0e-3)
        random_seed:    use same random number seed for Langevin dynamics of extended and physical system (default: system time)

    Output:
        bias_out.txt    text file containing i.a. CV, histogramm, bias, standard free energy and geometric free energy
    '''
    def __init__(self, MD, ats, method = 'meta-eABF', output_freq = 1000, friction = 1.0e-3, random_seed = None):
	
        # general parameters
        self.the_md     = MD
        self.method     = method
        self.out_freq   = output_freq
        self.friction   = friction
        
        self.ncoords    = len(ats)
        self.coord      = np.array([item[0] for item in ats])
        self.minx       = np.array([item[1] for item in ats])
        self.maxx       = np.array([item[2] for item in ats])
        self.dx         = np.array([item[3] for item in ats])

        (xi, delta_xi, div_delta_xi) = self.__get_coord()
        self.traj = np.array([xi])

        # get number of bins
        self.nbins_per_dim = np.array([1,1])
        self.grid          = []
        for i in range(len(self.coord)):
            self.nbins_per_dim[i] = int(np.ceil(np.abs(self.maxx[i]-self.minx[i])/self.dx[i])) if self.dx[i] > 0 else 1
            self.grid.append(np.linspace(self.minx[i]+self.dx[i]/2,self.maxx[i]-self.dx[i]/2,self.nbins_per_dim[i]))
        
        self.nbins = np.prod(self.nbins_per_dim)
        
        self.bias       = np.zeros((self.ncoords,self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)
        self.histogramm = np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)
        self.geom_corr  = np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)

        if method == 'ABF' or method == 'reference':
            # no more parameters to initialize
            pass

        elif method == 'eABF' or method == 'meta-eABF':
            # setup extended system for eABF or meta-eABF
            
            sigma         = np.array([item[4] for item in ats])
            self.k        = (kB_a*self.the_md.target_temp) / (sigma*sigma)
            self.ext_mass = np.array([item[5] for item in ats])
            
            self.ext_coords = np.copy(xi)
            self.etraj      = np.copy(self.traj)

            self.ext_forces  = np.zeros(self.ncoords)
            self.ext_momenta = np.zeros(self.ncoords)
            
            if type(random_seed) is int:
                random.seed(random_seed)
            else:
                print("\nNo seed was given for the random number generator of ABM so the system time is used!\n")
            
            for i in range(self.ncoords):
                # initialize extended system at target temp of MD simulation
            
                self.ext_momenta[i] = random.gauss(0.0,1.0)*np.sqrt(self.the_md.target_temp*self.ext_mass[i])
                TTT  = (np.power(self.ext_momenta, 2)/self.ext_mass).sum()
                TTT /= (self.ncoords)
                self.ext_momenta *= np.sqrt(self.the_md.target_temp/(TTT*au2k))
            
            # accumulators for czar estimator
            self.force_correction_czar = np.copy(self.bias)
            self.hist_z                = np.copy(self.histogramm)
            
            if method == 'meta-eABF':
                # parameters for meta-eABF or WTM-eABF
                
                self.metapot  = np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)
                self.abfforce = np.copy(self.bias)
                self.variance = np.array([item[6] for item in ats])
                self.maxBias  = [1.0/H_to_kJmol,1.0/H_to_kJmol]        

        elif method == 'metaD':
            # parameters for metaD or WT-metaD
            
            self.metapot  = np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)
            self.variance = np.array([item[4] for item in ats])
            self.maxBias  = [1.0/H_to_kJmol,1.0/H_to_kJmol]        
       
        else:
            print('\nMethod not implemented!')
            print('Available choices: reference, ABF, eABF, metaD or meta-eABF.')
            sys.exit(1)

        self.__print_parameters()

    # -----------------------------------------------------------------------------------------------------
    def reference(self, write_traj=True):
        '''get free energy from unbiased trajectory

        args:
            write_traj      (bool, True, write trajectory to CV_traj.dat)

        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, div_delta_xi) = self.__get_coord()
        self.traj = np.append(self.traj, [xi], axis = 0)

        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self.__get_bin(xi)
            self.histogramm[bink[1],bink[0]] += 1
            
            self.geom_corr[bink[1],bink[0]] += self.__get_gradient_correction(delta_xi)

        else:
            # confinement
            for i in range(self.ncoords):
                if xi[i] > self.maxx[i]:
                    self.the_md.forces -= 10.0/H_to_kJmol * (self.maxx[i]-xi[i]) * delta_xi[i] 

                elif xi[i] < self.minx[i]:
                    self.the_md.forces -= 10.0/H_to_kJmol * (self.minx[i]-xi[i]) * delta_xi[i]   

        if self.the_md.step%self.out_freq == 0:
            # write output

            if write_traj == True:
                self.write_traj(xi)
            
            self.dF = -kB_a*self.the_md.target_temp*np.log(self.histogramm, out=np.zeros_like(self.histogramm),where=(self.histogramm!=0))
            self.__get_geom_correction()
            self.write_output()
            self.write_restart()

        self.timing = time.perf_counter() - start
        

    # -----------------------------------------------------------------------------------------------------
    def ABF(self, N_full=100, write_traj=True):
        '''Adaptive biasing force method

        args:
            N_full:         (double, 100, number of sampels when full bias is applied to bin)
            write_traj      (bool, True, write trajectory to CV_traj.dat)

        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, div_delta_xi) = self.__get_coord()
        self.traj = np.append(self.traj, [xi], axis = 0)

        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self.__get_bin(xi)
            self.histogramm[bink[1],bink[0]] += 1
            
            self.geom_corr[bink[1],bink[0]] += self.__get_gradient_correction(delta_xi)

            for i in range(self.ncoords):

                # linear ramp function R(N,k)
                Rk = 1.0 if self.histogramm[bink[1],bink[0]] > N_full else self.histogramm[bink[1],bink[0]]/N_full

                # inverse gradient v_i
                delta_xi_n = np.linalg.norm(delta_xi[i])
                v_i = delta_xi[i]/(delta_xi_n*delta_xi_n)
		
                # apply biase force
                self.bias[i][bink[1],bink[0]] += np.dot(self.the_md.forces, v_i) - kB_a * self.the_md.target_temp * div_delta_xi[i]
                self.the_md.forces            -= Rk * (self.bias[i][bink[1],bink[0]]/self.histogramm[bink[1],bink[0]]) * delta_xi[i]
                
        else:
            # confinement
            for i in range(self.ncoords):
                if xi[i] > self.maxx[i]:
                    self.the_md.forces -= 10.0/H_to_kJmol * (self.maxx[i]-xi[i]) * delta_xi[i] 

                elif xi[i] < self.minx[i]:
                    self.the_md.forces -= 10.0/H_to_kJmol * (self.minx[i]-xi[i]) * delta_xi[i]   

        if self.the_md.step%self.out_freq == 0:
            # write output

            if write_traj == True:
                self.write_traj(xi)
            
            self.mean_force = self.__get_cond_avg(self.bias, self.histogramm)
            self.__F_from_Force(self.mean_force)
            self.write_output()
            self.write_restart()

        self.timing = time.perf_counter() - start
    
    #------------------------------------------------------------------------------------------------------
    def metaD(self, gaussian_height, update_int=50, WT_dT=2000, WT=True, grid=False, write_traj=True):
        '''Metadynamics and Well-Tempered Metadynamics

        args:
            gaussian_height     (double, -, heigth of gaussians for bias potential)
            WT                  (bool, True, use Well-Tempered metaD)
            grid                (bool, True, use grid to save bias between function calls)
            write_traj          (bool, True, write trajectory to CV_traj.dat)

        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, div_delta_xi) = self.__get_coord()

        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self.__get_bin(xi)
            self.histogramm[bink[1], bink[0]] += 1
            
            self.geom_corr[bink[1],bink[0]] += self.__get_gradient_correction(delta_xi)
                        
            # apply bias 
            bias = self.__get_metaD_bias(xi, bink, gaussian_height, update_int, self.the_md.step, WT_dT, WT, grid)
            for i in range(self.ncoords):
                
                if abs(bias[i]) >= self.maxBias[i]:
                    self.maxBias[i] = abs(bias[i])
                
                self.the_md.forces += bias[i] * delta_xi[i]
        
        else:
            # confinement 
            for i in range(self.ncoords):
                if xi[i] > self.maxx[i]:
                    self.the_md.forces -= 10.0*self.maxBias[i] * (self.maxx[i]-xi[i]) * delta_xi[i] 

                elif xi[i] < self.minx[i]:
                    self.the_md.forces -= 10.0*self.maxBias[i] * (self.minx[i]-xi[i]) * delta_xi[i]   
 
        self.traj = np.append(self.traj, [xi], axis = 0)
        
        if self.the_md.step%self.out_freq == 0:
            # write output

            if write_traj == True:
                self.write_traj(xi)
            
            self.__F_from_metaD(WT_dT, WT=WT)
            self.write_output()
            self.write_restart()
        
        self.timing = time.perf_counter() - start

    #------------------------------------------------------------------------------------------------------
    def eABF(self, N_full=100, write_traj = True):
        '''extended Adaptive Biasing Force method

        args:
	    N_full:         (double, 100, number of sampels when full bias is applied to bin)
            write_traj      (bool, True, write trajectory to CV_traj.dat)
        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, div_delta_xi) = self.__get_coord()

        self.__propagate_extended()

        if (self.ext_coords <= self.maxx).all() and (self.ext_coords >= self.minx).all():
            
            la_bin = self.__get_bin(xi, extended = True)
            self.histogramm[la_bin[1],la_bin[0]] += 1
             
            for i in range(self.ncoords):

                # harmonic coupling of extended coordinate to reaction coordinate
                dxi                 = self.ext_coords[i] - xi[i]
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]
                
                # apply biase force
                Rk = 1.0 if self.histogramm[la_bin[1],la_bin[0]] > N_full else self.histogramm[la_bin[1],la_bin[0]]/N_full
                self.bias[i][la_bin[1],la_bin[0]] -= self.k[i] * dxi
                self.ext_forces[i] += Rk * self.bias[i][la_bin[1],la_bin[0]]/self.histogramm[la_bin[1],la_bin[0]]
                  
        else:

            for i in range(self.ncoords):

                # harmonic coupling
                dxi                 = self.ext_coords[i] - xi[i]
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]

                # confinement
                if self.ext_coords[i] > self.maxx[i]:
                    self.ext_forces -= 1.0/H_to_kJmol * (self.maxx[i]-self.ext_coords[i]) 

                elif self.ext_coords[i] < self.minx[i]:
                    self.ext_forces -= 1.0/H_to_kJmol * (self.minx[i]-self.ext_coords[i])  

        if (xi <= self.maxx).all() and (xi >= self.minx).all():
            # accumulators for czar estimator
           
            bink = self.__get_bin(xi)
            self.hist_z[bink[1],bink[0]] += 1
            
            self.geom_corr[bink[1],bink[0]] += self.__get_gradient_correction(delta_xi)
            
            for i in range(self.ncoords):
                self.force_correction_czar[i][bink[1],bink[0]] += self.k[i] * (self.ext_coords[i] - self.grid[i][bink[i]])

        self.__up_momenta_extended()

        self.traj  = np.append(self.traj, [xi], axis = 0)
        self.etraj  = np.append(self.etraj, [self.ext_coords], axis = 0)

        if self.the_md.step%self.out_freq == 0:
            # write output 

            if write_traj == True:
                self.write_traj(xi,extended = True)
            
            self.__F_from_CZAR()
            self.write_output()
            self.write_restart()
        
        self.timing = time.perf_counter() - start

    #------------------------------------------------------------------------------------------------------
    def meta_eABF(self, N_full=100, gaussian_height=0, update_int=20, WT_dT=2000, WT = True, grid = True, write_traj = True):
        '''meta-eABF or WTM-eABF: combination of eABF with metadynamic

        args:
	    N_full:             (double, 100,  number of sampels when full bias is applied to bin)
            gaussian_height     (double, 0.2,  height of gaussians for metaD potential in kJ/mol)
            update_int          (int,    20,   intevall for deposition of gaussians)
            WT                  (bool,   True, use Well-Tempered metadynamics)
            grid                (bool,   True, store metadynamic bias on grid between function calls)
            write_traj          (bool,   True, write trajectory to CV_traj.dat)

        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, div_delta_xi) = self.__get_coord()

        self.__propagate_extended()

        self.traj  = np.append(self.traj, [xi], axis = 0)
        self.etraj  = np.append(self.etraj, [self.ext_coords], axis = 0)

        if (self.ext_coords <= self.maxx).all() and (self.ext_coords >= self.minx).all():

            la_bin = self.__get_bin(xi, extended = True)
            
            self.histogramm[la_bin[1],la_bin[0]] += 1
            meta_force = self.__get_metaD_bias(self.ext_coords, la_bin, gaussian_height, update_int, self.the_md.step, WT_dT, WT, grid)
            
            for i in range(self.ncoords):

                # harmonic coupling of extended coordinate to reaction coordinate
                dxi                 = self.ext_coords[i] - xi[i]
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]

                # metadynamics bias
                self.ext_forces[i] += meta_force[i]
      
                if abs(meta_force[i]) >= self.maxBias[i]:
                    self.maxBias[i] = abs(meta_force[i])

                # eABF bias
                Rk = 1.0 if self.histogramm[la_bin[1],la_bin[0]] > N_full else self.histogramm[la_bin[1],la_bin[0]]/N_full
                self.abfforce[i][la_bin[1],la_bin[0]] -= self.k[i] * dxi  
                self.ext_forces[i] += Rk * self.abfforce[i][la_bin[1],la_bin[0]]/self.histogramm[la_bin[1],la_bin[0]]  
            
        else:

            for i in range(self.ncoords):

                # harmonic coupling
                dxi                 = self.ext_coords[i] - xi[i]
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]

                # confinement
                if self.ext_coords[i] > self.maxx[i]:
                    self.ext_forces -= 10.0*self.maxBias[i] * (self.maxx[i]-self.ext_coords[i]) 

                if self.ext_coords[i] < self.minx[i]:
                    self.ext_forces -= 10.0*self.maxBias[i] * (self.minx[i]-self.ext_coords[i])  

        if (xi <= self.maxx).all() and (xi >= self.minx).all():
            # accumulators for czar estimator
            
            bink = self.__get_bin(xi)
            self.hist_z[bink[1],bink[0]] += 1
            
            self.geom_corr[bink[1],bink[0]] += self.__get_gradient_correction(delta_xi)

            for i in range(self.ncoords):
                self.force_correction_czar[i][bink[1],bink[0]] += self.k[i] * (self.ext_coords[i] - self.grid[i][bink[i]])
        
        self.__up_momenta_extended()

        self.traj  = np.append(self.traj, [xi], axis = 0)
        self.etraj  = np.append(self.etraj, [self.ext_coords], axis = 0)

        if self.the_md.step%self.out_freq == 0:
            # write output
            
            if write_traj == True:
                self.write_traj(xi,extended = True)
            
            self.__F_from_CZAR() 
            self.write_output()
            self.write_restart()
        
        self.timing = time.perf_counter() - start

    # -----------------------------------------------------------------------------------------------------
    def __get_angle(self):
        '''get bond angle between three atoms in rad

        args:
            -
        returns:
            angle        (double, -)
        '''    
        p1 = np.array([self.the_md.coords[3*self.atom[0]],self.the_md.coords[3*self.atom[0]+1],self.the_md.coords[3*self.atom[0]+2]],dtype=np.float)
        p2 = np.array([self.the_md.coords[3*self.atom[1]],self.the_md.coords[3*self.atom[1]+1],self.the_md.coords[3*self.atom[1]+2]],dtype=np.float)
        p3 = np.array([self.the_md.coords[3*self.atom[2]],self.the_md.coords[3*self.atom[2]+1],self.the_md.coords[3*self.atom[2]+2]],dtype=np.float)
                        
        q12 = p2-p1
        q23 = p2-p3
        
        q12_n = np.linalg.norm(q12)
        q23_n = np.linalg.norm(q23)
        
        q12_u = q12/q12_n  
        q23_u = q23/q23_n
        
        return np.arccos(np.dot(q12_u,q23_u))
        
    # -----------------------------------------------------------------------------------------------------
    def __get_torsion(self):
        '''get torsion angle between four atoms in rad

        args:
            -
        returns:
            torsion        (double, -)
        '''
        p1 = np.array([self.the_md.coords[3*self.atom[0]],self.the_md.coords[3*self.atom[0]+1],self.the_md.coords[3*self.atom[0]+2]],dtype=np.float)
        p2 = np.array([self.the_md.coords[3*self.atom[1]],self.the_md.coords[3*self.atom[1]+1],self.the_md.coords[3*self.atom[1]+2]],dtype=np.float)
        p3 = np.array([self.the_md.coords[3*self.atom[2]],self.the_md.coords[3*self.atom[2]+1],self.the_md.coords[3*self.atom[2]+2]],dtype=np.float)
        p4 = np.array([self.the_md.coords[3*self.atom[3]],self.the_md.coords[3*self.atom[3]+1],self.the_md.coords[3*self.atom[3]+2]],dtype=np.float)
        
        q12 = p1 - p2
        q23 = p3 - p2
        q34 = p4 - p3
        
        q23_u = q23 / np.linalg.norm(q23)
        
        n1 =  q12 - np.dot(q12,q23_u)*q23_u
        n2 =  q34 - np.dot(q34,q23_u)*q23_u
        
        return np.arctan2(np.dot(np.cross(q23_u,n1),n2),np.dot(n1,n2))
 
    # -----------------------------------------------------------------------------------------------------
    def __first_derivative_angle(self):
        '''gradient along bend angle 

        args:
            -
        returns:
            delta_xi        (array, -)
        ''' 
        p1 = np.array([self.the_md.coords[3*self.atom[0]],self.the_md.coords[3*self.atom[0]+1],self.the_md.coords[3*self.atom[0]+2]],dtype=np.float)
        p2 = np.array([self.the_md.coords[3*self.atom[1]],self.the_md.coords[3*self.atom[1]+1],self.the_md.coords[3*self.atom[1]+2]],dtype=np.float)
        p3 = np.array([self.the_md.coords[3*self.atom[2]],self.the_md.coords[3*self.atom[2]+1],self.the_md.coords[3*self.atom[2]+2]],dtype=np.float)
        
        q12 = p1-p2
        q23 = p2-p3
        q12_n = np.linalg.norm(q12)
        q23_n = np.linalg.norm(q23)
        q12_u = q12/q12_n
        q23_u = q23/q23_n
        
        dxi1 = np.cross(q12_u,np.cross(q12_u,-q23_u))
        dxi3 = np.cross(q23_u,np.cross(q12_u,-q23_u))
        dxi1 /= np.linalg.norm(dxi1)
        dxi3 /= np.linalg.norm(dxi3)
        dxi1 /= q12_n
        dxi3 /= q23_n
        
        # sum(dxi)=0
        dxi2 = - (dxi1 + dxi3)
        
        delta_xi = np.zeros(3*self.natoms)
        for dim in range(0,3):
            delta_xi[atom[0]*3+dim] += dxi1[dim]
            delta_xi[atom[1]*3+dim] += dxi2[dim]
            delta_xi[atom[2]*3+dim] += dxi3[dim]
        
        return delta_xi

    # -----------------------------------------------------------------------------------------------------
    def __first_derivative_torsion(self):
        '''gradient along torsion angle

        args:
            -
        returns:
            delta_xi        (array, -)
        ''' 
        p1 = np.array([self.the_md.coords[3*self.atom[0]],self.the_md.coords[3*self.atom[0]+1],self.the_md.coords[3*self.atom[0]+2]],dtype=np.float)
        p2 = np.array([self.the_md.coords[3*self.atom[1]],self.the_md.coords[3*self.atom[1]+1],self.the_md.coords[3*self.atom[1]+2]],dtype=np.float)
        p3 = np.array([self.the_md.coords[3*self.atom[2]],self.the_md.coords[3*self.atom[2]+1],self.the_md.coords[3*self.atom[2]+2]],dtype=np.float)
        p4 = np.array([self.the_md.coords[3*self.atom[3]],self.the_md.coords[3*self.atom[3]+1],self.the_md.coords[3*self.atom[3]+2]],dtype=np.float)
        
        q12 = p2 - p1
        q23 = p3 - p2
        q34 = p4 - p3
        
        q12_n = np.linalg.norm(q12)
        q23_n = np.linalg.norm(q23)
        q34_n = np.linalg.norm(q34)
        
        q12_u = q12 / q12_n
        q23_u = q23 / q23_n
        q34_u = q34 / q34_n
        
        cos_123 = np.dot(-q12_u,q23_u)
        cos_234 = np.dot(-q23_u,q34_u)
        
        sin2_123 = 1 - cos_123*cos_123
        sin2_234 = 1 - cos_234*cos_234
         
        dtau1 = - 1/(q12_n*sin2_123)*np.cross(-q12_u,-q23_u)
        dtau4 = - 1/(q34_n*sin2_234)*np.cross(-q34_u,-q23_u)
        
        # sum(dtau)=0 and rotation=0
        c_123 = ((q12_n*cos_123)/q23_n) - 1
        b_432 = ((q34_n*cos_234)/q23_n)
        
        dtau2 = c_123*dtau1 - b_432*dtau4
        dtau3 = -(dtau1 + dtau2 + dtau4)
        
        delta_xi = np.zeros(3*self.natoms)
        for dim in range(0,3):
            delta_xi[atom[0]*3+dim] += dtau1[dim]
            delta_xi[atom[1]*3+dim] += dtau2[dim]
            delta_xi[atom[2]*3+dim] += dtau3[dim]
            delta_xi[atom[3]*3+dim] += dtau4[dim]
        
        return delta_xi

    # -----------------------------------------------------------------------------------------------------
    def __get_coord(self):
        '''get CV

        args:
            -
        returns:
            -
        '''
        xi = np.array([])
        delta_xi     = [0 for i in range(self.ncoords)]
        div_delta_xi = [0 for i in range(self.ncoords)]

        for i in range(self.ncoords):

            if self.coord[i] == 0:
                pass

            elif self.coord[i] == 1:
                x = self.the_md.coords[0]
                xi = np.append(xi, x)
                delta_xi[i] += np.array([1,0])

            elif self.coord[i] == 2:
                y = self.the_md.coords[1]
                xi = np.append(xi, y)
                delta_xi[i] += np.array([0,1])

            elif self.coord[i] == 3:
                x = self.the_md.coords[0]
                y = self.the_md.coords[1]
                xi = np.append(xi, x + y)
                delta_xi[i] += np.array([1,1])

            elif self.coord[i] == 4:
                x = self.the_md.coords[0]
                y = self.the_md.coords[1]
                xi = np.append(xi, x/4.0 + y)
                delta_xi[i] += np.array([0.25,1])

            elif self.coord[i] == 5:
                x = self.the_md.coords[0]
                xi = np.append(xi, 1.0/x)
                delta_xi[i] += np.array([-1.0/(x*x),0])
                div_delta_xi[i] -= 2*x

            else:
                print("reaction coordinate not implemented!")
                sys.exit(0)

        return (xi, delta_xi, div_delta_xi)

    # -----------------------------------------------------------------------------------------------------
    def __get_bin(self, xi, extended = False):
        '''get current bin of CV or extended variable

        args:
           xi               (array, -)
           extended         (bool, False)
        returns:
           bink             (array, [ncol, nrow])
        '''
        X = xi if extended == False else self.ext_coords
        
        binX = [0,0]
        for i in range(self.ncoords):
            binX[i] = int(np.floor(np.abs(X[i]-self.minx[i])/self.dx[i]))
        
        return binX


    # -----------------------------------------------------------------------------------------------------
    def __get_metaD_bias(self, xi, bink, height, update_int, step, WT_dT, WT, grid):
        '''get Bias Potential and Force as sum of Gaussian kernels

        args:
            xi              (float, -, CV)
            bink            (int, -, Bin number of xi)
            height          (double, -, Gaussian height)
            update_int      (int, -, update intervall)
            step            (int, -, md step)
            WT              (bool, -, use WT-metaD)
            grid            (bool, -, use grid for force)

        returns:
            bias            (array, bias force per CV)
        '''
	# update bias every update_int's step and save on grid for free energy calculation
        # if grid == True this is also used as bias for dynamics 
       
        if self.ncoords == 1:
            # 1D
            if step%int(update_int) == 0:
        
                w = height/H_to_kJmol
                if WT == True:
                    w *= np.exp(-self.metapot[bink[1],bink[0]]/(kB_a*WT_dT))
        
                dx = self.grid[0] - xi[0]
                bias_factor = w * np.exp(-np.power(dx,2.0)/(2.0*self.variance[0]))
                 
                self.metapot += bias_factor
                self.bias[0] -= bias_factor.T * dx.T/self.variance[0]

            bias = [self.bias[0][bink[1],bink[0]]]
 
        else:
            # 2D
            if step%int(update_int) == 0:
        
                w = height/H_to_kJmol
                if WT == True:
                    w *= np.exp(-self.metapot[bink[1],bink[0]]/(kB_a*WT_dT))

                dx = self.grid[0] - xi[0]
                dy = self.grid[1] - xi[1]
    
                for i in range(self.nbins_per_dim[1]):
                    if abs(dy[i]) <= 3*self.variance[1]:
                        for j in range(self.nbins_per_dim[0]):
                            if abs(dx[j]) <= 3*self.variance[0]:
    
                                p1 = (dx[j]*dx[j])/self.variance[0]  
                                p2 = (dy[i]*dy[i])/self.variance[1]
                                gauss = np.exp(-(p1+p2)/2.0)
            
                                self.metapot[i,j] += w * gauss
                                self.bias[0][i,j] -= w * gauss * dx[j]/self.variance[0]
                                self.bias[1][i,j] -= w * gauss * dy[i]/self.variance[1]
                
            bias = [self.bias[0][bink[1],bink[0]],self.bias[1][bink[1],bink[0]]] 

        if grid == False:
            # get analytic bias potential and force every step
            # can become slow in long simulations 

            bias_factor = 0.0
            if self.ncoords == 1:
                # 1D                
                if step == 0:
                    self.center = np.array([xi[0]])
                    self.steps_in_bins = 0
    
                elif step%int(update_int) == 0:
                    self.center = np.append(self.center, xi[0])
                
                bias = [0.0]
                w0 = height/H_to_kJmol

                dx = self.center-xi[0]                     
                for ii, val in enumerate(dx):
                    if abs(val) <= 3*self.variance[0]:         	

                        w = w0
                        if WT == True:
                            w *= np.exp(-bias_factor/(kB_a*WT_dT))
                        
                        bias_factor += w * np.exp(-(val*val)/(2.0*self.variance[0]))
                        bias[0]     += bias_factor * val/self.variance[0]
            
            else:
                # 2D
                if step == 0:
                    self.center_x = np.array([xi[0]])
                    self.center_y = np.array([xi[1]])
               
                elif step%int(update_int) == 0:
                    self.center_x = np.append(self.center_x, xi[0])
                    self.center_y = np.append(self.center_y, xi[1])

                bias = [0.0,0.0] 

                w0 = height/H_to_kJmol
                dx = self.center_x - xi[0]
                dy = self.center_y - xi[1]               
 
                for i in range(len(dx)):
                    if abs(dx[i]) <= 3*self.variance[0] and abs(dy[i]) <= 3*self.variance[1]:
                            
                        w = w0
                        if WT == True:
                            w *= np.exp(-bias_factor/(kB_a*WT_dT))
                        
                        exp1  = (dx[i]*dx[i])/self.variance[0]  
                        exp2  = (dy[i]*dy[i])/self.variance[1]
                        gauss = w * np.exp(-(exp1+exp2)/2.0)

                        bias_factor += gauss
                        bias[0]     += gauss * dx[i]/self.variance[0]
                        bias[1]     += gauss * dy[i]/self.variance[1]
        
        return bias

    # -----------------------------------------------------------------------------------------------------
    def __get_gradient_correction(self, delta_xi):
        '''get correction for geometric free energie in current step
        
        args:
            delta_xi       	(array, -, gradients along all CV's)
        returns:
            correction		(double, -, correction for current step)
        '''
        if self.ncoords == 1:
            return np.linalg.norm(delta_xi[0])

        else:
            d = np.array([[0.0,0.0],[0.0,0.0]])
            d[0,0] = np.linalg.norm(delta_xi[0])
            d[1,1] = np.linalg.norm(delta_xi[1])
            d[1,0] = d[0,1] = np.sqrt(np.dot(delta_xi[0],delta_xi[1]))
            return np.linalg.det(d)

    # -----------------------------------------------------------------------------------------------------
    def __propagate_extended(self, langevin=True):
        '''Propagate momenta/coords of extended variable with Velocity Verlet

        args:
           langevin                (bool, False)
        returns:
           -
        '''
        if langevin==True:
            prefac    = 2.0 / (2.0 + self.friction*self.the_md.dt_fs)
            rand_push = np.sqrt(self.the_md.target_temp*self.friction*self.the_md.dt_fs*kB_a/2.0e0)
            self.ext_rand_gauss = np.zeros(shape=(len(self.ext_momenta),), dtype=np.double)
            for atom in range(len(self.ext_rand_gauss)):
                self.ext_rand_gauss[atom] = random.gauss(0, 1)

            self.ext_momenta += np.sqrt(self.ext_mass) * rand_push * self.ext_rand_gauss
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces
            self.ext_coords  += prefac * self.the_md.dt * self.ext_momenta / self.ext_mass

        else:
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces
            self.ext_coords  += self.the_md.dt * self.ext_momenta / self.ext_mass

    # -----------------------------------------------------------------------------------------------------
    def __up_momenta_extended(self, langevin=True):
        '''Update momenta of extended variables with Velocity Verlet

        args:
            langevin        (bool, True)
        returns:
            -
        '''
        if langevin==True:
            prefac = (2.0e0 - self.friction*self.the_md.dt_fs)/(2.0e0 + self.friction*self.the_md.dt_fs)
            rand_push = np.sqrt(self.the_md.target_temp*self.friction*self.the_md.dt_fs*kB_a/2.0e0)
            self.ext_momenta *= prefac
            self.ext_momenta += np.sqrt(self.ext_mass) * rand_push * self.ext_rand_gauss
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces
        else:
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces

    # -----------------------------------------------------------------------------------------------------
    def __get_cond_avg(self, a, hist):
        '''get conditional average

        args:
            a		   (array, -)
            hist           (array, -)
        returns:
            -
        '''
        # returns zero for bins without samples 
        mean_force = np.divide(a, hist, out=np.zeros_like(a), where=(hist!=0)) 
        return mean_force 

    # -----------------------------------------------------------------------------------------------------
    def __F_from_Force(self, mean_force):
        '''numeric on-the-fly integration of thermodynamic force to obtain free energy estimate 

        args:
            -
        returns:
            -
        '''
        self.dF = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)

        if self.ncoords == 1:
            # on-the-fly integraion only for 1D reaction coordinate

            for i in range(1,self.nbins_per_dim[0]):
                self.dF[0,i] = np.sum(mean_force[0][0,0:i]) * self.dx[0]

        self.__get_geom_correction()

    # -----------------------------------------------------------------------------------------------------
    def __F_from_metaD(self, WT_dT, WT=True):
        '''on-the-fly free energy estimate from metaD or WT-metaD bias potential

        args:
            WT              (bool, Well-Tempered metaD)
            grid            (bool, bias pot and force already saved on grid)
        returns:
            -
        '''
        self.dF = - self.metapot
        
        if WT==True:
            self.dF *= (self.the_md.target_temp + WT_dT)/WT_dT

        self.__get_geom_correction()

    # -----------------------------------------------------------------------------------------------------
    def __F_from_CZAR(self):
        '''on-the-fly CZAR estimate of unbiased thermodynamic force
           get unbiased free energy by integrating 
        
        args:
            -
        returns:
            -
        '''
        self.mean_force = np.array([np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])) for i in range(self.ncoords)], dtype=np.float64)
        
        # get ln(rho) and z-conditioned average force per bin
        log_rho   = np.log(self.hist_z, out=np.zeros_like(self.hist_z), where=(self.hist_z!=0))
        avg_force = self.__get_cond_avg(self.force_correction_czar, self.hist_z)         
        
        # czar estimate of thermodynamic force per bin 
        if self.ncoords == 1: 
            self.mean_force[0]  = - kB_a * self.the_md.target_temp * np.gradient(log_rho[0], self.grid[0]) + avg_force[0]
        
        else:
            der_log_rho = np.gradient(log_rho, self.grid[1], self.grid[0])
            self.mean_force[0] = - kB_a * self.the_md.target_temp * der_log_rho[1] + avg_force[0]
            self.mean_force[1] = - kB_a * self.the_md.target_temp * der_log_rho[0] + avg_force[1]    

        self.__F_from_Force(self.mean_force)
        self.__get_geom_correction(extended = True)
	
    # -----------------------------------------------------------------------------------------------------
    def __get_geom_correction(self, extended = False):
        '''get geometric free energy

        args:
            extended	(bool, False, True for methods with extended system)
        returns:
            -
        '''
        hist = self.histogramm if extended == False else self.hist_z
        grad_corr = self.__get_cond_avg(self.geom_corr, hist)
        grad_corr = np.log(grad_corr, out=np.zeros_like(grad_corr), where=(grad_corr!=0))
        
        self.geom_correction = kB_a * self.the_md.target_temp * grad_corr     

       
    # -----------------------------------------------------------------------------------------------------
    def read_bias(self, filename='restart_bias.dat'):
        '''read data from output file 
        '''
        data = np.loadtxt(filename)

        self.histogramm = data[:,0].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
        self.geom_corr  = data[:,1].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])

        if self.ncoords == 1:
            self.bias[0] = data[:,2].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
            if self.method == 'metaD':
                self.metapot = data[:,3].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
            elif self.method == 'eABF' or self.method == 'meta-eABF':
                self.hist_z = data[:,3].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                self.force_correction_czar[0] = data[:,4].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                if self.method == 'meta-eABF':
                    self.abfforce[0] = data[:,5].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])

        else:
            self.bias[0] = data[:,2].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
            self.bias[1] = data[:,3].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
            if self.method == 'metaD':
                self.metapot = data[:,4].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
            elif self.method == 'eABF' or self.method == 'meta-eABF':
                self.hist_z = data[:,4].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                self.force_correction_czar[0] = data[:,5].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                self.force_correction_czar[1] = data[:,6].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                if self.method == 'meta-eABF':
                    self.abfforce[0] = data[:,7].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                    self.abfforce[1] = data[:,8].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])

    # -----------------------------------------------------------------------------------------------------
    def write_restart(self):
        '''write convergence file

        args:
            -
        returns:
            -
        '''
        out = open(f"restart_bias.dat", "w")
        if len(self.coord) == 1:
            for i in range(self.nbins_per_dim[0]):
                row = (self.histogramm[0,i], self.geom_corr[0,i], self.bias[0][0,i])
                out.write("%8d\t%14df\t%14.10f" % row)
                if self.method == 'metaD':
                    out.write('\t%14.10f' % (self.metapot[0,i]))
                elif self.method == 'eABF' or self.method == 'meta-eABF':
                    out.write('\t%14.10f\t%14.10f' % (self.hist_z[0,i], self.force_correction_czar[0][0,i]))
                    if self.method == 'meta-eABF':
                        out.write('\t%14.10f' % (self.abfforce[0][0,i]))
                out.write('\n')

        else:
            for i in range(self.nbins_per_dim[1]):
                for j in range(self.nbins_per_dim[0]):
                    row = (self.histogramm[i,j], self.geom_corr[i,j], self.bias[0][i,j], self.bias[1][i,j])
                    out.write("%8d\t%8d\t%14.10f\t%14.10f" % row)
                    if self.method == 'metaD':
                        out.write('\t%14.10f' % (self.metapot[i,j]))
                    elif self.method == 'eABF' or self.method == 'meta-eABF':
                        out.write('\t%14.10f\t%14.10f\t%14.10f' % (self.hist_z[i,j], self.force_correction_czar[0][i,j], self.force_correction_czar[1][i,j]))
                        if self.method == 'meta-eABF':
                            out.write('\t%14.10f\t%14.10f' % (self.abfforce[0][i,j], self.abfforce[0][i,j]))
                    out.write('\n')

        out.close()

    # -----------------------------------------------------------------------------------------------------
    def write_traj(self, xi, extended = False):
        '''write trajectory of extended or normal ABF

        args:
            extended      (bool, False)
        returns:
            -
        '''
        if self.the_md.step == 0:
            traj_out = open("CV_traj.dat", "w")
            traj_out.write("%14s\t" % ("time [fs]"))
            for i in range(len(self.traj[0])):
                traj_out.write("%14s\t" % (f"Xi{i}"))
                if extended:
                    traj_out.write("%14s\t" % (f"eXi{i}"))
            traj_out.close()
	
        else:
            traj_out = open("CV_traj.dat", "a")
            for n in range(self.out_freq):
                traj_out.write("\n%14.6f\t" % ((self.the_md.step-self.out_freq+n)*self.the_md.dt*it2fs))
                for i in range(len(self.traj[0])):
                    traj_out.write("%14.6f\t" % (self.traj[-self.out_freq+n][i]))
                    if extended:
                        traj_out.write("%14.6f\t" % (self.etraj[-self.out_freq+n][i]))
            traj_out.close()
    
        self.traj  = np.array([xi])
        if self.method == 'eABF' or self.method == 'meta-eABF':
            self.etraj = np.array([self.ext_coords])

    # -----------------------------------------------------------------------------------------------------
    def write_output(self):
        '''write output of free energy calculations
        '''
        if self.method == 'metaD' or self.method == 'reference':
            self.mean_force = self.bias

        out = open(f"bias_out.dat", "w")
        if len(self.coord) == 1:
            head = ("Xi1", "Histogramm", "Bias", "dF", "geom_corr")
            out.write("%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
            for i in range(self.nbins_per_dim[0]):
                row = (self.grid[0][i], self.histogramm[0,i], self.mean_force[0][0,i], self.dF[0,i], self.geom_correction[0,i])
                out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)

        elif len(self.coord) == 2:
            if self.method == 'metaD' or 'reference':
                head = ("Xi1", "Xi1", "Histogramm", "Bias1", "Bias2", "geom_corr", "dF")
                out.write("%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
                for i in range(self.nbins_per_dim[1]):
                    for j in range(self.nbins_per_dim[0]):
                        row = (self.grid[1][i], self.grid[0][j], self.histogramm[i,j], self.mean_force[0][i,j], self.mean_force[1][i,j], self.geom_correction[i,j], self.dF[i,j])
                        out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)
                
            else:
                head = ("Xi1", "Xi1", "Histogramm", "Bias1", "Bias2", "geom_corr")
                out.write("%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
                for i in range(self.nbins_per_dim[1]):
                    for j in range(self.nbins_per_dim[0]):
                        row = (self.grid[1][i], self.grid[0][j], self.histogramm[i,j], self.mean_force[0][i,j], self.mean_force[1][i,j], self.geom_correction[i,j])
                        out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)

        out.close()

    # -----------------------------------------------------------------------------------------------------
    def __print_parameters(self):
        '''print parameters after init
        '''
        print("###############################################################################")
        print("Initialize {m} method:".format(m=self.method))
        for i in range(1,len(self.coord)+1):
            print("\n\tMinimum CV%d:\t\t\t\t%14.6f Bohr" % (i,self.minx[i-1]))
            print("\tMaximum CV%d:\t\t\t\t%14.6f Bohr" % (i,self.maxx[i-1]))
            print("\tBinwidth CV%d:\t\t\t\t%14.6f Bohr" % (i,self.dx[i-1]))
        print("\n\tTotel number of bins:\t\t\t%14.6f" % (self.nbins))

        for i in range(1,len(self.coord)+1):
            if self.method == 'eABF' or self.method == 'meta-eABF':
                print("\tSpring constant for extended variable:\t%14.6f Hartree/Bohr^2" % (self.k[i-1]))
                print("\tfictitious mass for extended variable:\t%14.6f a.u." % (self.ext_mass[i-1]))
        print("################################################################################\n")
