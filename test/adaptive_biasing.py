import os
import sys
import time
import random
import numpy as np

from CVs import * 

# bohr to angstrom
bohr2angs   = 0.52917721092e0   # 

# energy units
kB          = 1.380648e-23      # J / K
H_to_kJmol  = 2625.499639       #
H_to_J      = 4.359744e-18      #
kB_a        = kB / H_to_J       # Hartree / K
H2au        = 2.921264912428e-8 # 
au2k        = 315775.04e0       # 

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
        ats:            input parameters that have to be initialized beforehand
        method:         'reference', 'ABF', 'metaD', 'eABF' or 'meta-eABF'
        f_conf:         harmonic force constant outside of bins [kJ/mol]
        output_frec:    Number of steps between outputs (default: 1000)
        friction:       use same friction coefficient for Langevin dynamics of extended system and physical system (default: 1.0e-3)
        random_seed:    use same random number seed for Langevin dynamics of extended and physical system (default: system time)

    Output:
        bias_out.txt    text file containing i.a. CV, histogramm, bias, standard free energy and geometric free energy
    '''
    def __init__(self, MD, ats, method = 'meta-eABF', f_conf = 100, output_freq = 1000, friction = 1.0e-3, seed_in = None):
	
        # general parameters
        self.the_md   = MD
        self.method   = method
        self.f_conf   = f_conf
        self.out_freq = output_freq
        self.friction = friction
        
        # mass scaled coordinates
        self.sqrt_m = np.sqrt(self.the_md.masses)
   
        # definition of CVs 
        self.ncoords = len(ats)
        self.CV      = np.array([item[0] for item in ats])     
        self.minx    = np.array([item[2] for item in ats])
        self.maxx    = np.array([item[3] for item in ats])
        self.dx      = np.array([item[4] for item in ats])

        self.atoms = [[] for i in range(self.ncoords)]
        self.is_angle = [False for i in range(self.ncoords)]
    
        for i in range(self.ncoords):

            if hasattr(ats[i][1], "__len__"):
                # use center of mass for CV
                for index, a in enumerate(ats[i][1]):
                    self.atoms[i].append(np.array(a)-1)

            else:
                # use coordinates of atoms
                self.atoms[i].append(ats[i][1]-1)
      
	    # degree to rad 
            if self.CV[i] == 'angle' or self.CV[i] == 'torsion' or self.CV[i] == 'lin_comb_angles':
                self.is_angle[i] = True
                self.minx[i] = np.radians(self.minx[i])
                self.maxx[i] = np.radians(self.maxx[i])
                self.dx[i]   = np.radians(self.dx[i])

        (xi, delta_xi, div_delta_xi) = self.__get_coord()
        self.traj = np.array([xi])

        # get number of bins
        self.nbins_per_dim = np.array([1,1])
        self.grid          = []
        for i in range(self.ncoords):
            self.nbins_per_dim[i] = int(np.ceil(np.abs(self.maxx[i]-self.minx[i])/self.dx[i])) if self.dx[i] > 0 else 1
            self.grid.append(np.linspace(self.minx[i]+self.dx[i]/2,self.maxx[i]-self.dx[i]/2,self.nbins_per_dim[i]))

        self.nbins = np.prod(self.nbins_per_dim)
        
        self.bias       = np.zeros((self.ncoords,self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)
        self.histogramm = np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)
        self.geom_corr  = np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)
        self.CV_crit    = np.zeros((self.ncoords,self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)

        if method == 'reference' or method == 'ABF':
            pass

        elif method == 'eABF' or method == 'meta-eABF':
            # setup extended system for eABF or meta-eABF
            
            # force constant  
            sigma = np.array([item[5] for item in ats])
            for i in range(self.ncoords):
                if self.CV[i] == 'angle' or self.CV[i] == 'torsion' or self.CV[i] == 'lin_comb_angles':
                    sigma[i] = np.radians(sigma[i])
        
            self.k = (kB_a*self.the_md.target_temp) / (sigma*sigma)
            
            # mass in a.u.
            self.ext_mass = np.array([item[6] for item in ats])
            
            # langevin dynamic
            self.ext_coords = np.copy(xi)
            self.etraj      = np.copy(self.traj)

            self.ext_forces  = np.zeros(self.ncoords)
            self.ext_momenta = np.zeros(self.ncoords)
            
            if type(seed_in) is int:
                random.seed(seed_in)
            else:
                try:
                    random.setstate(seed_in)
                except:
                    print("\tThe provided seed was neither an int nor a state of random")
                    sys.exit(1)
           
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

                self.variance = np.array([item[5 if method == 'metaD' else 7] for item in ats])
                for i in range(self.ncoords):
                    if self.CV[i] == 'angle' or self.CV[i] == 'torsion' or self.CV[i] == 'lin_comb_angles':
                        self.variance[i] = np.radians(self.variance[i])
                    
                self.abfforce = np.copy(self.bias)
                self.metapot  = np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)

        elif method == 'metaD' or method == 'meta-eABF':
            # parameters for metaD or WT-metaD
           
            self.variance = np.array([item[5 if method == 'metaD' else 7] for item in ats])
            for i in range(self.ncoords):
              if self.CV[i] == 'angle' or self.CV[i] == 'torsion' or self.CV[i] == 'lin_comb_angles':
                  self.variance[i] = np.radians(self.variance[i])
                    
            self.metapot  = np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)
       
        else:
            print("\n----------------------------------------------------------------------")
            print("ERROR: Invalid keyword in definition of adaptove biasing method!")
            print('Available: reference, ABF, eABF, metaD, default: meta-eABF.')
            print("-----------------------------------------------------------------------")
            sys.exit(1)

        self.__print_parameters()

    # -----------------------------------------------------------------------------------------------------
    def ref(self, write_output=True, write_traj=True):
        '''get unbiased histogramm along CVs 
           can be used for equilibration of molecules prior to ABM simulation           

        args:
            write_output    (bool, True, write free energy to bias_out.dat)
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
                self.CV_crit[i][bink[1],bink[0]] += np.dot(delta_xi[i], self.the_md.forces)         

        else:
            for i in range(self.ncoords):

                # confinement
                max_diff = self.__diff_angles(self.maxx[i], xi[i]) 
                min_diff = self.__diff_angles(self.minx[i], xi[i])
                if abs(max_diff) < abs(min_diff):
                    self.the_md.forces -= self.f_conf/H_to_kJmol * max_diff * delta_xi[i]
                else:
                    self.the_md.forces -= self.f_conf/H_to_kJmol * min_diff * delta_xi[i]

        if self.the_md.step%self.out_freq == 0:
            # write output

            if write_traj == True:
                self.write_traj(xi, extended = False)
            
            if write_output == True:
                self.dF = -kB_a*self.the_md.target_temp*np.log(self.histogramm, out=np.zeros_like(self.histogramm),where=(self.histogramm!=0))
                self.__get_geom_correction()
                self.write_output()
                self.write_restart()

        self.timing = time.perf_counter() - start
        

    # -----------------------------------------------------------------------------------------------------
    def ABF(self, N_full=100, f_conf = 10, write_traj=True):
        '''Adaptive biasing force method

        args:
            N_full:         (double, 100, number of sampels when full bias is applied to bin)
            write_traj      (bool, True, write trajectory to CV_traj.dat)
           
        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, div_delta_xi) = self.__get_coord()

        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self.__get_bin(xi)
            self.histogramm[bink[1],bink[0]] += 1
            
            self.geom_corr[bink[1],bink[0]] += self.__get_gradient_correction(delta_xi)

            for i in range(self.ncoords):

                self.CV_crit[i][bink[1],bink[0]] += np.dot(delta_xi[i], self.the_md.forces)         

                # linear ramp function R(N,k)
                Rk = 1.0 if self.histogramm[bink[1],bink[0]] > N_full else self.histogramm[bink[1],bink[0]]/N_full

                # inverse gradient v_i
                delta_xi_n = np.linalg.norm(delta_xi[i])
                v_i = delta_xi[i]/(delta_xi_n*delta_xi_n)
            		
                # apply biase force
                self.bias[i][bink[1],bink[0]] += np.dot(self.the_md.forces, v_i) - kB_a * self.the_md.target_temp * div_delta_xi[i]
                self.the_md.forces            -= Rk * (self.bias[i][bink[1],bink[0]]/self.histogramm[bink[1],bink[0]]) * delta_xi[i]
                
        else:
            for i in range(self.ncoords):

                # confinement
                max_diff = self.__diff_angles(self.maxx[i], xi[i]) 
                min_diff = self.__diff_angles(self.minx[i], xi[i])
                if abs(max_diff) < abs(min_diff):
                    self.the_md.forces -= self.f_conf/H_to_kJmol * max_diff * delta_xi[i]
                else:
                    self.the_md.forces -= self.f_conf/H_to_kJmol * min_diff * delta_xi[i]

        self.traj = np.append(self.traj, [xi], axis = 0)

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

                self.CV_crit[i][bink[1],bink[0]] += np.dot(delta_xi[i], self.the_md.forces)         
                self.the_md.forces += bias[i] * delta_xi[i]
        
        else:
            for i in range(self.ncoords):

                # confinement
                max_diff = self.__diff_angles(self.maxx[i], xi[i]) 
                min_diff = self.__diff_angles(self.minx[i], xi[i])
                if abs(max_diff) < abs(min_diff):
                    self.the_md.forces -= self.f_conf/H_to_kJmol * max_diff * delta_xi[i]
                else:
                    self.the_md.forces -= self.f_conf/H_to_kJmol * min_diff * delta_xi[i]

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
            # lambda conditioned bias        
 
            la_bin = self.__get_bin(xi, extended = True)
            self.histogramm[la_bin[1],la_bin[0]] += 1
             
            for i in range(self.ncoords):

                # harmonic coupling of extended coordinate to reaction coordinate
                dxi                 = self.__diff_angles(self.ext_coords[i],xi[i])
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]
                
                # apply biase force
                Rk = 1.0 if self.histogramm[la_bin[1],la_bin[0]] > N_full else self.histogramm[la_bin[1],la_bin[0]]/N_full
                self.bias[i][la_bin[1],la_bin[0]] -= self.k[i] * dxi
                self.ext_forces[i] += Rk * self.bias[i][la_bin[1],la_bin[0]]/self.histogramm[la_bin[1],la_bin[0]]
                  
        else:
            # confinement of extended coordinate

            for i in range(self.ncoords):

                # harmonic coupling
                dxi                 = self.__diff_angles(self.ext_coords[i],xi[i])
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]

                # confinement of extended coordinate
                max_diff = self.__diff_angles(self.maxx[i], self.ext_coords[i]) 
                min_diff = self.__diff_angles(self.minx[i], self.ext_coords[i])
                if abs(max_diff) < abs(min_diff):
                    self.ext_forces -= self.f_conf/H_to_kJmol * max_diff
                else:
                    self.ext_forces -= self.f_conf/H_to_kJmol * min_diff
        
        if (xi <= self.maxx).all() and (xi >= self.minx).all():
            # xi-conditioned accumulators 
           
            bink = self.__get_bin(xi)
            self.hist_z[bink[1],bink[0]] += 1
            
            self.geom_corr[bink[1],bink[0]] += self.__get_gradient_correction(delta_xi)
            
            for i in range(self.ncoords):

                self.CV_crit[i][bink[1],bink[0]] += np.dot(delta_xi[i], self.the_md.forces)         

                dx = self.__diff_angles(self.ext_coords[i], self.grid[i][bink[i]])
                self.force_correction_czar[i][bink[1],bink[0]] += self.k[i] * dx 
         
        else:
            # confinement of physical coordinate

            max_diff = self.__diff_angles(self.maxx[i], xi[i]) 
            min_diff = self.__diff_angles(self.minx[i], xi[i])
            if abs(max_diff) < abs(min_diff):
                self.the_md.forces -= self.f_conf/H_to_kJmol * max_diff * delta_xi[i]
            else:
                self.the_md.forces -= self.f_conf/H_to_kJmol * min_diff * delta_xi[i]
                    
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
                dxi                 = self.__diff_angles(self.ext_coords[i], xi[i])
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]

                # metadynamics bias
                self.ext_forces[i] += meta_force[i]
      
                # eABF bias
                Rk = 1.0 if self.histogramm[la_bin[1],la_bin[0]] > N_full else self.histogramm[la_bin[1],la_bin[0]]/N_full
                self.abfforce[i][la_bin[1],la_bin[0]] -= self.k[i] * dxi  
                self.ext_forces[i] += Rk * self.abfforce[i][la_bin[1],la_bin[0]]/self.histogramm[la_bin[1],la_bin[0]]  
            
        else:

            for i in range(self.ncoords):

                # harmonic coupling
                dxi                 = self.__diff_angles(self.ext_coords[i], xi[i])
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]

                # confinement with constant force outside of bins
                max_diff = self.__diff_angles(self.maxx[i], self.ext_coords[i]) 
                min_diff = self.__diff_angles(self.minx[i], self.ext_coords[i])
                if abs(max_diff) < abs(min_diff):
                    self.ext_forces -= self.f_conf/H_to_kJmol * max_diff
                else:
                    self.ext_forces -= self.f_conf/H_to_kJmol * min_diff

        if (xi <= self.maxx).all() and (xi >= self.minx).all():
            
            bink = self.__get_bin(xi)
            self.geom_corr[bink[1],bink[0]]  += self.__get_gradient_correction(delta_xi)

            # accumulators for czar estimator
            self.hist_z[bink[1],bink[0]] += 1
            for i in range(self.ncoords):

                self.CV_crit[i][bink[1],bink[0]] += np.dot(delta_xi[i], self.the_md.forces)         

                dx = self.__diff_angles(self.ext_coords[i], self.grid[i][bink[i]])
                self.force_correction_czar[i][bink[1],bink[0]] += self.k[i] * dx
        
        else:
            # confinement of physical coordinate

            max_diff = self.__diff_angles(self.maxx[i], xi[i]) 
            min_diff = self.__diff_angles(self.minx[i], xi[i])
            if abs(max_diff) < abs(min_diff):
                self.the_md.forces -= self.f_conf/H_to_kJmol * max_diff * delta_xi[i]
            else:
                self.the_md.forces -= self.f_conf/H_to_kJmol * min_diff * delta_xi[i]

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
    def __get_coord(self):
        '''get collective variable from CVs.py

        args:
            -
        returns:
            -
        '''
        xi       = np.zeros(self.ncoords) 
        delta_xi = np.zeros((self.ncoords,self.the_md.natoms*3))
        div      = np.zeros(self.ncoords)

        for i in range(self.ncoords):

            if self.CV[i] == 'distance':
                     
                    # bond distance
                    (x, dx) = distance(self, self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx
                    div[i]      += 2.0/xi[i]

            elif self.CV[i] == 'projected_distance':
                     
                    # bond distance
                    (x, dx) = projected_distance(self, self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx
                    div[i]      += 0.0

            elif self.CV[i] == 'angle':

                    # bend angle
                    (x, dx) = angle(self, self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx
                    div[i]      += 1.0/np.tan(xi[i])  

            elif self.CV[i] == 'torsion':
                
                    # torsion angle
                    (x, dx) = torsion(self, self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx
                    div[i]      += 0.0  

            elif self.CV[i] == 'hBond':
            
                    # Hydrogen Bond
                    if self.the_md.step == 0 and self.method == 'ABF':
                        print('ERROR: Do not use with ABF, divergence not implemented!')
                        sys.exit(0)
           
                    (x, dx) = hBond(self, self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx
            
            elif self.CV[i] == 'lin_comb_dists':
                    
                    # linear combination of distances and projeced distances
                    if self.the_md.step == 0 and self.method == 'ABF':
                        print('ERROR: Do not use linear combination of CVs with ABF, divergence not implemented!')
                        sys.exit(0)

                    (x, dx) = lin_comb_dists(self, i, self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx

            elif self.CV[i] == 'lin_comb_angles':
                    
                    # linear combination of angles and dihedrals
                    if self.the_md.step == 0 and self.method == 'ABF':
                        print('ERROR: Do not use linear combination of CVs with ABF, divergence not implemented!')
                        sys.exit(0)

                    (x, dx) = lin_comb_dists(self, i, self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx

            elif self.CV[i] == 'lin_comb_hBonds':
                    
                    # linear combination of distances and projeced distances
                    if self.the_md.step == 0 and self.method == 'ABF':
                        print('ERROR: Do not use linear combination of CVs with ABF, divergence not implemented!')
                        sys.exit(0)

                    (x, dx) = lin_comb_hBonds(self, i, self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx
                              
            else:
                print("\n----------------------------------------------------------------------")
                print("ERROR: Invalid keyword in definition of collective variables!")
                print("Available: distance, angle, torsion, lin_comb_dists, lin_comb_angles")
                print("-----------------------------------------------------------------------")
                sys.exit(0)

        return (xi, delta_xi, div)

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
    def __diff_angles(self, a, b):
        '''returns difference of two angles in range (-pi,pi) or normal difference 

        args:
            a               (double, -, angle in rad)
            b               (double, -, angle in rad)
        returns:
            diff            (double, -, in rad)
        '''
        if self.is_angle == True: 
            diff = a - b
            if diff < -np.pi:  diff += 2*np.pi
            elif diff > np.pi: diff -= 2*np.pi
        else:
            diff = a - b
        return diff

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
        
                for i in range(self.nbins_per_dim[0]):
                    dx = self.__diff_angles(self.grid[0][i], xi[0])
                    if abs(dx) <= 3*self.variance[0]:
                        bias_factor = w * np.exp(-(dx*dx)/(2.0*self.variance[0]))

                        self.metapot[0,i] += bias_factor
                        self.bias[0][0,i] -= bias_factor * dx/self.variance[0]

            bias = [self.bias[0][bink[1],bink[0]]]
 
        else:
            # 2D
            if step%int(update_int) == 0:
        
                w = height/H_to_kJmol
                if WT == True:
                    w *= np.exp(-self.metapot[bink[1],bink[0]]/(kB_a*WT_dT))

                for i in range(self.nbins_per_dim[1]):
                    dy = self.__diff_angles(self.grid[1][i], xi[1])
                    if abs(dy) <= 3*self.variance[1]:
                        for j in range(self.nbins_per_dim[0]):
                            dx = self.__diff_angles(self.grid[0][j], xi[0])
                            if abs(dx) <= 3*self.variance[0]:
                                p1 = (dx*dx)/self.variance[0]  
                                p2 = (dy*dy)/self.variance[1]
                                gauss = np.exp(-(p1+p2)/2.0)
            
                                self.metapot[i,j] += w * gauss
                                self.bias[0][i,j] -= w * gauss * dx/self.variance[0]
                                self.bias[1][i,j] -= w * gauss * dy/self.variance[1]
                
            bias = [self.bias[0][bink[1],bink[0]],self.bias[1][bink[1],bink[0]]] 

        if grid == False:
            # get exact bias force every step
            # can become slow in long simulations 
            # TODO: use sorted dx for speedup          

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

                for ii, val in enumerate(self.center):
                    dx = self.__diff_angles(val,xi[0])                   
                    if abs(dx) <= 3*self.variance[0]:         	

                        w = w0
                        if WT == True:
                            w *= np.exp(-bias_factor/(kB_a*WT_dT))

                        bias_factor += w * np.exp(-(dx*dx)/(2.0*self.variance[0]))
                        bias[0]     += bias_factor * dx/self.variance[0]
            
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
                for i in range(len(dx)):
                    dx = self.__diff_angles(self.center_x[i], xi[0])
                    if abs(dx[i]) <= 3*self.variance[0]:
                        dy = self.__diff_angles(self.center_y[i], xi[1])          
                        if abs(dy[i]) <= 3*self.variance[1]:

                            w = w0
                            if WT == True:
                                w *= np.exp(-bias_factor/(kB_a*WT_dT))

                            exp1  = (dx*dx)/self.variance[0]  
                            exp2  = (dy*dy)/self.variance[1]
                            gauss = w * np.exp(-(exp1+exp2)/2.0)
    
                            bias_factor += gauss
                            bias[0]     += gauss * dx/self.variance[0]
                            bias[1]     += gauss * dy/self.variance[1]
        
        return bias

    # -----------------------------------------------------------------------------------------------------
    def __get_gradient_correction(self, delta_xi):
        '''get correction factor for geometric free energie in current step
           use mass-scalled coordinates to drop mass in calculation of free energy barrier 

        args:
            delta_xi       	(array, -, gradients along all CV's)
        returns:
            correction		(double, -, correction for current step)
        '''
        if self.ncoords == 1:
            q = self.sqrt_m * delta_xi[0]
            return np.linalg.norm(q)

        else:
            d = np.array([[0.0,0.0],[0.0,0.0]])
          
            q0 = self.sqrt_m * delta_xi[0]
            q1 = self.sqrt_m * delta_xi[1]

            d[0,0] = np.linalg.norm(q0)
            d[1,1] = np.linalg.norm(q1)
            d[1,0] = d[0,1] = np.sqrt(np.dot(q0,q1))
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
            cond_avg       (array, -)
        '''
        cond_avg = np.divide(a, hist, out=np.zeros_like(a), where=(hist!=0)) 
        return cond_avg 

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
            # on-the-fly integration only for 1D reaction coordinate

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
    def MW(self, MW_file = '../MW.dat', sync_interval = 20, trial = 0):
        '''Multiple walker strategy
           metadynamic/meta-eABF has to use grid!

        args:
            MW_file             (string, path to MW buffer)
            sync_interval       (int, intervall for sync with other walkers in steps)
            trial               (int, counter for recursive call)
        returns:
            -
        '''
        if self.the_md.step == 0:
            print('-------------------------------------------------------------')
            print('\tNew Multiple-Walker Instance created!')
            print('-------------------------------------------------------------')
            self.MW_histogramm = np.copy(self.histogramm)
            self.MW_geom_corr  = np.copy(self.geom_corr)
            self.MW_bias = np.copy(self.bias)
            self.MW_CV_crit = np.copy(self.CV_crit)
            if self.method == 'metaD': 
                self.MW_metapot = np.copy(self.metapot)
            elif self.method == 'eABF' or self.method == 'meta-eABF':
                self.MW_hist_z = np.copy(self.hist_z)
                self.MW_force_correction_czar = np.copy(self.force_correction_czar)
                if self.method == 'meta-eABF':
                    self.MW_abfforce = np.copy(self.abfforce)

        if self.the_md.step % sync_interval == 0:

            # check is MW file exists
            if os.path.isfile(MW_file):
                
                # avoid inconsitency by simultaneous access to buffer of multiple walkers
                if os.access(MW_file, os.W_OK) == False:
    
                    # grant write permission for local walker 
                    os.chmod(MW_file, 0o644)                
                    
                    # update local and MW accumulators
                    MW = np.loadtxt(MW_file)   
    
                    self.histogramm = self.__MW_update(self.histogramm, self.MW_histogramm, MW[:,0].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]))
                    self.MW_histogramm = np.copy(self.histogramm)
    
                    self.geom_corr = self.__MW_update(self.geom_corr, self.MW_geom_corr, MW[:,1].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]))
                    self.MW_geom_corr = np.copy(self.geom_corr)
    
                    if self.ncoords == 1:
    
                        self.bias = self.__MW_update(self.bias, self.MW_bias, [MW[:,2]])
                        self.MW_bias = np.copy(self.bias)
    
                        self.CV_crit = self.__MW_update(self.CV_crit, self.MW_CV_crit, [MW[:,3]])
                        self.MW_CV_crit = np.copy(self.CV_crit)
    
                        if self.method == 'metaD': 
                            self.metapot = self.__MW_update(self.metapot, self.MW_metapot, [MW[:,4]])
                            self.MW_metapot = np.copy(self.metapot)
    
                        elif self.method == 'eABF' or self.method == 'meta-eABF':
                            self.hist_z = self.__MW_update(self.hist_z, self.MW_hist_z, [MW[:,4]])
                            self.MW_hist_z = np.copy(self.hist_z)
                            self.force_correction_czar = self.__MW_update(self.force_correction_czar, self.MW_force_correction_czar, [MW[:,5]])
                            self.MW_force_correction_czar = np.copy(self.force_correction_czar)
    
                            if self.method == 'meta-eABF':
                                self.abfforce = self.__MW_update(self.abfforce, self.MW_abfforce, [MW[:,6]])
                                self.MW_abfforce = np.copy(self.abfforce)
                    
                    else:
                        
                        MW_new = np.array([MW[:,2].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]), MW[:,3].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])])
                        self.bias = self.__MW_update(self.bias, self.MW_bias, MW_new)
                        self.MW_bias = np.copy(self.bias)
       
                        MW_new = np.array([MW[:,4].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]), MW[:,5].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])])                 
                        self.CV_crit = self.__MW_update(self.CV_crit, self.MW_CV_crit, MW_new)
                        self.MW_CV_crit = np.copy(self.CV_crit)
    
                        if self.method == 'metaD': 
                            self.metapot = self.__MW_update(self.metapot, self.MW_metapot, MW[:,6].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]))
                            self.MW_metapot = np.copy(self.metapot)
    
                        elif self.method == 'eABF' or self.method == 'meta-eABF':
                            self.hist_z = self.__MW_update(self.hist_z, self.MW_hist_z, MW[:,6].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]))
                            self.MW_hist_z = np.copy(self.hist_z)
    
                            MW_new =  np.array([MW[:,7].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]), MW[:,8].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])])
                            self.force_correction_czar = self.__MW_update(self.force_correction_czar, self.MW_force_correction_czar, MW_new)
                            self.MW_force_correction_czar = np.copy(self.force_correction_czar)
    
                            if self.method == 'meta-eABF':
                                MW_new =  np.array([MW[:,9].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]), MW[:,10].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])])
                                self.abfforce = self.__MW_update(self.abfforce, self.MW_abfforce, MW_new)
                                self.MW_abfforce = np.copy(self.abfforce)
                    
                    if os.access(MW_file, os.W_OK):
                        self.write_restart(MW_file)
                        print('Bias synced with other walkers')
                     
                    else:
                        print('Failed to sync bias with other walkers')
                                          
      
                    # set MW_file back to read only
                    os.chmod(MW_file, 0o444)                
    
                elif trial < 10:

                    # try again
                    time.sleep(0.1)
                    self.MW(MW_file, sync_interval=sync_interval, trial=trial+1)
                 
                else:
                    print('Failed to sync bias with other walkers')
    
            else:
    
                # create MW buffer
                self.write_restart(MW_file)          
                os.chmod(MW_file, 0o444)                
    
    # -----------------------------------------------------------------------------------------------------
    def __MW_update(self, new, old, walkers):
        '''update accumulaters from MW buffer
        '''
        return walkers + (new - old)

    # -----------------------------------------------------------------------------------------------------
    def restart(self, filename='restart_bias.dat'):
        '''restart calculation from restart_bias.dat
        '''
        if os.path.isfile(filename):

            data = np.loadtxt(filename)
    
            self.histogramm = data[:,0].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
            self.geom_corr  = data[:,1].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
    
            if self.ncoords == 1:
                self.bias[0] = data[:,2].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                self.CV_crit[0] = data[:,3].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                if self.method == 'metaD':
                    self.metapot = data[:,4].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                elif self.method == 'eABF' or self.method == 'meta-eABF':
                    self.hist_z = data[:,4].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                    self.force_correction_czar[0] = data[:,5].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                    if self.method == 'meta-eABF':
                        self.abfforce[0] = data[:,6].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
    
            else:
                self.bias[0] = data[:,2].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                self.bias[1] = data[:,3].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                self.CV_crit[0] = data[:,4].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                self.CV_crit[1] = data[:,5].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                if self.method == 'metaD':
                    self.metapot = data[:,6].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                elif self.method == 'eABF' or self.method == 'meta-eABF':
                    self.hist_z = data[:,6].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                    self.force_correction_czar[0] = data[:,7].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                    self.force_correction_czar[1] = data[:,8].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                    if self.method == 'meta-eABF':
                        self.abfforce[0] = data[:,9].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                        self.abfforce[1] = data[:,10].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
            print('-------------------------------------------------------------')
            print('\tBias restarted from checkpoint file!')  
            print('-------------------------------------------------------------')

        else:
            print('-------------------------------------------------------------')
            print('\tRestart file not found.')  
            print('-------------------------------------------------------------')

    # -----------------------------------------------------------------------------------------------------
    def write_restart(self, filename='restart_bias.dat'):
        '''write relevant data for restart to txt file

        args:
            -
        returns:
            -
        '''
        out = open(filename, "w")
        if self.ncoords == 1:
            for i in range(self.nbins_per_dim[0]):
                row = (self.histogramm[0,i], self.geom_corr[0,i], self.bias[0][0,i], self.CV_crit[0][0,i])
                out.write("%14.10f\t%14.10f\t%14.10f\t%14.10f" % row)
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
                    row = (self.histogramm[i,j], self.geom_corr[i,j], self.bias[0][i,j], self.bias[1][i,j], self.CV_crit[0][0,i], self.CV_crit[1][0,i])
                    out.write("%14.10f\t%14.10f\t%14.10f\t%14.10f\t%14.10f\t%14.10f" % row)
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
        for i in range(self.ncoords):
            if self.CV[i] == 'angle' or self.CV[i] == 'torsion' or self.CV[i] == 'lin_comb_angles':
                self.traj  = np.degrees(self.traj)
                if extended == True:
                    self.etraj = np.degrees(self.etraj)
            
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
    def write_output(self, filename='bias_out.dat'):
        '''write output of free energy calculations
        '''
        if self.method == 'metaD' or self.method == 'reference':
            self.mean_force = self.bias

        crit = self.__get_cond_avg(self.CV_crit, self.histogramm)
        
        grid = np.copy(self.grid)
        for i in range(self.ncoords):
            if self.CV[i] =='angle' or self.CV[i] == 'torsion' or self.CV[i] == 'lin_comb_angles': 
                grid[i] = np.degrees(self.grid[i])
        
        out = open(filename, "w")
        if self.ncoords == 1:
            head = ("Xi1", "CV_crit", "Histogramm", "Bias", "dF", "geom_corr")
            out.write("%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
            for i in range(self.nbins_per_dim[0]):
                row = (grid[0][i], crit[0][0,i], self.histogramm[0,i], self.mean_force[0][0,i], self.dF[0,i], self.geom_correction[0,i])
                out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f" % row)
                out.write("\n")

        elif self.ncoords == 2:
            if self.method == 'metaD' or 'reference':
                head = ("Xi1", "Xi0", "CV_crit1", "CV_crit0", "Histogramm", "Bias1", "Bias0", "geom_corr", "dF")
                out.write("%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
                for i in range(self.nbins_per_dim[1]):
                    for j in range(self.nbins_per_dim[0]):
                        row = (grid[1][i], grid[0][j], crit[1][i,j], crit[0][i,j], self.histogramm[i,j], self.mean_force[1][i,j], self.mean_force[0][i,j], self.geom_correction[i,j], self.dF[i,j])
                        out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)
                
            else:
                head = ("Xi1", "Xi1", "Histogramm", "Bias1", "Bias2", "geom_corr")
                out.write("%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
                for i in range(self.nbins_per_dim[1]):
                    for j in range(self.nbins_per_dim[0]):
                        row = (grid[1][i], grid[0][j], self.histogramm[i,j], self.mean_force[0][i,j], self.mean_force[1][i,j], self.geom_correction[i,j])
                        out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)
                        if self.ethod == 'ABF':
                            out.write("\t%14.6f\t%14.6f" % (crit[0][i,j], crit[1][i,j]))
                        out.write("\n")

        out.close()

    # -----------------------------------------------------------------------------------------------------
    def __print_parameters(self):
        '''print parameters after init
        '''
        print("\nInitialize CV's for {m}:".format(m=self.method))
        for i in range(self.ncoords):
            if self.CV[i] == 'angle' or self.CV[i] == 'torsion' or self.CV[i] == 'lin_comb_angles':
                print("\n\tMinimum CV%d:\t\t%14.6f rad" % (i,self.minx[i]))
                print("\tMaximum CV%d:\t\t%14.6f rad" % (i,self.maxx[i]))
                print("\tBinwidth CV%d:\t\t%14.6f rad" % (i,self.dx[i]))
            else:
                print("\n\tMinimum CV%d:\t\t%14.6f A" % (i,self.minx[i]))
                print("\tMaximum CV%d:\t\t%14.6f A" % (i,self.maxx[i]))
                print("\tBinwidth CV%d:\t\t%14.6f A" % (i,self.dx[i]))
         
        print("\t---------------------------------------------")
        print("\tTotel number of bins:\t%14.6f" % (self.nbins))

        if self.method == 'eABF' or self.method == 'meta-eABF':
            print("\nInitialize extended Lagrangian:")
            for i in range(self.ncoords):
                if self.CV[i] == 'angle' or self.CV[i] == 'torsion' or self.CV[i] == 'lin_comb_angles':
                    print("\n\tspring constant:\t%14.6f Hartree/rad^2" % (self.k[i]))
                else:
                    print("\n\tspring constant:\t%14.6f Hartree/A^2" % (self.k[i]))
                print("\tfictitious mass:\t%14.6f a.u." % (self.ext_mass[i]))


