import sys
import time
import random
import numpy as np
import scipy.integrate as spi

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
                        Free energy estimated by integration of biasing force

                        ats = [[CV1, minx, maxx, dx],[CV2, ...]]

                        N_full:     Sampels per bin when full bias is applied, if N_bin < N_full: Bias = 1/N_bin * F_bias

        metaD           Metadynamics or Well-Tempered metadynamics for 1D or 2D CV's
                        Free energy estimated from biasing potential: dF= -(T+deltaT)/deltaT * V_bias

                        ats = [[CV1, minx, maxx, dx, variance, interval, DeltaT],[CV2,...]

                        variance:   Gaussian variance [Bohr]
                        interval:   Time intevall for deposition of Gaussians along CV [steps]
                        deltaT:     for Well-Tempered-metaD: deltaT -> 0            ordinary MD
                                                             500 < deltaT < 5000    WT-metaD
                                                             deltaT -> inf          standard metaD

        eABF            extended adaptive biasing force method for 1D or 2D CV's
                        Free energy obtained from CZAR estimator (Lesage, JPCB, 2016)

                        ats = [[CV1, minx, maxx, dx, N_full, sigma, tau],[CV2, ...]]

                        N_full:     Samples per bin where full bias is applied
                        sigma:      standard deviation between CV and fictitious particle [Bohr]
                                    both are connected by spring force with force constant k=1/(beta*sigma^2) [Hartree]
                        tau:        oscillation period of fictitious particle [fs]
                                    mass of fictitious particle: m = 1/beta * (tau/(2pi * sigma))^2 [a.u.]

        meta-eABF       Bias of extended coordinate by (WT) metaD + eABF (WTM-eABF or meta-eABF)
                        Free energy obtained from CZAR estimator (Lesage, JPCB, 2016)

                        ats = [[CV1, minx, maxx, dx, N_full, sigma, tau, variance, interval, DeltaT],[CV2,...]]

    Init parameters:
        MD:             MD object from InterfaceMD
        method:         'ABF', 'metaD', 'eABF' or 'meta-eABF'
        ats:            input parameters
        output_frec:    Number of steps between outputs + frequency of free energy calculation (default: 1000)
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
        self.traj       = np.array([xi])

        # get number of bins
        self.nbins_per_dim = np.array([1,1])
        self.grid          = []
        for i in range(len(self.coord)):
            self.nbins_per_dim[i] = int(np.ceil(np.abs(self.maxx[i]-self.minx[i])/self.dx[i])) if self.dx[i] > 0 else 1
            self.grid.append(np.linspace(self.minx[i]+self.dx[i]/2,self.maxx[i]-self.dx[i]/2,self.nbins_per_dim[i]))
        self.nbins = np.prod(self.nbins_per_dim)
	
        self.bias       = np.array([np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])) for i in range(self.ncoords)], dtype=np.float64)
        self.histogramm = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.int32)
        self.geom_corr  = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)

        if method == 'ABF' or method == 'eABF' or method == 'meta-eABF':
            # parameters special to ABF-like methods

            self.ramp_count = np.array([item[4] for item in ats])

            if method == 'eABF' or method == 'meta-eABF':
                # setup extended system for eABF or meta-eABF

                sigma       = np.array([item[5] for item in ats])
                tau         = np.array([item[6]*fs2au for item in ats])

                self.k      = (kB_a*self.the_md.target_temp) / (sigma*sigma)
                self.ext_mass = kB_a * H2au * self.the_md.target_temp * (tau/(2*np.pi*sigma)) * (tau/(2*np.pi*sigma))

                self.ext_coords     = np.copy(xi)
                self.ext_forces     = np.array([0.0 for i in range(self.ncoords)])
                self.ext_momenta    = np.array([0.0 for i in range(self.ncoords)])

                # Random Number Generator
                if type(random_seed) is int:
                    random.seed(random_seed)
                else:
                    print("\nNo seed was given for the random number generator of ABM so the system time is used!\n")

                for i in range(self.ncoords):
                    # initialize extended system at target temp of MD simulation

                    self.ext_momenta[i] = random.gauss(0.0,1.0)*np.sqrt(self.the_md.target_temp*self.ext_mass[i])
                    TTT = (np.power(self.ext_momenta, 2)/self.ext_mass).sum()
                    TTT /= (self.ncoords)
                    self.ext_momenta *= np.sqrt(self.the_md.target_temp/(TTT*au2k))

                self.etraj  = np.array([self.ext_coords])

                # accumulators for czar estimator
                self.force_correction_czar = np.array([np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])) for i in range(self.ncoords)], dtype=np.float64)
                self.hist_z = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)

                if method == 'meta-eABF':
                    # additional metadynamic parameters for meta-eABF or WTM-eABF
                    
                    self.metapot    = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)
                    self.abfforce   = np.copy(self.bias)
                    self.variance   = np.array([item[7] for item in ats])
                    self.update_int = np.array([item[8] for item in ats], dtype=np.int32)
                    self.WT_dT      = np.array([item[9] for item in ats])

        elif method == 'metaD':
            # parameters for metaD or WT-metaD
            
            self.metapot    = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)
            self.variance   = np.array([item[4] for item in ats])
            self.update_int = np.array([item[5] for item in ats], dtype=np.int32)
            self.WT_dT      = np.array([item[6] for item in ats])

        else:
            print('\nAdaptive biasing method not implemented!')
            print('Available choices: ABF, eABF, metaD or meta-eABF.')
            sys.exit(1)

        self.print_parameters()

    # -----------------------------------------------------------------------------------------------------
    def ABF(self, write_traj=True):
        '''Adaptive biasing force method

        args:
            N_full          (int, 100, number of samples per bin when full bias is applied)
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
            
            # gradient correction for geometric free energy
            self.geom_corr[bink[1],bink[0]] += self.__get_gradient_correction(delta_xi)

            for i in range(self.ncoords):

                # linear ramp function R(N,k)
                Rk = 1.0 if self.histogramm[bink[1],bink[0]] > self.ramp_count[i] else self.histogramm[bink[1],bink[0]]/self.ramp_count[i]

                # inverse gradient v_i
                delta_xi_n = np.linalg.norm(delta_xi[i])
                v_i = delta_xi[i]/(delta_xi_n*delta_xi_n)
		
                # apply biase force
                self.bias[i][bink[1],bink[0]] += np.dot(self.the_md.forces, v_i) + kB_a * self.the_md.target_temp * div_delta_xi[i]
                self.the_md.forces            -= Rk * (self.bias[i][bink[1],bink[0]]/self.histogramm[bink[1],bink[0]]) * delta_xi[i]

        if self.the_md.step%self.out_freq == 0:

            if write_traj == True:
                self.write_traj()
            
            # calculate free energy and write output
            self.mean_force = self.__get_mean(self.bias, self.histogramm)
            self.__F_from_Force(self.mean_force)
            self.write_output()
            self.write_conv()

        self.timing = time.perf_counter() - start
    
    #------------------------------------------------------------------------------------------------------
    def metaD(self, gaussian_height, WT=True, grid=True, write_traj=True):
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
            
            # gradient correction for geometric free energy
            self.gom_corr[bink[1],bink[0]] += self.__get_gradient_correction(delta_xi)
            
            # apply bias 
            self.__get_metaD_bias(xi, bink, gaussian_height, WT=WT, grid=grid)
            for i in range(self.ncoords):    
                self.the_md.forces += self.bias[i][bink[1],bink[0]] * delta_xi[i]
            
        self.traj = np.append(self.traj, [xi], axis = 0)
        
        if self.the_md.step%self.out_freq == 0:

            if write_traj == True:
                self.write_traj()
            
            # calculate free energy and write output
            self.__F_from_metaD(WT=WT, grid=grid)
            self.write_output()
            self.write_conv()
        
        self.timing = time.perf_counter() - start

    #------------------------------------------------------------------------------------------------------
    def eABF(self, write_traj = True):
        '''extended Adaptive Biasing Force method

        args:
            write_traj      (bool, True, write trajectory to CV_traj.dat)
        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, div_delta_xi) = self.__get_coord()

        self.__propagate_extended(friction=self.friction)

        if (self.ext_coords <= self.maxx).all() and (self.ext_coords >= self.minx).all():
            
            la_bin = self.__get_bin(xi, extended = True)
            
            # histogramm along extended coordinate for eABF bias
            self.histogramm[la_bin[1],la_bin[0]] += 1
             
            for i in range(self.ncoords):

                self.sum_gradient[la_bin[1],la_bin[0]] += np.linalg.norm(delta_xi[i])

                # harmonic coupling of exteded coordinate to reaction coordinate
                dxi                 = self.ext_coords[i] - xi[i]
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]
                
                # apply biase force
                Rk = 1.0 if self.histogramm[la_bin[1],la_bin[0]] > self.ramp_count[i] else self.histogramm[la_bin[1],la_bin[0]]/self.ramp_count[i]
                self.bias[i][la_bin[1],la_bin[0]] -= self.k[i] * dxi
                self.ext_forces[i] += Rk * self.bias[i][la_bin[1],la_bin[0]]/self.histogramm[la_bin[1],la_bin[0]]
                  
        else:

            for i in range(self.ncoords):

                # outside of bins only harmonic coupling without bias
                dxi                 = self.ext_coords[i] - xi[i]
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]

        # accumulators for czar estimator
        if (xi <= self.maxx).all() and (xi >= self.minx).all():
           
            bink = self.__get_bin(xi)
            self.hist_z[bink[1],bink[0]] += 1
            
            # gradient correction for geometric free energy
            self.gom_corr[bink[1],bink[0]] += self.__get_gradient_correction(delta_xi)
            
            for i in range(self.ncoords):
                self.force_correction_czar[i][bink[1],bink[0]] += self.k[i] * (self.ext_coords[i] - self.grid[i][bink[i]])

        self.__up_momenta_extended(friction=self.friction)

        self.traj  = np.append(self.traj, [xi], axis = 0)
        self.etraj  = np.append(self.etraj, [self.ext_coords], axis = 0)

        if self.the_md.step%self.out_freq == 0:

            if write_traj == True:
                self.write_traj(extended = True)
            
            self.__F_from_CZAR()
            self.write_output()
            self.write_conv()
        
        self.timing = time.perf_counter() - start

    #------------------------------------------------------------------------------------------------------
    def meta_eABF(self, gaussian_height, WT = True, grid = True, write_traj = True):
        '''meta-eABF or WTM-eABF: combination of eABF with metadynamic

        args:
            gaussian_height     (double, -, height of gaussians for metaD potential)
            WT                  (bool, True, use Well-Tempered metadynamics)
            grid                (bool, True, store metadynamic bias on grid between function calls)
            write_traj          (bool, True, write trajectory to CV_traj.dat)
        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, div_delta_xi) = self.__get_coord()

        self.__propagate_extended(friction=self.friction)

        self.traj  = np.append(self.traj, [xi], axis = 0)
        self.etraj  = np.append(self.etraj, [self.ext_coords], axis = 0)

        if (self.ext_coords <= self.maxx).all() and (self.ext_coords >= self.minx).all():

            la_bin = self.__get_bin(xi, extended = True)
            
            # histogramm along extended coordinate for eABF bias
            self.histogramm[la_bin[1],la_bin[0]] += 1
            
            for i in range(self.ncoords):

                self.sum_gradient[la_bin[1],la_bin[0]] += np.linalg.norm(delta_xi[i])

                # harmonic coupling of exteded coordinate to reaction coordinate
                dxi                 = self.ext_coords[i] - xi[i]
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]
                self.ext_forces[i]  = self.k[i] * dxi

                # WTM bias
                self.ext_forces[i] += self.bias[i][la_bin[1],la_bin[0]] 
            
                # eABF bias
                Rk = 1.0 if self.histogramm[la_bin[1],la_bin[0]] > self.ramp_count[i] else self.histogramm[la_bin[1],la_bin[0]]/self.ramp_count[i]
                self.abfforce[i][la_bin[1],la_bin[0]] -= self.k[i] * dxi
                self.ext_forces[i] += Rk * self.abfforce[i][la_bin[1],la_bin[0]]/self.histogramm[la_bin[1],la_bin[0]]

        else:

            for i in range(self.ncoords):

                # outside of bins only harmonic coupling without bias
                dxi                 = self.ext_coords[i] - xi[i]
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]

        # accumulators for czar estimator
        if (xi <= self.maxx).all() and (xi >= self.minx).all():
            
            # histogramm along physical coordinate 
            bink = self.__get_bin(xi)
            self.hist_z[bink[1],bink[0]] += 1
            
            # gradient correction for geometric free energy
            self.gom_corr[bink[1],bink[0]] += self.__get_gradient_correction(delta_xi)

            # force correction  
            for i in range(self.ncoords):
                self.force_correction_czar[i][bink[1],bink[0]] += self.k[i] * (self.ext_coords[i] - self.grid[i][bink[i]])
        
        self.__up_momenta_extended(friction=self.friction)

        self.traj  = np.append(self.traj, [xi], axis = 0)
        self.etraj  = np.append(self.etraj, [self.ext_coords], axis = 0)

        if self.the_md.step%self.out_freq == 0:
            
            if write_traj == True:
                self.write_traj(extended = True)
            
            # calculate free energy and write output
            self.__F_from_CZAR() 
            self.write_output()
            self.write_conv()
        
        self.timing = time.perf_counter() - start

    # -----------------------------------------------------------------------------------------------------
    def __get_coord(self):
        '''get CV

        args:
            -
        returns:
            -
        '''
        xi = np.array([])
        delta_xi	 = [0 for i in range(self.ncoords)]
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
                div_dela_xi[i] = 2*xi[i]

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
           bink             (array, [col, row])
        '''
        X = xi if extended == False else self.ext_coords
        
        binX = [0,0]
        for i in range(self.ncoords):
            binX[i] = int(np.floor(np.abs(X[i]-self.minx[i])/self.dx[i]))
        
        return binX


    # -----------------------------------------------------------------------------------------------------
    def __get_metaD_bias(self, xi, bink, height, WT = True, grid = True):
        '''get Bias Potential and Force as sum of Gaussian kernels

        args:
            xi              (float, -, CV)
            bink            (int, -, Bin number of xi)
            height          (double, -, Gaussian height)
            WT              (bool, True, use WT-metaD)
            grid            (bool, True, use grid for force)
        returns:
            bias_force      (double, MetaD force on xi)
        '''
        if grid == True:

            if self.ncoords == 1:
                
                if self.the_md.step%self.update_int[0] == 0:
                    # update bias every update_int's step and save on grid

                    w = height/H_to_kJmol
                    if WT == True:
                        # scale height according to Well-Tempered metaD
                        w *= np.exp(-self.metapot[bink[1],bink[0]]/(kB_a*self.WT_dT[0]))

                    # get new Gaussian 
                    dx = self.grid[0] - xi[0]
                    bias_factor = w * np.exp(-0.5*np.power(dx ,2.0)/self.variance[0])
                    
                    # update bias potential and force 
                    self.metapot += bias_factor
                    self.bias[0] -= bias_factor.T * dx.T/self.variance[0]
                
            else:

                if self.the_md.step%self.update_int[0] == 0:

                    w = height/H_to_kJmol
                    if WT == True:
                        w *= np.exp(-self.metapot[bink[1],bink[0]]/(kB_a*self.WT_dT[0]))
                    
                    # add 2D Gaussian to bias potential 
                    for i in range(self.nbins_per_dim[1]):
                        for j in range(self.nbins_per_dim[0]):

                            exp1 = np.power(self.grid[0][j]-xi[0],2.0)/self.variance[0]  
                            exp2 = np.power(self.grid[1][i]-xi[1],2.0)/self.variance[1]
                            gauss = np.exp(-0.5*(exp1+exp2))

                            self.metapot[i,j] += w * gauss
                            
                            self.bias[0][i,j] -= w * gauss * (self.grid[0][j]-xi[0])/self.variance[0]
                            self.bias[1][i,j] -= w * gauss * (self.grid[1][i]-xi[1])/self.variance[1]

        else:
            # not yet implemented
            pass
        
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
            d[0,0] = np.dot(delta_xi[0],delta_xi[0])
            d[1,1] = np.dot(delta_xi[1],delta_xi[1])
            d[1,0] = np.dot(delta_xi[0],delta_xi[1])
            d[0,1] = np.dot(delta_xi[1],delta_xi[0])
            return np.linalg.det(np.sqrt(d))

    # -----------------------------------------------------------------------------------------------------
    def __propagate_extended(self, langevin=True, friction=1.0e-3):
        '''Propagate momenta/coords of extended variable with Velocity Verlet

        args:
           langevin                (bool, False)
           friction                (float, 10^-3 1/fs)
        returns:
           -
        '''
        if langevin==True:
            prefac    = 2.0 / (2.0 + friction*self.the_md.dt_fs)
            rand_push = np.sqrt(self.the_md.target_temp*friction*self.the_md.dt_fs*kB_a/2.0e0)
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
    def __up_momenta_extended(self, langevin=True, friction=1.0e-3):
        '''Update momenta of extended variables with Velocity Verlet

        args:
            langevin        (bool, True)
            friction        (float, 1.0e-3)
        returns:
            -
        '''
        if langevin==True:
            prefac = (2.0e0 - friction*self.the_md.dt_fs)/(2.0e0 + friction*self.the_md.dt_fs)
            rand_push = np.sqrt(self.the_md.target_temp*friction*self.the_md.dt_fs*kB_a/2.0e0)
            self.ext_momenta *= prefac
            self.ext_momenta += np.sqrt(self.ext_mass) * rand_push * self.ext_rand_gauss
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces
        else:
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces

    # -----------------------------------------------------------------------------------------------------
    def __get_mean(self, a, b):
        '''get mean of accumulated forces by division with histogramm

        args:
            a		(array, -)
        returns:
            -
        '''
        # returns zero for bins without samples 
        mean_force = np.divide(a, b, out=np.zeros_like(a), where=(b!=0)) 
        return mean_force 

    # -----------------------------------------------------------------------------------------------------
    def __F_from_Force(self, mean_force):
        '''numeric on-the-fly integration of mean force to obtain free energy estimate 

        args:
            -
        returns:
            -
        '''
        self.dF = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)

        if self.ncoords == 1:
            # numeric integration with Simpson rule

            for i in range(1,self.nbins_per_dim[0]):
                self.dF[0,i] = np.sum(mean_force[0][0,0:i]) * self.dx[0]
            self.dF -= self.dF.min()

        else:
            # use Simpson rule twice for integration in 2D
            
            for i in range(1,self.nbins_per_dim[1]):
                for j in range(1,self.nbins_per_dim[0]):
                    
                    if self.histogramm[i,j] == 0:
                        self.dF[i,j] == 0.0
                    else:
                        self.dF[i,j] += spi.simps(mean_force[0][i,0:j], self.grid[0][0:j])/2.0
                        self.dF[i,j] += spi.simps(mean_force[1][0:i,j], self.grid[1][0:i])/2.0
            
            self.dF -= self.dF.min()

        # get geometric free energie
        self.__F_geom_from_F()

    # -----------------------------------------------------------------------------------------------------
    def __F_from_metaD(self, WT=True, grid = True):
        '''on-the-fly free energy estimate from metaD or WT-metaD bias potential

        args:
            WT              (bool, Well-Tempered metaD)
            grid            (bool, bias pot and force already saved on grid)
        returns:
            -
        '''
        if grid == False:
            # not yet implemented
            pass

        # get standard free energie
        self.dF = - self.metapot
        if WT==True:
            self.dF *= (self.the_md.target_temp + self.WT_dT[0])/self.WT_dT[0]
        self.dF -= self.dF.min()

        # get geometric free energie
        self.__F_geom_from_F()

    # -----------------------------------------------------------------------------------------------------
    def __F_from_CZAR(self):
        '''on-the-fly CZAR estimate for unbiased free energy 
        
        args:
            -
        returns:
            -
        '''
        self.mean_force = np.array([np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])) for i in range(self.ncoords)], dtype=np.float64)
        
        # get ln(rho) and z-conditioned average force per bin
        log_rho   = np.log(self.hist_z, out=np.zeros_like(self.hist_z), where=(self.hist_z!=0))
        avg_force = self.__get_mean(self.force_correction_czar, self.hist_z)         
        
        if self.ncoords == 1: 
            # CZAR estimate for dF(z)/dz    
            self.mean_force[0]  = - kB_a * self.the_md.target_temp * np.gradient(log_rho[0], self.grid[0]) + avg_force[0]
        
        else:
            # partial derivatives of log(rho(z1,z2))
            der_log_rho = np.gradient(log_rho, self.grid[1], self.grid[0])
             
            # CZAR forces dF(z1,z2)/dz1 and dF(z1,z2)/dz2
            self.mean_force[0] = - kB_a * self.the_md.target_temp * der_log_rho[1] + avg_force[0]
            self.mean_force[1] = - kB_a * self.the_md.target_temp * der_log_rho[0] + avg_force[1]    

        # integrate czar estimate to get free energy
        self.__F_from_Force(self.mean_force)
        
        # get geometric free energie
        self.__F_geom_from_F(extended = True)
	
    # -----------------------------------------------------------------------------------------------------
    def __F_geom_from_F(self, extended = False):
        '''get geometric free energy

        args:
            extended	(bool, False, True for methods with extended system)
        returns:
            F_g     	(array, geometric free energy)
        '''
        hist = self.histogramm if extended == False else self.hist_z
        
        log_grad_corr = self.__get_mean(self.geom_corr, hist)
        log_grad_corr = np.log(log_grad_corr, out=np.zeros_like(log_grad_corr), where=(log_grad_corr!=0))
                
 
        self.dF_geom  = self.dF - kB_a * self.the_md.target_temp * log_grad_corr
        self.dF_geom -= self.dF_geom.min()

    # -----------------------------------------------------------------------------------------------------
    def write_traj(self, extended = False):
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
            traj_out.write("%14s" % ("timing"))
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

    # -----------------------------------------------------------------------------------------------------
    def write_conv(self):
        '''write convergence file

        args:
            -
        returns:
            -
        '''
        sum_F = np.sum(np.sum(self.dF)) 
        
        if self.the_md.step == 0:
            conv_out = open("bias_conv.dat", "w")
            conv_out.write("%14s\t%14s\n" % ("time [fs]", "change of dF"))
            conv_out.close()
        
        conv_out = open("bias_conv.dat", "a")
        conv_out.write("%14.6f\t%14.6f\n" % (self.the_md.step*self.the_md.dt*it2fs, sum_F/self.nbins))
        conv_out.close()
        

    # -----------------------------------------------------------------------------------------------------
    def write_output(self):
        '''write output of free energy calculations
        '''
        if self.method == 'metaD':
            self.mean_force = self.bias

        if self.method == 'meta-eABF':
            self.mean_force = self.bias + self.__get_mean(self.abfforce, self.hist_z)
	
        out = open(f"bias_out.dat", "w")
        if len(self.coord) == 1:
            head = ("Xi1", "Histogramm", "Bias", "dF", "dF geom")
            out.write("%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
            for i in range(self.nbins_per_dim[0]):
                row = (self.grid[0][i], self.histogramm[0,i], self.mean_force[0][0,i], self.dF[0,i], self.dF_geom[0,i])
                out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)

        elif len(self.coord) == 2:
            head = ("Xi1", "Xi1", "Histogramm", "Bias", "dF", "dF geom")
            out.write("%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
            for i in range(self.nbins_per_dim[1]):
                for j in range(self.nbins_per_dim[0]):
                    row = (self.grid[1][i], self.grid[0][j], self.histogramm[i,j], self.mean_force[0][i,j]+self.mean_force[1][i,j], self.dF[i,j], self.dF_geom[i,j])
                    out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)

        out.close()

    # -----------------------------------------------------------------------------------------------------
    def print_parameters(self):
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
            if self.method == 'ABF' or self.method == 'eABF' or self.method == 'meta-eABF':
                print("\n\tN_full Xi%d:\t\t\t\t%14.6f steps/bin" % (i,self.ramp_count[i-1]))
                if self.method == 'eABF' or self.method == 'meta-eABF':
                    print("\tSpring constant for extended variable:\t%14.6f Hartree/Bohr^2" % (self.k[i-1]))
                    print("\tfictitious mass for extended variable:\t%14.6f a.u." % (self.ext_mass[i-1]))
                    if self.method == 'meta-eABF':
                        print("\tGaussian variance%d:\t\t\t%14.6f Bohr" % (i,self.variance[i-1]))
                        print("\ttime intervall for update of bias:\t%14.6f steps" % (self.update_int[i-1]))
                        print("\tdT for WT-metaD:\t\t\t%14.6f K" % (self.WT_dT[i-1]))

            if self.method == 'metaD':
                print("\n\tGaussian variance%d:\t\t\t%14.6f Bohr" % (i,self.variance[i-1]))
                print("\ttime intervall for update of bias:\t%14.6f steps" % (self.update_int[i-1]))
                print("\tdT for WT-metaD:\t\t\t%14.6f K" % (self.WT_dT[i-1]))
        print("################################################################################\n")
