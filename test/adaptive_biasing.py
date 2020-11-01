import sys
import time
import numpy as np
import pandas as pd
import random

# energy units
kB          = 1.380648e-23      # J / K
H_to_kJmol  = 2625.499639
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

                        ats = [[CV1, minx, maxx, dx, N_full],[CV2, ...]]

                        N_full:     Sampels per bin when full bias is applied, if N_bin < N_full: Bias = 1/N_bin * F_bias
                                    save choice: N_full >= 200

        metaD           Metadynamics or Well-Tempered metadynamics for 1D or 2D CV's
                        Free energy estimated from biasing potential: dF= -(T+deltaT)/deltaT * V_bias

                        ats = [[CV1, minx, maxx, dx, height, variance, interval, DeltaT],[CV2,...]

                        height:     Gaussian height [kJ/mol]
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

                        ats = [[CV1, minx, maxx, dx, N_full, sigma, tau, height, variance, interval, DeltaT],[CV2,...]]

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

        self.coord      = np.array([item[0] for item in ats])
        self.minx       = np.array([item[1] for item in ats])
        self.maxx       = np.array([item[2] for item in ats])
        self.dx         = np.array([item[3] for item in ats])

        (xi, delta_xi, div_delta_xi) = self.__get_coord()
        self.traj       = np.array([xi])

        # get number of bins
        self.nbins_per_dim = np.array([1,1])
        for i in range(len(self.coord)):
            self.nbins_per_dim[i] = int(np.ceil(np.abs(self.maxx[i]-self.minx[i])/self.dx[i])) if self.dx[i] > 0 else 1
        self.nbins = np.prod(self.nbins_per_dim)

        self.grid         = [np.linspace(self.minx[0],self.maxx[0],self.nbins_per_dim[0])]
        self.bias         = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)
        self.histogramm   = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.int32)
        self.sum_gradient = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)

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
                self.ext_natoms     = len(self.coord)
                self.ext_forces     = np.array([0.0 for i in range(self.ext_natoms)])
                self.ext_momenta    = np.array([0.0 for i in range(self.ext_natoms)])

                # Random Number Generator
                if type(random_seed) is int:
                    random.seed(random_seed)
                else:
                    print("\nNo seed was given for the random number generator of ABM so the system time is used!\n")

                for i in range(self.ext_natoms):
                    # initialize extended system at target temp of MD simulation

                    self.ext_momenta[i] = random.gauss(0.0,1.0)*np.sqrt(self.the_md.target_temp*self.ext_mass[i])
                    TTT = (np.power(self.ext_momenta, 2)/self.ext_mass).sum()
                    TTT /= (self.ext_natoms)
                    self.ext_momenta *= np.sqrt(self.the_md.target_temp/(TTT*au2k))

                self.etraj  = np.array([self.ext_coords])

                if method == 'meta-eABF':
                    # additional metadynamics parameters for meta-eABF

                    self.abfforce   = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)
                    self.metaforce  = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)

                    self.height     = np.array([item[7]/H_to_kJmol for item in ats])
                    self.variance   = np.array([item[8] for item in ats])
                    self.update_int = np.array([item[9] for item in ats], dtype=np.int32)
                    self.WT_dT      = np.array([item[10] for item in ats])

        elif method == 'metaD':
            # parameters for metaD or WT-metaD

            self.metaforce  = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)

            self.height     = np.array([item[4]/H_to_kJmol for item in ats])
            self.variance   = np.array([item[5] for item in ats])
            self.update_int = np.array([item[6] for item in ats], dtype=np.int32)
            self.WT_dT      = np.array([item[7] for item in ats])

        else:
            print('\nAdaptive biasing method not implemented!')
            print('Available choices: ABF, eABF, metaD or meta-eABF.')
            sys.exit(1)

        self.print_parameters()

    # -----------------------------------------------------------------------------------------------------
    def ABF(self, write_traj=True):
        '''Adaptive biasing force method

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

            for i in range(len(self.coord)):

                self.histogramm[i,bink[i]] += 1

                # ramp function R(N,k)
                Rk = 1.0 if self.histogramm[i,bink[i]] > self.ramp_count[i] else self.histogramm[i,bink[i]]/self.ramp_count[i]

                # inverse gradient v_i
                delta_xi_n = np.linalg.norm(delta_xi[i])
                v_i = delta_xi[i]/(delta_xi_n*delta_xi_n)
                self.sum_gradient[i,bink[i]] += delta_xi_n

                # apply biase force
                self.bias[i,bink[i]] += np.dot(self.the_md.forces, v_i) + kB_a*self.the_md.target_temp*div_delta_xi[i]
                self.the_md.forces -= Rk * (self.bias[i,bink[i]]/self.histogramm[i,bink[i]]) * delta_xi[i]

        self.timing = time.perf_counter() - start

        if write_traj == True:
            self.__write_traj()

        if self.the_md.step%self.out_freq == 0:
            # calculate free energy and write output
            self.__F_from_ABF()
            self.__write_output()
            self.__write_conv()

    #------------------------------------------------------------------------------------------------------
    def metaD(self, WT=True, grid=True, write_traj=True):
        '''Metadynamics and Well-Tempered Metadynamics

        args:
            WT              (bool, True, use Well-Tempered metaD)
            grid            (bool, True, use grid to save bias between function calls)
            write_traj      (bool, True, write trajectory to CV_traj.dat)
        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, div_delta_xi) = self.__get_coord()

        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self.__get_bin(xi)
            self.histogramm[bink] += 1

            for i in range(len(self.coord)):

                # for geometric correction of free energy
                self.sum_gradient[bink] += np.linalg.norm(delta_xi[i])

                # apply bias force
                bias_force = self.__calc_metaD_bias(xi,bink,WT=WT,grid=grid)
                self.the_md.forces -= bias_force * delta_xi[i]

        self.traj = np.append(self.traj, [xi], axis = 0)
        self.timing = time.perf_counter() - start

        if write_traj == True:
            self.__write_traj()

        if self.the_md.step%self.out_freq == 0:

            # calculate free energy and write output
            self.__F_from_metaD(WT=WT, grid=grid)
            self.__write_output()

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

            for i in range(len(self.coord)):

                self.histogramm[i,la_bin[i]] += 1

                # harmonic coupling of exteded coordinate to reaction coordinate
                dxi                 = self.ext_coords[i] - xi[i]
                self.ext_forces[i]  = self.k * dxi
                self.the_md.forces -= self.k * dxi * delta_xi[i]

                # for geometric correction of free energy
                self.sum_gradient[i,la_bin[i]] += np.linalg.norm(delta_xi[i])

                print()
                # ramp function R(N,k)
                Rk = 1.0 if self.histogramm[i,la_bin[i]] > self.ramp_count[i] else self.histogramm[i,la_bin[i]]/self.ramp_count[i]

                # apply biase force
                self.bias[i,la_bin[i]] += self.k * dxi
                self.ext_forces        -= Rk * self.bias[i,la_bin[i]]/self.histogramm[i,la_bin[i]]

        else:

            for i in range(len(self.coord)):

                # outside of bins only harmonic coupling without bias
                dxi                 = self.ext_coords[i] - xi[i]
                self.ext_forces[i]  = self.k * dxi
                self.the_md.forces -= self.k * dxi * delta_xi[i]

        self.__up_momenta_extended(friction=self.friction)

        self.traj  = np.append(self.traj, [xi], axis = 0)
        self.etraj  = np.append(self.etraj, [self.ext_coords], axis = 0)

        self.timing = time.perf_counter() - start

        if write_traj == True:
            self.__write_traj(extended = True)

        if self.the_md.step%self.out_freq == 0:
            self.__F_from_ABF()     # eABF/naive
            self.__F_from_CZAR()    # eABF/CZAR
            self.__write_output()
            self.__write_conv()

    #------------------------------------------------------------------------------------------------------
    def meta_eABF(self, WT = True, grid = True, write_traj = True):
        '''meta-eABF or WTM-eABF: combination of eABF with metadynamic

        args:
            WT              (bool, True, use Well-Tempered metadynamics)
            grid            (bool, True, store metadynamic bias on grid between function calls)
            write_traj      (bool, True, write trajectory to CV_traj.dat)
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

            for i in range(len(self.coord)):

                self.histogramm[i,la_bin[i]] += 1

                # for geometric correction of free energy
                self.sum_gradient[i,la_bin[i]] += np.linalg.norm(delta_xi[i])

                # harmonic coupling of exteded coordinate to reaction coordinate
                dxi                 = self.ext_coords[i] - xi[i]
                self.ext_forces[i]  = self.k * dxi
                self.the_md.forces -= self.k * dxi * delta_xi[i]

                # eABF bias
                Rk = 1.0 if self.histogramm[i,la_bin[i]] > self.ramp_count[i] else self.histogramm[i,la_bin[i]]/self.ramp_count[i]
                self.abfforce[i,la_bin[i]] += self.k * dxi
                self.ext_forces -= Rk * self.abfforce[i,la_bin[i]]/self.histogramm[i,la_bin[i]]

                # metaD bias
                metaforce = self.__calc_metaD_bias(self.ext_coords, la_bin[i], WT=WT, grid=grid)
                self.ext_forces -= metaforce

        else:

            for i in range(len(self.coord)):

                # outside of bins only harmonic coupling without bias
                dxi                 = self.ext_coords[i] - xi[i]
                self.ext_forces[i]  = self.k * dxi
                self.the_md.forces -= self.k * dxi * delta_xi[i]

        self.__up_momenta_extended(friction=self.friction)

        self.traj  = np.append(self.traj, [xi], axis = 0)
        self.etraj  = np.append(self.etraj, [self.ext_coords], axis = 0)

        self.timing = time.perf_counter() - start

        if write_traj == True:
            self.__write_traj(extended = True)

        if self.the_md.step%self.out_freq == 0:
            # calculate free energy and write output
            self.__F_from_CZAR()
            self.__write_output()
            self.__write_conv()

    # -----------------------------------------------------------------------------------------------------
    def __get_coord(self):
        '''get CV

        args:
            -
        returns:
            -
        '''
        xi = np.array([])
        delta_xi = [0 for i in range(len(self.coord))]
        div_delta_xi = [0 for i in range(len(self.coord))]

        for i in range(len(self.coord)):

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
           bink             (array, [bin0, bin1, bink])
        '''
        X = self.ext_coords if extended == True else xi

        bink = [0 for i in range(3)]

        if len(self.coord) == 1:
            bink[0] = int(np.floor(abs(X[0]-self.minx[0])/self.dx[0]))
            bink[1] = 0
            bink[2] = int(np.floor(abs(X[0]-self.minx[0])/self.dx[0]))
        else:
            bink[0] = int(np.floor(abs(X[0]-self.minx[0])/self.dx[0]))
            bink[1] = int(np.floor(abs(X[1]-self.minx[1])/self.dx[1]))
            bink[2] = bink[0] + self.nbins_per_dim[0] * bink[1]

        return bink

    # -----------------------------------------------------------------------------------------------------
    def __calc_metaD_bias(self, xi, bink, WT = True, grid = True):
        '''get Bias Potential and Force as sum of Gaussian kernels

        args:
            xi              (float, -, CV)
            bink            (int, -, Bin number of xi)
            WT              (bool, True, use WT-metaD)
            grid            (bool, True, use grid for force)
        returns:
            bias_force      (double, MetaD force on xi)
        '''
        if grid == True:
            # update bias every update_int's step and save on grid

            if self.the_md.step%self.update_int[0] == 0:

                w = self.height[0]
                if WT == True:
                    w *= np.exp(-self.bias[0][bink]/(kB_a*self.WT_dT))

                dx = self.grid - xi[0]
                bias_factor = w * np.exp(-0.5*np.power(dx,2.0)/self.variance)

                self.bias += bias_factor.T
                self.metaforce += bias_factor.T * dx.T/self.variance

            bias_force = self.metaforce[0][bink]

        else:
            # calculate exact bias every timestep

            # not working !!!!!!
            bias_pot = 0
            bias_force = 0
            for j in range(0,len(self.traj),self.update_int[i]):

                w = self.height[i]
                if WT == True:
                    w *= np.exp(-bias_pot/(kB_a*self.WT_dT))

                dx = self.traj[j] - xi[i]
                bias_factor = w * np.exp(-0.5*(dx*dx)/self.variance[i])

                bias_pot    += bias_factor
                bias_force  += bias_factor * dx/self.variance[i]

        return bias_force

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
    def __get_mean(self):
        '''get mean force form sum of instanteneous forces

        args:
            -
        returns:
            -
        '''
        self.mean_force = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)
        for dim in range(len(self.coord)):
            for i in range(self.nbins):
                if self.histogramm[dim,i] > 0:
                    self.mean_force[dim,i] += self.bias[dim,i]/self.histogramm[dim,i]

    # -----------------------------------------------------------------------------------------------------
    def __F_from_ABF(self):
        '''on-the-fly integration of ABF force to obtain free energy estimate of ABF or eABF/naive

        args:
            -
        returns:
            -
        '''
        self.dF = np.array([0.0 for i in range(self.nbins)])
        if len(self.coord)==1:

            self.__get_mean()
            for i in range(self.nbins):
                self.dF[i] = np.sum(self.mean_force[0:i])*self.dx[0]
            self.dF -= self.dF.min()
            self.dF_geom = self.__F_geom_from_F(self.dF)

        else:
            pass

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
            # save matadynamic bias potential and force on grid

            self.bias *= 0
            self.metaforce *= 0
            for dim in range(len(self.coord)):
                for j in range(0,len(self.traj),self.update_int[dim]):
                    for i in range(self.nbins):

                        if WT == True:
                            WT_factor = np.exp(-self.bias[dim][i]/(kB_a*self.WT_dT))

                        dx = self.traj[j] - self.grid[i]
                        self.bias[dim][i] += self.height[dim] * WT_factor[dim] * np.exp(-0.5*np.sum(dx*dx)/self.variance[dim])
                        self.metaforce[dim][i] -= self.height[dim] * WT_factor[dim] * np.exp(-0.5*np.sum(dx*dx)/self.variance[dim]) * np.sum(dx)/self.variance[dim]

        # get standard free energie
        self.dF = - self.bias[0]
        if WT==True:
            self.dF *= (self.the_md.target_temp + self.WT_dT)/self.WT_dT
        self.dF -= self.dF.min()

        # get geometric free energie
        self.dF_geom = self.__F_geom_from_F(self.dF)

    # -----------------------------------------------------------------------------------------------------
    def __F_from_CZAR(self):
        '''on-the-fly CZAR free energy estimate from trajectory of CV and extended coordinate
           the outher 2 bins are cut due to numeric integration

        args:
            -
        returns:
            -
        '''
        traj         = pd.DataFrame()
        traj['z']    = self.traj[:,0]
        traj['la']   = self.etraj[:,0]

        self.dF_czar = np.array([0.0 for i in range(self.nbins)])
        dFbar        = np.array([0.0 for i in range(self.nbins)])
        ln_z         = np.array([0.0 for i in range(self.nbins)])
        dln_z        = np.array([0.0 for i in range(self.nbins)])

        # get histogramm along CV and mean harmonic force per bin from trajectory
        for i in range(self.nbins):
            z = traj[traj.iloc[:,0].between(self.minx[0]+i*self.dx[0],self.minx[0]+i*self.dx[0]+self.dx[0])]
            if len(z) > 0:
                ln_z[i] = np.log(len(z))
                dFbar[i] += self.k * (np.mean(z.iloc[:,1]) - self.grid[0][i])

        # numeric derivative of ln(rho(z)) by five point stencil
        for i in range(2,self.nbins-2):
            dln_z[i] += (-ln_z[i+2] + 8*ln_z[i+1] - 8*ln_z[i-1] + ln_z[i-2]) / (12.0*self.dx[0])

        dFbar -= kB_a * self.the_md.target_temp * dln_z

        # get F(z) from F'(z)
        for i in range(self.nbins):
            self.dF_czar[i] += np.sum(dFbar[0:i])*self.dx[0]
        self.dF_czar -= self.dF_czar.min()

        # get F^G(z)
        self.dF_czar_geom = self.__F_geom_from_F(self.dF_czar)

    # -----------------------------------------------------------------------------------------------------
    def __F_geom_from_F(self, F):
        '''get geometric free energy

        args:
            F       (array, standard free energy)
        returns:
            F_g     (array, geometric free energy)
        '''
        self.mean_grad = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)
        geom_corr      = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)
        for dim in range(len(self.coord)):
            for i in range(self.nbins):
                if self.histogramm[dim,i] > 0:
                    self.mean_grad[dim,i] = self.sum_gradient[dim,i]/self.histogramm[dim,i]
                    geom_corr[dim,i] = kB_a * self.the_md.target_temp * np.log(self.mean_grad[dim,i])

        F_g = F - geom_corr
        F_g -= F_g.min()

        return F_g


    # -----------------------------------------------------------------------------------------------------
    def __write_traj(self, extended = False):
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

        traj_out = open("CV_traj.dat", "a")
        traj_out.write("\n%14.6f\t" % (self.the_md.step*self.the_md.dt*it2fs))
        for i in range(len(self.traj[0])):
            traj_out.write("%14.6f\t" % (self.traj[-1][i]))
            if extended:
                traj_out.write("%14.6f\t" % (self.etraj[-1][i]))
        traj_out.close()

    # -----------------------------------------------------------------------------------------------------
    def __write_conv(self):
        '''write convergence file

        args:
            -
        returns:
            -
        '''
        if self.the_md.step == 0:
            conv_out = open("bias_conv.dat", "w")
            conv_out.write("%14s\t" % ("time [fs]"))
            for dim in range(len(self.coord)):
                for i in range(self.nbins):
                    conv_out.write("%14s\t" % (f"Bin{i}"))
            conv_out.close()

        conv_out = open("bias_conv.dat", "a")
        conv_out.write("\n%14.6f\t" % (self.the_md.step*self.the_md.dt*it2fs))
        for i in range(self.nbins):
            conv_out.write("%14.6f\t" % (self.mean_force[0][i]))
        conv_out.close()

    # -----------------------------------------------------------------------------------------------------
    def __write_output(self):
        '''write output of free energy calculations
        '''
        if len(self.coord) == 1:

            out = open(f"bias_out.dat", "w")
            if (self.method == 'ABF'):
                head = ("Bin", "Xi", "Histogramm", "Mean Grad", "Mean Force", "dF", "dF geom")
                out.write("%6s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
                for i in range(len(self.bias[0])):
                    row = (i, self.grid[0][i], self.histogramm[0,i], self.mean_grad[0,i], self.mean_force[0,i], self.dF[i], self.dF_geom[i])
                    out.write("%6d\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)
                out.close()

            elif (self.method == 'eABF'):
                head = ("Bin", "Xi", "Hist (la)", "Mean Grad", "Mean Force", "dF/naive", "dF/naive geom", "dF/CZAR", "dF/CZAR geom")
                out.write("%6s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
                for i in range(len(self.bias[0])):
                    row = (i, self.grid[0][i], self.histogramm[0,i], self.mean_grad[0,i], self.mean_force[0,i], self.dF[i], self.dF_geom[0,i], self.dF_czar[i], self.dF_czar_geom[0,i])
                    out.write("%6d\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)
                out.close()

            elif (self.method == 'metaD'):
                head = ("Bin", "Xi", "Histogramm", "Mean Grad", "Bias Pot", "Bias Force", "dF", "dF geom")
                out.write("%6s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
                for i in range(len(self.bias[0])):
                    row = (i, self.grid[i], self.histogramm[i], self.mean_grad[i], self.bias[0][i],self.metaforce[0][i], self.dF[i], self.dF_geom[i])
                    out.write("%6d\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)
                out.close()

            elif (self.method == 'meta-eABF'):
                head = ("Bin", "Xi", "Histogramm", "Mean Grad", "eABF bias", "MetaD bias", "dF", "dF geom", "dF/CZAR", "dF/CZAR geom")
                out.write("%6s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
                for i in range(len(self.bias[0])):
                    row = (i, self.grid[i], self.histogramm[i], self.mean_grad[i], self.abfforce[0][i], self.metaforce[0][i],self.dF_czar[i], self.dF_czar_geom[i])
                    out.write("%6d\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)
                out.close()


        else:
            # 2D reaction coordinate
            pass

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
                print("\n\tN_full Xi%d:\t\t\t\t%14.6f steps\n" % (i,self.ramp_count[i-1]))
                if self.method == 'eABF' or self.method == 'meta-eABF':
                    print("\tSpring constant for extended variable:\t%14.6f Hartree/Bohr^2" % (self.k[i-1]))
                    print("\tfictitious mass for extended variable:\t%14.6f a.u." % (self.ext_mass[i-1]))
                    if self.method == 'meta-eABF':
                        print("\n\tGaussian height%d:\t\t\t%14.6f Hartree" % (i,self.height[i-1]))
                        print("\tGaussian variance%d:\t\t\t%14.6f Bohr" % (i,self.variance[i-1]))
                        print("\ttime intervall for update of bias:\t%14.6f steps" % (self.update_int[i-1]))
                        print("\n\tdT for WT-metaD:\t\t\t%14.6f K" % (self.WT_dT[i-1]))

            if self.method == 'metaD':
                print("\n\tGaussian height%d:\t\t\t%14.6f Hartree" % (i,self.height[i-1]))
                print("\tGaussian variance%d:\t\t\t%14.6f Bohr" % (i,self.variance[i-1]))
                print("\ttime intervall for update of bias:\t%14.6f steps" % (self.update_int[i-1]))
                print("\tdT for WT-metaD:\t\t\t%14.6f K" % (self.WT_dT[i-1]))
        print("################################################################################\n")
