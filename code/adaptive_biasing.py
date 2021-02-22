#!/usr/bin/env python

import os
import sys
import time
import random
import numpy as np

from CVs import *

# constants
H_to_kJmol = 2625.499639       #
kB         = 1.380648e-23      # J / K
H_to_J     = 4.359744e-18      #
kB_a       = kB / H_to_J       # Hartree / K
au2k       = 315775.04e0       #
it2fs      = 1.0327503e0       # fs per iteration
bohr2angs  = 0.52917721092e0   #

class ABM:
    '''Class for adaptive biasing methods for 1D or 2D collective variables (CVs)

    Available methods:

        ref: Unbiased simulation, restraint to region of interest in CV space
             Can be used for equilibration or free energy estimation from unbiased simulation

        ABF: Adaptive biasing force method, CVs have to be orthogonal to constraints and to each other!
             1D: On-the-fly free energy estimate,
             2D: Postprocessing step (FEM integration) necessary

        MtD: Metadynamics or Well-Tempered metadynamics for 1D or 2D CV's
             On-the-fly free energy estimate: A(CV) = -(T+deltaT)/deltaT * U_bias

        eABF: Enhanced sampling of extended-system with adaptive biasing method for 1D or 2D CV's
              Unbiased estimate of thermodynamic force obtained from CZAR estimator (Lesage, JPCB, 2016)
              1D: On-the-fly free energy estimate,
              2D: Postprocessing step (FEM integration) necessary

        meta-eABF: Enhanced sampling of extended-system by (WT)-MtD + eABF (WTM-eABF or meta-eABF)
                   Unbiased estimate for thermodynamic force obtained from CZAR estimator (Lesage, JPCB, 2016)
                   1D: On-the-fly free energy estimate,
                   2D: Postprocessing step (FEM integration) necessary

        MW: Multiple-Walker strategy, can be used in combination with all abm methods
            shares bias with other walkers for faster convergence

    Definition of CV:

        basic: 	   [['keyword', [index 1, index 2, ...], min, max, bin width],[possible second dimension]]
        MtD:	   [[..., variance],[...]]
        eABF:      [[..., sigma, mass],[...]]
        meta-eABF: [[..., sigma, mass, variance],[...]]

        keywords:  'distance': distance 12 in range [0,inf] in Angstrom
                   'projected_distance': distance 12 projected on vector 13 in range [-inf,inf] in Angstrom
                   'angle': bend angle 123 in range [-180,180] in degrees
                   'torsion': torsion angle 1234 in range [-180,180] in degrees
                   'lin_comb_dists': linear combination of distances or projected distances
                   'lin_comb_angles': linear combination of bend or torsion angles

        indices can ether be index of single atom or array of indices to use center of mass

    Args:
        MD (obj): object of class MD
        CV (array): definition of Collective Variable
        method (string, meta-eABF): available:'ref', 'ABF', 'MtD', 'eABF' or 'meta-eABF'
        f_conf (double, 100): restraining force to range of interest [kJ/mol])
        output_frec (int, 100): Number of steps between outputs
        friction (double, 1.0e-3): friction coefficient for Langevin dynamics of extended system
        seed_in (int, -): random seed for Langevin dynamics of extended system

    '''
    def __init__(self, MD, CV, method = 'meta-eABF', f_conf = 100, output_freq = 100, friction = 1.0e-3, seed_in = None):

        # general parameters
        self.the_md   = MD
        self.method   = method
        self.out_freq = output_freq
        self.friction = friction

        # definition of CVs
        self.ncoords = len(CV)
        self.CV      = np.array([item[0] for item in CV])
        self.minx    = np.array([item[2] for item in CV])
        self.maxx    = np.array([item[3] for item in CV])
        self.dx      = np.array([item[4] for item in CV])

        self.atoms    = [[] for i in range(self.ncoords)]
        self.is_angle = [False for i in range(self.ncoords)]
        self.f_conf   = [f_conf/H_to_kJmol for i in range(self.ncoords)]

        for i in range(self.ncoords):

            if hasattr(CV[i][1], "__len__"):
                # use center of mass of group of atoms for CV
                for index, a in enumerate(CV[i][1]):
                    self.atoms[i].append(np.array(a)-1)

            else:
                # use coordinates of atoms
                self.atoms[i].append(CV[i][1]-1)

	    # unit conversion
            if self.CV[i] == 'angle' or self.CV[i] == 'torsion' or self.CV[i] == 'lin_comb_angles':
                self.is_angle[i] = True
                self.minx[i] = np.radians(self.minx[i])
                self.maxx[i] = np.radians(self.maxx[i])
                self.dx[i]   = np.radians(self.dx[i])

            else:
                self.minx[i]   /= bohr2angs
                self.maxx[i]   /= bohr2angs
                self.dx[i]     /= bohr2angs
                self.f_conf[i] *= bohr2angs*bohr2angs

        (xi, _, _) = self._get_coord()
        self.traj = np.array([xi])
        self.temp = np.array([self.the_md.temp])

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

        if method == 'ref' or method == 'ABF':
            pass

        elif method == 'eABF' or method == 'meta-eABF':
            # setup extended system for eABF or meta-eABF

            # force constant
            self.sigma = np.array([item[5] for item in CV])
            for i in range(self.ncoords):
                if self.is_angle[i]:
                    self.sigma[i] = np.radians(self.sigma[i])
                else:
                    self.sigma[i] /= bohr2angs

            self.k = (kB_a*self.the_md.target_temp) / (self.sigma*self.sigma)

            # mass in a.u.
            self.ext_mass = np.array([item[6] for item in CV])

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

                self.ext_momenta[i] = random.gauss(0.0,1.0)*np.sqrt(MD.target_temp*self.ext_mass[i])
                TTT  = (np.power(self.ext_momenta, 2)/self.ext_mass).sum()
                TTT /= (self.ncoords)
                self.ext_momenta *= np.sqrt(MD.target_temp/(TTT*au2k))

            # accumulators for czar estimator
            self.force_correction_czar = np.copy(self.bias)
            self.hist_z                = np.copy(self.histogramm)

            if method == 'meta-eABF':

                self.variance = np.array([item[7] for item in CV])
                for i in range(self.ncoords):
                    if self.is_angle[i]:
                        self.variance[i] = np.radians(self.variance[i])
                    else:
                        self.variance[i] /= bohr2angs

                self.abfforce = np.copy(self.bias)
                self.metapot  = np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)

        elif method == 'MtD':
            # parameters for MtD or WT-MtD

            self.variance = np.array([item[5] for item in CV])
            for i in range(self.ncoords):
                if self.is_angle[i]:
                    self.variance[i] = np.radians(self.variance[i])
                else:
                    self.variance[i] /= bohr2angs

            self.metapot  = np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0]), dtype=np.float64)

        else:
            print("\n----------------------------------------------------------------------")
            print("ERROR: Invalid keyword in definition of adaptove biasing method!")
            print('Available: ref, ABF, eABF, MtD, default: meta-eABF.')
            print("-----------------------------------------------------------------------")
            sys.exit(1)

        self._print_parameters()

    # -----------------------------------------------------------------------------------------------------
    def ref(self, write_output=True, write_traj=True):
        '''constrain CV to range of interest and get free energy from unbiased probability density
           can be used for equilibration prior to ABM simulation

        args:
            write_output (bool, True): write free energy estimate to abm.out)
            write_traj (bool, True): write trajectory to CV_traj.dat)

        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, _) = self._get_coord()
        self.traj = np.append(self.traj, [xi], axis = 0)
        self.temp = np.append(self.temp, self.the_md.temp)

        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self._get_bin(xi)
            self.histogramm[bink[1],bink[0]] += 1

            # correction for kinetic
            self.geom_corr[bink[1],bink[0]] += self._get_gradient_correction(delta_xi)

            for i in range(self.ncoords):
                self.CV_crit[i][bink[1],bink[0]] += abs(np.dot(delta_xi[i], self.the_md.forces))

        else:
            for i in range(self.ncoords):

                # confinement
                if xi[i] > self.maxx[i]:
                    diff = self._diff(self.maxx[i], xi[i], self.is_angle[i])
                    self.the_md.forces -= self.f_conf[i] * diff/self.dx[i] * delta_xi[i]

                elif xi[i] < self.minx[i]:
                    diff = self._diff(self.minx[i], xi[i], self.is_angle[i])
                    self.the_md.forces -= self.f_conf[i] * diff/self.dx[i] * delta_xi[i]

        if self.the_md.step%self.out_freq == 0:
            # write output

            if write_traj == True:
                self.write_traj(xi, extended = False)

            if write_output == True:
                self.dF = -kB_a*self.the_md.target_temp*np.log(self.histogramm, out=np.zeros_like(self.histogramm),where=(self.histogramm!=0))
                self._get_geom_correction()
                self.write_output()
                self.write_restart()

        self.timing = time.perf_counter() - start


    # -----------------------------------------------------------------------------------------------------
    def ABF(self, N_full=100, write_traj=True):
        '''Adaptive biasing force method

        args:
            N_full (double, 100): linear ramp for bias = N_bin/N_full if N_full > N_bin else 1
            write_traj (bool, True): write trajectory to CV_traj.dat

        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, div_delta_xi) = self._get_coord()

        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self._get_bin(xi)
            self.histogramm[bink[1],bink[0]] += 1

            self.geom_corr[bink[1],bink[0]] += self._get_gradient_correction(delta_xi)

            for i in range(self.ncoords):

                self.CV_crit[i][bink[1],bink[0]] += abs(np.dot(delta_xi[i], self.the_md.forces))

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
                if xi[i] > self.maxx[i]:
                    diff = self._diff(self.maxx[i], xi[i], self.is_angle[i])
                    self.the_md.forces -= self.f_conf[i] * diff/self.dx[i] * delta_xi[i]

                elif xi[i] < self.minx[i]:
                    diff = self._diff(self.minx[i], xi[i], self.is_angle[i])
                    self.the_md.forces -= self.f_conf[i] * diff/self.dx[i] * delta_xi[i]


        self.traj = np.append(self.traj, [xi], axis = 0)
        self.temp = np.append(self.temp, self.the_md.temp)

        if self.the_md.step%self.out_freq == 0:
            # write output

            if write_traj == True:
                self.write_traj(xi)

            self.mean_force = self._get_cond_avg(self.bias, self.histogramm)
            self._F_from_Force(self.mean_force)
            self.write_output()
            self.write_restart()

        self.timing = time.perf_counter() - start

    #------------------------------------------------------------------------------------------------------
    def MtD(self, gaussian_height=1.0, update_int=50, WT_dT=2000, WT=True, grid=True, write_traj=True):
        '''Metadynamics and Well-Tempered Metadynamics

        args:
            gaussian_height (double, 1.0): heigth of gaussians for bias potential [kJ/mol]
            WT (bool, True): use Well-Tempered MtD
            WT_dT (double, 2000): only used if WT=True
            grid (bool, True): use coarse grained bias force accumulated on grid, faster in long simulations
            write_traj (bool, True): write trajectory to CV_traj.dat

        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, _) = self._get_coord()

        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self._get_bin(xi)

            self.histogramm[bink[1], bink[0]] += 1
            self.geom_corr[bink[1],bink[0]] += self._get_gradient_correction(delta_xi)

            # apply metadynamic bias
            bias = self._get_MtD_bias(xi, bink, gaussian_height, update_int, self.the_md.step, WT_dT, WT, grid, False)
            for i in range(self.ncoords):
                self.CV_crit[i][bink[1],bink[0]] += abs(np.dot(delta_xi[i], self.the_md.forces))
                self.the_md.forces += bias[i] * delta_xi[i]

                # damp repulsion of systems out of bins with harmonic force near margin
                if xi[i] < (self.minx[i]+2*self.dx[i]):

                    diff = self._diff(self.minx[i]+2*self.dx[i], xi[i], self.is_angle[i])
                    self.the_md.forces -= self.f_conf[i] * diff/self.dx[i] * delta_xi[i]

                elif xi[i] > (self.maxx[i]-2*self.dx[i]):

                    diff = self._diff(self.maxx[i]-2*self.dx[i], xi[i], self.is_angle[i])
                    self.the_md.forces -= self.f_conf[i] * diff/self.dx[i] * delta_xi[i]

        else:

            # get repulsive MtD potential outside of bins to avoid discontinuity of force
            meta_force = self._get_MtD_bias(xi, 0, gaussian_height, update_int, self.the_md.step, WT_dT, WT, False, True)

            for i in range(self.ncoords):

                # confinement
                if xi[i] > self.maxx[i]:

                    diff = self._diff(self.maxx[i]-2*self.dx[i], xi[i], self.is_angle[i])
                    self.the_md.forces -= self.f_conf[i] * diff/self.dx[i] * delta_xi[i]

                elif xi[i] < self.minx[i]:

                    diff = self._diff(self.minx[i]+2*self.dx[i], xi[i], self.is_angle[i])
                    self.the_md.forces -= self.f_conf[i] * diff/self.dx[i] * delta_xi[i]

        self.traj = np.append(self.traj, [xi], axis = 0)
        self.temp = np.append(self.temp,self.the_md.temp)

        if self.the_md.step%self.out_freq == 0:
            # write output

            if write_traj == True:
                self.write_traj(xi)

            self._F_from_MtD(WT_dT, WT=WT)
            self.write_output()
            self.write_restart()

        self.timing = time.perf_counter() - start

    #------------------------------------------------------------------------------------------------------
    def eABF(self, N_full=100, write_traj = True):
        '''extended Adaptive Biasing Force method

        args:
	    N_full (double, 100): linear ramp for bias = N_bin/N_full if N_full > N_bin else 1
            write_traj (bool, True): write trajectory to CV_traj.dat

        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, _) = self._get_coord()

        self._propagate_extended()

        margin = 2*self.sigma

        if (self.ext_coords <= self.maxx).all() and (self.ext_coords >= self.minx).all():
            # lambda conditioned bias force

            la_bin = self._get_bin(xi, extended = True)
            self.histogramm[la_bin[1],la_bin[0]] += 1

            for i in range(self.ncoords):

                # harmonic coupling of extended coordinate to reaction coordinate
                dxi                 = self._diff(self.ext_coords[i], xi[i], self.is_angle[i])
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]

                # apply biase force
                Rk = 1.0 if self.histogramm[la_bin[1],la_bin[0]] > N_full else self.histogramm[la_bin[1],la_bin[0]]/N_full
                self.bias[i][la_bin[1],la_bin[0]] -= self.k[i] * dxi
                self.ext_forces[i] += Rk * self.bias[i][la_bin[1],la_bin[0]]/self.histogramm[la_bin[1],la_bin[0]]

                # confinement of extended variable with harmonic force near margin
                if self.ext_coords[i] < (self.minx[i]+margin[i]):

                    diff = self._diff(self.minx[i]+margin[i], self.ext_coords[i], self.is_angle[i])
                    self.ext_forces[i] -= self.f_conf[i] * diff/self.dx[i]

                elif self.ext_coords[i] > (self.maxx[i]-margin[i]):

                    diff = self._diff(self.maxx[i]-margin[i], self.ext_coords[i], self.is_angle[i])
                    self.ext_forces[i] -= self.f_conf[i] * diff/self.dx[i]

        else:

            for i in range(self.ncoords):

                # harmonic coupling
                dxi                 = self._diff(self.ext_coords[i],xi[i], self.is_angle[i])
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]

                # confinement of extended coordinate
                if self.ext_coords[i] > self.maxx[i]:

                    diff = self._diff(self.maxx[i]-margin[i], self.ext_coords[i], self.is_angle[i])
                    self.ext_forces[i] -= self.f_conf[i] * diff/self.dx[i]

                elif self.ext_coords[i] < self.minx[i]:

                    diff = self._diff(self.minx[i]+margin[i], self.ext_coords[i], self.is_angle[i])
                    self.ext_forces[i] -= self.f_conf[i] * diff/self.dx[i]

        if (xi <= self.maxx).all() and (xi >= self.minx).all():
            # xi-conditioned accumulators for CZAR

            bink = self._get_bin(xi)
            self.hist_z[bink[1],bink[0]] += 1

            self.geom_corr[bink[1],bink[0]] += self._get_gradient_correction(delta_xi)

            for i in range(self.ncoords):

                self.CV_crit[i][bink[1],bink[0]] += abs(np.dot(delta_xi[i], self.the_md.forces))

                dx = self._diff(self.ext_coords[i], self.grid[i][bink[i]], self.is_angle[i])
                self.force_correction_czar[i][bink[1],bink[0]] += self.k[i] * dx

        self._up_momenta_extended()

        self.traj = np.append(self.traj, [xi], axis = 0)
        self.etraj = np.append(self.etraj, [self.ext_coords], axis = 0)
        self.temp = np.append(self.temp, self.the_md.temp)

        if self.the_md.step%self.out_freq == 0:
            # write output

            if write_traj == True:
                self.write_traj(xi,extended = True)

            self._F_from_CZAR()
            self.write_output()
            self.write_restart()

        self.timing = time.perf_counter() - start

    #------------------------------------------------------------------------------------------------------
    def meta_eABF(self, N_full=100, gaussian_height=1.0, update_int=20, WT_dT=2000, WT = True, grid = True, write_traj = True):
        '''meta-eABF or WTM-eABF: combination of eABF with metadynamic

        args:
	    N_full (double, 100): linear ramp for ABF force = N_bin/N_full if N_full > N_bin
            gaussian_height (double, 1.0): height of gaussians for MtD potential in kJ/mol
            update_int (int, 20): intevall for deposition of gaussians in steps
            WT (bool, True): use Well-Tempered metadynamics (WTM)
            WT_dT (double, 2000): effective temperature for WTM
            grid (bool, True): accumulate metadynamic bias force on grid with same bin width than ABF force
            write_traj (bool, True): write trajectory to CV_traj.dat

        returns:
            -
        '''
        start = time.perf_counter()

        (xi, delta_xi, _) = self._get_coord()

        self._propagate_extended()

        margin = 2*self.sigma

        if (self.ext_coords <= self.maxx).all() and (self.ext_coords >= self.minx).all():

            la_bin = self._get_bin(xi, extended = True)

            self.histogramm[la_bin[1],la_bin[0]] += 1
            meta_force = self._get_MtD_bias(self.ext_coords, la_bin, gaussian_height, update_int, self.the_md.step, WT_dT, WT, grid, False)

            for i in range(self.ncoords):

                # harmonic coupling of extended coordinate to reaction coordinate
                dxi                 = self._diff(self.ext_coords[i], xi[i], self.is_angle[i])
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]

                # metadynamics bias
                self.ext_forces[i] += meta_force[i]

                # eABF bias
                Rk = 1.0 if self.histogramm[la_bin[1],la_bin[0]] > N_full else self.histogramm[la_bin[1],la_bin[0]]/N_full
                self.abfforce[i][la_bin[1],la_bin[0]] -= self.k[i] * dxi
                self.ext_forces[i] += Rk * self.abfforce[i][la_bin[1],la_bin[0]]/self.histogramm[la_bin[1],la_bin[0]]

                # damp repulsion by MtD potential with harmonic force near margin
                if self.ext_coords[i] < (self.minx[i]+margin[i]):

                    diff = self._diff(self.minx[i]+margin[i], self.ext_coords[i], self.is_angle[i])
                    self.ext_forces[i] -= self.f_conf[i] * diff/self.dx[i]

                elif self.ext_coords[i] > (self.maxx[i]-margin[i]):

                    diff = self._diff(self.maxx[i]-margin[i], self.ext_coords[i], self.is_angle[i])
                    self.ext_forces[i] -= self.f_conf[i] * diff/self.dx[i]

        else:

            # get repulsive MtD potential outside of bins to avoid discontinuity of force
            meta_force = self._get_MtD_bias(self.ext_coords, 0, gaussian_height, update_int, self.the_md.step, WT_dT, WT, False, True)

            for i in range(self.ncoords):

                # harmonic coupling of CV and ext coord
                dxi                 = self._diff(self.ext_coords[i], xi[i], self.is_angle[i])
                self.ext_forces[i]  = self.k[i] * dxi
                self.the_md.forces -= self.k[i] * dxi * delta_xi[i]

                # confinement with MtD bias of nearest bin + harmonic restraing force
                if self.ext_coords[i] > self.maxx[i]:

                    diff = self._diff(self.maxx[i]-margin[i], self.ext_coords[i], self.is_angle[i])
                    self.ext_forces[i] += meta_force[i] - self.f_conf[i] * diff/self.dx[i]

                elif self.ext_coords[i] < self.minx[i]:

                    diff = self._diff(self.minx[i]+margin[i], self.ext_coords[i], self.is_angle[i])
                    self.ext_forces[i] += meta_force[i] - self.f_conf[i] * diff/self.dx[i]

        if (xi <= self.maxx).all() and (xi >= self.minx).all():
            # xi conditioned accumulators for CZAR

            bink = self._get_bin(xi)
            self.geom_corr[bink[1],bink[0]] += self._get_gradient_correction(delta_xi)

            self.hist_z[bink[1],bink[0]] += 1
            for i in range(self.ncoords):

                self.CV_crit[i][bink[1],bink[0]] += abs(np.dot(delta_xi[i], self.the_md.forces))

                dx = self._diff(self.ext_coords[i], self.grid[i][bink[i]], self.is_angle[i])
                self.force_correction_czar[i][bink[1],bink[0]] += self.k[i] * dx

        self._up_momenta_extended()

        self.traj  = np.append(self.traj, [xi], axis = 0)
        self.etraj  = np.append(self.etraj, [self.ext_coords], axis = 0)
        self.temp = np.append(self.temp, self.the_md.temp)

        if self.the_md.step%self.out_freq == 0:
            # write output

            if write_traj == True:
                self.write_traj(xi,extended = True)

            self._F_from_CZAR()
            self.write_output()
            self.write_restart()

        self.timing = time.perf_counter() - start

    # -----------------------------------------------------------------------------------------------------
    def _get_bin(self, xi, extended = False):
        '''get current bin of CV or extended variable

        args:
           xi (array): Current value of collective variable
           extended (bool, False): True for extended system based methods

        returns:
           bink (array): [ncol, nrow]
        '''
        X = xi if extended == False else self.ext_coords

        binX = [-1,-1]
        for i in range(self.ncoords):
            binX[i] = int(np.floor(np.abs(X[i]-self.minx[i])/self.dx[i]))

        return binX

    # -----------------------------------------------------------------------------------------------------
    def _diff(self, a, b, is_angle):
        '''returns difference of two angles in range (-pi,pi) or normal difference

        args:
            a (double)
            b (double)

        returns:
            diff (double)
        '''
        if is_angle:
            diff = a - b
            if diff < -np.pi:  diff += 2*np.pi
            elif diff > np.pi: diff -= 2*np.pi
        else:
            diff = a - b
        return diff

    # -----------------------------------------------------------------------------------------------------
    def _get_MtD_bias(self, xi, bink, height, update_int, step, WT_dT, WT, grid, out_of_bounds):
        '''get metadynamics bias potential and force

        args:
            xi (float): collective variable
            bink (int): current bin of xi
            height (float): Gaussian height
            update_int (int): intervall for deposition of Gaussians
            step (int): md step
            WT (bool): if Ture use WTM
            grid (bool): use grid for force
            out_of_bond (bool): get force from hills near margin if system outside of grid

        returns:
            bias (array): bias force on CV
        '''
	# update bias every update_int's step and save on grid for free energy calculation
        # if grid == True also coarse graind force is used for bias of dynamics

        if self.ncoords == 1 and out_of_bounds == False:
            # 1D

            if step%int(update_int) == 0:

                # save centers to calculate bias in case system leafs bins
                if step == 0:
                    self.center = np.array([xi[0]])
                else:
                    self.center = np.append(self.center, xi[0])

                w = height/H_to_kJmol
                if WT == True:
                    w *= np.exp(-self.metapot[bink[1],bink[0]]/(kB_a*WT_dT))

                for i in range(self.nbins_per_dim[0]):
                    dx = self._diff(self.grid[0][i], xi[0], self.is_angle[0])
                    if abs(dx) <= 3*self.variance[0]:
                        bias_factor = w * np.exp(-(dx*dx)/(2.0*self.variance[0]))

                        self.metapot[0,i] += bias_factor
                        self.bias[0][0,i] -= bias_factor * dx/self.variance[0]

            bias = [self.bias[0][bink[1],bink[0]]]

        elif self.ncoords == 2 and out_of_bounds == False:
            # 2D
            if step%int(update_int) == 0:

                # save centers to calculate bias in case system leafs bins
                if step == 0:
                    self.center_x = np.array([xi[0]])
                    self.center_y = np.array([xi[1]])
                else:
                    self.center_x = np.append(self.center_x, xi[0])
                    self.center_y = np.append(self.center_y, xi[1])

                w = height/H_to_kJmol
                if WT == True:
                    w *= np.exp(-self.metapot[bink[1],bink[0]]/(kB_a*WT_dT))

                for i in range(self.nbins_per_dim[1]):
                    dy = self._diff(self.grid[1][i], xi[1], self.is_angle[1])
                    if abs(dy) <= 3*self.variance[1]:
                        for j in range(self.nbins_per_dim[0]):
                            dx = self._diff(self.grid[0][j], xi[0], self.is_angle[0])
                            if abs(dx) <= 3*self.variance[0]:
                                p1 = (dx*dx)/self.variance[0]
                                p2 = (dy*dy)/self.variance[1]
                                gauss = np.exp(-(p1+p2)/2.0)

                                self.metapot[i,j] += w * gauss
                                self.bias[0][i,j] -= w * gauss * dx/self.variance[0]
                                self.bias[1][i,j] -= w * gauss * dy/self.variance[1]

            bias = [self.bias[0][bink[1],bink[0]],self.bias[1][bink[1],bink[0]]]

        if grid == False:
            # get exact bias force of step
            # can get slow in long simulations

            bias_factor = 0.0
            if self.ncoords == 1:
                # 1D

                if self.the_md.step == 0 and out_of_bounds == True:
                    self.center = np.array([])

                bias = [0.0]
                w0 = height/H_to_kJmol

                for ii, val in enumerate(self.center):
                    dx = self._diff(val,xi[0], self.is_angle[0])
                    if abs(dx) <= 3*self.variance[0]:

                        w = w0
                        if WT == True:
                            w *= np.exp(-bias_factor/(kB_a*WT_dT))

                        bias_factor += w * np.exp(-(dx*dx)/(2.0*self.variance[0]))
                        bias[0]     += bias_factor * dx/self.variance[0]

            else:
                # 2D

                if self.the_md.step == 0 and out_of_bounds == True:
                    self.center_x = np.array([])
                    self.center_y = np.array([])

                bias = [0.0,0.0]
                w0 = height/H_to_kJmol

                for i, x in enumerate(self.center_x):
                    dx = self._diff(x, xi[0], self.is_angle[0])
                    if abs(dx) <= 3*self.variance[0]:
                        dy = self._diff(self.center_y[i], xi[1], self.is_angle[1])
                        if abs(dy) <= 3*self.variance[1]:

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
    def _get_gradient_correction(self, delta_xi):
        '''get mass of collective variable for TS theory and kinetics
           results not yet varified!!!

        args:
            delta_xi (array): gradients of CV's
        returns:
            m_xi (double): coordinate dependent mass of collective variable)
        '''
        if self.ncoords == 1:
            return 1.0/np.dot(delta_xi[0], (1.0/self.the_md.masses)*delta_xi[0])

    # -----------------------------------------------------------------------------------------------------
    def _propagate_extended(self, langevin=True):
        '''Propagate momenta/coords of extended variable with Velocity Verlet

        args:
           langevin (bool, True)
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
    def _up_momenta_extended(self, langevin=True):
        '''Update momenta of extended variables with Velocity Verlet

        args:
            langevin (bool, True)
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
    def _get_cond_avg(self, a, hist):
        '''get hist conditioned average of a

        args:
            a (array)
            hist (array): histogramm along CV (prob density)
        returns:
            cond_avg (array)
        '''
        cond_avg = np.divide(a, hist, out=np.zeros_like(a), where=(hist!=0))
        return cond_avg

    # -----------------------------------------------------------------------------------------------------
    def _F_from_Force(self, mean_force):
        '''numeric on-the-fly integration of thermodynamic force to obtain free energy estimate of 1D CVs

        args:
            mean_force (array)
        returns:
            -
        '''
        self.dF = np.array(np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])), dtype=np.float64)

        if self.ncoords == 1:
            # on-the-fly integration only for 1D reaction coordinate

            for i in range(1,self.nbins_per_dim[0]):
                self.dF[0,i] = np.sum(mean_force[0][0,0:i]) * self.dx[0]

        self._get_geom_correction()

    # -----------------------------------------------------------------------------------------------------
    def _F_from_MtD(self, WT_dT=0.0, WT=True):
        '''on-the-fly free energy estimate from MtD or WTM bias potential

        args:
            WT_dT (double, 0.0): effective temperature for WTM
            WT (bool, True): if True use Well-Tempered MtD
        returns:
            -
        '''
        self.dF = - self.metapot

        if WT==True:
            self.dF *= (self.the_md.target_temp + WT_dT)/WT_dT

        self._get_geom_correction()

    # -----------------------------------------------------------------------------------------------------
    def _F_from_CZAR(self):
        '''on-the-fly CZAR estimate of unbiased thermodynamic force
           get unbiased free energy by integrating czar estimate

        args:
            -
        returns:
            -
        '''
        self.mean_force = np.array([np.zeros((self.nbins_per_dim[1],self.nbins_per_dim[0])) for i in range(self.ncoords)], dtype=np.float64)

        # get ln(rho) and z-average force per bin
        log_rho   = np.log(self.hist_z, out=np.zeros_like(self.hist_z), where=(self.hist_z!=0))
        avg_force = self._get_cond_avg(self.force_correction_czar, self.hist_z)

        # czar estimate of thermodynamic force per bin
        if self.ncoords == 1:
            self.mean_force[0] = - kB_a * self.the_md.target_temp * np.gradient(log_rho[0], self.grid[0]) + avg_force[0]

        else:
            der_log_rho = np.gradient(log_rho, self.grid[1], self.grid[0])
            self.mean_force[0] = - kB_a * self.the_md.target_temp * der_log_rho[1] + avg_force[0]
            self.mean_force[1] = - kB_a * self.the_md.target_temp * der_log_rho[0] + avg_force[1]

        self._F_from_Force(self.mean_force)
        self._get_geom_correction(extended = True)

    # -----------------------------------------------------------------------------------------------------
    def _get_geom_correction(self, extended = False):
        '''get geometric free energy

        args:
            extended (bool, False): True for extended system methods
        returns:
            -
        '''
        hist = self.histogramm if extended == False else self.hist_z
        self.geom_correction = self._get_cond_avg(self.geom_corr, hist)

    # -----------------------------------------------------------------------------------------------------
    def MW(self, MW_file = '../MW.dat', sync_interval = 20, trial = 0):
        '''Multiple walker strategy
           metadynamic/meta-eABF has to use grid!

        args:
            MW_file (string, '../MW.dat'): path to buffer file for shared bias
            sync_interval (int, 20): intervall between syncs with other walkers in steps
            trial (int, 0): internally used to handle recursive function calls
                            if access is denied more than 10 times the walker will share its progres at the next sync intervall

        returns:
            -
        '''
        if self.the_md.step == 0 and trial == 0:
            print('-------------------------------------------------------------')
            print('\tNew Multiple-Walker Instance created!')
            print('-------------------------------------------------------------')
            self.MW_histogramm = np.copy(self.histogramm)
            self.MW_geom_corr  = np.copy(self.geom_corr)
            self.MW_bias = np.copy(self.bias)
            self.MW_CV_crit = np.copy(self.CV_crit)
            if self.method == 'MtD':
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

                    self.histogramm = self._MW_update(self.histogramm, self.MW_histogramm, MW[:,0].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]))
                    self.MW_histogramm = np.copy(self.histogramm)

                    self.geom_corr = self._MW_update(self.geom_corr, self.MW_geom_corr, MW[:,1].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]))
                    self.MW_geom_corr = np.copy(self.geom_corr)

                    if self.ncoords == 1:

                        self.bias = self._MW_update(self.bias, self.MW_bias, [MW[:,2]])
                        self.MW_bias = np.copy(self.bias)

                        self.CV_crit = self._MW_update(self.CV_crit, self.MW_CV_crit, [MW[:,3]])
                        self.MW_CV_crit = np.copy(self.CV_crit)

                        if self.method == 'MtD':
                            self.metapot = self._MW_update(self.metapot, self.MW_metapot, [MW[:,4]])
                            self.MW_metapot = np.copy(self.metapot)

                        elif self.method == 'eABF' or self.method == 'meta-eABF':
                            self.hist_z = self._MW_update(self.hist_z, self.MW_hist_z, [MW[:,4]])
                            self.MW_hist_z = np.copy(self.hist_z)
                            self.force_correction_czar = self._MW_update(self.force_correction_czar, self.MW_force_correction_czar, [MW[:,5]])
                            self.MW_force_correction_czar = np.copy(self.force_correction_czar)

                            if self.method == 'meta-eABF':
                                self.abfforce = self._MW_update(self.abfforce, self.MW_abfforce, [MW[:,6]])
                                self.MW_abfforce = np.copy(self.abfforce)

                    else:

                        MW_new = np.array([MW[:,2].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]), MW[:,3].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])])
                        self.bias = self._MW_update(self.bias, self.MW_bias, MW_new)
                        self.MW_bias = np.copy(self.bias)

                        MW_new = np.array([MW[:,4].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]), MW[:,5].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])])
                        self.CV_crit = self._MW_update(self.CV_crit, self.MW_CV_crit, MW_new)
                        self.MW_CV_crit = np.copy(self.CV_crit)

                        if self.method == 'MtD':
                            self.metapot = self._MW_update(self.metapot, self.MW_metapot, MW[:,6].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]))
                            self.MW_metapot = np.copy(self.metapot)

                        elif self.method == 'eABF' or self.method == 'meta-eABF':
                            self.hist_z = self._MW_update(self.hist_z, self.MW_hist_z, MW[:,6].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]))
                            self.MW_hist_z = np.copy(self.hist_z)

                            MW_new =  np.array([MW[:,7].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]), MW[:,8].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])])
                            self.force_correction_czar = self._MW_update(self.force_correction_czar, self.MW_force_correction_czar, MW_new)
                            self.MW_force_correction_czar = np.copy(self.force_correction_czar)

                            if self.method == 'meta-eABF':
                                MW_new =  np.array([MW[:,9].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0]), MW[:,10].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])])
                                self.abfforce = self._MW_update(self.abfforce, self.MW_abfforce, MW_new)
                                self.MW_abfforce = np.copy(self.abfforce)

                    if os.access(MW_file, os.W_OK):
                        self.write_restart(MW_file)
                        print('Bias synced with {f}!'.format(f=MW_file))

                    else:
                        print('Failed to sync bias with {f}!'.format(f=MW_file))


                    # set MW_file back to read only
                    os.chmod(MW_file, 0o444)

                elif trial < 10:

                    # try again
                    time.sleep(0.1)
                    self.MW(MW_file, sync_interval=sync_interval, trial=trial+1)

                else:
                    print('Failed to sync bias with {f}!'.format(f=MW_file))

            else:

                # create MW buffer
                self.write_restart(MW_file)
                os.chmod(MW_file, 0o444)

    # -----------------------------------------------------------------------------------------------------
    def _MW_update(self, new, old, walkers):
        '''update accumulators from MW buffer
        '''
        return walkers + (new - old)

    # -----------------------------------------------------------------------------------------------------
    def restart(self, filename='restart_bias.dat'):
        '''restart calculation from checkpoint file

        args:
            filename (string, 'restart_bias.dat'): filename for checkpoint file)

        returns:
            -
        '''
        if os.path.isfile(filename):

            data = np.loadtxt(filename)

            self.histogramm = data[:,0].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
            self.geom_corr  = data[:,1].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])

            if self.ncoords == 1:
                self.bias[0] = data[:,2].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                self.CV_crit[0] = data[:,3].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                if self.method == 'MtD':
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
                if self.method == 'MtD':
                    self.metapot = data[:,6].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                elif self.method == 'eABF' or self.method == 'meta-eABF':
                    self.hist_z = data[:,6].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                    self.force_correction_czar[0] = data[:,7].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                    self.force_correction_czar[1] = data[:,8].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                    if self.method == 'meta-eABF':
                        self.abfforce[0] = data[:,9].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
                        self.abfforce[1] = data[:,10].reshape(self.nbins_per_dim[1],self.nbins_per_dim[0])
            print('-------------------------------------------------------------')
            print('\tBias restarted from {f}!'.format(f=filename))
            print('-------------------------------------------------------------')

        else:
            print('-------------------------------------------------------------')
            print('\tRestart file not found.')
            print('-------------------------------------------------------------')

    # -----------------------------------------------------------------------------------------------------
    def write_restart(self, filename='restart_bias.dat'):
        '''write relevant data for restart to .txt file

        args:
            filename (string, 'restart_bias.dat'): filename for checkpoint file

        returns:
            -
        '''
        out = open(filename, "w")
        if self.ncoords == 1:
            # for 1D CV
            for i in range(self.nbins_per_dim[0]):
                row = (self.histogramm[0,i], self.geom_corr[0,i], self.bias[0][0,i], self.CV_crit[0][0,i])
                out.write("%14.10f\t%14.10f\t%14.10f\t%14.10f" % row)
                if self.method == 'MtD':
                    out.write('\t%14.10f' % (self.metapot[0,i]))
                elif self.method == 'eABF' or self.method == 'meta-eABF':
                    out.write('\t%14.10f\t%14.10f' % (self.hist_z[0,i], self.force_correction_czar[0][0,i]))
                    if self.method == 'meta-eABF':
                        out.write('\t%14.10f' % (self.abfforce[0][0,i]))
                out.write('\n')

        else:
            # for 2D CV
            for i in range(self.nbins_per_dim[1]):
                for j in range(self.nbins_per_dim[0]):
                    row = (self.histogramm[i,j], self.geom_corr[i,j], self.bias[0][i,j], self.bias[1][i,j], self.CV_crit[0][i,j], self.CV_crit[1][i,j])
                    out.write("%14.10f\t%14.10f\t%14.10f\t%14.10f\t%14.10f\t%14.10f" % row)
                    if self.method == 'MtD':
                        out.write('\t%14.10f' % (self.metapot[i,j]))
                    elif self.method == 'eABF' or self.method == 'meta-eABF':
                        out.write('\t%14.10f\t%14.10f\t%14.10f' % (self.hist_z[i,j], self.force_correction_czar[0][i,j], self.force_correction_czar[1][i,j]))
                        if self.method == 'meta-eABF':
                            out.write('\t%14.10f\t%14.10f' % (self.abfforce[0][i,j], self.abfforce[0][i,j]))
                    out.write('\n')

        out.close()

    # -----------------------------------------------------------------------------------------------------
    def write_traj(self, xi, extended = False):
        '''write trajectory of extended or normal ABF at output times

        args:
            xi (double, -)
            extended (bool, False)

        returns:
            -
        '''
        # convert units to degree and Angstrom
        for i in range(self.ncoords):
            if self.is_angle[i]:
                self.traj[:,i] = np.degrees(self.traj[:,i])
                if extended:
                    self.etraj[:,i] = np.degrees(self.etraj[:,i])
            else:
                self.traj[:,i] = self.traj[:,i] * bohr2angs
                if extended:
                    self.etraj[:,i] = self.etraj[:,i] * bohr2angs

        if self.the_md.step == 0:
            # start new file in first step

            traj_out = open("CV_traj.dat", "w")
            traj_out.write("%14s\t" % ("time [fs]"))
            for i in range(len(self.traj[0])):
                traj_out.write("%14s\t" % (f"Xi{i}"))
                if extended:
                    traj_out.write("%14s\t" % (f"eXi{i}"))
            traj_out.write("%14s" % ("temp [K]"))
            traj_out.close()

        else:
            # append new steps of trajectory since last output

            traj_out = open("CV_traj.dat", "a")
            for n in range(self.out_freq):
                traj_out.write("\n%14.6f\t" % ((self.the_md.step-self.out_freq+n)*self.the_md.dt*it2fs))
                for i in range(len(self.traj[0])):
                    traj_out.write("%14.6f\t" % (self.traj[-self.out_freq+n][i]))
                    if extended:
                        traj_out.write("%14.6f\t" % (self.etraj[-self.out_freq+n][i]))
                traj_out.write("%14.6f" % (self.temp[-self.out_freq+n]))
            traj_out.close()

        # start new traj arrays for next output
        self.traj = np.array([xi])
        self.temp = np.array([self.the_md.temp])
        if self.method == 'eABF' or self.method == 'meta-eABF':
            self.etraj = np.array([self.ext_coords])

    # -----------------------------------------------------------------------------------------------------
    def write_output(self, filename='abm.out'):
        '''write output of free energy calculations

        args:
            filename (string, 'abm.out'): filename for output

        returns:
            -
        '''
        if self.method == 'MtD' or self.method == 'ref':
            self.mean_force = self.bias

        crit = self.CV_crit

        # convert units of CV to degree or Angstrom
        grid = np.copy(self.grid)
        for i in range(self.ncoords):
            if self.is_angle[i]:
                grid[i] = np.degrees(self.grid[i])
            else:
                grid[i] = self.grid[i] * bohr2angs

        out = open(filename, "w")
        if self.ncoords == 1:
            head = ("Xi1", "CV_crit", "<m_xi>", "Histogramm", "Bias", "dF")
            out.write("%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
            for i in range(self.nbins_per_dim[0]):
                row = (grid[0][i], crit[0][0,i], self.geom_correction[0,i], self.histogramm[0,i], self.mean_force[0][0,i], self.dF[0,i])
                out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f" % row)
                out.write("\n")

        elif self.ncoords == 2:
            if self.method == 'MtD' or 'ref':
                head = ("Xi1", "Xi0", "CV_crit1", "CV_crit0", "Histogramm", "Bias1", "Bias0", "dF")
                out.write("%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
                for i in range(self.nbins_per_dim[1]):
                    for j in range(self.nbins_per_dim[0]):
                        row = (grid[1][i], grid[0][j], crit[1][i,j], crit[0][i,j], self.histogramm[i,j], self.mean_force[1][i,j], self.mean_force[0][i,j], self.dF[i,j])
                        out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)

            else:
                head = ("Xi1", "Xi1", "CV_crit1", "CV_crit0", "Histogramm", "Bias1", "Bias2")
                out.write("%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
                for i in range(self.nbins_per_dim[1]):
                    for j in range(self.nbins_per_dim[0]):
                        row = (grid[1][i], grid[0][j], crit[1][i,j], crit[0][i,j], self.histogramm[i,j], self.mean_force[0][i,j], self.mean_force[1][i,j])
                        out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % row)
                        out.write("\n")

        out.close()

    # -----------------------------------------------------------------------------------------------------
    def _print_parameters(self):
        '''print out parameters after init
        '''
        for i in range(self.ncoords):
            print("\nInitialize {CV} for {m}:".format(CV=self.CV[i],m=self.method))
            if self.is_angle[i]:
                print("\n\tMinimum CV%d:\t\t%14.6f degree" % (i, np.degrees(self.minx[i])))
                print("\tMaximum CV%d:\t\t%14.6f degree" % (i,   np.degrees(self.maxx[i])))
                print("\tBinwidth CV%d:\t\t%14.6f degree" % (i,  np.degrees(self.dx[i])))
            else:
                print("\n\tMinimum CV%d:\t\t%14.6f Angstorm" % (i,self.minx[i]*bohr2angs))
                print("\tMaximum CV%d:\t\t%14.6f Angstrom" % (i,self.maxx[i]*bohr2angs))
                print("\tBinwidth CV%d:\t\t%14.6f Angstrom" % (i,self.dx[i]*bohr2angs))

        print("\t---------------------------------------------")
        print("\tTotel number of bins:\t%14.6f" % (self.nbins))

        if self.method == 'eABF' or self.method == 'meta-eABF':
            print("\nInitialize extended Lagrangian:")
            for i in range(self.ncoords):
                if self.is_angle[i]:
                    print("\n\tspring constant:\t%14.6f Hartree/rad^2" % (self.k[i]))
                else:
                    print("\n\tspring constant:\t%14.6f Hartree/a_0^2" % (self.k[i]))
                print("\tfictitious mass:\t%14.6f a.u." % (self.ext_mass[i]))


#######################################################################################################
#                         Collecitve Variables for free energy methods
#######################################################################################################
    def _get_coord(self):
        '''Collective variables

        args:
            -

        returns:
            (xi (array), delta_xi (array), div (array))): Collective variable and its gradient, divergence
        '''
        xi       = np.zeros(self.ncoords)
        delta_xi = np.zeros((self.ncoords,self.the_md.natoms*3))
        div      = np.zeros(self.ncoords)

        for i in range(self.ncoords):

            if self.CV[i] == 'distance':

                    # bond distance
                    (x, dx) = self.distance(self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx
                    div[i]      += 2.0/xi[i]

            elif self.CV[i] == 'projected_distance':

                    # bond distance
                    (x, dx) = self.projected_distance(self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx
                    div[i]      += 0.0

            elif self.CV[i] == 'angle':

                    # bend angle
                    (x, dx) = self.angle(self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx
                    div[i]      += 1.0/np.tan(xi[i])

            elif self.CV[i] == 'torsion':

                    # torsion angle
                    (x, dx) = self.torsion(self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx
                    div[i]      += 0.0

            elif self.CV[i] == 'lin_comb_dists':

                    # linear combination of distances and projeced distances
                    if self.the_md.step == 0 and self.method == 'ABF':
                        print('ERROR: Do not use linear combination of CVs with ABF, divergence not implemented!')
                        sys.exit(0)

                    (x, dx) = self.lin_comb_dists(i, self.atoms[i])
                    xi[i]       += x
                    delta_xi[i] += dx

            elif self.CV[i] == 'lin_comb_angles':

                    # linear combination of angles and dihedrals
                    if self.the_md.step == 0 and self.method == 'ABF':
                        print('ERROR: Do not use linear combination of CVs with ABF, divergence not implemented!')
                        sys.exit(0)

                    (x, dx) = self.lin_comb_dists(i, self.atoms[i])
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
    def _mass_center(self, atoms):
        '''get center of mass and overall mass of group of atoms
        '''
        m = 0
        c = np.zeros(3)
        if hasattr(atoms, "__len__"):
            for index, a in enumerate(atoms):
                m += self.the_md.mass[a]
                c += self.the_md.mass[a] * np.array([self.the_md.coords[3*a],self.the_md.coords[3*a+1],self.the_md.coords[3*a+2]],dtype=np.float)
            p = c/m

        else:
            atoms = int(atoms)
            m = self.the_md.mass[atoms]
            p = np.array([self.the_md.coords[3*atoms],self.the_md.coords[3*atoms+1],self.the_md.coords[3*atoms+2]],dtype=np.float)

        return (p, m)

    # -----------------------------------------------------------------------------------------------------
    def _mass_weights(self, mass_group, atoms):
        '''get mass weights of atoms for gradient of group of atoms
        '''
        coords = np.zeros((3,3*self.the_md.natoms),dtype=np.float)
        if hasattr(atoms, "__len__"):
            for index, a in enumerate(atoms):
                coords[0,3*a]   = self.the_md.mass[a]/mass_group
                coords[1,3*a+1] = self.the_md.mass[a]/mass_group
                coords[2,3*a+2] = self.the_md.mass[a]/mass_group
        else:
            atoms = int(atoms)
            coords[0,3*atoms]   = 1.0
            coords[1,3*atoms+1] = 1.0
            coords[2,3*atoms+2] = 1.0

        return coords

    # -----------------------------------------------------------------------------------------------------
    def distance(self, atoms):
        '''distance between two mass centers in Angstrom in range(0, inf)
        '''
        if len(atoms) != 2:
            print("ERROR: Invalid number of centers in definition of distance!")
            sys.exit(0)

        (p1,m1) = self._mass_center(atoms[0])
        (p2,m2) = self._mass_center(atoms[1])

        # get distance
        r12 = (p2-p1)
        xi  = np.linalg.norm(r12)

        # get gradient
        w1 = self._mass_weights(m1, atoms[0])
        w2 = self._mass_weights(m2, atoms[1])

        dxi = r12/xi
        grad_xi =- np.dot(dxi, w1)
        grad_xi += np.dot(dxi, w2)

        return (xi, grad_xi)

    #-----------------------------------------------------------------------------------------------------
    def projected_distance(self, atoms):
        '''distance between group 1 and 2 projected on vector between 2 3 in Angstrom in range(-inf,inf)
        '''
        if len(atoms) != 3:
            print("ERROR: Invalid number of centers in definition of projected distance!")
            sys.exit(0)

        (p1,m1) = self._mass_center(atoms[0])
        (p2,m2) = self._mass_center(atoms[1])
        (p3,m3) = self._mass_center(atoms[2])

        # get projected distance
        r12 = (p2-p1)
        r23 = (p3-p2)

        e   = r23/np.linalg.norm(r23)
        xi  = np.dot(r12,e)

        # get gradient
        w1 = self._mass_weights(m1, atoms[0])
        w2 = self._mass_weights(m2, atoms[1])

        grad_xi =- np.dot(e, w1)
        grad_xi += np.dot(e, w2)

        return (xi, grad_xi)


    # -----------------------------------------------------------------------------------------------------
    def angle(self, atoms):
        '''get angle between three mass centers in range(-pi,pi)
        '''
        if len(atoms) != 3:
            print("ERROR: Invalid number of centers in definition of angle!")
            sys.exit(0)

        (p1,m1) = self._mass_center(atoms[0])
        (p2,m2) = self._mass_center(atoms[1])
        (p3,m3) = self._mass_center(atoms[2])

        # get angle
        q12 = p1-p2
        q23 = p2-p3

        q12_n = np.linalg.norm(q12)
        q23_n = np.linalg.norm(q23)

        q12_u = q12/q12_n
        q23_u = q23/q23_n

        xi = np.arccos(np.dot(-q12_u,q23_u))

        # get gradient
        w1 = self._mass_weights(m1, atoms[0])
        w2 = self._mass_weights(m2, atoms[1])
        w3 = self._mass_weights(m3, atoms[2])

        dxi1 = np.cross(q12_u,np.cross(q12_u,-q23_u))
        dxi3 = np.cross(q23_u,np.cross(q12_u,-q23_u))
        dxi1 /= np.linalg.norm(dxi1)
        dxi3 /= np.linalg.norm(dxi3)
        dxi1 /= q12_n
        dxi3 /= q23_n
        dxi2 =- dxi1-dxi3

        grad_xi =+ np.dot(dxi1, w1)
        grad_xi += np.dot(dxi2, w2)
        grad_xi += np.dot(dxi3, w3)

        return (xi, grad_xi)

    # -----------------------------------------------------------------------------------------------------
    def torsion(self, atoms):
        '''torsion angle between four mass centers in range(-pi,pi)
        '''
        if len(atoms) != 4:
            print("ERROR: Invalid number of centers in definition of torsion!")
            sys.exit(0)

        (p1,m1) = self._mass_center(atoms[0])
        (p2,m2) = self._mass_center(atoms[1])
        (p3,m3) = self._mass_center(atoms[2])
        (p4,m4) = self._mass_center(atoms[3])

        # get torsion
        q12 = p2 - p1
        q23 = p3 - p2
        q34 = p4 - p3

        q12_n = np.linalg.norm(q12)
        q23_n = np.linalg.norm(q23)
        q34_n = np.linalg.norm(q34)

        q12_u = q12 / q12_n
        q23_u = q23 / q23_n
        q34_u = q34 / q34_n

        n1 = -q12 - np.dot(-q12,q23_u)*q23_u
        n2 =  q34 - np.dot(q34,q23_u)*q23_u

        xi = np.arctan2(np.dot(np.cross(q23_u,n1),n2),np.dot(n1,n2))

        # get gradient
        w1 = self._mass_weights(m1, atoms[0])
        w2 = self._mass_weights(m2, atoms[1])
        w3 = self._mass_weights(m3, atoms[2])
        w4 = self._mass_weights(m4, atoms[3])

        cos_123 = np.dot(-q12_u,q23_u)
        cos_234 = np.dot(-q23_u,q34_u)

        sin2_123 = 1 - cos_123*cos_123
        sin2_234 = 1 - cos_234*cos_234

        dtau1 = - 1/(q12_n*sin2_123)*np.cross(-q12_u,-q23_u)
        dtau4 = - 1/(q34_n*sin2_234)*np.cross(-q34_u,-q23_u)

        # sum(grad)=0 and rotation=0
        c_123 = ((q12_n*cos_123)/q23_n) - 1
        b_432 = ((q34_n*cos_234)/q23_n)

        dtau2 = c_123*dtau1 - b_432*dtau4
        dtau3 = -(dtau1 + dtau2 + dtau4)

        grad_xi =+ np.dot(dtau1, w1)
        grad_xi += np.dot(dtau2, w2)
        grad_xi += np.dot(dtau3, w3)
        grad_xi += np.dot(dtau4, w4)

        return (xi, grad_xi)

    # -----------------------------------------------------------------------------------------------------
    def lin_comb_dists(self, i, atoms):
        '''linear combination distances and projected distances
        '''
        CVs = np.array([])
        grad_CVs = np.zeros(3*self.the_md.natoms,dtype=np.float)

        for index, CV in enumerate(atoms):

            if len(CV) == 3:

                # distance
                a = CV[0] + 1.0
                (x,dx) = distance(self, np.array(CV[1:]))

                CVs = np.append(CVs, a*x)
                grad_CVs += a*dx

            elif len(CV) == 4:

                # projected distance
                a = CV[0] + 1.0
                (x,dx) = projected_distance(self, np.array(CV[1:]))

                CVs = np.append(CVs, a*x)
                grad_CVs += a*dx

            else:
                print("ERROR: Invalid number of centers in definition of distance!")
                sys.exit(0)

        CV = CVs.sum()
        write_lc_traj(i, CVs)

        return (CV, grad_CVs)

    # -----------------------------------------------------------------------------------------------------
    def lin_comb_angles(self, i, atoms):
        '''linear combination of angles or torsion
        '''
        CVs = np.array([])
        grad_CVs = np.zeros(3*self.the_md.natoms,dtype=np.float)
        for index, CV in enumerate(atoms):

            if len(CV) == 4:

                # angle
                a = CV[0] + 1.0
                (x,dx) = angle(self, np.array(CV[1:]))

                CVs = np.append(CVs, a*x)
                grad_CVs += a*dx

            elif len(CV) == 5:

                # torsion
                a = CV[0] + 1.0
                (x,dx) = torsion(self, np.array(CV[1:]))

                CVs = np.append(CVs, a*x)
                grad_CVs += a*dx

            else:
                print("ERROR: Invalid number of centers in definition of CV!")
                sys.exit(0)

        CV = CVs.sum()

        write_lc_traj(i, CVs)

        return (CV, grad_CVs)

    # -----------------------------------------------------------------------------------------------------
    def write_lc_traj(i, CVs):
        '''write seperate traj for single contributions of linear combination
        '''
        traj_out = open("lc_traj_%d.dat" % (i+1), "a")
        for j in range(len(CVs)):
            traj_out.write("%14.6f\t" % (CVs[j]))
        traj_out.write("\n")
        traj_out.close()
