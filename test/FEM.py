import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import interpolate 
import time

H_in_kJmol = 2625.499639
R          = 8.314 # J/(K*mol)
T          = 300   # K
RT         = R*T/1000.0 # kJ / mol
RT_au      = RT/2625.5
kB         = 1.380648e-23      # J / K
H_to_J     = 4.359744e-18      #
kB_a       = kB / H_to_J       # Hartree / K

class FEM:
    '''integration of 2D ABF gradients with finite element method
    '''
    def __init__(self, inputname='bias_out.dat', outputname='free_energy'):
        
        print("\n#######################################################")
        print("\tFEM integration of 2D ABF data")
        print("#######################################################\n")
        print("Initialize spline functions.\n")
        
        self.outname = outputname        
        data = np.loadtxt(inputname, skiprows=1)
        
        # coordinates of bins
        xi_1 = data[:,0]
        xi_2 = data[:,1]
        
        minxi_1 = xi_1.min()
        maxxi_1 = xi_1.max()
        minxi_2 = xi_2.min()
        maxxi_2 = xi_2.max()
        
        dxi_2 = xi_2[1]-xi_2[0]
        self.x_bins = int((maxxi_2-minxi_2)/dxi_2 + 1)
        self.y_bins = int(xi_1.shape[0]/self.x_bins)
        xi_1 = xi_1.reshape(self.y_bins, self.x_bins)
        xi_2 = xi_2.reshape(self.y_bins, self.x_bins)
        dxi_1 = xi_1[1,0]-xi_1[0,0]

        # control points
        self.dy = dxi_1 / 4
        self.dx = dxi_2 / 4
        self.minx = minxi_2
        self.maxx = maxxi_2
        self.miny = minxi_1
        self.maxy = maxxi_1

        self.x = np.arange(self.minx, self.maxx+self.dx, self.dx)
        self.y = np.arange(self.miny, self.maxy+self.dy, self.dy)
        self.lenx = len(self.x)
        self.leny = len(self.y)      
        self.xx, self.yy = np.meshgrid(self.x,self.y)
        
        # coefficient matrix
        self.alpha = np.full((int(self.x_bins*self.y_bins),), 0.0, dtype=np.float)
        print("%30s:\t%8d" % ("Number of coefficients", self.alpha.size))
        
        # interploate gradient to control points
        origD = [data[:,3].reshape(self.y_bins, self.x_bins), data[:,4].reshape(self.y_bins, self.x_bins)]
        D = [np.zeros(shape=(self.leny,self.lenx), dtype=np.float), np.zeros(shape=(self.leny,self.lenx), dtype=np.float)]
        for xy in range(2):
            D[xy][::4, ::4] = origD[xy]
            D[xy][::4,2::4] = 0.5*D[xy][::4,:-4:4] + 0.5*D[xy][::4,4::4] 
            D[xy][::4,1::4] = 0.5*D[xy][::4,2::4]  + 0.5*D[xy][::4,:-4:4]
            D[xy][::4,3::4] = 0.5*D[xy][::4,2::4]  + 0.5*D[xy][::4,4::4] 
            D[xy][2::4]     = 0.5*D[xy][:-4:4]     + 0.5*D[xy][4::4]
            D[xy][1::4]     = 0.5*D[xy][2::4]      + 0.5*D[xy][:-4:4]
            D[xy][3::4]     = 0.5*D[xy][2::4]      + 0.5*D[xy][4::4] 
        self.D = np.asarray(D) * H_in_kJmol 
        self.D = self.D[:,:-1,:-1]
        print("%30s:\t%8d" % ("Elements in gradient matrix", self.D.size))
        
        # initialize pyramid functions 
        self.B = []
        self.gradB = []
        for center_y in xi_1[:,0]:
            for center_x in xi_2[0]:
                self.B.append(self.pyramid(self.x, self.y, dxi_2, dxi_1, center_x, center_y)) 
                self.gradB.append(self.gradPyr(self.x, self.y, dxi_2, dxi_1, center_x, center_y))
         
        self.B = np.asarray(self.B)
        self.gradB = np.asarray(self.gradB)
        self.gradB = self.gradB[:,:,:-1,:-1]
        print("%30s:\t%8d" % ("Elements in gradB matrix", self.gradB.size))

    #----------------------------------------------------------------------------------------
    def pyramid(self, x, y, dx, dy, cx, cy):
        '''pyramid function
        '''
        return_func = np.zeros(shape=(self.leny,self.lenx), dtype=np.float)
        for ii, val_y in enumerate(y):
            for jj, val_x in enumerate(x):
                if val_y >= (cy - dy) and val_y < cy:
                    if val_x >= (cx - dx) and val_x < cx:
                        return_func[ii, jj] = ((val_y - cy)/dy + 1)*((val_x - cx)/dx + 1)
                    if val_x < (cx + dx) and val_x >= cx:
                        return_func[ii, jj] = ((val_y - cy)/dy + 1)*((cx - val_x)/dx + 1)
                if val_y < (cy + dy) and val_y >= cy:
                    if val_x >= (cx - dx) and val_x < cx:
                        return_func[ii, jj] = ((cy - val_y)/dy + 1)*((val_x - cx)/dx + 1)
                    if val_x < (cx + dx) and val_x >= cx:
                        return_func[ii, jj] = ((cy - val_y)/dy + 1)*((cx - val_x)/dx + 1)
        #
        return return_func

    #----------------------------------------------------------------------------------------
    def gradPyr(self, x, y, dx, dy, cx, cy):
        '''gradients of pyramid function
        '''
        deriv_x = np.zeros(shape=(len(y),len(x)), dtype=np.float)
        deriv_y = np.zeros(shape=(len(y),len(x)), dtype=np.float)
        for ii, val_y in enumerate(y):
            for jj, val_x in enumerate(x):
                if val_y >= (cy - dy) and val_y < cy:
                    if val_x >= (cx - dx) and val_x < cx:
                        deriv_x[ii, jj] = ((val_y - cy)/dy + 1)*(1.0/dx)
                        deriv_y[ii, jj] = (1.0/dy)*((val_x - cx)/dx + 1)
                    if val_x < (cx + dx) and val_x >= cx:
                        deriv_x[ii, jj] = ((val_y - cy)/dy + 1)*(-1.0/dx)
                        deriv_y[ii, jj] = (1.0/dy)*((cx - val_x)/dx + 1)
                if val_y < (cy + dy) and val_y >= cy:
                    if val_x >= (cx - dx) and val_x < cx:
                        deriv_x[ii, jj] = ((cy - val_y)/dy + 1)*(1.0/dx)
                        deriv_y[ii, jj] = (-1.0/dy)*((val_x - cx)/dx + 1)
                    if val_x < (cx + dx) and val_x >= cx:
                        deriv_x[ii, jj] = ((cy - val_y)/dy + 1)*(-1.0/dx)
                        deriv_y[ii, jj] = (-1.0/dy)*((cx - val_x)/dx + 1)
        #
        return [deriv_x, deriv_y]

    #----------------------------------------------------------------------------------------
    def error_power(self, alpha):
        '''
        '''
        a_gradB = np.zeros(shape=self.gradB.shape[1:])
        for ii, a in enumerate(alpha):
            a_gradB += a*self.gradB[ii]
        a_gradB_D = a_gradB - self.D
        return np.power(a_gradB_D, 2).sum()

    #--------------------------------------------------------------------------------------
    def error_rmsd(self, alpha):
        '''
        '''
        a_gradB = np.zeros(shape=self.gradB.shape[1:])
        for ii, a in enumerate(alpha):
            a_gradB += a*self.gradB[ii]
        a_gradB_D = a_gradB - self.D
        err = np.power(a_gradB_D, 2).sum(axis=0)
        err = err.mean()
        return np.sqrt(err)

    #--------------------------------------------------------------------------------------
    def BFGS(self, maxiter=15000, ftol=1e-10, gtol=1e-5, error_function='rmsd'):
        '''BFGS minimization of error function
        '''        
        if error_function == 'rmsd':
            print("\nerror = RMSD")
            self.error = self.error_rmsd
        else:
            print("\nerror = diff^2")
            self.error = self.error_power 
 
        options={
            'disp': True,
            'gtol': gtol,
            'maxiter': maxiter,
        }
        
        self.it = 0
        self.err0 = 0
        err = self.error(self.alpha) 
        err0 = err
        
        print("\nStarting BFGS optimization of coefficents.")
        print("--------------------------------------------------------")
        print("%6s\t%14s\t%14s\t%14s" % ("Iter", "Error [kJ/mol]", "Change Error", "Wall Time [s]")) 
        print("--------------------------------------------------------")
        print("%6d\t%14.6f\t%14.6f\t%14.6f" % (self.it, err, 0.0, 0.0)) 

        self.start = time.perf_counter()
        result = opt.minimize(self.error, self.alpha, method='BFGS', tol=ftol, callback=self.BFGS_progress, options=options)
        
        self.alpha = result.x
        self.get_F()

    #--------------------------------------------------------------------------------------
    def BFGS_progress(self, alpha):
         '''callback function to display BFGS progress
         ''' 
         self.it += 1
         err = self.error(alpha)
         print("%6d\t%14.6f\t%14.6f\t%14.6f" % (self.it, err, err-self.err0, time.perf_counter()-self.start))
         self.err0 = err
         self.start = time.perf_counter()        

         return False

    #--------------------------------------------------------------------------------------
    def get_F(self):
        '''
        ''' 
        F_surface = np.zeros(shape=(self.leny,self.lenx), dtype=np.float64)
        fitted_grad_x = np.zeros(shape=(self.leny-1,self.lenx-1), dtype=np.float64)
        fitted_grad_y = np.zeros(shape=(self.leny-1,self.lenx-1), dtype=np.float64)

        for ii, a in enumerate(self.alpha):
            F_surface     += a*self.B[ii]
            fitted_grad_x += a*self.gradB[ii][0]
            fitted_grad_y += a*self.gradB[ii][1]
    
        prob_surface = np.exp(-F_surface/RT) 
        prob_surface /= prob_surface.sum()*self.dx*self.dy
        F_surface = -RT*np.log(prob_surface)       
 
        estim_err_x = np.abs(fitted_grad_x - self.D[0]) 
        estim_err_y = np.abs(fitted_grad_y - self.D[1]) 
         
        self.write_output(F_surface, prob_surface, estim_err_x, estim_err_y) 
        self.plot_F(F_surface, estim_err_x, estim_err_y)

    # -----------------------------------------------------------------------------------------------------
    def write_output(self, F_surface, prob_surface, estim_err_x, estim_err_y):
        '''write output of free energy calculations
        '''
        out = open(f"%s.dat" % (self.outname), "w")

        head = ("Xi1", "Xi1", "error x", "error y", "probability", "free energy [kJ/mol]")
        out.write("%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
        for i in range(self.leny-1):
            for j in range(self.lenx-1):
                row = (self.y[i], self.x[j], estim_err_x[i,j], estim_err_y[i,j], prob_surface[i,j], F_surface[i,j])
                out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.10f\t%14.10f\n" % row)

        out.close()
	
    #--------------------------------------------------------------------------------------
    def plot_F(self, A_surface, estim_err_x, estim_err_y):
        '''plot free energy surface 
        '''
        # Plotting
        plt.rcParams["figure.figsize"] = [8,14]
        fig, axs = plt.subplots(3)
        #
        im0 = axs[0].imshow(estim_err_x, origin='lower', extent=[self.xx.min(), self.xx.max(), self.yy.min(), self.yy.max()], zorder=1, interpolation='None', cmap='viridis', aspect='auto')
        cb0 = plt.colorbar(im0, ax=axs[0], fraction=0.044)
        cb0.outline.set_linewidth(2.5)
        cb0.ax.set_ylabel('Diff Gradient', fontsize=18)
        cb0.ax.tick_params(length=4,width=2.5,labelsize=18, pad=10, direction='in')
        #
        im1 = axs[1].imshow(estim_err_y, origin='lower', extent=[self.xx.min(), self.xx.max(), self.yy.min(), self.yy.max()], zorder=1, interpolation='None', cmap='viridis', aspect='auto')
        cb1 = plt.colorbar(im1, ax=axs[1], fraction=0.044)
        cb1.outline.set_linewidth(2.5)
        cb1.ax.set_ylabel('Diff Gradient', fontsize=18)
        cb1.ax.tick_params(length=4,width=2.5,labelsize=18, pad=10, direction='in')
        #
        CS = axs[2].contour(self.xx, self.yy, A_surface, colors='black', zorder=3)
        plt.clabel(CS, CS.levels, inline='true', fontsize=10, fmt="%5.3f")
        im2 = axs[2].imshow(A_surface, origin='lower', extent=[self.xx.min(), self.xx.max(), self.yy.min(), self.yy.max()], zorder=1, interpolation='None', cmap='viridis', aspect='auto')
        cb2 = plt.colorbar(im2, ax=axs[2], fraction=0.044)
        cb2.outline.set_linewidth(2.5)
        cb2.ax.set_ylabel('Free Energy', fontsize=18)
        cb2.ax.tick_params(length=4,width=2.5,labelsize=18, pad=10, direction='in')
        #
        for ax in axs:
            ax.set_ylim([self.miny, self.maxy])
            ax.set_xlim([self.minx, self.maxx])
            ax.set_xlabel(r'$\xi_1$', fontsize=20)
            ax.set_ylabel(r'$\xi_2$', fontsize=20)
            ax.spines['bottom'].set_linewidth('3')
            ax.spines['top'].set_linewidth('3')
            ax.spines['left'].set_linewidth('3')
            ax.spines['right'].set_linewidth('3')
            ax.tick_params(axis='y',length=6,width=3,labelsize=20, pad=10, direction='in')
            ax.tick_params(axis='x',length=6,width=3,labelsize=20, pad=10, direction='in')
        #
        plt.tight_layout()
        plt.savefig(f"%s.png" % (self.outname), dpi=400)
        plt.close()
