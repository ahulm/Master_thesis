import numpy as np
import random

it2fs = 1.0327503e0
kB      = 1.380648e-23      # J / K
H_in_J  = 4.359744e-18
H_in_kJmol = 2625.499639
kB_a    = kB / H_in_J       # Hartree / K
au2k  = 315775.04e0
T = 300.0

# -----------------------------------------------------------------------------------------------------
def get_angle(self, ats):
	
	atom1 = int(ats[0])-1
	atom2 = int(ats[1])-1
	atom3 = int(ats[2])-1
		
	p1 = np.array([self.coords[3*atom1+0],self.coords[3*atom1+1],self.coords[3*atom1+2]],dtype=np.float64)
	p2 = np.array([self.coords[3*atom2+0],self.coords[3*atom2+1],self.coords[3*atom2+2]],dtype=np.float64)
	p3 = np.array([self.coords[3*atom3+0],self.coords[3*atom3+1],self.coords[3*atom3+2]],dtype=np.float64)
			
	q12 = p2-p1
	q23 = p2-p3
	
	q12_n = np.linalg.norm(q12)
	q23_n = np.linalg.norm(q23)
	
	q12_u = q12/q12_n  
	q23_u = q23/q23_n

	alpha = np.arccos(np.dot(q12_u,q23_u))
	
	return np.degrees(alpha)


# -----------------------------------------------------------------------------------------------------
def get_torsion_angle(self, ats):
	
	atom1 = int(ats[0])-1
	atom2 = int(ats[1])-1
	atom3 = int(ats[2])-1
	atom4 = int(ats[3])-1
	
	p1 = np.array([self.coords[3*atom1+0],self.coords[3*atom1+1],self.coords[3*atom1+2]],dtype=np.float64)
	p2 = np.array([self.coords[3*atom2+0],self.coords[3*atom2+1],self.coords[3*atom2+2]],dtype=np.float64)
	p3 = np.array([self.coords[3*atom3+0],self.coords[3*atom3+1],self.coords[3*atom3+2]],dtype=np.float64)
	p4 = np.array([self.coords[3*atom4+0],self.coords[3*atom4+1],self.coords[3*atom4+2]],dtype=np.float64)

	q12 = p1 - p2
	q23 = p3 - p2
	q34 = p4 - p3
	
	q23_u = q23 / np.linalg.norm(q23)
	
	n1 =  q12 - np.dot(q12,q23_u)*q23_u
	n2 =  q34 - np.dot(q34,q23_u)*q23_u
	
	torsion = np.degrees(np.arctan2(np.dot(np.cross(q23_u,n1),n2),np.dot(n1,n2)))
	
	return torsion


# -----------------------------------------------------------------------------------------------------
def first_derivative_angle(self, ats):

	atom1 = int(ats[0])-1
	atom2 = int(ats[1])-1
	atom3 = int(ats[2])-1

	p1 = np.array([self.coords[3*atom1+0],self.coords[3*atom1+1],self.coords[3*atom1+2]],dtype=np.float64)
	p2 = np.array([self.coords[3*atom2+0],self.coords[3*atom2+1],self.coords[3*atom2+2]],dtype=np.float64)
	p3 = np.array([self.coords[3*atom3+0],self.coords[3*atom3+1],self.coords[3*atom3+2]],dtype=np.float64)

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
	
	dxi2 = - (dxi1 + dxi3)
	
	delta_xi = np.zeros(3*self.natoms)
	for dim in range(0,3):
		delta_xi[atom1*3+dim] += dxi1[dim]
		delta_xi[atom2*3+dim] += dxi2[dim]
		delta_xi[atom3*3+dim] += dxi3[dim]
	
	return delta_xi


# -----------------------------------------------------------------------------------------------------
def first_derivative_torsion(self, ats):

	atom1 = int(ats[0])-1
	atom2 = int(ats[1])-1
	atom3 = int(ats[2])-1
	atom4 = int(ats[3])-1
	
	p1 = np.array([self.coords[3*atom1+0],self.coords[3*atom1+1],self.coords[3*atom1+2]],dtype=np.float64)
	p2 = np.array([self.coords[3*atom2+0],self.coords[3*atom2+1],self.coords[3*atom2+2]],dtype=np.float64)
	p3 = np.array([self.coords[3*atom3+0],self.coords[3*atom3+1],self.coords[3*atom3+2]],dtype=np.float64)
	p4 = np.array([self.coords[3*atom4+0],self.coords[3*atom4+1],self.coords[3*atom4+2]],dtype=np.float64)

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
	
	c_123 = ((q12_n*cos_123)/q23_n) - 1
	b_432 = ((q34_n*cos_234)/q23_n)
		
	dtau2 = c_123*dtau1 - b_432*dtau4
	dtau3 = -(dtau1 + dtau2 + dtau4)

	delta_xi = np.zeros(3*self.natoms)
	for dim in range(0,3):
		delta_xi[atom1*3+dim] += dtau1[dim]
		delta_xi[atom2*3+dim] += dtau2[dim]
		delta_xi[atom3*3+dim] += dtau3[dim]
		delta_xi[atom4*3+dim] += dtau4[dim]

	return delta_xi

def get_coord(self, ats):

    xi = np.array([])
    delta_xi = [0 for i in range(len(ats))]

    for i in range(len(ats)):

        if ats[i][0] == 1:
            x = self.coords[0]
            xi = np.append(xi, x)
            delta_xi[i] += np.array([1,0])
        
        elif ats[i][0] == 2:
            y = self.coords[1]
            xi = np.append(xi, y)
            delta_xi[i] += np.array([0,1])
        
        elif ats[i][0] == 3:
            x = self.coords[0]
            y = self.coords[1]
            xi = np.append(xi, x + y)
            delta_xi[i] += np.array([1,1])
        
        elif ats[i][0] == 4:
            x = self.coords[0]
            y = self.coords[1]
            xi = np.append(xi, x/4.0 + y)
            delta_xi[i] += np.array([0.25,1])

    return (xi, delta_xi)


# -----------------------------------------------------------------------------------------------------
def extend_system(self, index, xi):
	'''add extended variable to system
	
	Args:
	  index		(int, index of extended variable)
	  xi		(double, reaction coordinate)

	Returns:
	  -
	'''
	random.seed(np.random.randint(2147483647))
	
	if index == 0:
		self.ext_mass = np.array([])
		self.ext_masses = np.array([])
		self.ext_coords = np.array([])
		self.ext_momenta = np.array([])
		self.ext_forces = np.array([]) 	
		self.ext_natoms = 0 
	
	self.ext_mass = np.append(self.ext_mass, self.mass)
	self.ext_masses = np.append(self.ext_masses, self.mass)		
	self.ext_coords = np.append(self.ext_coords, xi[index])
	self.ext_forces = np.append(self.ext_forces, np.zeros(1)) 	
	self.ext_natoms += 1
	
	self.ext_momenta = np.append(self.ext_momenta, np.zeros(1))
	for i in range(self.ext_natoms):
		self.ext_momenta[i] = random.gauss(0.0,1.0)*np.sqrt(self.target_temp*self.ext_mass[i])	
		TTT = (np.power(self.ext_momenta, 2)/self.ext_masses).sum()
		TTT /= (self.ext_natoms)
		self.ext_momenta *= np.sqrt(T/(TTT*au2k))
	


# -----------------------------------------------------------------------------------------------------
def propagate_extended(self, langevin=True, friction=1.0e-3):
	'''Propagate momenta/coords of extended variable with Velocity Verlet
	
	Args:
	   langevin                (bool, False)
	   friction                (float, 10^-3 1/fs)
	Returns:
	   -
	'''
	
	if langevin==True:
		prefac    = 2.0 / (2.0 + friction*self.dt_fs)
		rand_push = np.sqrt(self.target_temp*friction*self.dt_fs*kB_a/2.0e0)
		self.ext_rand_gauss = np.zeros(shape=(len(self.ext_momenta),), dtype=np.double)
		for atom in range(len(self.ext_rand_gauss)):
			self.ext_rand_gauss[atom]   = random.gauss(0, 1)
		
		self.ext_momenta += np.sqrt(self.ext_masses) * rand_push * self.ext_rand_gauss
		self.ext_momenta -= 0.5e0 * self.dt * self.ext_forces
		self.ext_coords  += prefac * self.dt * self.ext_momenta / self.ext_masses
	
	else:
		self.ext_momenta -= 0.5e0 * self.dt * self.ext_forces
		self.ext_coords  += self.dt * self.ext_momenta / self.ext_masses


# -----------------------------------------------------------------------------------------------------
def up_momenta_extended(self, langevin=False, friction=1.0e-3):
	'''Update momenta of extended variables with Velocity Verlet
	
	Args:
	   -
	
	Returns:
	   -
	'''
	
	if langevin==True:
		prefac = (2.0e0 - friction*self.dt_fs)/(2.0e0 + friction*self.dt_fs)
		rand_push = np.sqrt(self.target_temp*friction*self.dt_fs*kB_a/2.0e0)
		self.ext_momenta *= prefac
		self.ext_momenta += np.sqrt(self.ext_masses) * rand_push * self.ext_rand_gauss
		self.ext_momenta -= 0.5e0 * self.dt * self.ext_forces
	else:
		self.ext_momenta -= 0.5e0 * self.dt * self.ext_forces


# -----------------------------------------------------------------------------------------------------
def write_traj(self, extended = False):
	'''write trajectory of extended or normal ABF

	Args:
	  extended	(bool, False)
	
	Returns:
	  -
	'''	
	if self.step == 0:
		
		traj_out = open("abf_traj.dat", "w")
		traj_out.write("%14s\t" % ("time [fs]"))
		for i in range(len(self.traj[0])):
			traj_out.write("%14s\t" % (f"Xi{i}"))
			if extended:
				traj_out.write("%14s\t" % (f"eXi{i}"))

		traj_out.close()
	
	traj_out = open("abf_traj.dat", "a")
	traj_out.write("\n%14.6f\t" % (self.step*self.dt*it2fs))
	for i in range(len(self.traj[0])):
		traj_out.write("%14.6f\t" % (self.traj[-1][i]))
		if extended:
			traj_out.write("%14.6f\t" % (self.etraj[-1][i]))
	traj_out.close()


# -----------------------------------------------------------------------------------------------------
def write_output(self, ats):
	'''write output of ABF or eABF calculations

	Args:
	  ats:	(array, atom indizes)

	Returns:
	  -
	'''
	if len(self.minx) == 1:
		# 1D reaction coordinate
		abf_out = open(f"abf_out.txt", "w")
		abf_out.write("%6s\t%14s\t%14s\t%14s\t%14s\n" % ("Bin", "Xi", "Count", "Sum Forces", "Mean Force"))
		for i in range(len(self.biases[0])):
			abf_out.write("%6d\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % (i, i*self.dx[0]+self.minx[0], self.bin_list[i], self.biases[0][i], self.biases[0][i]/self.bin_list[i] if self.bin_list[i] > 0 else 0))
		abf_out.close()
	
	else:
		# 2D reaction coordinate 
		abf_out = open("abf_out.txt", "w")
		abf_out.write("%6s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % ("Bin","Xi1","Xi2","Count","Sum Forces1", "Sum Forces 2", "Mean Force 1","Mean Force 2"))
		
		Nbins0 = int(np.floor(np.abs(self.maxx[0]-self.minx[0])/self.dx[0]))
		Nbins1 = int(np.floor(np.abs(self.maxx[1]-self.minx[1])/self.dx[1]))
		
		b = 0	
		for i in range(Nbins0):

			bin0 = i*ats[0][3]+ats[0][1]

			for j in range(Nbins1):
			
				bin1 = j*ats[1][3]+ats[1][1]	
				abf_out.write("%6d\t%14.6f\t%14.6f\t" % (b, bin0, bin1))
				b += 1				
				
				bin_ij = i + Nbins0*j
				abf_out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.6f\n" % (self.bin_list[bin_ij], self.biases[0][bin_ij], self.biases[1][bin_ij], self.biases[0][bin_ij]/self.bin_list[bin_ij] if self.bin_list[bin_ij] > 0 else 0, self.biases[1][bin_ij]/self.bin_list[bin_ij] if self.bin_list[bin_ij] > 0 else 0))			

		abf_out.close()

