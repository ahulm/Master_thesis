import numpy as np
import sys

bohr2angs   = 0.52917721092e0    

# -----------------------------------------------------------------------------------------------------
def mass_center(MD, atoms):
    '''get center of mass and overall mass of group of atoms
    '''   
    m = 0
    c = np.zeros(3)
    if hasattr(atoms, "__len__"):
        for index, a in enumerate(atoms): 
            m += MD.mass[a]
            c += MD.mass[a] * np.array([MD.coords[3*a],MD.coords[3*a+1],MD.coords[3*a+2]],dtype=np.float)
        p = c/m

    else:
        atoms = int(atoms)
        m = MD.mass[atoms]
        p = np.array([MD.coords[3*atoms],MD.coords[3*atoms+1],MD.coords[3*atoms+2]],dtype=np.float)

    return (p, m)

# -----------------------------------------------------------------------------------------------------
def mass_weights(MD, mass_group, atoms):
    '''get mass weights of atoms for gradient of group of atoms
    '''
    coords = np.zeros((3,3*MD.natoms),dtype=np.float)
    if hasattr(atoms, "__len__"):
        for index, a in enumerate(atoms): 
            coords[0,3*a]   = MD.mass[a]/mass_group
            coords[1,3*a+1] = MD.mass[a]/mass_group
            coords[2,3*a+2] = MD.mass[a]/mass_group
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
        print("ERROR: Invalid number of centers in definition of CV!")
        sys.exit(0)

    (p1,m1) = mass_center(self.the_md, atoms[0])
    (p2,m2) = mass_center(self.the_md, atoms[1]) 

    # get distance
    r12 = (p2-p1) * bohr2angs 
    xi  = np.linalg.norm(r12)
    
    # get gradient
    w1 = mass_weights(self.the_md, m1, atoms[0])
    w2 = mass_weights(self.the_md, m2, atoms[1])

    dxi = r12/xi
    grad_xi =- np.dot(dxi, w1)
    grad_xi += np.dot(dxi, w2)
     
    return (xi, grad_xi)

#-----------------------------------------------------------------------------------------------------
def projected_distance(self, atoms):
    '''distance between group 1 and 2 projected on vector between 2 3 in Angstrom in range(-inf,inf) 
    '''
    if len(atoms) != 3:
        print("ERROR: Invalid number of centers in definition of CV!")
        sys.exit(0)

    (p1,m1) = mass_center(self.the_md, atoms[0])
    (p2,m2) = mass_center(self.the_md, atoms[1])
    (p3,m3) = mass_center(self.the_md, atoms[2])

    # get projected distance
    r12 = (p2-p1) * bohr2angs
    r23 = (p3-p2) * bohr2angs
    
    e   = r23/np.linalg.norm(r23)
    xi  = np.dot(r12,e)
        
    # get gradient
    w1 = mass_weights(self.the_md, m1, atoms[0])
    w2 = mass_weights(self.the_md, m2, atoms[1])
    #w3 = mass_weights(self.the_md, m3, atoms[2])

    grad_xi =- np.dot(e, w1)
    grad_xi += np.dot(e, w2)
    #grad_xi += np.dot(e, 0.5*w3)

    return (xi, grad_xi)


# -----------------------------------------------------------------------------------------------------
def angle(self, atoms):
    '''get angle between three mass centers in range(-pi,pi)
    '''    
    if len(atoms) != 3:
        print("ERROR: Invalid number of centers in definition of CV!")
        sys.exit(0)

    (p1,m1) = mass_center(self.the_md, atoms[0])
    (p2,m2) = mass_center(self.the_md, atoms[1]) 
    (p3,m3) = mass_center(self.the_md, atoms[2]) 

    # get angle
    q12 = p1-p2
    q23 = p2-p3
    
    q12_n = np.linalg.norm(q12)
    q23_n = np.linalg.norm(q23)
    
    q12_u = q12/q12_n  
    q23_u = q23/q23_n
   
    xi = np.arccos(np.dot(-q12_u,q23_u))
 
    # get gradient
    w1 = mass_weights(self.the_md, m1, atoms[0])
    w2 = mass_weights(self.the_md, m2, atoms[1])
    w3 = mass_weights(self.the_md, m3, atoms[2])

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
        print("ERROR: Invalid number of centers in definition of CV!")
        sys.exit(0)

    (p1,m1) = mass_center(self.the_md, atoms[0])
    (p2,m2) = mass_center(self.the_md, atoms[1]) 
    (p3,m3) = mass_center(self.the_md, atoms[2]) 
    (p4,m4) = mass_center(self.the_md, atoms[3]) 

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
    w1 = mass_weights(self.the_md, m1, atoms[0])
    w2 = mass_weights(self.the_md, m2, atoms[1])
    w3 = mass_weights(self.the_md, m3, atoms[2])
    w4 = mass_weights(self.the_md, m4, atoms[3])

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
def hBond(self, atoms, n=6, m=8):
    '''hydrogen bond as coordination number between the donor and acceptor atom
     
    args:
        self		(object of bias class)
        atoms           (array, [index hydrogen, index acceptor, index donor, cutoff distance]) 
    returns:
        xi              (double, dimensionless number between 0 and 1)
        grad_xi         (array, N-atoms)
    '''
    (p2,m2) = mass_center(self.the_md, atoms[1]) 
    (p3,m3) = mass_center(self.the_md, atoms[2]) 
    
    if hasattr(atoms[0], "__len__"):
        diff_0 = 0.0
        h_index = atoms[0][0]
        for index, a in enumerate(atoms[0]): 
            c_new = np.array([self.the_md.coords[3*a],self.the_md.coords[3*a+1],self.the_md.coords[3*a+2]],dtype=np.float)
            diff_1 = np.linalg.norm(c_new - p2)
            diff_2 = np.linalg.norm(c_new - p3)
            diff_new = diff_1 + diff_2
            if diff_new > diff_0:
                diff_new = diff_0
                h_index  = a

        (p1,m1) = mass_center(self.the_md, h_index)

    else:
        (p1,m1) = mass_center(self.the_md, atoms[0])    

    # coord number
    d0 = atoms[3] 
    r1 = p1-p2
    r2 = p1-p2

    d1 = np.linalg.norm(r1)
    d2 = np.linalg.norm(r2)
    
    xi =  (1-np.power(d1/d0,n))/(1-np.power(d1/d0,m))
    xi += (1-np.power(d2/d0,n))/(1-np.power(d2/d0,m))    
    
    # gradient
    w1 = mass_weights(self.the_md, m1, atoms[0])
    w2 = mass_weights(self.the_md, m2, atoms[1])
    w3 = mass_weights(self.the_md, m3, atoms[2])

    r1_d = r1/d0
    r2_d = r2/d0

    r1_d_n = np.power(r1_d,n)
    r1_d_m = np.power(r1_d,m)
    r2_d_n = np.power(r2_d,n)
    r2_d_m = np.power(r2_d,m)
    
    dxi1 =+ ( r1_d_n * ((n-m)*r1_d_m-n) + m*r1_d_m ) / ( -r1 * np.power(r1_d_m-1, 2.0) )
    dxi2 = -dxi1

    dxi1 =+ ( r2_d_n * ((n-m)*r2_d_m-n) + m*r2_d_m ) / ( -r2 * np.power(r2_d_m-1, 2.0) )
    dxi3 = -dxi1-dxi2

    grad_xi =+ np.dot(dxi1, w1)
    grad_xi += np.dot(dxi2, w2)
    grad_xi += np.dot(dxi3, w3)
     
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
            (x,dx) = distance(self, np.array(CV[1:]))
          
            CVs = np.append(CV[0]*CVs, x)
            grad_CVs += CV[0]*dx

        elif len(CV) == 4:
   
            # projected distance
            (x,dx) = projected_distance(self, np.array(CV[1:]))
          
            CVs = np.append(CV[0]*CVs, x)
            grad_CVs += CV[0]*dx

        else:
            print("ERROR: Invalid number of centers in definition of CV!")
            sys.exit(0)

    write_lc_traj(i, CVs)
  
    return (CVs.mean(), grad_CVs) 

# -----------------------------------------------------------------------------------------------------
def lin_comb_angles(self, i, atoms):
    '''linear combination of angles or torsion
    '''
    CVs = np.array([])
    grad_CVs = np.zeros(3*self.the_md.natoms,dtype=np.float)
    for index, CV in enumerate(atoms):

        if len(CV) == 3:

            # angle
            (x,dx) = angle(self, np.array(CV))

            CVs = np.append(CVs, x)
            grad_CVs += dx
        
        elif len(CV) == 4:

            # torsion
            (x,dx) = torsion(self, np.array(CV))

            CVs = np.append(CVs, x)
            grad_CVs += dx

        else:
            print("ERROR: Invalid number of centers in definition of CV!")
            sys.exit(0)
    
    write_lc_traj(i, CVs)

    return (CVs.mean(), grad_CVs) 

# -----------------------------------------------------------------------------------------------------
def lin_comb_hBonds(self, i, atoms):
    '''linear combination of angles or torsion
    '''
    CVs = np.array([])
    grad_CVs = np.zeros(3*self.the_md.natoms,dtype=np.float)
    for index, CV in enumerate(atoms):

        (x,dx) = hBond(self, np.array(CV))

        CVs = np.append(CVs, x)
        grad_CVs += dx

    write_lc_traj(i, CVs)

    return (CVs.mean(), grad_CVs) 

# -----------------------------------------------------------------------------------------------------
def write_lc_traj(i, CVs):
    '''write traj of single contributions of linear combination
    '''
    traj_out = open("lc_traj_%d.dat" % (i+1), "a")
    for j in range(len(CVs)):
        traj_out.write("%14.6f\t" % (CVs[j]))
    traj_out.write("\n")
    traj_out.close()


