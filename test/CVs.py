import numpy as np
import sys

bohr2angs   = 0.52917721092e0    

# -----------------------------------------------------------------------------------------------------
def mass_center(MD, atoms):
    '''get center of mass of group of atoms
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

    grad_xi =- np.dot(e, w1)
    grad_xi += np.dot(e, w2)

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
def lin_comb_dists(self, atoms):
    '''linear combination distances and projected distances 
    '''
    CVs = np.array([])
    grad_CVs = np.zeros(3*self.the_md.natoms,dtype=np.float)
    for index, CV in enumerate(atoms):

        if len(CV) == 2:

            # distance
            (x,dx) = distance(self, np.array(CV))
          
            CVs = np.append(CVs, x)
            grad_CVs += dx

        elif len(CV) == 3:
   
            # projected distance
            (x,dx) = projected_distance(self, np.array(CV))
          
            CVs = np.append(CVs, x)
            grad_CVs += dx

        else:
            print("ERROR: Invalid number of centers in definition of CV!")
            sys.exit(0)

    return (CVs.mean(), grad_CVs) 

# -----------------------------------------------------------------------------------------------------
def lin_comb_angles(self, atoms):
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
    
    return (CVs.mean(), grad_CVs) 



