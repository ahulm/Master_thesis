import numpy as np
from numpy import inf

H_in_J = 4.359744e-18
H_in_kJmol = 2625.499639 # Hartree/kJmol
kB      = 1.380648e-23      # J / K
kB_a = kB / H_in_J
bohr2angs = 0.52917721092e0

def ABF_estimator(data):
    '''Integrate A'(z) 
    '''
    df = data.copy()
    dxi = df.iloc[1,1]-df.iloc[0,1]
    df['dA'] = [0.0 for i in range(len(df))]
    for i in range(len(df)):
        df.iloc[i,5] = np.sum(np.array(df.iloc[0:i,4],dtype=np.float64))*dxi/bohr2angs
    df.iloc[:,5] -= df.iloc[:,5].min()
    df.iloc[:,5] *= H_in_kJmol
    
    return df

def ABF2D_integrator(F, dx, dy):
    '''Integrate A'(z1,z2) using Simpsons rule
    '''
    dA = np.copy(F) * 0.0
    
    for i in range(1,len(dA[0,:])-1):
        for j in range(1,len(dA[:,0])-1):
            
            p1 = (F[j-1,i-1] + 4*F[j-1,i] + F[j-1,i+1]) * (dx*bohr2angs)/3
            p2 = (F[j,i-1]   + 4*F[j,i]   + F[j,i+1])   * (dx*bohr2angs)/3
            p3 = (F[j+1,i-1] + 4*F[j+1,i] + F[j+1,i+1]) * (dx*bohr2angs)/3
            
            dA[j,i] = abs(p1 + 4*p2 + p3) * (dy*bohr2angs)/3
    
    dA -= dA.min()
    dA *= H_in_kJmol
    
    return dA

def CZAR_estimator(traj, minx, maxx, dx, sigma, T):
    '''Estimate A'(z) from biased eABF dynamics
    '''
    k = (kB_a*T)/(sigma*sigma)
    Nbins = int((maxx-minx)/dx)

    print('\n\tk =\t%14.6f Hartree/rad^2' % (k))
    print('\tdx =\t%14.6f' % (dx))
    print('\tminx =\t%14.6f' % (minx))
    print('\tmaxx =\t%14.6f' % (maxx))
    print('\tnbins =\t%14d\n' % (Nbins))

    f = np.array([0.0 for i in range(Nbins)])
    dxi = np.array([0.0 for i in range(Nbins)])
    bin_counts = np.array([0.0 for i in range(Nbins)])
    
    for i in range(len(f)):
        
        dxi[i] = minx+i*dx+dx/2 
        z = traj[traj.iloc[:,1].between(minx+i*dx,minx+i*dx+dx)]
        bin_counts[i] = len(z)      
        
        if bin_counts[i] > 0:
            f[i] += k * (np.mean(z.iloc[:,2]) - (minx+i*dx+dx/2))
    
    S = np.log(bin_counts)
    S[S == -inf] = 0.0
    S[S == inf] = 0.0

    t1 = np.array([0.0 for i in range(len(S))])
    for i in range(2,len(S)-2):
        t1[i] += (-S[i-2] + 8*S[i-1] - 8*S[i+1] + S[i+2]) / (12.0*dx)
    
    czar = kB_a * T * t1 + f

    dA = np.array([0.0 for i in range(len(czar))])
    for i in range(len(dA)):
        dA[i] += np.sum(czar[0:i])*dx
    dA -= dA.min()
    dA *= H_in_kJmol

    return [dxi,dA]

def ZhengYang_estimator(traj, abf_out, sigma, T):
    
    k = (kB_a*T)/(sigma*sigma)
    dx      =   (abf_out.iloc[1,1] - abf_out.iloc[0,1]) 
    minx    =   (abf_out.iloc[0,1]) 
    maxx    =   (abf_out.iloc[-1,1]+dx) 
    
    print('\n\tk =\t%14.6f Hartree/rad^2' % (k))
    print('\tdx =\t%14.6f' % (dx))
    print('\tminx =\t%14.6f' % (minx))
    print('\tmaxx =\t%14.6f\n' % (maxx))

    f = np.array([0.0 for i in range(len(abf_out))])
    dxi = np.array([0.0 for i in range(len(abf_out))])
    
    for i in range(len(f)):
        
        dxi[i] += minx+i*dx+dx/2 
        la = traj[traj.iloc[:,1].between(minx+i*dx,minx+i*dx+dx)]

        if len(la) > 0:
            
            mean_x = np.mean(la.iloc[:,1])
            var_x = np.var(la.iloc[:,1])
            
            for j in range(len(la)):           
                f[i] += -kB_a * T * (la.iloc[j,1]-mean_x)/var_x + k * (la.iloc[j,2] - la.iloc[j,1])
                
            f[i] /= len(la) 
    
    dA = np.array([0.0 for i in range(len(czar))])
    for i in range(len(dA)):
        dA[i] += np.sum(f[0:i])*dx
    dA -= dA.min()
    dA *= H_in_kJmol

    return [dxi,dA]


