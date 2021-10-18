import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import pi, cos, sin


def KP(list):
    output = np.kron(list[0], list[1])
    for arr in list[2:]:
        output = np.kron(output, arr)
    
    return output



def overlap_matrix(OME, evecs, filled_bands, direction, plot=False):
    
    k_shape = evecs.shape[:-2]
    dim = len(k_shape)
    N_sites = np.prod(k_shape)
    N_occ = len(filled_bands)
    
    # Create momentum-space grid
    ks = [range(val) for val in k_shape] # List of 1D arrays with indices for each dimension
    Ks = np.meshgrid(*ks, indexing='ij') # Meshgrid of the previous lists
    Ks = np.moveaxis(Ks, 0, -1) # Move component dimension to last axis
    
    Ks = np.reshape(Ks, (N_sites, dim))
    
    O_occ = np.zeros([N_sites*N_occ] * 2, complex)
    
    # kn_list = [ [[tuple(k), n] for n in filled_bands] for k in Ks ]
    kn_list = []
    for k in Ks:
        for n in filled_bands:
            kn_list.append( [tuple(k), n] )
    
    for idx_kn, kn in enumerate(kn_list):
        for idx_pm, pm in enumerate(kn_list):
            O_occ[idx_pm, idx_kn] = OME(evecs, pm, kn, direction)
    
    if plot:
        fig, ax = plt.subplots()
        #neg = ax.matshow(np.angle(O_occ) * np.abs(O_occ))
        neg = ax.matshow(np.log10(np.abs(O_occ)))
        fig.colorbar(neg, ax=ax, location='right', anchor=(0, 0.3), shrink=0.7)
        plt.show()
    
    return O_occ

# Polarization / Dipole moment

def overlap_Pmatrix_element(evecs, pm, kn, direction):
    # evecs: momentum dimensions, sublattice index, band index
    # kn is tuple giving momentum indices and band index
    # pm = [(p1, p2, ...), m]
    k_shape = evecs.shape[:-2]
    N_orb   = evecs.shape[-2]
    N_sites = np.prod(k_shape)
    
    # Tuples for indexing evecs array
    kn_index = kn[0] + (slice(None), kn[1])
    pm_index = pm[0] + (slice(None), pm[1])
    v_prod = np.sum( np.conj(evecs[pm_index]) * evecs[kn_index] )
    
    # Create position-space grid
    rs = [range(val) for val in k_shape] # List of 1D arrays with indices for each dimension
    Rs = np.meshgrid(*rs, indexing='ij') # Meshgrid of the previous lists
    Rs = np.moveaxis(Rs, 0, -1) # Move component dimension to last axis
    
    # Compute Bloch factors in position space
    p_minus_k        = np.array(pm[0]) - np.array(kn[0])
    p_minus_k_scaled = p_minus_k * 2.*np.pi/np.array(k_shape, float)
    R_dot_p_minus_k = np.sum(Rs * p_minus_k_scaled, axis=-1) # Take sum along last axis
    
    # Compute polarization factor in position space
    UP = Rs[...,direction] * 2.*np.pi / float(k_shape[direction])
    
    # Take the product
    Rsum = np.sum( np.exp(1.j*R_dot_p_minus_k) * np.exp(1.j*UP) ) / N_sites
    
    # Take the product to find the overlap
    O_pm_kn = v_prod * Rsum
    
    return O_pm_kn

def pol_al(evecs, direc):
    # Create position-space grid
    k_shape = evecs.shape[:-2]
    N_orb   = evecs.shape[-2]
    rs = [range(val) for val in k_shape] # List of 1D arrays with indices for each dimension
    Rs = np.meshgrid(*rs, indexing='ij') # Meshgrid of the previous lists
    R_direction = Rs[direc]
    
    n_al = 0.5 * N_orb * np.sum(R_direction) / R_direction.size
    
    return n_al

def polarization(evecs, filled_bands, direction):
    O_occ = overlap_matrix(overlap_Pmatrix_element, evecs, filled_bands, direction)
    
    det_O_occ = np.linalg.det(O_occ)
    
    pol = np.imag(np.log(det_O_occ)) / (2.*np.pi)
    
    return pol    

# Quadrupole moment

def overlap_Qmatrix_element(evecs, pm, kn, directions):
    # evecs: momentum dimensions, sublattice index, band index
    # kn is tuple giving momentum indices and band index
    # pm = [(p1, p2, ...), m]
    k_shape = evecs.shape[:-2]
    N_orb   = evecs.shape[-2]
    N_sites = np.prod(k_shape)
    
    # Tuples for indexing evecs array
    kn_index = kn[0] + (slice(None), kn[1])
    pm_index = pm[0] + (slice(None), pm[1])
    v_prod = np.sum( np.conj(evecs[pm_index]) * evecs[kn_index] )
    
    # Create position-space grid
    rs = [range(val) for val in k_shape] # List of 1D arrays with indices for each dimension
    Rs = np.meshgrid(*rs, indexing='ij') # Meshgrid of the previous lists
    Rs = np.moveaxis(Rs, 0, -1) # Move component dimension to last axis
    
    # Compute Bloch factors in position space
    p_minus_k        = np.array(pm[0]) - np.array(kn[0])
    p_minus_k_scaled = p_minus_k * 2.*np.pi/np.array(k_shape, float)
    R_dot_p_minus_k = np.sum(Rs * p_minus_k_scaled, axis=-1) # Take sum along last axis
    
    # Compute polarization factor in position space
    UQ = Rs[...,directions[0]] * Rs[...,directions[1]] * 2.*np.pi / float(k_shape[directions[0]] * k_shape[directions[1]])
    
    # Take the product
    Rsum = np.sum( np.exp(1.j*R_dot_p_minus_k) * np.exp(1.j*UQ) ) / N_sites
    
    # Take the product to find the overlap
    O_pm_kn = v_prod * Rsum
    
    return O_pm_kn

def q_al(evecs, direcs):
    # Create position-space grid
    k_shape = evecs.shape[:-2]
    N_orb   = evecs.shape[-2]
    rs = [range(val) for val in k_shape] # List of 1D arrays with indices for each dimension
    Rs = np.array(np.meshgrid(*rs, indexing='ij')) # Meshgrid of the previous lists
    
    ans = 0.5 * N_orb * np.sum(Rs[direcs[0], ...] * Rs[direcs[1], ...]) / Rs[direcs[0],...].size
    
    return ans

def quad_moment(evecs, filled_bands, directions):
    O_occ = overlap_matrix(overlap_Qmatrix_element, evecs, filled_bands, directions)
    
    det_O_occ = np.linalg.det(O_occ)
    
    pol = np.imag(np.log(det_O_occ)) / (2.*np.pi)
    
    return pol    

if __name__ == "__main__":
    np.set_printoptions(linewidth=750)
    
    # Convenience aliases for Pauli matrices
    pauli0 = np.eye(2)
    pauli1 = np.array([[0,1], 
                       [1,0]])
    pauli2 = np.array([[0,-1j], 
                      [1j,0]])
    pauli3 = np.array([[1,0], 
                       [0,-1]])
    
    scenario = 'BBH'
    
    if scenario=='SSH':
        
        def hamSSH(params_dict, N):
            ''' Return the Bloch hamiltonian. Broadcasts in k. '''

            t1 = params_dict['t1']
            t2 = params_dict['t2']
            m  = params_dict['m']

            k = np.linspace(0., 2.*np.pi, num=N, endpoint=False)
            k_shape = k.shape
            mat = np.zeros(k_shape + (2,2), complex)

            mat[...,0,1] += t1 + t2*np.exp(+1.j*k)
            mat[...,1,0] += t1 + t2*np.exp(-1.j*k)
            mat[...,0,0] +=   m/2.
            mat[...,1,1] += - m/2.

            return mat
    
        params = {'t1':1.5, 't2':1., 'm':0.}
    
        direction = 0
        filled_bands = [0]
    
        verbose = False
    
        for N in range(30, 41):
            print('N = {}'.format(N))
            mat = hamSSH(params, N)
            if verbose: print('mat.shape = {}'.format(mat.shape))
        
            evals, evecs = np.linalg.eigh(mat)
            if verbose: print('evecs.shape = {}'.format(evecs.shape))
        
            n_al = pol_al(evecs, direction)
            pol = polarization(evecs, filled_bands, direction)
        
            print('(pol - n_al) mod 1 = {}'.format((pol - n_al) % 1.))
    
    if scenario=='BBH':
        
        def hamBBH(params_dict, N_list):
            ''' Return the Bloch hamiltonian. Broadcasts in k. 
            First dimension of k are the kx,ky components.
            '''
            # Unpack arguments
            gamma_x = params_dict['gamma_x']
            gamma_y = params_dict['gamma_y']
            lambd = params_dict['lambd']
            delta = params_dict['delta']
    
            # Matrices that enter into hamiltonian (4 by 4)
            Gamma0 =   np.kron(pauli3, pauli0)
            Gamma1 = - np.kron(pauli2, pauli1)
            Gamma2 = - np.kron(pauli2, pauli2)
            Gamma3 = - np.kron(pauli2, pauli3)
            Gamma4 =   np.kron(pauli1, pauli0)
            
            k_shape = tuple(N_list)
            dim = len(k_shape)
            N_sites = np.prod(k_shape)
    
            # Create momentum-space grid
            ks = [np.arange(val) * 2.*np.pi / val for val in k_shape] # List of 1D arrays with indices for each dimension
            Kx, Ky = np.meshgrid(*ks, indexing='ij') # Meshgrid of the previous lists
            
            Kx = Kx[...,None,None] # Add length-one axes to broadcast against orbital axes
            Ky = Ky[...,None,None]
    
            mat = np.zeros(k_shape + (4,4), complex)
    
            # x-direction terms
            mat += gamma_x * Gamma4 
            mat += lambd * ( cos(Kx) * Gamma4 + sin(Kx) * Gamma3 )
            # y-direction terms
            mat += gamma_y * Gamma2
            mat += lambd * ( cos(Ky) * Gamma2 + sin(Ky) * Gamma1 )
            # On-site mass term
            mat += delta * Gamma0
    
            return mat#, Kx, Ky
        
        
        params_dict = {'gamma_x':0.5, 
               'gamma_y':0.5, 
               'lambd':1., 
               'delta':0.}
        
        Nx, Ny = 20,20
        
#         print('H_array.shape = {}'.format(H_array.shape))
#         evals = np.linalg.eigvalsh(H_array)
#         print('evals.shape = {}'.format(evals.shape))
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         # Plot the surface.
#         print(Kx.shape)
#         print(Ky.shape)
#         print(evals.shape)
#         for n in range(evals.shape[-1]):
#             surf = ax.plot_surface(np.squeeze(Kx), np.squeeze(Ky), evals[...,n], cmap=cm.coolwarm, linewidth=0, antialiased=False)
#         plt.show()
        
        directions = [0,1]
        filled_bands = [0,1]
    
        verbose = False
    
        mat = hamBBH(params_dict, [Nx, Ny])
        if verbose: print('mat.shape = {}'.format(mat.shape))
        
        evals, evecs = np.linalg.eigh(mat)
        if verbose: print('evecs.shape = {}'.format(evecs.shape))
        
        #OM = overlap_matrix(overlap_Qmatrix_element, evecs, filled_bands, directions, plot=False)
        #print(OM.shape)
            
        qal = q_al(evecs, directions)
        print('qal = {}'.format(qal))
        
        q = quad_moment(evecs, filled_bands, directions)
        print('q = {}'.format(q))
    
        print('(q - qal) mod 1 = {}'.format((q - qal) % 1.))

