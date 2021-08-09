import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi, cos, sin

pauli0 = np.eye(2)

pauli1 = np.array([[0, 1],
                   [1, 0]])
pauli2 = np.array([[0, -1j],
                   [1j, 0]])
pauli3 = np.array([[1, 0],
                   [0, -1]])

def KP(list):
    output = np.kron(list[0], list[1])
    for arr in list[2:]:
        output = np.kron(output, arr)
    
    return output

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

def overlap_matrix_element(evecs, pm, kn, direction):
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

def overlap_matrix_det(evecs, filled_bands, direction):
    
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
            O_occ[idx_pm, idx_kn] = overlap_matrix_element(evecs, pm, kn, direction)
                    
    
    return O_occ

def polarization(evecs, filled_bands, direction):
    O_occ = overlap_matrix_det(evecs, filled_bands, direction)
    det_O_occ = np.linalg.det(O_occ)
    
    pol = np.imag(np.log(det_O_occ)) / (2.*np.pi)
    
    return pol

def pol_al(evecs):
    # Create position-space grid
    k_shape = evecs.shape[:-2]
    N_orb   = evecs.shape[-2]
    rs = [range(val) for val in k_shape] # List of 1D arrays with indices for each dimension
    Rs = np.meshgrid(*rs, indexing='ij') # Meshgrid of the previous lists
    R_direction = Rs[direction]
    
    n_al = 0.5 * N_orb * np.sum(R_direction) / R_direction.size
    
    return n_al




if __name__ == "__main__":
    np.set_printoptions(linewidth=750)
    
    params = {'t1':0.5, 't2':1., 'm':0.001}
    
    direction = 0
    filled_bands = [0]
    
    verbose = False
    
    for N in range(10, 100):
        print('N = {}'.format(N))
        mat = hamSSH(params, N)
        if verbose: print('mat.shape = {}'.format(mat.shape))
        
        evals, evecs = np.linalg.eigh(mat)
        if verbose: print('evecs.shape = {}'.format(evecs.shape))
        
        n_al = pol_al(evecs)
        pol = polarization(evecs, filled_bands, direction)
        
        print('(pol - n_al) mod 1 = {}'.format((pol - n_al) % 1.))
