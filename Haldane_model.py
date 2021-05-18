import numpy as np
from numpy import sqrt, cos, sin
import misc as mi
import matplotlib.pyplot as plt

def HaldaneHamiltonian(k, params):
    t1 = params['t1']
    t2 = params['t2']
    phi = params['phi']
    M = params['M']
    
    # Convenience aliases for Pauli matrices
    s0 = np.eye(2)
    s1 = np.array([[0,1], 
                       [1,0]])
    s2 = np.array([[0,-1j], 
                      [1j,0]])
    s3 = np.array([[1,0], 
                       [0,-1]])
    
    K1, K2 = k # Split momentum into its components
    K1 = 2.*np.pi * K1[...,None,None] # Add length-one axes to broadcast against orbital axes
    K2 = 2.*np.pi * K2[...,None,None]
    
    k_shape = k.shape[1:] # Shape of momentum arrays (excluding kx,ky components)
    mat = np.zeros(k_shape + (2,2), complex)
    
    cos_a = cos((-K1-2.*K2)/3.) + cos((2.*K1+K2)/3.) + cos((K2-K1)/3.)
    sin_a = sin((-K1-2.*K2)/3.) + sin((2.*K1+K2)/3.) + sin((K2-K1)/3.)
    cos_b = cos(K1) + cos(K2) + cos(- K1 - K2)
    sin_b = sin(K1) + sin(K2) + sin(- K1 - K2)
    
    mat += 2. * t2 * cos(phi) * cos_b * s0
    mat += t1 * cos_a * s1
    mat += t1 * sin_a * s2
    mat += (M - 2.*t2*sin(phi)*sin_b) * s3
    
    return mat

def find_gaps(Hamiltonian, params, Nlist):
    
    linspaces_list = []
    for N in Nlist:
        linspaces_list.append( np.linspace(-0.5, 0.5, N, endpoint=False) )
    
    
    k = np.array( np.meshgrid(*linspaces_list, indexing='ij') )
    
    ham = HaldaneHamiltonian(k, params)
    
    evals = np.linalg.eigvalsh(ham)
    
    Ngaps = evals.shape[-1] - 1
    
    gaps_array = evals[..., 1:] - evals[..., :-1]
    
    gaps_min = np.amin(gaps_array, axis=tuple(range(gaps_array.ndim-1)))
    
    return np.squeeze(gaps_min)

if __name__ == "__main__":
    np.set_printoptions(linewidth=750)
    
    params = {'t1':1., 't2':0.3, 'phi':0., 'M':0. }
    Nlist = [200, 200]
    
    N_M = 41
    N_phi = 41
    M_array = params['t2'] * np.linspace(-6., 6., N_M, endpoint=True)
    phi_array = np.linspace(-np.pi, np.pi, N_phi, endpoint=True)
    
    gap_array = np.zeros(M_array.shape + phi_array.shape)
    
    for M_idx, M in enumerate(M_array):
        params['M'] = M
        for phi_idx, phi in enumerate(phi_array):
            params['phi'] = phi
            gap_array[M_idx, phi_idx] = find_gaps(HaldaneHamiltonian, params, Nlist)
    
    fig, ax = plt.subplots()
    ax.contourf(M_array, phi_array, gap_array, 25)
    plt.show()