import numpy as np
from numpy import sqrt, cos, sin
import misc as mi
import matplotlib.pyplot as plt

def HaldaneHamiltonian(k, params, periodic=False):
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
    
    k = np.asarray(k)
    K1, K2 = k # Split momentum into its components
    K1 = 2.*np.pi * K1[...,None,None] # Add length-one axes to broadcast against orbital axes
    K2 = 2.*np.pi * K2[...,None,None]
    
    k_shape = k.shape[1:] # Shape of momentum arrays (excluding kx,ky components)
    mat = np.zeros(k_shape + (2,2), complex)
    
    if periodic:
        cos_a = 1. + cos(K1) + cos(-K2)
        sin_a = - (sin(K1) + sin(-K2))
    else:
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

def find_gaps_Hal():
    params = {'t1':1., 't2':0.3, 'phi':0., 'M':0. }
    Nlist = [200, 200]
    
    N_M = 31
    N_phi = 31
    M_array = params['t2'] * np.linspace(-10., 10., N_M, endpoint=True)
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

def link_variable(evecs, verbose=True):
    ''' Calculate Wilson line elements between adjacent momentum points using evecs_occ. 
    Last two dimensions of evecs_occ are orbitals and bands, previous ones are momentum
    directions. If unitary==True, the purely "unitary part" of the Wilson line elements 
    are returned
    '''
    
    # Last two dimensions are orbitals and bands, previous ones are momentum
    Nbands = evecs.shape[-1]
    k_shape = evecs.shape[:-2]
    D = len(k_shape)
    
    if verbose:
        mi.sprint('Nbands',Nbands)
        mi.sprint('D',D)
    
    # Create return array
    U = np.zeros((D,) + k_shape + (Nbands,), complex)
    
    for d in range(D):
        U_temp = np.sum( evecs.conj() * np.roll(evecs, -1, axis=d), axis=-2 )
        U_temp *= 1. / np.abs(U_temp)
        
        U[d,...] = U_temp
    
    return U

def Chern(linkvar, imagtol=1.e-15):
    
    # Only makes sense for two spatial dimensions
    assert linkvar.ndim==4
    assert linkvar.shape[0]==2
    
    factor1 = U[0,...]
    factor2 = np.roll(U[1,...], -1, 1)
    factor3 = np.roll(U[0,...], -1, 2)
    factor4 = U[1,...]
    
    F12 = np.log( factor1*factor2 / (factor3*factor4) ) / 1.j
    
    F12_imag = np.imag(F12)
    F12_imag_max = np.amax(np.abs(F12_imag))
    
    if F12_imag_max>imagtol:
        print('WARNING: In Chern(), imaginary parts were larger than tolerance. Largest is {}'.format(F12_imag_max))
    else:
        F12 = np.real(F12)
    
    C = np.sum( F12, axis=(0,1) ) / 2.*np.pi
    
    return C

def check_equiv_of_hamiltonians():
    params = {'t1':1., 't2':0.3, 'phi':np.pi/2., 'M':0. }
    N1, N2 = 1000, 1100
    k1 = np.linspace(-0.5, 0.5, N1, endpoint=False)
    k2 = np.linspace(-0.5, 0.5, N2, endpoint=False)
    k = np.array(np.meshgrid(k1,k2, indexing='ij'))
    
    ham1 = HaldaneHamiltonian(k, params, periodic=True)
    ham2 = HaldaneHamiltonian(k, params, periodic=False)
    print(np.amax(np.abs( np.linalg.eigvalsh(ham1)-np.linalg.eigvalsh(ham2) )))
    
    k1 = np.linspace(2.5, 3.5, N1, endpoint=False)
    k2 = np.linspace(-0.5, 0.5, N2, endpoint=False)
    k = np.array(np.meshgrid(k1,k2, indexing='ij'))
    ham1_t = HaldaneHamiltonian(k, params, periodic=True)
    ham2_t = HaldaneHamiltonian(k, params, periodic=False)
    print(np.amax(np.abs( ham1-ham1_t )))
    print(np.amax(np.abs( ham2-ham2_t )))
    
    return

if __name__ == "__main__":
    np.set_printoptions(linewidth=750)
    
    params = {'t1':1., 't2':0.3, 'phi':np.pi/2., 'M':0. }
    N1, N2 = 1000, 1100
    
    gap = find_gaps(HaldaneHamiltonian, params, [N1,N2])
    mi.sprint('gap',gap)
    
    k1 = np.linspace(-0.5, 0.5, N1, endpoint=False)
    k2 = np.linspace(-0.5, 0.5, N2, endpoint=False)
    k = np.array(np.meshgrid(k1,k2, indexing='ij'))
    ham = HaldaneHamiltonian(k, params)
    
    evals, evecs = np.linalg.eigh(ham)
    
    U = link_variable(evecs)
    mi.sprint('U.shape', U.shape)
    
    C = Chern(U)
    mi.sprint('C', C)