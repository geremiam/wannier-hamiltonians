import numpy as np
from numpy import sqrt, cos, sin
import misc as mi
import matplotlib.pyplot as plt

import multiprocessing
import joblib

def find_gaps(evals):
    
    gaps_array = evals[..., 1:] - evals[..., :-1]
    
    gaps_min = np.amin(gaps_array, axis=tuple(range(gaps_array.ndim-1)))
    
    return np.squeeze(gaps_min)

def link_variable(evecs, verbose=False):
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

def Chern_fromlinkvar(linkvar, imagtol=1.e-14):
    
    # Only makes sense for two spatial dimensions
    assert linkvar.ndim==4
    assert linkvar.shape[0]==2
    
    factor1 = linkvar[0,...]
    factor2 = np.roll(linkvar[1,...], -1, axis=0)
    factor3 = np.roll(linkvar[0,...], -1, axis=1)
    factor4 = linkvar[1,...]
    
    F12 = np.log( factor1*factor2 / (factor3*factor4) ) / 1.j
    
    F12_imag = np.imag(F12)
    F12_imag_max = np.amax(np.abs(F12_imag))
    
    if F12_imag_max>imagtol:
        print('WARNING: In Chern_fromlinkvar(), imaginary parts were larger than tolerance. Largest is {}'.format(F12_imag_max))
    else:
        F12 = np.real(F12)
    
    C = np.sum( F12, axis=(0,1) ) / (2.*np.pi)
    
    return C

def Chern(evecs, imagtol=1.e-14, verbose=False):
    U = link_variable(evecs, verbose=verbose)
    C = Chern_fromlinkvar(U, imagtol=imagtol)
    return C

# #######################################################################################

def HaldaneHamiltonian(k, params, periodic=True):
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

def plot_Chern_Haldane(parallelize=False, plot=False):
    t1 = 1.
    t2 = 0.3
    
    N1 = 150
    N2 = N1+1
    k1 = np.linspace(-0.5, 0.5, N1, endpoint=False)
    k2 = np.linspace(-0.5, 0.5, N2, endpoint=False)
    k = np.array(np.meshgrid(k1,k2, indexing='ij'))
    
    N_M = 70
    N_phi = N_M+1
    M_array = t2 * np.linspace(-7., 7., N_M, endpoint=True)
    phi_array = np.linspace(-np.pi, np.pi, N_phi, endpoint=True)
    ################################################################################
    def foo(M, phi):
        params = {'t1':t1, 't2':t2}
        params['M'] = M
        params['phi'] = phi
        
        evals, evecs = np.linalg.eigh( HaldaneHamiltonian(k, params) )
        
        gaps = find_gaps(evals)
        Ch = Chern(evecs)
        
        return gaps, Ch
    
    if parallelize:
        num_cores = multiprocessing.cpu_count()
        mi.sprint('num_cores',num_cores)
        output = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(foo)(M, phi) for M in M_array for phi in phi_array)
        gap_array, Chern_array = zip(*output) # "Transpose" the output
        
        # Reshape the output
        gap_array = np.reshape(np.array(gap_array), (N_M, N_phi))
        Chern_array = np.reshape(np.array(Chern_array), (N_M, N_phi,2))
    
    else:
        gap_array = np.zeros(M_array.shape + phi_array.shape)
        Chern_array = np.zeros(M_array.shape + phi_array.shape + (2,))
        
        params = {'t1':t1, 't2':t2}
        
        for M_idx, M in enumerate(M_array):
            params['M'] = M
            for phi_idx, phi in enumerate(phi_array):
                params['phi'] = phi
                
                evals, evecs = np.linalg.eigh( HaldaneHamiltonian(k, params) )
                
                gap_array[M_idx, phi_idx] = find_gaps(evals)
                Chern_array[M_idx, phi_idx,:] = Chern(evecs)
    
    mi.sprint('gap_array.shape', gap_array.shape)
    mi.sprint('Chern_array.shape', Chern_array.shape)
    ################################################################################
    if plot:
        fig, ax = plt.subplots(1,3, sharex=True, sharey=True)
        ax[0].pcolormesh(phi_array, M_array, gap_array, shading='nearest', cmap='magma')
        ax[1].pcolormesh(phi_array, M_array, Chern_array[...,0], shading='nearest', cmap='coolwarm')
        ax[2].pcolormesh(phi_array, M_array, Chern_array[...,1], shading='nearest', cmap='coolwarm')
        plt.show()
    
    return gap_array, Chern_array

if __name__ == "__main__":
    np.set_printoptions(linewidth=750)
    
    plot_Chern_Haldane(parallelize=True, plot=True)