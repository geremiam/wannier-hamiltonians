import numpy as np
import misc as mi


def link_variable(evecs, metric=None, verbose=False):
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
        print('Nbands = {}'.format(Nbands))
        print('D = {}'.format(D))
    
    # Create return array
    U = np.zeros((D,) + k_shape + (Nbands,), complex)
    
    for d in range(D):
        if metric is None:
            U_temp = np.sum( evecs.conj() * np.roll(evecs, -1, axis=d), axis=-2 )
        else:
            norms = np.sum( evecs.conj() * (metric @ evecs), axis=-2 )
            U_temp = np.sum( evecs.conj() * np.roll(evecs, -1, axis=d), axis=-2 ) / norms
        
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

# #######################################################################################

def Chern(evecs, metric=None, imagtol=1.e-14, verbose=False):
    U = link_variable(evecs, metric=metric, verbose=verbose)
    C = Chern_fromlinkvar(U, imagtol=imagtol)
    return C
