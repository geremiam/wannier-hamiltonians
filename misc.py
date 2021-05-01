import numpy as np

def sprint(string, val, nl=False):
    if nl:
        print(string + ' =\n{}'.format(val))
    else:
        print(string + ' = {}'.format(val))

def stackedidentity_like(a):
    ''' Given input a of shape (..., N, N), returns a stack of shape (..., N, N) of 
    identity matrices in the last two dimensions.
    '''
    a_shape = a.shape
    assert a_shape[-1]==a_shape[-2], 'Last two dimensions were expected to be equal in length, but are not.'
    
    output = np.zeros_like(a)
    # Weird advanced indexing stuff
    # https://stackoverflow.com/questions/59897165/create-identity-matrices-with-arbitrary-shape-with-numpy
    I = np.arange(a_shape[-1])
    output[...,I,I] = 1
    
    return output

def eigvalsu(a, atol=1.e-13, shift_branch=False):
    ''' Eigenvalues of a unitary matrix in the form of phases.
    The eigenvalues of a unitary matrix are all on the unit circle. This functions returns 
    the eigenvalue phases theta, where e^{i theta}, in ascending order. By default, the 
    principle branch is used; if shift_branch, phases are shifted to [0, 2pi[.
    The function checks that a^dagger a = 1 within tolerance tol, elementwise. '''
    
    # Check that matrix is unitary.
    deviation = np.amax(np.abs(a @ np.conj(np.swapaxes(a, -1, -2)) - stackedidentity_like(a)))
    isunitary = deviation < atol
    assert isunitary, 'Argument of eigvalsu() was expected to be unitary, but is not. Largest deviation is {}.'.format(deviation)
    
    evals = np.linalg.eigvals(a) # Evaluate eigenvalues
    
    phases = np.log(evals) / 1j # Get argument of exponential
    
    if shift_branch:
        phases += 2.*np.pi*(np.real(phases)<0.)
    
    # Check that imaginary parts are zero
    phases_maximag = np.amax(np.abs(np.imag(phases)))
    assert phases_maximag < atol, 'Phases have large imaginary parts: phases_maximag = {}'.format(phases_maximag)
    
    phases = np.real(phases) # Keep only real part
    
    # Use energies to get sorted indices
    sorted_inds = np.argsort(phases, axis=-1)
    
    # Use the sorted indices to sort evals and evecs
    phases = np.take_along_axis(phases, sorted_inds,         axis=-1)
    #T        = np.take_along_axis(T,        sorted_inds[None,:], axis=-1)
    
    return phases

def polardecomp(mat):
    ''' A broadcasting polar decomposition using SVD. Returns U and P s.t. mat = U @ P.'''
    
    # Decomposition mat = (u * s[...,None,:]) @ vh
    u, s, vh = np.linalg.svd(mat)
    v = np.conj(np.swapaxes(vh, -1, -2))
    
    # Get polar decomposition from SVD
    P = (v * s[...,None,:]) @ vh
    U = u @ vh
    
    return U, P