'''
Variety of mathematical or non-mathematical tools
'''
import numpy as np
import scipy.linalg

def findinarray(arr,val):
    ''' Finds all instances of "val" in a numpy array, returning a list of indices at 
    which it occurs. Only works for values that have exact representation. '''
    return [tuple(i) for i in np.argwhere(arr==val)]

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

def check_unitary(a, atol=1.e-13, elementwise=False):
    ''' Check that matrix is unitary. '''
    
    if elementwise:
        deviation = np.amax(np.abs(a @ np.conj(np.swapaxes(a, -1, -2)) - stackedidentity_like(a)), axis=(-2,-1))
        isunitary = deviation < atol
        retval = isunitary, deviation
    
    else:
        deviation = np.amax(np.abs(a @ np.conj(np.swapaxes(a, -1, -2)) - stackedidentity_like(a)))
        isunitary = deviation < atol
        retval = isunitary, deviation
    
    return retval

def eigvalsu(a, atol=1.e-13, shift_branch=False):
    ''' Eigenvalues of a unitary matrix in the form of phases.
    The eigenvalues of a unitary matrix are all on the unit circle. This functions returns 
    the eigenvalue phases theta, where e^{i theta}, in ascending order. By default, the 
    principle branch is used; if shift_branch, phases are shifted to [0, 2pi[.
    The function checks that a^dagger a = 1 within tolerance tol, elementwise. '''
    
    # Check that matrix is unitary.
    isunitary, deviation = check_unitary(a, atol=atol)
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

def eigu(a, atol=1.e-13, shift_branch=False):
    ''' Eigenvalues of a unitary matrix in the form of phases.
    The eigenvalues of a unitary matrix are all on the unit circle. This functions returns 
    the eigenvalue phases theta, where e^{i theta}, in ascending order. By default, the 
    principle branch is used; if shift_branch, phases are shifted to [0, 2pi[.
    The function checks that a^dagger a = 1 within tolerance tol, elementwise. '''
    
    # Check that matrix is unitary.
    isunitary, deviation = check_unitary(a, atol=atol)
    assert isunitary, 'Argument of eigvalsu() was expected to be unitary, but is not. Largest deviation is {}.'.format(deviation)
    
    evals, evecs = np.linalg.eig(a) # Evaluate eigenvalues using vectorized function
    
    # Unfortunately, this function is not guaranteed to return orthonormal eigenvectors,
    # and so may cause problems when there are (near) degeneracies. For normal matrices, 
    # Schur decomposition is equivalent to eigendecomposition and will yeild orthonormal 
    # eigenvectors, so for the matrices in "a" that don't yield orthonormal eigenvectors, 
    # we call the (non-vectorized) routine scipy.linalg.schur().
    
    # Check that the eigenvectors are orthonormal
    isunitary, deviation = check_unitary(evecs, atol=atol, elementwise=True)
    nonunitary_indices = findinarray(isunitary,False)
    
    if nonunitary_indices!=[]:
        print('Using Schur decomposition to find orthonormal eigenvectors for the matrices at the following indices:\n{}.'.format(nonunitary_indices))
        print('Deviations from eigenbases being unitary are\n{}'.format(deviation[isunitary==False]))
    
    for idx in nonunitary_indices:        
        idx1 = idx + (slice(None),) # For evals
        idx2 = idx + (slice(None), slice(None)) # For Hamiltonian and evecs
        
        T, Z = scipy.linalg.schur(a[idx2], output='complex')
        
        evals[idx1] = np.diag(T)
        evecs[idx2] = Z
    
    ################################################################################
    
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
    phases = np.take_along_axis(phases, sorted_inds,             axis=-1)
    evecs  = np.take_along_axis(evecs,  sorted_inds[...,None,:], axis=-1)
    
    # Check that we have a correct solution to the eigenproblem
    assert np.allclose(a @ evecs, np.exp(1j*phases)[...,None,:]*evecs, rtol=0., atol=atol), 'Incorrect eigenproblem solution'
    
    # Check that final eigenvectors are unitary
    isunitary, deviation = check_unitary(evecs, atol=atol)
    assert isunitary, 'evecs were expected to be orthonormal, but are not. Largest deviation is {}.'.format(deviation)
    
    return phases, evecs

def paraunitary_diag_evecs(hamiltonian, tau3, verbose=False):
    ''' No broadcasting. Uses Cholesky decomposition approach to get paraunitary 
    transformation. Only works if 'hamiltonian' is positive definite. '''
    
    assert np.allclose(hamiltonian, hamiltonian.T.conj(), rtol=1e-05, atol=1e-08), 'The hamiltonian should be Hermitian, but is not.'
    assert np.all(np.linalg.eigvalsh(hamiltonian)>0.), 'Error: matrix is not positive-definite: evals = {}'.format(np.linalg.eigvalsh(hamiltonian))
    
    K_ct = np.linalg.cholesky(hamiltonian) # Convention is contrary to that in paper
    K    = K_ct.conj().T
    W = K @ tau3 @ K_ct
    
    # Energies are the same as the eigenvalues of tau3 @ hamiltonian
    energies, W_evecs = np.linalg.eigh(W) # eigh gives orthonormal eigenvectors
    
    assert (hamiltonian.shape[-1]%2==0), 'Dimension of Hamiltonian is expected to be even, but is not.'
    n = hamiltonian.shape[-1] // 2 # Get (half-)size of hamiltonian
    
    assert np.all(energies[..., :n] < 0.), 'Expected the first half of the energies to be negative; was not.'
    assert np.all(energies[..., n:] > 0.), 'Expected the second half of the energies to be positive; was not.'
    
    T = np.linalg.solve(K, W_evecs * np.sqrt(np.abs(energies))[...,None,:])
    
    assert np.allclose(tau3 @ hamiltonian @ T, T @ np.diag(energies), rtol=1e-05, atol=1e-08), 'Eigendecomposition is incorrect. (A)'
    
    # Energies are purely real because 'hamiltonian' is positive definite
    energies = np.real(energies)
    
    # Use energies to get sorted indices
    sorted_inds = np.argsort(energies, axis=-1)
    if verbose: mt.smartprint('sorted_inds',sorted_inds)
    
    # Use the sorted indices to sort evals and evecs
    energies = np.take_along_axis(energies, sorted_inds,         axis=-1)
    T        = np.take_along_axis(T,        sorted_inds[None,:], axis=-1)
    
    assert np.allclose(tau3 @ hamiltonian @ T, T @ np.diag(energies), rtol=1e-05, atol=1e-08), 'Eigendecomposition is incorrect. (B)'
    
    return energies, T

def polardecomp(mat):
    ''' A broadcasting polar decomposition using SVD. Returns U and P s.t. mat = U @ P.'''
    
    # Decomposition mat = (u * s[...,None,:]) @ vh
    u, s, vh = np.linalg.svd(mat)
    v = np.conj(np.swapaxes(vh, -1, -2))
    
    # Get polar decomposition from SVD
    P = (v * s[...,None,:]) @ vh
    U = u @ vh
    
    return U, P

if __name__ == "__main__":
    np.set_printoptions(linewidth=750)
    
    a1 = np.reshape(np.arange(4*4),(4,4))*np.array([1,1,1,1]) - 0.2j*np.reshape(np.arange(4*4),(4,4))
    a2 = np.reshape(np.arange(4*4),(4,4))*np.array([1,1,1,-1]) - 0.2j*np.reshape(np.arange(4*4),(4,4))
    a3 = np.reshape(np.arange(4*4),(4,4))*np.array([1,1,-1,-1]) - 0.2j*np.reshape(np.arange(4*4),(4,4))
    a4 = np.reshape(np.arange(4*4),(4,4))*np.array([1,-1,-1,-1]) - 0.2j*np.reshape(np.arange(4*4),(4,4))
    
    U1, P1 = polardecomp(a1)
    U2, P2 = polardecomp(a2)
    U3, P3 = polardecomp(a3)
    U4, P4 = polardecomp(a4)
    
    U = np.array([[U1,U2],[U3,U4]])
    
    phases, evecs = eigu(U)
    
