'''
Variety of mathematical or non-mathematical tools
'''
import numpy as np
import scipy.linalg

def savearray(f, x):
    ''' Saves a multidimensional numpy array "x" to a file "f". '''
    if len(x.shape)==0:
        np.savetxt(f, x)
    elif len(x.shape)==1:
        np.savetxt(f, np.array([x]))
    elif len(x.shape)==2:
        np.savetxt(f, x)
    else:
        xshape = x.shape
        # Array with the same shape as x, but without the last two dimensions. 
        # Only used to iterate over
        aux_array = np.zeros(xshape[:-2])
        
        previndex = tuple(np.zeros(len(aux_array.shape), int)) # Zero tuple of correct length
        
        for index, val in np.ndenumerate(aux_array):
            diff = np.array(index, int) - np.array(previndex,int) # Incrementation in index
            Nnewlines = np.sum(diff!=0)
            for i in range(Nnewlines):
                f.write('\n')
            
            fullindex = index + (slice(None),slice(None)) # Tuple for slicing x
            np.savetxt(f, x[fullindex]) # Write 2*2 slice of x to file
            
            previndex = index
    return

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

def generate_posdef(n, verbose=False):
  rng = np.random.default_rng()
  L = rng.uniform(low=-1., high=1., size=(n,n)) + 1.j*rng.uniform(low=-1., high=1., size=(n,n))
  
  posdef = L @ L.T.conj()
  
  if verbose:
      print('det(posdef) = {}'.format(np.linalg.det(posdef)))
      print('eigvalsh(posdef) = {}'.format(np.linalg.eigvalsh(posdef)))
  
  assert np.all(np.linalg.eigvalsh(posdef)>0.), 'Error: matrix is not positive-definite: evals = {}'.format(np.linalg.eigvalsh(posdef))
  
  return posdef

def KP(list):
    ''' Performs a Kronecker product of the arrays in "list". '''
    output = np.kron(list[0], list[1])
    for arr in list[2:]:
        output = np.kron(output, arr)
    
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
    ''' Uses Cholesky decomposition approach to get paraunitary 
    transformation. Only works if 'hamiltonian' is positive definite. Energies (and 
    states) are ordered from lowest to highest. '''
    
    assert np.allclose(hamiltonian, np.swapaxes(hamiltonian.conj(),-1,-2), rtol=1e-05, atol=1e-08), 'The hamiltonian should be Hermitian, but is not.'
    assert np.all(np.linalg.eigvalsh(hamiltonian)>0.), 'Error: matrix is not positive-definite: evals = {}'.format(np.linalg.eigvalsh(hamiltonian))
    
    K_ct = np.linalg.cholesky(hamiltonian) # Convention is contrary to that in paper
    K    = np.swapaxes( np.conj(K_ct), -1, -2 )
    W = K @ tau3 @ K_ct
    
    # Energies are the same as the eigenvalues of tau3 @ hamiltonian
    energies, W_evecs = np.linalg.eigh(W) # eigh gives orthonormal eigenvectors
    
    assert (hamiltonian.shape[-1]%2==0), 'Dimension of Hamiltonian is expected to be even, but is not.'
    n = hamiltonian.shape[-1] // 2 # Get (half-)size of hamiltonian
    
    assert np.all(energies[..., :n] < 0.), 'Expected the first half of the energies to be negative; was not.'
    assert np.all(energies[..., n:] > 0.), 'Expected the second half of the energies to be positive; was not.'
    
    T = np.linalg.solve(K, W_evecs * np.sqrt(np.abs(energies))[...,None,:])
    
    assert np.allclose(tau3 @ hamiltonian @ T, T * energies[...,None,:], rtol=1e-05, atol=1e-08), 'Eigendecomposition is incorrect. (A)'
    
    # Energies are purely real because 'hamiltonian' is positive definite
    energies = np.real(energies)
    
    # Use energies to get sorted indices
    sorted_inds = np.argsort(energies, axis=-1)
    if verbose: mt.smartprint('sorted_inds',sorted_inds)
    
    # Use the sorted indices to sort evals and evecs
    energies = np.take_along_axis(energies, sorted_inds,             axis=-1)
    T        = np.take_along_axis(T,        sorted_inds[...,None,:], axis=-1)
    
    assert np.allclose(tau3 @ hamiltonian @ T, T * energies[...,None,:], rtol=1e-05, atol=1e-08), 'Eigendecomposition is incorrect. (B)'
    
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
    
    n = 2
    N1 = 3
    N2 = 5
    
    mat = np.zeros([N1,N2,2*n,2*n], complex)
    
    for i1 in range(N1):
      for i2 in range(N2):
        mat[i1,i2,:,:] = generate_posdef(2*n)
    
    sprint('mat.shape', mat.shape)
    sprint('evals', np.linalg.eigvalsh(mat))
    
    tau3 = np.kron(np.array([[-1.,0.],[0.,1.]]), np.eye(n))
    sprint('tau3',tau3)
    
    energies, T = paraunitary_diag_evecs(mat, tau3)
    
    sprint('energies', energies)
    
    print(np.amax(np.abs(np.swapaxes(T.conj(),-1,-2) @ tau3 @ T - tau3)))