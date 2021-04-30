import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def smartprint(string, val, nl=False):
    if nl:
        print(string + ' =\n{}'.format(val))
    else:
        print(string + ' = {}'.format(val))

def polardecomp(mat):
    ''' A broadcasting polar decomposition using SVD. Returns U and P s.t. mat = U @ P.'''
    
    # Decomposition mat = (u * s[...,None,:]) @ vh
    u, s, vh = np.linalg.svd(mat)
    v = np.conj(np.swapaxes(vh, -1, -2))
    
    # Get polar decomposition from SVD
    P = (v * s[...,None,:]) @ vh
    U = u @ vh
    
    return U, P

def hamSSH(k, params_dict):
    ''' Return the Bloch hamiltonian. Broadcasts in k. '''
    
    t1 = params_dict['t1']
    t2 = params_dict['t2']
    m  = params_dict['m']
    e0 = params_dict['e0']
    
    k = np.atleast_1d(k)
    k_shape = k.shape
    mat = np.zeros(k_shape + (3,3), complex)
    
    mat[...,0,1] += t1 + t2*np.exp(+1.j*k)
    mat[...,1,0] += t1 + t2*np.exp(-1.j*k)
    mat[...,0,0] +=   m/2.
    mat[...,1,1] += - m/2.
    mat[...,2,2] += e0
    
    return mat

def Wilson_line_elements(evecs_occ, unitary=True):
    ''' Calculate Wilson line elements between adjacent momentum points using evecs_occ. '''
    T = evecs_occ
    Tdagger = np.conj(np.swapaxes(evecs_occ, -1, -2))
    
    F = np.roll(Tdagger, -1, 0) @ T
    
    if unitary:
        U, P = polardecomp(F)
        F = U
    
    return F

def Wilson_loop(F, basepoint):
    ''' Calculates the (large) Wilson loop, starting from the index "basepoint". First
    dimension of "F" is momentum, second and third are matrix dimensions. '''
    F = np.roll(F, -basepoint, 0) # Rolls the basepoint to the start of the array
    
    W = np.eye(F.shape[-1]) # Identity matrix of the same size as eigenvectors
    
    for val in F: # Multiply the 
        W = val @ W
    
    return W

def compute_pol(hamfunc, params_dict, filled_bands, N, ret_evals=False):
    
    k = np.linspace(0., 2.*np.pi, N, endpoint=False)
    ham = hamfunc(k, params_dict)
    
    evals, evecs = np.linalg.eigh(ham)
    
    evecs_occ = evecs[...,filled_bands]
    
    F = Wilson_line_elements(evecs_occ, unitary=True)
    
    basepoint = 0
    W = Wilson_loop(F, basepoint)
    
    p = np.log(np.linalg.det(W)) / (2.j*np.pi)
    
    if np.abs(np.imag(p))>1.e-14:
        print('WARNING: polarization has large imaginary part. p = {}'.format(p))
    else:
        p = np.real(p)
    
    if ret_evals:
        ret = p, k, evals
    else:
        ret = p
    
    return ret

def Wannier_centers(hamfunc, params_dict, filled_bands, N):
    
    k = np.linspace(0., 2.*np.pi, N, endpoint=False)
    ham = hamfunc(k, params_dict)
    
    evals, evecs = np.linalg.eigh(ham)
    
    evecs_occ = evecs[...,filled_bands]
    
    F = Wilson_line_elements(evecs_occ, unitary=True)
    
    basepoint = 0
    W = Wilson_loop(F, basepoint)
    
    centers = np.log(np.linalg.eigvals(W)) / (2.j*np.pi)
    
    if np.abs(np.amax(np.imag(centers)))>1.e-14:
        print('WARNING: centers have large imaginary part. centers = {}'.format(centers))
    else:
        centers = np.real(centers)
    
    return centers

def compare_polarizations():
    ''' Compare polarizations from different bands and with different masses. '''
    np.set_printoptions(linewidth=750)
    
    params_dict = {'t1':0.99, 
                   't2':1., 
                   'm':0.,
                   'e0':0.}
    
    N = 100
#     filled_bands = [0,1]
    
    m_array = np.array([-0.1, -0.001, 0., 0.001, 0.1])
    smartprint('m_array', m_array)
    t1_array = np.linspace(-1.5, 1.5, 201, endpoint=True)
    smartprint('t1_array', t1_array)
    
    p_array  = np.zeros(t1_array.shape + m_array.shape + (3,))
    
    for band_idx, band in enumerate([0,1,2]):
        filled_bands = [band]
        for m_idx, m in enumerate(m_array):
            for t1_idx, t1 in enumerate(t1_array):
                
                params_dict['m'] = m
                params_dict['t1'] = t1
                
                p_array[t1_idx, m_idx, band_idx] = compute_pol(hamSSH, params_dict, filled_bands, N)
    
    # Plot p as a function of parameters
    fig, ax = plt.subplots()
    ax.plot(t1_array, np.sum(p_array, axis=-1))
#     ax.plot(t1_array, p_array[...,1], ls='--')
#     ax.plot(t1_array, p_array[...,2], ls=':')
    plt.show()

def plot_polarizations():
    np.set_printoptions(linewidth=750)
    
    params_dict = {'t1':0.99, 
                   't2':1., 
                   'm':0.,
                   'e0':0.}
    
    N = 100
    filled_bands = [0,1]
    
    centers = Wannier_centers(hamSSH, params_dict, filled_bands, N)
    smartprint('centers', centers)
    
    # Plot the band structure
#     p, k, evals = compute_pol(hamSSH, params_dict, filled_bands, N, ret_evals=True)
#     fig, ax = plt.subplots()
#     ax.plot(k, evals)
#     plt.show()
    
    m_array = np.array([-0.1, -0.001, 0., 0.001, 0.1])
    smartprint('m_array', m_array)
    t1_array = np.linspace(-1.5, 1.5, 201, endpoint=True)
    smartprint('t1_array', t1_array)
    
    p_array  = np.zeros(t1_array.shape + m_array.shape)
    
    for m_idx, m in enumerate(m_array):
        for t1_idx, t1 in enumerate(t1_array):
            
            params_dict['m'] = m
            params_dict['t1'] = t1
            
            p_array[t1_idx, m_idx] = compute_pol(hamSSH, params_dict, filled_bands, N)
    
    # Plot p as a function of parameters
    fig, ax = plt.subplots()
    ax.plot(t1_array, p_array)
    plt.show()

if __name__ == "__main__":
    compare_polarizations()