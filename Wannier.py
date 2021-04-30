import numpy as np
from numpy import sin, cos, pi
import scipy.linalg

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def smartprint(string, val, nl=False):
    if nl:
        print(string + ' =\n{}'.format(val))
    else:
        print(string + ' = {}'.format(val))

def stackedidentity_like(a):
    a_shape = a.shape
    assert a_shape[-1]==a_shape[-2], 'Last two dimensions were expected to be equal in length, but are not.'
    
    output = np.zeros_like(a)
    # Weird advanced indexing stuff
    # https://stackoverflow.com/questions/59897165/create-identity-matrices-with-arbitrary-shape-with-numpy
    I = np.arange(a_shape[-1])
    output[...,I,I] = 1
    
    return output

def eigvalsu(a, atol=1.e-13, shift_branch=False):
    ''' Eigenvalues of a unitary matrix in the form of a phase.
    The eigenvalues of a unitary matrix are all on the unit circle. This functions the 
    eigenvalue phases theta, where e^{i theta}, with the principle branch and in 
    ascending order. '''
    # Check that matrix is unitary.
    deviation = np.amax(np.abs(a @ np.conj(np.swapaxes(a, -1, -2)) - stackedidentity_like(a)))
    isunitary = deviation < atol
    assert isunitary, 'Argument of eigvalsu() was expected to be unitary, but is not. Largest deviation is {}.'.format(deviation)
    
    evals = np.linalg.eigvals(a) # Evaluate eigenvalues
    
    phases = np.log(evals) / 1j # Get argument of exponential
    
    if shift_branch:
        phases += 2.*pi*(np.real(phases)<0.)
    
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

def hamBBH(k, params_dict):
    ''' Return the Bloch hamiltonian. Broadcasts in k. 
    First dimension of k are the kx,ky components.
    '''
    # Unpack arguments
    gamma = params_dict['gamma']
    lambd = params_dict['lambd']
    delta = params_dict['delta']
    
    # Convenience aliases for Pauli matrices
    pauli0 = np.eye(2)
    pauli1 = np.array([[0,1], 
                       [1,0]])
    pauli2 = np.array([[0,-1j], 
                      [1j,0]])
    pauli3 = np.array([[1,0], 
                       [0,-1]])
    
    # Matrices that enter into hamiltonian (4 by 4)
    Gamma0 =   np.kron(pauli3, pauli0)
    Gamma1 = - np.kron(pauli2, pauli1)
    Gamma2 = - np.kron(pauli2, pauli2)
    Gamma3 = - np.kron(pauli2, pauli3)
    Gamma4 =   np.kron(pauli1, pauli0)
    
    Kx, Ky = k # Split momentum into its components
    Kx = Kx[...,None,None] # Add length-one axes to broadcast against orbital axes
    Ky = Ky[...,None,None]
    
    k_shape = k.shape[1:] # Shape of momentum arrays (excluding kx,ky components)
    mat = np.zeros(k_shape + (4,4), complex)
    
    # x-direction terms
    mat += gamma * Gamma4 
    mat += lambd * ( cos(Kx) * Gamma4 + sin(Kx) * Gamma3 )
    # y-direction terms
    mat += gamma * Gamma2
    mat += lambd * ( cos(Ky) * Gamma2 + sin(Ky) * Gamma1 )
    # On-site mass term
    mat += delta * Gamma0
    
    return mat

def Wilson_line_elements2D(evecs_occ, unitary=True):
    ''' Calculate Wilson line elements between adjacent momentum points using evecs_occ. '''
    
    assert len(evecs_occ.shape)==4
    
    T = evecs_occ
    Tdagger = np.conj(np.swapaxes(evecs_occ, -1, -2))
    
    Fx = np.roll(Tdagger, -1, 0) @ T
    Fy = np.roll(Tdagger, -1, 1) @ T
    
    if unitary:
        Ux, Px = polardecomp(Fx)
        Fx = Ux
        
        Uy, Py = polardecomp(Fy)
        Fy = Uy
    
    return np.array([Fx, Fy])

def Wilson_line_elements(evecs_occ, unitary=True, verbose=True):
    ''' Calculate Wilson line elements between adjacent momentum points using evecs_occ. '''
    
    # Last two dimensions are orbitals and bands, previous ones are momentum
    Nocc = evecs_occ.shape[-1]
    k_shape = evecs_occ.shape[:-2]
    D = len(k_shape)
    
    if verbose:
        smartprint('Nocc',Nocc)
        smartprint('D',D)
    
    # Create return array
    F = np.zeros((D,) + k_shape + (Nocc,Nocc), complex)
    
    # Aliases for convenience
    T = evecs_occ
    Tdagger = np.conj(np.swapaxes(evecs_occ, -1, -2))
    
    for d in range(D):
        F_temp = np.roll(Tdagger, -1, d) @ T
        
        if unitary:
            U_temp, P_temp = polardecomp(F_temp)
            F_temp = U_temp
        
        F[d,...] = F_temp
    
    return F

def Wilson_loop_directional(Fdirectional, axis, basepoint, atol=1.e-13):
    ''' Calculates the (large) Wilson loop, starting from the index "basepoint". Last two
    dimensions of "F" are matrix dimensions, others are momentum axes. 
    Output W has one less momentum axis.
    '''
    
    axis_len = Fdirectional.shape[axis] # Get length of axis to multiply over
    
    # Roll the basepoint to the start of the array for convenience
    Fdirectional = np.roll(Fdirectional, -basepoint, axis)
    
    Wdirectional = np.take(Fdirectional, 0, axis=axis) # Define initial value of W
    
    for i in range(1,axis_len): # START LOOP AT SECOND VALUE
        Wdirectional = np.take(Fdirectional, i, axis=axis) @ Wdirectional # Left-multiply throughout entire axis
    
    # Check that W matrices are unitary
    deviation = np.amax(np.abs(Wdirectional @ np.conj(np.swapaxes(Wdirectional, -1, -2)) - stackedidentity_like(Wdirectional)))
    isunitary = deviation < atol
    if not isunitary:
        print('WARNING: Wilson_loop_directional did not ouput a unitary matrix. Largest deviation is {}.'.format(deviation))
    
    return Wdirectional

def Wilson_loops(F, basepoints):
    
    Wilsonloops_list = []
    
    for idx_axis, Fdirectional in enumerate(F):
        W = Wilson_loop_directional(Fdirectional, idx_axis, basepoints[idx_axis])
        Wilsonloops_list.append(W)
    
    return Wilsonloops_list

# def compute_pol(hamfunc, params_dict, filled_bands, N, ret_evals=False):
#     
#     k = np.linspace(0., 2.*pi, N, endpoint=False)
#     ham = hamfunc(k, params_dict)
#     
#     evals, evecs = np.linalg.eigh(ham)
#     
#     evecs_occ = evecs[...,filled_bands]
#     
#     F = Wilson_line_elements(evecs_occ, unitary=True)
#     
#     basepoint = 0
#     W = Wilson_loop(F, basepoint)
#     
#     p = np.log(np.linalg.det(W)) / (2.j*pi)
#     
#     if np.abs(np.imag(p))>1.e-14:
#         print('WARNING: polarization has large imaginary part. p = {}'.format(p))
#     else:
#         p = np.real(p)
#     
#     if ret_evals:
#         ret = p, k, evals
#     else:
#         ret = p
#     
#     return ret

# def Wannier_centers(hamfunc, params_dict, filled_bands, N):
#     
#     k = np.linspace(0., 2.*pi, N, endpoint=False)
#     ham = hamfunc(k, params_dict)
#     
#     evals, evecs = np.linalg.eigh(ham)
#     
#     evecs_occ = evecs[...,filled_bands]
#     
#     F = Wilson_line_elements(evecs_occ, unitary=True)
#     
#     basepoint = 0
#     W = Wilson_loop(F, basepoint)
#     
#     centers = np.log(np.linalg.eigvals(W)) / (2.j*pi)
#     
#     if np.abs(np.amax(np.imag(centers)))>1.e-14:
#         print('WARNING: centers have large imaginary part. centers = {}'.format(centers))
#     else:
#         centers = np.real(centers)
#     
#     return centers

def plot_bandstructure():
    params_dict = {'gamma':0.5, 
                   'lambd':1., 
                   'delta':0.}
    
    Nx, Ny = 50, 50
    kx = np.linspace(0., 2.*pi, Nx, endpoint=False)
    ky = np.linspace(0., 2.*pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    smartprint('k.shape', k.shape)
    
    H_array = hamBBH(k, params_dict)
    smartprint('H_array.shape', H_array.shape)
    evals = np.linalg.eigvalsh(H_array)
    smartprint('evals.shape', evals.shape)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    for n in range(evals.shape[-1]):
        surf = ax.plot_surface(k[0], k[1], evals[...,n], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

def calculation_Wilsonloops():
    params_dict = {'gamma':1.1, 
                   'lambd':1., 
                   'delta':0.}
    
    Nx, Ny = 800, 800
    kx = np.linspace(-pi, pi, Nx, endpoint=False)
    ky = np.linspace(-pi, pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    smartprint('k.shape', k.shape)
    
    H_array = hamBBH(k, params_dict)
    smartprint('H_array.shape', H_array.shape)
    evals, evecs = np.linalg.eigh(H_array)
    
    filled_bands = [0,1]
    smartprint('filled_bands',filled_bands)
    evecs_occ = evecs[...,filled_bands]
    
    F = Wilson_line_elements(evecs_occ, unitary=True)
    smartprint('F.shape', F.shape)
    
    basepoints = [0,0]
    Wilsonloops_list = Wilson_loops(F, basepoints)
    
    Wx = Wilsonloops_list[0]
    smartprint('Wx.shape',Wx.shape)
    Wannierbands_x = eigvalsu(Wx, shift_branch=True) / (2.*pi)
    smartprint('Wannierbands_x.shape',Wannierbands_x.shape)
    
    Wy = Wilsonloops_list[1]
    smartprint('Wy.shape',Wy.shape)
    Wannierbands_y = eigvalsu(Wy, shift_branch=True) / (2.*pi)
    smartprint('Wannierbands_y.shape',Wannierbands_y.shape)
    
    fig, ax = plt.subplots(1,2)
    ax[0].plot(ky, Wannierbands_x)
    ax[0].set_xlabel(r'$k_y$')
    ax[1].plot(kx, Wannierbands_y)
    ax[1].set_xlabel(r'$k_x$')
    for axis in ax.flatten():
        axis.axhline(0, color='C7')
        axis.axhline(1, color='C7')
    plt.show()
    
    return

if __name__ == "__main__":
    np.set_printoptions(linewidth=750)
    
    calculation_Wilsonloops()
