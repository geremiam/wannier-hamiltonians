import numpy as np
from numpy import sin, cos, pi
import scipy.linalg

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import misc as m

def hamBBH(k, params_dict):
    ''' Return the Bloch hamiltonian. Broadcasts in k. 
    First dimension of k are the kx,ky components.
    '''
    # Unpack arguments
    gamma_x = params_dict['gamma_x']
    gamma_y = params_dict['gamma_y']
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
    mat += gamma_x * Gamma4 
    mat += lambd * ( cos(Kx) * Gamma4 + sin(Kx) * Gamma3 )
    # y-direction terms
    mat += gamma_y * Gamma2
    mat += lambd * ( cos(Ky) * Gamma2 + sin(Ky) * Gamma1 )
    # On-site mass term
    mat += delta * Gamma0
    
    return mat

def Wilson_line_elements(evecs_occ, unitary=True, verbose=True):
    ''' Calculate Wilson line elements between adjacent momentum points using evecs_occ. '''
    
    # Last two dimensions are orbitals and bands, previous ones are momentum
    Nocc = evecs_occ.shape[-1]
    k_shape = evecs_occ.shape[:-2]
    D = len(k_shape)
    
    if verbose:
        m.sprint('Nocc',Nocc)
        m.sprint('D',D)
    
    # Create return array
    F = np.zeros((D,) + k_shape + (Nocc,Nocc), complex)
    
    # Aliases for convenience
    T = evecs_occ
    Tdagger = np.conj(np.swapaxes(evecs_occ, -1, -2))
    
    for d in range(D):
        F_temp = np.roll(Tdagger, -1, axis=d) @ T
        
        if unitary:
            U_temp, P_temp = m.polardecomp(F_temp)
            F_temp = U_temp
        
        F[d,...] = F_temp
    
    return F

def Wilson_loop_directional(Fdirectional, axis, basepoint, atol=1.e-13):
    ''' Calculates the (large) Wilson loop, starting from the index "basepoint". Last two
    dimensions of "Fdirectional" are matrix dimensions, others are momentum axes. "axis" 
    specifies direction of the Wilson line elements in "Fdirectional".
    Output W has one less momentum axis.
    '''
    
    axis_len = Fdirectional.shape[axis] # Get length of axis to multiply over
    
    if basepoint=='all': # In this case, every possible basepoint is calculated
        
        Wlist = [] # List to temporarily store the W's for the different basepoints in the "axis" direction
        
        axis_indices = np.arange(axis_len) # Array of indices along the "axis" direction
        
        # Loop over indices in "axis" direction, taking them as a basepoint one by one
        for bpoint_loc in axis_indices:
            # Reordered array of indices, starting with "bpoint_loc"
            axis_indices_reordered = np.roll(axis_indices, -bpoint_loc)
            
            # Define initial value of W
            Wdir_temp = np.take(Fdirectional, axis_indices_reordered[0], axis=axis)
            
            for i in axis_indices_reordered[1:]: # START LOOP AT SECOND VALUE
                Wdir_temp = np.take(Fdirectional, i, axis=axis) @ Wdir_temp # Left-multiply throughout entire axis
            
            Wlist.append(Wdir_temp) # Append the value for the current basepoint to the list
        
        Wdirectional = np.stack(Wlist, axis=axis) # Turn the list into properly stacked array
    
    else:
        
        # Roll the basepoint to the start of the array for convenience
        Fdirectional = np.roll(Fdirectional, -basepoint, axis=axis)
        
        Wdirectional = np.take(Fdirectional, 0, axis=axis) # Define initial value of W
        
        for i in range(1,axis_len): # START LOOP AT SECOND VALUE
            Wdirectional = np.take(Fdirectional, i, axis=axis) @ Wdirectional # Left-multiply throughout entire axis
        
        # Check that W matrices are unitary
        deviation = np.amax(np.abs(Wdirectional @ np.conj(np.swapaxes(Wdirectional, -1, -2)) - m.stackedidentity_like(Wdirectional)))
        isunitary = deviation < atol
        if not isunitary:
            print('WARNING: Wilson_loop_directional did not ouput a unitary matrix. Largest deviation is {}.'.format(deviation))
    
    return Wdirectional

def Wilson_loops(F, basepoints):
    
    if basepoints=='all':
        Wilsonloops_list = []
        
        for idx_axis, Fdirectional in enumerate(F):
            W = Wilson_loop_directional(Fdirectional, idx_axis, 'all')
            Wilsonloops_list.append(W)
        
        retval = np.array( Wilsonloops_list )
    
    else:
        Wilsonloops_list = []
        
        for idx_axis, Fdirectional in enumerate(F):
            W = Wilson_loop_directional(Fdirectional, idx_axis, basepoints[idx_axis])
            Wilsonloops_list.append(W)
        
        retval = Wilsonloops_list
    
    return retval

def Wannier_states(evecs_occ, Wilsonloops):
    
    phases, evecs_Wil = m.eigu(Wilsonloops)
    m.sprint('phases.shape', phases.shape)
    m.sprint('evecs_occ.shape', evecs_occ.shape)
    m.sprint('evecs_Wil.shape', evecs_Wil.shape)
    
    # Perform sum-product over band index n (filled bands only)
    Wannierstates = np.sum( evecs_occ[...,None]*evecs_Wil[...,None,:,:], axis=-2 )
    m.sprint('Wannierstates.shape', Wannierstates.shape)
    
    return Wannierstates

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

def plot_bandstructure():
    params_dict = {'gamma_x':0.5, 
                   'gamma_y':0.5, 
                   'lambd':1., 
                   'delta':0.}
    
    Nx, Ny = 50, 50
    kx = np.linspace(0., 2.*pi, Nx, endpoint=False)
    ky = np.linspace(0., 2.*pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    m.sprint('k.shape', k.shape)
    
    H_array = hamBBH(k, params_dict)
    m.sprint('H_array.shape', H_array.shape)
    evals = np.linalg.eigvalsh(H_array)
    m.sprint('evals.shape', evals.shape)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    for n in range(evals.shape[-1]):
        surf = ax.plot_surface(k[0], k[1], evals[...,n], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

def calculation_Wilsonloops():
    params_dict = {'gamma_x':0.5, 
                   'gamma_y':1.5, 
                   'lambd':1., 
                   'delta':0.}
    
    Nx, Ny = 800, 800
    kx = np.linspace(-pi, pi, Nx, endpoint=False)
    ky = np.linspace(-pi, pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    m.sprint('k.shape', k.shape)
    
    H_array = hamBBH(k, params_dict)
    m.sprint('H_array.shape', H_array.shape)
    evals, evecs = np.linalg.eigh(H_array)
    
    filled_bands = [0,1]
    m.sprint('filled_bands',filled_bands)
    evecs_occ = evecs[...,filled_bands]
    
    F = Wilson_line_elements(evecs_occ, unitary=True)
    m.sprint('F.shape', F.shape)
    
    basepoints = [0,0]
    Wilsonloops_list = Wilson_loops(F, basepoints)
    
    Wx = Wilsonloops_list[0]
    m.sprint('Wx.shape',Wx.shape)
    Wannierbands_x = m.eigvalsu(Wx, shift_branch=True) / (2.*pi)
    m.sprint('Wannierbands_x.shape',Wannierbands_x.shape)
    
    Wy = Wilsonloops_list[1]
    m.sprint('Wy.shape',Wy.shape)
    Wannierbands_y = m.eigvalsu(Wy, shift_branch=True) / (2.*pi)
    m.sprint('Wannierbands_y.shape',Wannierbands_y.shape)
    
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
    
    params_dict = {'gamma_x':0.5, 
                   'gamma_y':1., 
                   'lambd':1., 
                   'delta':0.}
    
    Nx, Ny = 10, 12
    kx = np.linspace(-pi, pi, Nx, endpoint=False)
    ky = np.linspace(-pi, pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    m.sprint('k.shape', k.shape)
    
    H_array = hamBBH(k, params_dict)
    m.sprint('H_array.shape', H_array.shape)
    evals, evecs = np.linalg.eigh(H_array)
    
    filled_bands = [0,1]
    m.sprint('filled_bands',filled_bands)
    evecs_occ = evecs[...,filled_bands]
    
    F = Wilson_line_elements(evecs_occ, unitary=True)
    m.sprint('F.shape', F.shape)
    
    Wilsonloops = Wilson_loops(F, 'all')
    m.sprint('Wilsonloops.shape', Wilsonloops.shape)
    
    Wannierstates = Wannier_states(evecs_occ, Wilsonloops)
    
    
    Wannier_sector = [0]
    m.sprint('Wannier_sector',Wannier_sector)
    Wannierstates_sector = Wannierstates[...,Wannier_sector]
    
    F_Wannier = Wilson_line_elements(Wannierstates_sector, unitary=True)

