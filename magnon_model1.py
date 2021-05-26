'''
Driver file for computing the Wannier Hamiltonians and Wannier polarizations for the BBH 
model. The model is defined here and the tools from the module Wannier_toolbox are used.
'''
import numpy as np
from numpy import sin, cos, pi

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import misc as mi
import Wannier_toolbox as WT

def ham_tBBH(k, params):
    ''' Return the Bloch hamiltonian. Broadcasts in k. 
    First dimension of k are the kx,ky components.
    '''
    # Unpack arguments
    J = params['J']
    K1h = params['K1h']
    K2h = params['K2h']
    K1v = params['K1v']
    K2v = params['K2v']
    
    # Convenience aliases for Pauli matrices
    s0 = np.eye(2)
    s1 = np.array([[0,1], 
                   [1,0]])
    s2 = np.array([[0,-1j], 
                   [1j,0]])
    s3 = np.array([[1,0], 
                   [0,-1]])
    
    # Matrices that enter into hamiltonian (4 by 4)
    G000 = mi.KP([s0, s0, s0])
    G10x = mi.KP([s1, s0, s0+s1])
    G21x = mi.KP([s2, s1, s0+s1])
    G22x = mi.KP([s2, s2, s0+s1])
    G23x = mi.KP([s2, s3, s0+s1])
    
    Kx, Ky = k # Split momentum into its components
    Kx = Kx[...,None,None] # Add length-one axes to broadcast against orbital axes
    Ky = Ky[...,None,None]
    
    k_shape = k.shape[1:] # Shape of momentum arrays (excluding kx,ky components)
    mat = np.zeros(k_shape + G000.shape, complex)
    
    # Dimer term
    mat += J * G000
    # x-direction terms
    mat += K1h/2. * G10x
    mat += K2h/2. * ( cos(Kx) * G10x - sin(Kx) * G23x )
    # y-direction terms
    mat += - K1v/2. * G22x
    mat += - K2v/2. * ( cos(Ky) * G22x + sin(Ky) * G21x )
    
    tau3 = mi.KP([s0, s0, s3])
    
    return mat, tau3

def plot_bandstructure(params):
    
    Nx, Ny = 50, 50
    kx = np.linspace(0., 2.*pi, Nx, endpoint=False)
    ky = np.linspace(0., 2.*pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    mi.sprint('k.shape', k.shape)
    
    H_array, tau3 = ham_tBBH(k, params)
    mi.sprint('H_array.shape', H_array.shape)
    evals = mi.eigvals_paraunitary(H_array, tau3)
    mi.sprint('evals.shape', evals.shape)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    for n in range(evals.shape[-1]):
        surf = ax.plot_surface(k[0], k[1], evals[...,n], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

def plot_Wannierbands(params, plot=True):
    
    Nx, Ny = 200, 220
    kx = np.linspace(-pi, pi, Nx, endpoint=False)
    ky = np.linspace(-pi, pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    mi.sprint('k.shape', k.shape)
    
    H_array, tau3 = ham_tBBH(k, params)
    mi.sprint('H_array.shape', H_array.shape)
    evals, evecs = mi.eig_paraunitary(H_array, tau3)
    
    filled_bands = [4,5]
    mi.sprint('filled_bands',filled_bands)
    evecs_occ = evecs[...,filled_bands]
    
    basepoints = [0,0]
    Wilsonloops_list = WT.Wilson_loops(evecs_occ, basepoints, metric=tau3)
    
    Wx = Wilsonloops_list[0]
    mi.sprint('Wx.shape',Wx.shape)
    px = WT.pol(Wx, atol=1.e-14, branchtol=0., verbose=True)
    mi.sprint('px', px)
    Wannierbands_x = mi.eigvalsu(Wx, shift_branch=True) / (2.*pi)
    mi.sprint('Wannierbands_x.shape',Wannierbands_x.shape)
    
    Wy = Wilsonloops_list[1]
    mi.sprint('Wy.shape',Wy.shape)
    py = WT.pol(Wy, atol=1.e-14, branchtol=0., verbose=True)
    mi.sprint('py', py)
    Wannierbands_y = mi.eigvalsu(Wy, shift_branch=True) / (2.*pi)
    mi.sprint('Wannierbands_y.shape',Wannierbands_y.shape)
    
    if plot:
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

def calculate_Wannierpol(params):
    np.set_printoptions(linewidth=750)
    
    Nx, Ny = 100, 150
    kx = np.linspace(-pi, pi, Nx, endpoint=False)
    ky = np.linspace(-pi, pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    mi.sprint('k.shape', k.shape)
    
    H_array, tau3 = ham_tBBH(k, params)
    mi.sprint('H_array.shape', H_array.shape)
    evals, evecs = mi.eig_paraunitary(H_array, tau3)
    
    filled_bands = [4,5]
    mi.sprint('filled_bands',filled_bands)
    evecs_occ = evecs[...,filled_bands]
    mi.sprint('evecs_occ.shape',evecs_occ.shape)
    
    Wilsonloops = WT.Wilson_loops(evecs_occ, 'all', metric=tau3)
    mi.sprint('Wilsonloops.shape', Wilsonloops.shape)
    
    Wannierstates = WT.Wannier_states(evecs_occ, Wilsonloops)
    mi.sprint('Wannierstates.shape', Wannierstates.shape)
    
    Wannier_sector = [0]
    mi.sprint('Wannier_sector',Wannier_sector)
    Wannierstates_sector = Wannierstates[...,Wannier_sector]
    mi.sprint('Wannierstates_sector.shape',Wannierstates_sector.shape)
    
    basepoints = [0,0]
    W_xx, W_xy = WT.Wilson_loops(Wannierstates_sector[0,...], basepoints, metric=tau3)
    W_yx, W_yy = WT.Wilson_loops(Wannierstates_sector[1,...], basepoints, metric=tau3)
    
    mi.sprint('W_xy.shape',W_xy.shape)
    mi.sprint('W_yx.shape',W_yx.shape)
    
    
    verbose = True
    p_xy = WT.pol(W_xy, atol=1.e-14, branchtol=1.e-8, verbose=verbose)
    mi.sprint('p_xy',p_xy)
    
    p_yx = WT.pol(W_yx, atol=1.e-14, branchtol=1.e-8, verbose=verbose)
    mi.sprint('p_yx',p_yx)
    
    return


if __name__ == "__main__":
    np.set_printoptions(linewidth=750)
    
    params_tBBH = {'J':0.3, 'K1h':0.11, 'K1v':0.0,
                            'K2h':0.1, 'K2v':0.1}
    mi.sprint('params_tBBH',params_tBBH)
    
    plot_bandstructure(params_tBBH)
    print()
    plot_Wannierbands(params_tBBH)
    print()
    calculate_Wannierpol(params_tBBH)
