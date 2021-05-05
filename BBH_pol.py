'''
Driver file for computing the Wannier Hamiltonians and Wannier polarizations for the BBH 
model. The model is defined here and the tools from the module Wannier_toolbox are used.
'''
import numpy as np
from numpy import sin, cos, pi
import scipy.linalg

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import misc as mi
import Wannier_toolbox as WT

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

def plot_bandstructure():
    params_dict = {'gamma_x':0.5, 
                   'gamma_y':0.5, 
                   'lambd':1., 
                   'delta':0.}
    
    Nx, Ny = 50, 50
    kx = np.linspace(0., 2.*pi, Nx, endpoint=False)
    ky = np.linspace(0., 2.*pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    mi.sprint('k.shape', k.shape)
    
    H_array = hamBBH(k, params_dict)
    mi.sprint('H_array.shape', H_array.shape)
    evals = np.linalg.eigvalsh(H_array)
    mi.sprint('evals.shape', evals.shape)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    for n in range(evals.shape[-1]):
        surf = ax.plot_surface(k[0], k[1], evals[...,n], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

def plot_Wannierbands(plot=True):
    params_dict = {'gamma_x':0.5, 
                   'gamma_y':1.5, 
                   'lambd':1., 
                   'delta':0.}
    
    Nx, Ny = 300, 400
    kx = np.linspace(-pi, pi, Nx, endpoint=False)
    ky = np.linspace(-pi, pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    mi.sprint('k.shape', k.shape)
    
    H_array = hamBBH(k, params_dict)
    mi.sprint('H_array.shape', H_array.shape)
    evals, evecs = np.linalg.eigh(H_array)
    
    filled_bands = [0,1]
    mi.sprint('filled_bands',filled_bands)
    evecs_occ = evecs[...,filled_bands]
    
    basepoints = [0,0]
    Wilsonloops_list = WT.Wilson_loops(evecs_occ, basepoints)
    
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

def calculate_Wannierpol():
    np.set_printoptions(linewidth=750)
    
    params_dict = {'gamma_x':0.99, 
                   'gamma_y':1.01, 
                   'lambd':1., 
                   'delta':0.}
    mi.sprint('params_dict',params_dict)
    
    Nx, Ny = 100, 150
    kx = np.linspace(-pi, pi, Nx, endpoint=False)
    ky = np.linspace(-pi, pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    mi.sprint('k.shape', k.shape)
    
    H_array = hamBBH(k, params_dict)
    mi.sprint('H_array.shape', H_array.shape)
    evals, evecs = np.linalg.eigh(H_array)
    
    filled_bands = [0,1]
    mi.sprint('filled_bands',filled_bands)
    evecs_occ = evecs[...,filled_bands]
    mi.sprint('evecs_occ.shape',evecs_occ.shape)
    
    Wilsonloops = WT.Wilson_loops(evecs_occ, 'all')
    mi.sprint('Wilsonloops.shape', Wilsonloops.shape)
    
    Wannierstates = WT.Wannier_states(evecs_occ, Wilsonloops)
    mi.sprint('Wannierstates.shape', Wannierstates.shape)
    
    Wannier_sector = [0]
    mi.sprint('Wannier_sector',Wannier_sector)
    Wannierstates_sector = Wannierstates[...,Wannier_sector]
    mi.sprint('Wannierstates_sector.shape',Wannierstates_sector.shape)
    
    basepoints = [0,0]
    W_xx, W_xy = WT.Wilson_loops(Wannierstates_sector[0,...], basepoints)
    W_yx, W_yy = WT.Wilson_loops(Wannierstates_sector[1,...], basepoints)
    
    mi.sprint('W_xy.shape',W_xy.shape)
    mi.sprint('W_yx.shape',W_yx.shape)
    
    
    verbose = False
    p_xy = WT.pol(W_xy, atol=1.e-14, branchtol=1.e-8, verbose=verbose)
    mi.sprint('p_xy',p_xy)
    
    p_yx = WT.pol(W_yx, atol=1.e-14, branchtol=1.e-8, verbose=verbose)
    mi.sprint('p_yx',p_yx)
    
    return


if __name__ == "__main__":
    calculate_Wannierpol()
