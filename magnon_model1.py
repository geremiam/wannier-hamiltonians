'''
Driver file for computing the Wannier Hamiltonians and Wannier polarizations for the BBH 
model. The model is defined here and the tools from the module Wannier_toolbox are used.
'''
import argparse
import numpy as np
from numpy import sin, cos, pi

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import misc as mi
import Wannier_toolbox as WT

def ham_AFM(k, params):
    ''' Return the Bloch hamiltonian. Broadcasts in k. 
    First dimension of k are the kx,ky components.
    '''
    # Unpack arguments
    Jx = params['Jx']
    Jy = params['Jy']
    Jz = params['Jz']
    Jp = (Jx + Jy) / 2.
    Jm = (Jx - Jy) / 2.
    
    k = np.asarray(k)
    Kx, Ky = k # Split momentum into its components
    Kx = Kx[...,None,None] # Add length-one axes to broadcast against orbital axes
    Ky = Ky[...,None,None]
    
    k_shape = k.shape[1:] # Shape of momentum arrays (excluding kx,ky components)
    mat = np.zeros(k_shape + (4,4), complex)
    
    gamma = 1. + np.exp(-1.j*Kx) + np.exp(-1.j*Ky) + np.exp(-1.j*(Kx+Ky))
    
    block = np.array([[Jm, Jp],
                      [Jp, Jm]])
    
    # Populate the matrix
    mat += 4. * Jz * np.eye(4)
    mat[...,0:2,2:4] += block * gamma
    mat[...,2:4,0:2] += block * np.conj(gamma)
    
    tau3 = np.diag([1., -1., 1., -1.])
    
    return mat, tau3

def plot_bandstructure(ham, params):
    
    Nx, Ny = 101, 101
    kx = np.linspace(-pi, pi, Nx, endpoint=False)
    ky = np.linspace(-pi, pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    mi.sprint('k.shape', k.shape)
    
    H_array, tau3 = ham(k, params)
    mi.sprint('H_array.shape', H_array.shape)
    evals = mi.eigvals_paraunitary(H_array, tau3)
    mi.sprint('evals.shape', evals.shape)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    for n in range(evals.shape[-1]):
        surf = ax.plot_surface(k[0], k[1], evals[...,n], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

def plot_Wannierbands(ham, params, plot=True):
    
    Nx, Ny = 201, 211
    kx = np.linspace(-pi, pi, Nx, endpoint=False)
    ky = np.linspace(-pi, pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    mi.sprint('k.shape', k.shape)
    
    H_array, tau3 = ham(k, params)
    mi.sprint('H_array.shape', H_array.shape)
    evals, evecs = mi.eig_paraunitary(H_array, tau3)
    
    filled_bands = [2,3]
    mi.sprint('filled_bands',filled_bands)
    evecs_occ = evecs[...,filled_bands]
    
    basepoints = [0,0]
    Wilsonloops_list = WT.Wilson_loops(evecs_occ, basepoints, force_unitary=False, metric=tau3)
    
    Wx = Wilsonloops_list[0]
    
    mi.sprint('Largest deviation from unitarity', np.amax(np.abs(Wx @ np.swapaxes(Wx.conj(), -1, -2) - mi.stackedidentity_like(Wx))) )
    
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

def calculate_Wannierpol(ham, params):
    np.set_printoptions(linewidth=750)
    
    Nx, Ny = 201, 201
    kx = np.linspace(-pi, pi, Nx, endpoint=False)
    ky = np.linspace(-pi, pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    mi.sprint('k.shape', k.shape)
    
    H_array, tau3 = ham(k, params)
    mi.sprint('H_array.shape', H_array.shape)
    evals, evecs = mi.eig_paraunitary(H_array, tau3)
    
    filled_bands = [2,3]
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
    parser = argparse.ArgumentParser()
    parser.prog = "magnon_model1.py"
    parser.description = "Wannier band analysis for the magnon band structure of an AFM."
    parser.add_argument("--bandstructure", action="store_true", help="Plot bandstructure")
    parser.add_argument("--Wannierbands", action="store_true", help="Plot Wannier bands")
    parser.add_argument("--Wannierpol", action="store_true", help="Calculate Wannier polarization")
    args = parser.parse_args()
    
    np.set_printoptions(linewidth=750)
    
    params = {'Jx':0.999, 'Jy':0.5, 'Jz':1.}
    mi.sprint('params',params)
    
    H_zerommentum, tau3 = ham_AFM([0.,0.], params)
    print('Energies at k = 0 are {}.'.format(np.sort(np.linalg.eigvals(tau3 @ H_zerommentum))))
    
    if args.bandstructure:
        print()
        plot_bandstructure(ham_AFM, params)
    if args.Wannierbands:
        print()
        plot_Wannierbands(ham_AFM, params)
    if args.Wannierpol:
        print()
        calculate_Wannierpol(ham_AFM, params)
