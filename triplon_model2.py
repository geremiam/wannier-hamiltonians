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

def ham_tSOC(k, params):
    ''' Return the Bloch hamiltonian. Broadcasts in k. 
    First dimension of k are the kx,ky components.
    '''
    # Unpack arguments
    J = params['J']
    h = params['h']
    hs = params['hs']
    K1 = params['K1']
    K2 = params['K2']
    K = params['K']
    Gamma = params['Gamma']
    D = params['D']
    
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
    G023 = mi.KP([s0, s2, s3])
    G323 = mi.KP([s3, s2, s3])
    G00x = mi.KP([s0, s0, s0+s1])
    G10x = mi.KP([s1, s0, s0+s1])
    G20x = mi.KP([s2, s0, s0+s1])
    G21x = mi.KP([s2, s1, s0+s1])
    G22x = mi.KP([s2, s2, s0+s1])
    G23x = mi.KP([s2, s3, s0+s1])
    G31x = mi.KP([s3, s1, s0+s1])
    G32x = mi.KP([s3, s2, s0+s1])
    
    Kx, Ky = k # Split momentum into its components
    Kx = Kx[...,None,None] # Add length-one axes to broadcast against orbital axes
    Ky = Ky[...,None,None]
    
    k_shape = k.shape[1:] # Shape of momentum arrays (excluding kx,ky components)
    mat = np.zeros(k_shape + G000.shape, complex)
    
    # Dimer term
    mat += J * G000
    # Magnetic field terms
    mat += - hs * G323
    mat += - h  * G023
    # x-direction terms
    mat += K1/2. * G10x 
    mat += K2/2. * ( cos(Kx) * G10x + sin(Kx) * G20x )
    # y-direction terms
    mat += cos(Ky) * (K*G00x + Gamma*G31x)
    mat += sin(Ky) * D * G32x
    
    tau3 = mi.KP([s0, s0, s3])
    
    return mat, tau3

def crit_fields(params_tSOC, verbose=False):
    J = params_tSOC['J']
    D = params_tSOC['D']
    K1 = params_tSOC['K1']
    K2 = params_tSOC['K2']
    
    hsc = ( np.sqrt(J**2 + 2*J*D) - np.sqrt(J**2 - 2*J*D) ) / 2.
    
    fp = np.sqrt( 4.*D**2 + (np.abs(K1) + np.abs(K2))**2 )
    fm = np.sqrt( 4.*D**2 + (np.abs(K1) - np.abs(K2))**2 )
    
    hc1 = ( np.sqrt(J**2 + J*fm) - np.sqrt(J**2 - J*fm) ) / 2.
    hc2 = ( np.sqrt(J**2 + J*fp) - np.sqrt(J**2 - J*fp) ) / 2.
    
    if verbose:
        print('hc1, hc2 = {}, {}'.format(hc1, hc2))
        print('hsc = {}'.format(hsc))
    return hc1, hc2, hsc

def plot_bandstructure(params_dict):
    
    Nx, Ny = 50, 50
    kx = np.linspace(0., 2.*pi, Nx, endpoint=False)
    ky = np.linspace(0., 2.*pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    mi.sprint('k.shape', k.shape)
    
    H_array, tau3 = ham_tSOC(k, params_dict)
    mi.sprint('H_array.shape', H_array.shape)
    evals = mi.eigvals_paraunitary(H_array, tau3)
    mi.sprint('evals.shape', evals.shape)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    for n in range(evals.shape[-1]):
        surf = ax.plot_surface(k[0], k[1], evals[...,n], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

def plot_Wannierbands(params_dict, plot=True):
    
    Nx, Ny = 200, 210
    kx = np.linspace(-pi, pi, Nx, endpoint=False)
    ky = np.linspace(-pi, pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    mi.sprint('k.shape', k.shape)
    
    H_array, tau3 = ham_tSOC(k, params_dict)
    mi.sprint('H_array.shape', H_array.shape)
    evals, evecs = mi.eig_paraunitary(H_array, tau3)
    
    filled_bands = [4,5]
    mi.sprint('filled_bands',filled_bands)
    evecs_occ = evecs[...,filled_bands]
    
    basepoints = [0,0]
    Wilsonloops_list = WT.Wilson_loops(evecs_occ, basepoints, metric=tau3)
    
    shift_branch = False
    
    Wx = Wilsonloops_list[0]
    mi.sprint('Wx.shape',Wx.shape)
    px = WT.pol(Wx, atol=1.e-14, branchtol=0., verbose=True)
    mi.sprint('px', px)
    Wannierbands_x = mi.eigvalsu(Wx, shift_branch=shift_branch) / (2.*pi)
    mi.sprint('Wannierbands_x.shape',Wannierbands_x.shape)
    
    Wy = Wilsonloops_list[1]
    mi.sprint('Wy.shape',Wy.shape)
    py = WT.pol(Wy, atol=1.e-14, branchtol=0., verbose=True)
    mi.sprint('py', py)
    Wannierbands_y = mi.eigvalsu(Wy, shift_branch=shift_branch) / (2.*pi)
    mi.sprint('Wannierbands_y.shape',Wannierbands_y.shape)
    
    if plot:
        fig, ax = plt.subplots(1,2)
        ax[0].plot(ky, Wannierbands_x)
        ax[0].set_xlabel(r'$k_y$')
        ax[1].plot(kx, Wannierbands_y)
        ax[1].set_xlabel(r'$k_x$')
        for axis in ax.flatten():
            if shift_branch:
                axis.axhline(0, color='C7')
                axis.axhline(1, color='C7')
#             else:
#                 axis.axhline(-0.5, color='C7')
#                 axis.axhline(0.5, color='C7')
        plt.show()
    
    return

def calculate_Wannierpol(params_dict):
    
    mi.sprint('params_dict',params_dict)
    
    Nx, Ny = 200, 250
    kx = np.linspace(-pi, pi, Nx, endpoint=False)
    ky = np.linspace(-pi, pi, Ny, endpoint=False)
    k = np.array( np.meshgrid(kx, ky, indexing='ij') )
    mi.sprint('k.shape', k.shape)
    
    H_array, tau3 = ham_tSOC(k, params_dict)
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
    p_xy = WT.pol(W_xy, atol=1.e-14, branchtol=1.e-3, verbose=verbose)
    mi.sprint('p_xy',p_xy)
    
    p_yx = WT.pol(W_yx, atol=1.e-14, branchtol=1.e-3, verbose=verbose)
    mi.sprint('p_yx',p_yx)
    
    return


if __name__ == "__main__":
    np.set_printoptions(linewidth=750)
    
    params_tSOC = {
    'J':1.,
    'h':0.13,
    'hs':0.,
    'K1':0.14,
    'K2':0.08,
    'K':0.03,
    'Gamma':0.11,
    'D':0.10
    }
    
    hc1, hc2, hsc = crit_fields(params_tSOC, verbose=True)
    
    params_tSOC['h'] = 1.1 * hc2
    mi.sprint('h', params_tSOC['h'])
    
    plot_bandstructure(params_tSOC)
    print()
    plot_Wannierbands(params_tSOC)
    print()
    calculate_Wannierpol(params_tSOC)
