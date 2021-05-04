import numpy as np
import misc as mi

# Internal routines #####################################################################

def Wilson_line_elements(evecs_occ, unitary=True, verbose=False):
    ''' Calculate Wilson line elements between adjacent momentum points using evecs_occ. 
    Last two dimensions of evecs_occ are orbitals and bands, previous ones are momentum
    directions. If unitary==True, the purely "unitary part" of the Wilson line elements 
    are returned.
    '''
    
    # Last two dimensions are orbitals and bands, previous ones are momentum
    Nocc = evecs_occ.shape[-1]
    k_shape = evecs_occ.shape[:-2]
    D = len(k_shape)
    
    if verbose:
        mi.sprint('Nocc',Nocc)
        mi.sprint('D',D)
    
    # Create return array
    F = np.zeros((D,) + k_shape + (Nocc,Nocc), complex)
    
    # Aliases for convenience
    T = evecs_occ
    Tdagger = np.conj(np.swapaxes(evecs_occ, -1, -2))
    
    for d in range(D):
        F_temp = np.roll(Tdagger, -1, axis=d) @ T
        
        if unitary:
            U_temp, P_temp = mi.polardecomp(F_temp)
            F_temp = U_temp
        
        F[d,...] = F_temp
    
    return F

def Wilson_loop_directional(Fdirectional, axis, basepoint, atol=1.e-13):
    ''' Based on the wilson line elements Fdirectional in a specific direction (which must
    be specified in "axis"), all Wilson loops in the "axis" direction whose "axis"-
    direction basepoints are "basepoint" are calculated. If basepoint=='all', Wilson loops 
    with all basepoints are returned. Return is "Wdirectional".
    
    Fdirectional.shape = (Momentum axes) + (Nocc, Nocc)
    
    If basepoint=='all',
        Wdirectional.shape = (Momentum axes) + (Nocc, Nocc)
    Otherwise,
        Wdirectional.shape = (Momentum axes other than "axis") + (Nocc, Nocc)
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
        deviation = np.amax(np.abs(Wdirectional @ np.conj(np.swapaxes(Wdirectional, -1, -2)) - mi.stackedidentity_like(Wdirectional)))
        isunitary = deviation < atol
        if not isunitary:
            print('WARNING: Wilson_loop_directional did not ouput a unitary matrix. Largest deviation is {}.'.format(deviation))
    
    return Wdirectional

# External routines #####################################################################

def Wilson_loops(evecs_occ, basepoints):
    '''
    Returns Wilson loops in every direction from different basepoints. 
    Return is Wilsonloops.
    
    If basepoint=='all',
        Wilsonloops.shape = (D,) + (Momentum axes) + (Nocc, Nocc)
    Otherwise, Wdirectional is a length-D list of Wilson loops in the direction possible
    direction. Each list element has the shape 
        (Momentum axes except the loop direction) + (Nocc, Nocc)
    '''
    
    F = Wilson_line_elements(evecs_occ, unitary=True, verbose=False)
    
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

def Wannier_states(evecs_occ, Wilsonloops, verbose=False):
    '''
    Calculates the Wannier basis states as defined by BBH.
    
                                                       alpha   n
    evecs_occ.shape   =        (Momentum dimensions) + (Norb, Nocc)
    Wilsonloops.shape = (D,) + (Momentum dimensions) + (Nocc, Nocc)
                                                         n     j
    Return:
    Wannierstates.shape = (D,) + (Momentum dimensions) + (Norb, Nocc)
                                                         alpha   j

    '''
    
    phases, evecs_Wil = mi.eigu(Wilsonloops)
    
    # Perform sum-product over band index n (filled bands only)
    Wannierstates = np.sum( evecs_occ[...,None]*evecs_Wil[...,None,:,:], axis=-2 )
    
    if verbose:
        mi.sprint('evecs_occ.shape', evecs_occ.shape)
        mi.sprint('evecs_Wil.shape', evecs_Wil.shape)
        mi.sprint('Wannierstates.shape', Wannierstates.shape)
    
    return Wannierstates


if __name__ == "__main__":
    pass
