#!/usr/bin/env python

# Author: Alexandre Fabre <alexandre.fabre22@gmail.com>

import os
import numpy as np


import mne
from mne.source_space import SourceSpaces
from mne.time_frequency.csd import csd_multitaper
#from mne.source_estimate import SourceEstimate
#from beamformer import dics_source_power
#from beamformer import dics_source_power_epochs
from mne.beamformer import tf_dics, make_dics, apply_dics_csd

from mne import make_forward_solution
from mne.connectivity.spectral import (_epoch_spectral_connectivity,
                                       spectral_connectivity)

from .data import create_param_dict

from .data import read_serialize


def forward_model(subject, raw, fname_trans, src, subjects_dir, force_fixed=False, surf_ori=False, name='single-shell'):
    """construct forward model

    Parameters
    ----------
    subject : str
        The name of subject
    raw : instance of rawBTI
        functionnal data
    fname_trans : str
        The filename of transformation matrix
    src : instance of SourceSpaces | list
        Sources of each interest hemisphere
    subjects_dir : str
        The subjects directory
    force_fixed: Boolean
        Force fixed source orientation mode
    name : str
        Use to save output
       

    Returns
    -------
    fwd : instance of Forward
    -------
    Author : Alexandre Fabre
    #"""
    # files to save step
    
    #fname_bem_model = os.path.join(subjects_dir , '{0}/bem/{0}-{1}-bem.fif'.format(subject, name))
    #fname_bem_sol = os.path.join(subjects_dir ,  '{0}/bem/{0}-{1}-bem-sol.fif'.format(subject, name))
    #fname_fwd = os.path.join(subjects_dir , '{0}/fwd/{0}-{1}-fwd.fif'.format(subject, name))

    if not os.path.exists(os.path.join(subjects_dir ,subject, 'bem')): ## ajout
        os.makedirs(os.path.join(subjects_dir ,subject,'bem'))
        
    if not os.path.exists(os.path.join(subjects_dir ,subject, 'fwd')): ## ajout
        os.makedirs(os.path.join(subjects_dir ,subject,'fwd')) 
        
    fname_bem_model = os.path.join(subjects_dir ,subject, 'bem','{0}-{1}-bem.fif'.format(subject, name))
    fname_bem_sol = os.path.join(subjects_dir ,subject,  'bem','{0}-{1}-bem-sol.fif'.format(subject, name))
    fname_fwd = os.path.join(subjects_dir ,subject, 'fwd','{0}-{1}-fwd.fif'.format(subject, name))


    # Make bem model: single-shell model. Depends on anatomy only.
    model = mne.make_bem_model(subject, conductivity=[0.3], subjects_dir=subjects_dir)
    mne.write_bem_surfaces(fname_bem_model, model)

    # Make bem solution. Depends on anatomy only.
    bem_sol = mne.make_bem_solution(model)
    mne.write_bem_solution(fname_bem_sol, bem_sol)

    # bem_sol=mne.read_bem_solution(fname_bem_sol)

    if len(src) == 2:
            # gather sources each the two hemispheres
            lh_src, rh_src = src
            src = lh_src + rh_src

    # Compute forward operator, commonly referred to as the gain or leadfield matrix.
    #fwd = make_forward_solution(info = raw.info, trans = fname_trans, src = src, bem = bem_sol, fname_fwd, mindist=0.0)
    fwd = make_forward_solution(info = raw.info, trans = fname_trans, src = src, bem = bem_sol, mindist=0.0)
    mne.write_forward_solution(fname_fwd, fwd, overwrite = True)
	
    ## Set orientation of the source
    #if force_fixed:
        ## Force fixed
        #fwd = mne.read_forward_solution(fname_fwd, force_fixed=True)
    #elif surf_ori:
        ## Surface normal
        #fwd = mne.read_forward_solution(fname_fwd, surf_ori=True)
    #else:
        ## Free like a bird
		##fwd = mne.read_forward_solution(fname_fwd)

    return fwd

def compute_on_epochs_dics(epochs, fwd, src, fmin, fmax, tmin,
                          tmax, n_fft,
                          bandwidth, adaptive = False,
                          low_bias = True, projs=None, verbose=None):


    #### Perform Average CSD and DICS ########################################################
    # Compute cross-spectral density csd matrix (nChans, nChans, nTapers, nTrials) on average to compute DICS filters
    avg_csds = csd_multitaper(epochs, fmin=fmin, fmax=fmax, tmin=tmin,
                        tmax=tmax, n_fft=n_fft,
                        bandwidth=bandwidth, adaptive= adaptive,
                        low_bias=low_bias, projs=projs, verbose=verbose)
    
    # Compute the spatial filters for each vertex, using two approaches .
    # (dans l'exemple tiré du tuto de MNE, normalize_fwd = False est aussi utilisé)
    filters = make_dics(info = epochs.info, forward = fwd , csd = avg_csds, reg=0.05, pick_ori='max-power', normalize_fwd=True) 

    # Compute the DICS power map by computing each csd and applying the spatial filters to the CSD matrix for each of the epochs
    csds = csd_multitaper(epochs, fmin=fmin, fmax=fmax, tmin=tmin,
                        tmax=tmax, n_fft=n_fft,
                        bandwidth=bandwidth, adaptive= adaptive,
                        low_bias=low_bias, projs=projs, verbose=verbose, on_epochs = True)
    
    #print (csds)
    
    #### Applying filters on CSD obtained for each epoch
    powers_csd = []
    
    for csd in csds:
        power_csd, f = apply_dics_csd(csd = csd, filters = filters, src = src)
        powers_csd.append(power_csd.data)
        
    # Append time slices
    powers_csd = np.array(powers_csd)[:,:,0]
    powers_csd = np.transpose(powers_csd)
    
    return powers_csd


def compute_average_dics(epochs, fwd, src,fmin, fmax, tmin,
                          tmax, n_fft,
                          bandwidth, mode = "multitaper", adaptive = False,
                          low_bias = True, projs=None, verbose=None):


    #### Perform Average CSD and DICS ########################################################
    # Compute cross-spectral density csd matrix (nChans, nChans, nTapers, nTrials) on average to compute DICS filters
    if mode == "multitaper":
        avg_csds = csd_multitaper(epochs, fmin=fmin, fmax=fmax, tmin=tmin,
                        tmax=tmax, n_fft=n_fft,
                        bandwidth=bandwidth, adaptive= adaptive,
                        low_bias=low_bias, projs=None, verbose=None)
    elif mode == 'fourier':
        avg_csds = csd_fourier(epochs, fmin=fmin, fmax=fmax, tmin=tmin,
                        tmax=tmax, n_fft=n_fft,
                        projs=None, verbose=None)
    # Compute the spatial filters for each vertex, using two approaches .
    # (dans l'exemple tiré du tuto de MNE, normalize_fwd = False est aussi utilisé)
    filters = make_dics(info = epochs.info, forward = fwd , csd = avg_csds, reg=0.05, pick_ori='max-power', normalize_fwd=True) 

    #### Applying filters on avearge CSD
    power_csd, f = apply_dics_csd(csd = avg_csds, filters = filters, src = src)
        
    print (power_csd.data)
    
    #0/0
    
    ## Append time slices
    #powers_csd = np.array(powers_csd)[:,:,0]
    #powers_csd = np.transpose(powers_csd)
    
    return power_csd.data


def get_epochs_dics(epochs, fwd, src = None, tmin=None, tmax=None, tstep=0.005, win_lengths=0.2, mode='multitaper',
                    fmin=0, fmax= np.inf, n_fft=None, mt_bandwidth=None,
                    mt_adaptive=False, mt_low_bias=True, projs=None, verbose=None,
                    reg=0.01, label=None, pick_ori=None, on_epochs=True,
                    avg_tapers=True):
    """construct forward model

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs
    fwd : instance of Forward
        The solution of forward problem
    tmin : float | None
        Minimum time instant to consider. If None start at first sample
    tmax : float | None
        Maximum time instant to consider. If None end at last sample
    tstep : float | None 
        Time window step. If None, it's as large as the time interval (tmin - tmax)
    win_lengths: float | None
        The window size. If None, it's as large as the time interval (tmin - tmax)
    mode : 'multitaper' | 'fourier'
        Spectrum estimation mode
    fmin: float
        Minimum frequency of interest
    fmax : float
        Maximum frequency of interest
    n_fft : int | None
        Length of the FFT. If None the exact number of samples between tmin and
        tmax will be used.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'multitaper' mode.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
        Only used in 'multitaper' mode.
    mt_low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth. Only used in 'multitaper' mode.
    projs : list of Projection | None
        List of projectors to use in CSD calculation, or None to indicate that
        the projectors from the epochs should be inherited.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    on_epochs : bool
        Average on epoch
    on_tapers : bool
        Averge on tapers in csd computing

    Returns
    -------
    stc : list of SourceEstimate
    -------
    Author : Alexandre Fabre
    """
    

    # Default values
    if tmin is None:
        tmin = epochs.times[0]
    if tmax is None:
        tmax = epochs.times[-1] - win_lengths

    print (epochs.times)
    assert tmin > epochs.times[0], "Error, tmin {} is shorter than first epoch {}".format(tmin , epochs.times[0])
    assert tmax < epochs.times[-1], "Error, tmax {} is larger than last epoch {}".format(tmax , epochs.times[-1])
    
    
    # Multiplying by 1e3 to avoid numerical issues (?)
    n_time_steps = int(((tmax - win_lengths - tmin) * 1e3) // (tstep * 1e3))
    
    print (((tmax - tmin) * 1e3) // (tstep * 1e3))
    print (int(((tmax - tmin) * 1e3) // (tstep * 1e3)))
    print (n_time_steps)
    
    time_windows = [(round(tmin + i_time * tstep,3),round(tmin + i_time * tstep + win_lengths,3)) for i_time in range(n_time_steps) if (tmin + i_time * tstep + win_lengths< epochs.times[-1])]
    print (time_windows)
    

    # Init power and time
    power = []
    time  = np.zeros(n_time_steps)

    print('Computing cross-spectral density from epochs...')
    for i_time, (win_tmin,win_tmax) in enumerate(time_windows):

        print('   From {0}s to {1}s'.format(win_tmin, win_tmax))
        time[i_time] = (win_tmin + win_tmax)/2.0

        if on_epochs:
            assert mode == 'multitaper', "Error, on_epochs not implemented yet for 'fourier' mode"
            powers_csd = compute_on_epochs_dics(epochs = epochs, fwd = fwd, src = src, fmin=fmin, fmax=fmax, tmin=win_tmin,
                          tmax=win_tmax, n_fft=n_fft,
                          bandwidth=mt_bandwidth, adaptive= mt_adaptive,
                          low_bias=mt_low_bias, projs=projs, verbose=verbose)
            
            power.append(powers_csd)
        else: 
            
            power_csd = compute_average_dics(epochs = epochs, fwd = fwd, src = src,fmin=fmin, fmax=fmax, tmin=win_tmin,
                          tmax=win_tmax, n_fft=n_fft,
                          bandwidth=mt_bandwidth, mode = mode, adaptive= mt_adaptive,
                          low_bias=mt_low_bias, projs=projs, verbose=verbose)
            
            power.append(power_csd)
            

    return power, time



def source2atlas(data, baseline, atlas):
    '''
    Transform source estimates to atlas-based
    i) log transform
    ii) takes zscore wrt baseline
    iii) average across sources within the same area
    '''
    print (np.array(data).shape)
    print (np.array(baseline).shape)
    
    # Dimensions
    n_time_points, n_src, n_trials = np.array(data).shape 
    

    # Take z-score of event related data with respect to baseline activity
    z_value = z_score(data, baseline)
    print ("Z value:")
    print (z_value.shape)
    
    # Extract power time courses and sort them for each parcel (area in MarsAtlas)
    power_sources = area_activity(z_value, atlas)
    print("power sources:")
    print (len(power_sources))
    
    # Take average time course for each parcel across sources within an area (n_epochs, n_areas, n_times)
    dims = np.array(power_sources).shape
    power_atlas = np.zeros((n_trials, np.prod(dims), n_time_points))
    print ("power_atlas:")
    print (power_atlas.shape)
    
    narea = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            power_singlearea = power_sources[i][j]
            power_singlearea = np.mean(power_singlearea, axis=0)
            power_atlas[:, narea, :] = power_singlearea
            narea += 1

    # Get names and lobes of areas
    names = []
    lobes = []
    for hemi in ['lh', 'rh']:
        for i in range(len(atlas[hemi])):
            names.append(atlas['lh'][i].name + '_' + hemi)
            lobes.append(atlas['lh'][i].lobe + '_' + hemi)

    # Transform to list
    names = np.array(names).T
    names = names.tolist()
    lobes = np.array(lobes).T
    lobes = lobes.tolist()

    return power_atlas, names, lobes



def z_score(data, baseline):
    """ z-score of source power wrt baseline period (noise)

    Parameters
    ----------
    data : instance of SourceEstimate | array
        The studied data
    baseline : instance of SourceEstimate | array
        The baseline

    Returns
    -------
    z_value : array
        The data transformed to z-value

    -------
    Author : Andrea Brovelli and Alexandre Fabre
    """
    # Dimensions
    n_time_points, n_src, n_trials = np.array(data).shape

    # Take log to make data approximately Gaussian
    data_log = np.log(data)
    baseline_log = np.log(baseline)

    # Mean and std of baseline
    mean = baseline_log.mean(axis=0)
    std = baseline_log.std(axis=0)

    # Take z-score wrt baseline
    z_value = np.zeros((n_src, n_trials, n_time_points))
    for i in range(n_time_points):
        # Compute z-score
        value = (data_log[i] - mean) / std
        # Store
        z_value[..., i] = value

    return z_value



def area_activity(data, obj):
    """ Mean activity for each atlas area

    Parameters
    ----------
    data : array
        Data activities
    obj : (list | dict) of array | (list | dict) of object with index_pack_src attribute
        Allows to select sources to build regions

    Returns
    -------
    fwd : instance of Forward
    -------
    Author : Andrea Brovelli and Alexandre Fabre
    """
    
    #print ('Obj:')
    print ("obj:")
    print (len(obj))
    print (obj.keys())
    
    obj_dict = create_param_dict(obj)
    
    print("obj_dict:")
    #print(obj_dict)
    print (len(obj_dict))
    
    for key in obj_dict:
        
        if hasattr(obj_dict[key], 'index_pack_src'):
            print('yes')
            obj[key] = list(map( lambda x : x.index_pack_src, obj_dict[key]))
        else:
            print('no')
            obj[key] = obj_dict[key]

    # src number across each hemisphere
    nb_src = {}

    for hemi in ['lh', 'rh']:
        nb_src[hemi] = sum([len(v.index_pack_src) for v in obj[hemi] if v.index_pack_src is not None])

    print ("nb_src: ",nb_src)
    
    # Organize in hemispheres
    regions = []
    start = 0
    for hemi in ['lh', 'rh']:
        values = data[start:start+nb_src[hemi]]
        start += nb_src[hemi]
        hemi_regions = [values[region.index_pack_src] for region in obj[hemi] if region.index_pack_src is not None]
        regions.append(hemi_regions)

    print ("regions:")
    print (len(regions))
    print (len(regions[0]))
    
    #print (regions[0][0].shape)
    
    return regions



if __name__=="__main__":
    pass