#from functools import reduce
#from glob import glob
import os
#import os.path as op
#from shutil import copyfile, copytree

#import pytest
#import numpy as np
#from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           #assert_array_equal)

import mne

from mne.utils import run_tests_if_main

#_TempDir, requires_fs_or_nibabel, requires_nibabel,                       requires_freesurfer, run_subprocess,                       requires_mne, requires_version, 
#from mne.datasets import testing
#from mne.transforms import (Transform, apply_trans, rotation, translation,
                            #scaling)
#from mne.coreg import (fit_matched_points, create_default_subject, scale_mri,
                       #_is_mri_subject, scale_labels, scale_source_space,
                       #coregister_fiducials)
#from mne.io.constants import FIFF
#from mne.utils import _TempDir, run_tests_if_main, requires_nibabel
#from mne.source_space import write_source_spaces

from mne.bv2mne.source_analysis import (forward_model,
                             get_epochs_dics,
                             source2atlas)

from mne.bv2mne import get_brain

subject = "subject_01"
figure = None

# Project 's directory
test_data_dir = os.path.join("data", "bv")
subj_data_dir = os.path.join(test_data_dir, subject)


############################## for get_brain and get_sources ###############################
# Surface files
fname_surf_L = os.path.join(subj_data_dir , "surf", '{0}_Lwhite.gii'.format(subject))
fname_surf_R = os.path.join(subj_data_dir , "surf",'{0}_Rwhite.gii'.format(subject))

# MarsAtlas texture files
fname_tex_L = os.path.join(subj_data_dir , "tex",'{0}_Lwhite_parcels_marsAtlas.gii'.format(subject))
fname_tex_R = os.path.join(subj_data_dir , "tex",'{0}_Rwhite_parcels_marsAtlas.gii'.format(subject))

# Transformatio file from BV to MNE
trans = os.path.join(subj_data_dir , "ref",'{0}-trans.trm'.format(subject))

# MarsAtas files
fname_atlas = os.path.join(test_data_dir , "MarsAtlas_BV_2015.txt")
fname_color = os.path.join(test_data_dir , "MarsAtlas.ima")

# MarsAtlas volumetric parcellation file
fname_vol = os.path.join(subj_data_dir , "vol",'{0}_parcellation.nii.gz'.format(subject))

name_lobe_vol = ['Subcortical']

############################################## for forward_model

###  anat data
# File to align coordinate frames meg2mri computed using mne analyze (interactive gui)
fname_trans = os.path.join(subj_data_dir , "trans",'{0}-trans.fif'.format(subject))

###################
### fonctional fif data (requires some preprocessing) / This is for a given session

########################################### for forward_model ant get_epochs_dics
# Epoched event-of-interest data
fname_event = os.path.join(subj_data_dir , 'prep','{0}_bline-epo.fif'.format(subject)) # modified, from the original architecture, the data are in a directory per session 

# Epoched baseline data
fname_baseline = os.path.join(subj_data_dir , 'prep','{0}_bline-epo.fif'.format(subject))

# File to align coordinate frames meg2mri computed using mne analyze (interactive gui) - result of test_forward
fname_fwd_cort = os.path.join(subj_data_dir , "fwd",'{0}-singleshell-cortex-fwd.fif'.format(subject))
fname_fwd_subcort = os.path.join(subj_data_dir , "fwd",'{0}-singleshell-subcort-fwd.fif'.format(subject))
    
from mne.bv2mne.brain import (Brain, load_brain_src_as_fif)
from mne.bv2mne.surface import Surface
from mne.bv2mne.volume import Volume

from mne.forward import Forward
from mne.source_space import SourceSpaces

from nose.tools import assert_raises, assert_true

#################################################################################################################################################################"

#################################################################################################################################################################

def test_get_brain_get_sources():

    # Create brian object
    brain = get_brain(subject, fname_surf_L, fname_surf_R, fname_tex_L, fname_tex_R,
                      0, fname_vol, name_lobe_vol, trans, fname_atlas, fname_color)

    assert_true(isinstance(brain,Brain))
    
    ## Assert all surfaces are OK within Brain
    print([isinstance(surf, Surface) for hemi in brain.surfaces.keys() for surf in brain.surfaces[hemi]])

    assert_true((all(isinstance(surf, Surface) for hemi in brain.surfaces.keys() for surf in brain.surfaces[hemi])))

    ## Assert all volumes are OK within Brain
    print([isinstance(vol, Volume) for hemi in brain.volumes.keys() for vol in brain.volumes[hemi]])

    assert_true((all(isinstance(vol, Volume) for hemi in brain.volumes.keys() for vol in brain.volumes[hemi])))

    ## Create source space on surface and volume
    src = brain.get_sources(space=5, distance='euclidean')

    print (src)
    
    ### src should have 2 lists, one for surfaces, one for volumes
    assert_true(len(src) == 2)
    
    ### each sublist should have 2 lists, one for each hemisphere
    assert_true(len(src[0]) == 2 and len(src[1]) == 2)
    
    ## all elements in the sublist should be SourceSpaces objects
    assert_true(all(isinstance(src_i_i, SourceSpaces) for src_i in src for src_i_i in src_i ))
    
    ## get_sources also modifies the brain object, and add to all surfaces and volumes a field index_pack_src
    assert_true(all(surf.index_pack_src is not None for hemi in brain.surfaces.keys() for surf in brain.surfaces[hemi]))
    assert_true(all(vol.index_pack_src is not None for hemi in brain.volumes.keys() for vol in brain.volumes[hemi]))
    
    #brain.set_index()
    
    #### method pour eviter de sauver les sources en pickle 
    # ne fonctionne pas pour l'instant, probleme avec les sources subcorticales
    #save_brain_src_as_fif(src, os.path.join(test_data_dir , "src"), subject)
    


def test_forward_model():

    ## loading event of interest
    epochs_event = mne.read_epochs(fname_event)

    ######### using only surface (cortical) information
    
    ### loading cortical src
    
    from mne.bv2mne.brain import load_brain_src_as_fif
    
    src = load_brain_src_as_fif(os.path.join(subj_data_dir,"src"),subject)
    
    print (src)
    
    print('\n----------------------------------------------\n DICS for cortical soures\n----------------------------------------------\n')
    # Forward model for cortical sources (fix the source orientation normal to cortical surface)
    fwd_cort = forward_model(subject, epochs_event, fname_trans, src[0], subjects_dir = test_data_dir, force_fixed=True,
                                name='singleshell-cortex')
    
    assert_true(isinstance(fwd_cort,Forward))
    
    
    print('\n----------------------------------------------\n DICS for subcortical soures\n----------------------------------------------\n')
    # Forward model for cortical sources (fix the source orientation normal to cortical surface)
    fwd_sub = forward_model(subject, epochs_event, fname_trans, src[1], subjects_dir = test_data_dir, force_fixed=True,
                                name='singleshell-subcort')
    
    assert_true(isinstance(fwd_sub,Forward))
    
    


def test_get_epochs_dics():
    
    #Functional parameters

    # High-gamma activity (HGA) parameters
    fmin = 88
    fmax = 92
    mt_bandwidth = 60
    
    # Time parameters
    win_lengths = 0.2
    tstep = 0.05 #0.005 c'es trop long
    
    # Initial time points of multitaper window
    t_event= [0, 0.49] # 0.5 error of discrete timings

    ###  forward models
    #fwd = mne.read_forward_solution(fname_fwd)
    fwd_cort = mne.read_forward_solution(fname_fwd_cort)
    fwd_subcort = mne.read_forward_solution(fname_fwd_subcort)
    
    print (fwd_cort)
    print (fwd_subcort)
    
    ####################
    #### fonctional fif data (requires some preprocessing) / This is for a given session
    
    
    # Epoched event-of-interest data
    epochs_event = mne.read_epochs(fname_event)
    
    ## test ajout des sources
    
    ##################################### necessite de refaire Brain (sinon en le sauvant en pickle...)

    #name_lobe_vol = ['Subcortical']

    ## Create brain object
    #brain = get_brain(subject, fname_surf_L, fname_surf_R, fname_tex_L, fname_tex_R,
                      #0, fname_vol, name_lobe_vol, trans, fname_atlas, fname_color)

    #print (brain.surfaces)
    
    ## necessite de reappliquer la fonction get_sources, sinon les index_pack_src (dans brain) ne sont pas remplis ...
    #src = brain.get_sources(space=5, distance='euclidean')
    
    #################" 
    src = load_brain_src_as_fif(os.path.join(subj_data_dir,"src"),subject)
    
    #src = []
    
    #for hemi in ['lh','rh']:
        #fname_src  = os.path.join(subj_data_dir,"src","source_" + subject + "_vol_" + hemi + "-src.fif")
        ##fname_src  = os.path.join(subj_data_dir,"src","source_" + subject + "_" + hemi + "-src.fif")
        #hemi_src = mne.read_source_spaces(fname_src)
    
        #print (hemi_src)
    
        #src.append(hemi_src)
    
    print('\n----------------------------------------------\n Epoched DICS for cortical soures\n----------------------------------------------\n')

    epoch_power_event, epoch_time_event = get_epochs_dics(epochs_event, fwd_cort, src = src[0][0], tmin=t_event[0], tmax=t_event[1], tstep=tstep, ## sinon c'est bien trop long...
                                      win_lengths=win_lengths, mode='multitaper',
                                      fmin=fmin, fmax=fmax, mt_bandwidth=mt_bandwidth,
                                      mt_adaptive=False, on_epochs=True, avg_tapers=False, pick_ori=None)
    
    ### list of power should have the length corresponding to the number of windows
    n_windows = int(((t_event[1] - win_lengths - t_event[0]) * 1e3) // (tstep * 1e3))
    assert_true(len(epoch_power_event) == n_windows)
    
    print('\n----------------------------------------------\n Average DICS for cortical soures\n----------------------------------------------\n')

    average_power_event, average_time_event = get_epochs_dics(epochs_event, fwd_cort, src = src[0][0], tmin=t_event[0], tmax=t_event[1], tstep=tstep, ## sinon c'est bien trop long...
                                      win_lengths=win_lengths, mode='multitaper',
                                      fmin=fmin, fmax=fmax, mt_bandwidth=mt_bandwidth,
                                      mt_adaptive=False, on_epochs=False, avg_tapers=False, pick_ori=None)
    
    ### list of power should have the length corresponding to the number of windows
    assert_true(len(average_power_event) == n_windows)
    
    
    ### both element in the list of power (taken randomly as the first) should have the same length (corresponding to the number of sources)
    assert_true(len(epoch_power_event[0]) == len(average_power_event[0]))
    
    print (len(average_power_event[0]))
    print (average_power_event[0].shape)

def test_source2atlas():

    #####Functional parameters

    # High-gamma activity (HGA) parameters
    fmin = 88
    fmax = 92
    mt_bandwidth = 60
    
    # Time parameters
    win_lengths = 0.2
    tstep = 0.05 #0.005 c'est trop long
    
    # Initial time points of multitaper window
    t_event = [0, 0.49] # 0.5 error of discrete timings
    t_bline = [-1,-0.5]
    
    ###  anat data
    # File to align coordinate frames meg2mri computed using mne analyze (interactive gui)
    
    fwd_cort = mne.read_forward_solution(fname_fwd_cort)
    
    print (fwd_cort)
    
    ######## reading epochs
    epochs_event = mne.read_epochs(fname_event)
    
    epochs_baseline = mne.read_epochs(fname_baseline)
    ## test ajout des sources (Warning in apply_dics_csd)
    
    ##################################### necessite de refaire Brain (sinon en le sauvant en pickle...)

    #name_lobe_vol = ['Subcortical']

    ## Create brain object
    #brain = get_brain(subject, fname_surf_L, fname_surf_R, fname_tex_L, fname_tex_R,
                      #0, fname_vol, name_lobe_vol, trans, fname_atlas, fname_color)

    #print (brain.surfaces)
    
    ## necessite de reappliquer la fonction get_sources, sinon les index_pack_src (dans brain) ne sont pas remplis ...
    #src = brain.get_sources(space=5, distance='euclidean')
    
    #print (brain.surfaces)
    
    ##################################################################################################################################################
    print('\n----------------------------------------------\n DICS for cortical sources\n----------------------------------------------\n')
    power_event, time_event = get_epochs_dics(epochs_event, fwd_cort, src = src[0][0],  ### src[0] car seule la provenance des sources (surface ou volume) est lue par la suite)
                                      tmin=t_event[0], tmax=t_event[1], tstep=tstep, 
                                      win_lengths=win_lengths, mode='multitaper',
                                      fmin=fmin, fmax=fmax, mt_bandwidth=mt_bandwidth,
                                      mt_adaptive=False, on_epochs=True, avg_tapers=False, pick_ori=None)
    
    print (power_event, time_event)
    
    power_baseline, time_baseline = get_epochs_dics(epochs_baseline, fwd_cort, src = src[0][0], 
                                         tmin=t_bline[0], tmax=t_bline[1], tstep=tstep,
                                         win_lengths=win_lengths, mode='multitaper',
                                         fmin=fmin, fmax=fmax, mt_bandwidth=mt_bandwidth,
                                         mt_adaptive=False, on_epochs=True, avg_tapers=False, pick_ori=None)
    
    ##################################### 
    print('\n----------------------------------------------\n Sources to atlas\n----------------------------------------------\n')
    power_atlas_cortical, area_names, area_lobes = source2atlas(power_event, power_baseline, brain.surfaces)

    print ("Results:")
    
    print (power_atlas_cortical)
    print (area_names)
    print (area_lobes)
    
    
run_tests_if_main()

if __name__ == '__main__':
    
    ### test fonction par fonction
    #test_get_brain()
    #test_forward_model() 
    test_get_epochs_dics()
    #test_source2atlas()## OK 
    
    
