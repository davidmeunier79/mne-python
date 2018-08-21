#from functools import reduce
#from glob import glob
import os
#import os.path as op
#from shutil import copyfile, copytree

#import pytest
#import numpy as np
#from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           #assert_array_equal)

#import mne
#from mne.datasets import testing
#from mne.transforms import (Transform, apply_trans, rotation, translation,
                            #scaling)
#from mne.coreg import (fit_matched_points, create_default_subject, scale_mri,
                       #_is_mri_subject, scale_labels, scale_source_space,
                       #coregister_fiducials)
#from mne.io.constants import FIFF
#from mne.utils import _TempDir, run_tests_if_main, requires_nibabel
#from mne.source_space import write_source_spaces


from mne.bv2mne import get_brain


def test_get_brain():

    """ Display sources of the Occipital cortex in the left hemiphere
        and display sources of the Thalamus in the right hemisphere"""

    subject = "subject_01"
    figure = None
    
    # Project 's directory
    test_data_dir = os.path.join("data")
    
    # Surface files
    fname_surf_L = os.path.join(test_data_dir , "bv", "surf", '{0}_Lwhite.gii'.format(subject))
    fname_surf_R = os.path.join(test_data_dir , "bv", "surf",'{0}_Rwhite.gii'.format(subject))
    
    print(fname_surf_L)
    #0/0
    
    # MarsAtlas texture files
    fname_tex_L = os.path.join(test_data_dir , "bv", "tex",'{0}_Lwhite_parcels_marsAtlas.gii'.format(subject))
    fname_tex_R = os.path.join(test_data_dir , "bv", "tex",'{0}_Rwhite_parcels_marsAtlas.gii'.format(subject))
    
    # Transformatio file from BV to MNE
    trans = os.path.join(test_data_dir , "bv", "ref",'{0}-trans.trm'.format(subject))
    
    # MarsAtas files
    fname_atlas = os.path.join(test_data_dir , "bv", "MarsAtlas_BV_2015.txt")
    fname_color = os.path.join(test_data_dir , "bv", "MarsAtlas.ima")
    
    # MarsAtlas volumetric parcellation file
    fname_vol = os.path.join(test_data_dir , "bv","vol",'{0}_parcellation.nii.gz'.format(subject))
    
    name_lobe_vol = ['Subcortical']

    # Create brian object
    brain = get_brain(subject, fname_surf_L, fname_surf_R, fname_tex_L, fname_tex_R,
                      0, fname_vol, name_lobe_vol, trans, fname_atlas, fname_color)

    
    # To show MarsAtlas parcels
    #brain.show() ### Ne fonctionne pas...
    ## To show the left frontal areas (problem in insula)
    #brain.show(hemi='lh', lobe=['Frontal']) ### Ne fonctionne pas...
    #brain.show(hemi='lh', lobe=['Frontal'], name=['Insula']) ### Ne fonctionne pas...
    
    ## Create source space on surface and volume
    src = brain.get_sources(space=5, distance='euclidean')

    ## Display sources in frontal lobe
    #brain.show_sources(src[0], hemi='lh', lobe=['Frontal'], sphere_color=(0.7, 0.7, 0.7)) ### Ne fonctionne pas...
    
    ## Display sources in occipital lobe
    #brain.show_sources(src[0], hemi='lh', lobe=['Occipital']) ### N'affiche rien
    
    ## The show_brain option = True does not work because it calls a FS mesh which is not correctly oriented
    #figure = brain.show_sources(src[0], hemi='lh', lobe=['Occipital'], figure=figure, opacity=1, show_brain=False) ### Ne fonctionne pas...
    
    ## Display sources in the motor cortex
    #brain.show_sources(src[0], hemi='lh', lobe=['Frontal'], name=['Mdl'], opacity=1) ## Ne fonctionne pas...

    brain.set_index()

    ## Does no work for hemi='all'
    #brain.show_sources(src[1], hemi='lh', lobe=['Subcortical'], name=['Thal'], opacity=1) # bug (pas mieux avec 0.1 -> 1)


#@testing.requires_testing_data
#def test_scale_mri():
    #"""Test creating fsaverage and scaling it."""
    ## create fsaverage using the testing "fsaverage" instead of the FreeSurfer
    ## one
    #tempdir = _TempDir()
    #fake_home = testing.data_path()
    #create_default_subject(subjects_dir=tempdir, fs_home=fake_home,
                           #verbose=True)
    #assert _is_mri_subject('fsaverage', tempdir), "Creating fsaverage failed"

    #fid_path = op.join(tempdir, 'fsaverage', 'bem', 'fsaverage-fiducials.fif')
    #os.remove(fid_path)
    #create_default_subject(update=True, subjects_dir=tempdir,
                           #fs_home=fake_home)
    #assert op.exists(fid_path), "Updating fsaverage"

    ## copy MRI file from sample data (shouldn't matter that it's incorrect,
    ## so here choose a small one)
    #path_from = op.join(testing.data_path(), 'subjects', 'sample', 'mri',
                        #'T1.mgz')
    #path_to = op.join(tempdir, 'fsaverage', 'mri', 'orig.mgz')
    #copyfile(path_from, path_to)

    ## remove redundant label files
    #label_temp = op.join(tempdir, 'fsaverage', 'label', '*.label')
    #label_paths = glob(label_temp)
    #for label_path in label_paths[1:]:
        #os.remove(label_path)

    ## create source space
    #print('Creating surface source space')
    #path = op.join(tempdir, 'fsaverage', 'bem', 'fsaverage-%s-src.fif')
    #src = mne.setup_source_space('fsaverage', 'ico0', subjects_dir=tempdir,
                                 #add_dist=False)
    #write_source_spaces(path % 'ico-0', src)
    #mri = op.join(tempdir, 'fsaverage', 'mri', 'orig.mgz')
    #print('Creating volume source space')
    #vsrc = mne.setup_volume_source_space(
        #'fsaverage', pos=50, mri=mri, subjects_dir=tempdir,
        #add_interpolator=False)
    #write_source_spaces(path % 'vol-50', vsrc)

    ## scale fsaverage
    #os.environ['_MNE_FEW_SURFACES'] = 'true'
    #scale = np.array([1, .2, .8])
    #scale_mri('fsaverage', 'flachkopf', scale, True, subjects_dir=tempdir,
              #verbose='debug')
    #del os.environ['_MNE_FEW_SURFACES']
    #assert _is_mri_subject('flachkopf', tempdir), "Scaling fsaverage failed"
    #spath = op.join(tempdir, 'flachkopf', 'bem', 'flachkopf-%s-src.fif')

    #assert op.exists(spath % 'ico-0'), "Source space ico-0 was not scaled"
    #assert os.path.isfile(os.path.join(tempdir, 'flachkopf', 'surf',
                                       #'lh.sphere.reg'))
    #vsrc_s = mne.read_source_spaces(spath % 'vol-50')
    #pt = np.array([0.12, 0.41, -0.22])
    #assert_array_almost_equal(apply_trans(vsrc_s[0]['src_mri_t'], pt * scale),
                              #apply_trans(vsrc[0]['src_mri_t'], pt))
    #scale_labels('flachkopf', subjects_dir=tempdir)

    ## add distances to source space
    #mne.add_source_space_distances(src)
    #src.save(path % 'ico-0', overwrite=True)

    ## scale with distances
    #os.remove(spath % 'ico-0')
    #scale_source_space('flachkopf', 'ico-0', subjects_dir=tempdir)
    #ssrc = mne.read_source_spaces(spath % 'ico-0')
    #assert ssrc[0]['dist'] is not None


#@testing.requires_testing_data
#@requires_nibabel()
#def test_scale_mri_xfm():
    #"""Test scale_mri transforms and MRI scaling."""
    ## scale fsaverage
    #tempdir = _TempDir()
    #os.environ['_MNE_FEW_SURFACES'] = 'true'
    #fake_home = testing.data_path()
    ## add fsaverage
    #create_default_subject(subjects_dir=tempdir, fs_home=fake_home,
                           #verbose=True)
    ## add sample (with few files)
    #sample_dir = op.join(tempdir, 'sample')
    #os.mkdir(sample_dir)
    #os.mkdir(op.join(sample_dir, 'bem'))
    #for dirname in ('mri', 'surf'):
        #copytree(op.join(fake_home, 'subjects', 'sample', dirname),
                 #op.join(sample_dir, dirname))
    #subject_to = 'flachkopf'
    #spacing = 'oct2'
    #for subject_from in ('fsaverage', 'sample'):
        #if subject_from == 'fsaverage':
            #scale = 1.  # single dim
        #else:
            #scale = [0.9, 2, .8]  # separate
        #src_from_fname = op.join(tempdir, subject_from, 'bem',
                                 #'%s-%s-src.fif' % (subject_from, spacing))
        #src_from = mne.setup_source_space(
            #subject_from, spacing, subjects_dir=tempdir, add_dist=False)
        #write_source_spaces(src_from_fname, src_from)
        #print(src_from_fname)
        #vertices_from = np.concatenate([s['vertno'] for s in src_from])
        #assert len(vertices_from) == 36
        #hemis = ([0] * len(src_from[0]['vertno']) +
                 #[1] * len(src_from[0]['vertno']))
        #mni_from = mne.vertex_to_mni(vertices_from, hemis, subject_from,
                                     #subjects_dir=tempdir)
        #if subject_from == 'fsaverage':  # identity transform
            #source_rr = np.concatenate([s['rr'][s['vertno']]
                                        #for s in src_from]) * 1e3
            #assert_allclose(mni_from, source_rr)
        #if subject_from == 'fsaverage':
            #overwrite = skip_fiducials = False
        #else:
            #with pytest.raises(IOError, match='No fiducials file'):
                #scale_mri(subject_from, subject_to,  scale,
                          #subjects_dir=tempdir)
            #skip_fiducials = True
            #with pytest.raises(IOError, match='already exists'):
                #scale_mri(subject_from, subject_to,  scale,
                          #subjects_dir=tempdir, skip_fiducials=skip_fiducials)
            #overwrite = True
        #scale_mri(subject_from, subject_to, scale, subjects_dir=tempdir,
                  #verbose='debug', overwrite=overwrite,
                  #skip_fiducials=skip_fiducials)
        #if subject_from == 'fsaverage':
            #assert _is_mri_subject(subject_to, tempdir), "Scaling failed"
        #src_to_fname = op.join(tempdir, subject_to, 'bem',
                               #'%s-%s-src.fif' % (subject_to, spacing))
        #assert op.exists(src_to_fname), "Source space was not scaled"
        ## Check MRI scaling
        #fname_mri = op.join(tempdir, subject_to, 'mri', 'T1.mgz')
        #assert op.exists(fname_mri), "MRI was not scaled"
        ## Check MNI transform
        #src = mne.read_source_spaces(src_to_fname)
        #vertices = np.concatenate([s['vertno'] for s in src])
        #assert_array_equal(vertices, vertices_from)
        #mni = mne.vertex_to_mni(vertices, hemis, subject_to,
                                #subjects_dir=tempdir)
        #assert_allclose(mni, mni_from, atol=1e-3)  # 0.001 mm
    #del os.environ['_MNE_FEW_SURFACES']


#def test_fit_matched_points():
    #"""Test fit_matched_points: fitting two matching sets of points"""
    #tgt_pts = np.random.RandomState(42).uniform(size=(6, 3))

    ## rotation only
    #trans = rotation(2, 6, 3)
    #src_pts = apply_trans(trans, tgt_pts)
    #trans_est = fit_matched_points(src_pts, tgt_pts, translate=False,
                                   #out='trans')
    #est_pts = apply_trans(trans_est, src_pts)
    #assert_array_almost_equal(tgt_pts, est_pts, 2, "fit_matched_points with "
                              #"rotation")

    ## rotation & translation
    #trans = np.dot(translation(2, -6, 3), rotation(2, 6, 3))
    #src_pts = apply_trans(trans, tgt_pts)
    #trans_est = fit_matched_points(src_pts, tgt_pts, out='trans')
    #est_pts = apply_trans(trans_est, src_pts)
    #assert_array_almost_equal(tgt_pts, est_pts, 2, "fit_matched_points with "
                              #"rotation and translation.")

    ## rotation & translation & scaling
    #trans = reduce(np.dot, (translation(2, -6, 3), rotation(1.5, .3, 1.4),
                            #scaling(.5, .5, .5)))
    #src_pts = apply_trans(trans, tgt_pts)
    #trans_est = fit_matched_points(src_pts, tgt_pts, scale=1, out='trans')
    #est_pts = apply_trans(trans_est, src_pts)
    #assert_array_almost_equal(tgt_pts, est_pts, 2, "fit_matched_points with "
                              #"rotation, translation and scaling.")

    ## test exceeding tolerance
    #tgt_pts[0, :] += 20
    #pytest.raises(RuntimeError, fit_matched_points, tgt_pts, src_pts, tol=10)


#run_tests_if_main()

if __name__ == '__main__':
    test_get_brain()
