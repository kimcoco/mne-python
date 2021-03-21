"""
.. _ex-hilbert-beamformer:

============================================================================
Compute time-frequency activation in source space with a Hilbert beamformer
============================================================================

Compute and plot time-frequency activation in source space with a Hilbert
beamformer pipeline, using LCMV beamforming on Hilbert transformed data and
plotting the envelope and inter-trial coherence (ITC) in source space.
This example uses MNE-Python's :ref:`somato dataset <somato-dataset>`, a data
set from an experiment with somatosensory stimulation, and shows beta and gamma
band activation in the source.
"""
# Author: Britta Westner <britta.wstnr@gmail.com>
#
#
# License: BSD (3-clause)
import os
import os.path as op
import numpy as np

import mne
from mne.datasets import somato
from mne.beamformer import make_lcmv, apply_lcmv_epochs

from mne_bids import get_head_mri_trans, BIDSPath, read_raw_bids

import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img

import warnings

print(__doc__)

###############################################################################

# is this needed?
mne.set_log_level('WARNING')

# Preparing everything:
data_path = somato.data_path()
meg_dir = op.join(data_path, 'sub-01/meg/')
freesurf_dir = op.join(data_path, 'derivatives/freesurfer/subjects/')
subject = '01'

output_path = op.join(meg_dir, 'output/')
if not op.isdir(output_path):
    os.makedirs(output_path)

t1_fname = op.join(freesurf_dir, subject, 'mri/T1.mgz')
bem_fname = op.join(freesurf_dir, subject, 'bem/01-5120-bem-sol.fif')
fwd_fname = op.join(output_path, 'somato_vol-fwd.fif')


# BIDS path
bids_path = BIDSPath(subject='01', root=data_path, task='somato',
                     datatype='meg')

# epoching parameters:
event_id, tmin, tmax = 1, -.2, 2
baseline = (-.2, -0.05)

###############################################################################
# Load raw data and head-MRI transform
# ------------------------------------

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    raw = read_raw_bids(bids_path)

    # get the head-MRI transform from JSON file
    transf = get_head_mri_trans(bids_path=bids_path)

###############################################################################
# Compute or load forward solution
# --------------------------------
# As always with source reconstruction, we need a forward model: we choose a
# volume forward model with 10 mm grid resolution based on a Boundary Element
# Model. The forward model will be created and saved to disk - unless it is
# already present in the output directory:

if not op.isfile(fwd_fname):

    # use a volume source grid
    src = mne.setup_volume_source_space(subject, pos=10., mri=t1_fname,
                                        bem=bem_fname,
                                        subjects_dir=freesurf_dir)

    # make leadfield
    fwd = mne.make_forward_solution(raw.info, trans=transf, src=src,
                                    bem=bem_fname, meg=True, eeg=False,
                                    n_jobs=1)
    mne.write_forward_solution(fwd_fname, fwd, overwrite=False)

else:

    fwd = mne.read_forward_solution(fwd_fname)

###############################################################################
# ITC function for source space
# -----------------------------
# This is a little helper function to compute intertrial coherence in
# source space:

def compute_source_itc(stcs):
    n_trials = len(stcs)

    tmp = np.zeros(stcs[0].data.shape, dtype=np.complex128)
    for stc in stcs:
        # divide by amplitude and sum angles
        tmp += stc.data / abs(stc.data)

    # take absolute value and normalize
    itc = abs(tmp) / n_trials

    return itc

###############################################################################
# Define frequency bands for iteration
# ------------------------------------
# We need to define the frequency bands for which we want to estimate source
# activity. We will look at somatosensory beta and gamma activity here and
# define three specific frequency bands: the beta band and two gamma bands:

# this would be a container to use for several freq bands.
# further below, we show, how you would loop over them.
iter_freqs = [
        ('Beta', 18, 28),  # keep transition width in mind
]

###############################################################################
# Hilbert beamformer pipeline
# ---------------------------
# Now everything is in place to enter the actual pipeline! We will loop over
# our three frequency bands and bring the data to source space for each of them.
# We will estimate the Hilbert envelope and intertrial coherence of the
# activity in every frequency band and keep that for later visualization. The
# setup of the pipeline is as follows:
#
# Rationale:
# ~~~~~~~~~~
# 1. _filter raw data_ : we will filter the data according to the frequency
#    bands.
# 2. _compute the covariance matrices_ : the filtered data will be epoched and
#    used to compute the covariance and noise covariance matrices for the
#    beamformer.
# 3. _spatial filter_ : we will compute the spatial filter for the frequency
#    band at hand.
# 4. _compute the Hilbert transform on raw data_ : we will take the filtered
#    raw data to compute the analytic signal (Hilbert transform) and create
#    epochs of that.
# 5. _apply the spatial filter_ : finally, we will apply the spatial filter on
#    the Hilbert data. From the source space output, we will compute intertrial
#    coherence using the real and imaginary part of the Hilbert signal, as well
#    as the envelope by taking the absolute value.

# initialize output lists:
hilbert_stcs = []  # hilbert envelope across freq bands
itcs = []  # ITC across freq bands

# loop over freq bands (this kind of loop would run, if several freq bands were
# defined above):
for ii, (band, fmin, fmax) in enumerate(iter_freqs):

    # print message to keep track:
    print('Processing frequency band %i of %i: %s'
          % (ii+1, len(iter_freqs), band))

    # we filter the data below, so we need to reload
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = read_raw_bids(bids_path, extra_params=dict(preload=True))

    # pick channels from raw
    raw.pick_types(meg=True)

    # 1. bandpass filter the data
    raw.filter(fmin, fmax, n_jobs=1, l_trans_bandwidth=7,
               h_trans_bandwidth=7, fir_design='firwin')

    # 2. compute the covariance matrices
    events, event_dict = mne.events_from_annotations(raw)
    epochs_cov = mne.Epochs(raw, events, event_id, tmin, tmax,
                            baseline=baseline, preload=True)

    # manually setting the rank, since data is severly rank deficient
    # the rank can be checked by using data_cov.plot(raw.info)
    data_cov = mne.compute_covariance(epochs_cov, tmin=0, tmax=None,
                                      rank=dict(meg=64))

    noise_cov = mne.compute_covariance(epochs_cov, tmin=None, tmax=-.05,
                                       rank=dict(meg=64))

    # 3. compute spatial filter
    # note that the data is severly rank deficient due to the maxfiltering
    # we thus manually set the rank here so the beamformer can take care of it
    filters = make_lcmv(epochs_cov.info, fwd, data_cov=data_cov,
                        noise_cov=noise_cov, pick_ori='max-power',
                        weight_norm='nai', reg=0.05, rank=dict(meg=64))

    del epochs_cov  # care for memory

    # 4. compute hilbert transform on raw data and epoch
    raw.apply_hilbert(n_jobs=1, envelope=False)
    epochs_hilb = mne.Epochs(raw, events, event_id, tmin, tmax,
                             baseline=None, preload=True)
    epochs_hilb._raw = None  # care for memory

    # 5. use spatial filter on hilbert data
    stcs = apply_lcmv_epochs(epochs_hilb, filters, max_ori_out='signed')

    # 6. compute intertrial coherence
    itcs.append(compute_source_itc(stcs))

    # 7. compute Hilbert envelope
    for stc in stcs:
        stc.data[:, :] = np.abs(stc.data)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stc.data = np.array(stc.data, 'float64')

    hilbert_stcs.append(stcs)

print('Done.')

###############################################################################
# Plot Hilbert amplitude in source space
# --------------------------------------
# So, next let's plot the output we got! We will start with visualizing the
# Hilbert amplitude (envelope) in source space: we will plot the spatial maps
# for the maximum time point as well as the time course in the maximum voxel
# for the three frequency bands.

# setup plotting:
colors = ['midnightblue', 'orangered', 'lightseagreen']
x_ax = np.linspace(hilbert_stcs[0][0].times.min()*1000,
                   hilbert_stcs[0][0].times.max()*1000,
                   hilbert_stcs[0][0].data.shape[1])

# we loop again over our 3 frequency bands:
mean_all = []
for ii, (color, stc_freq) in enumerate(zip(colors, hilbert_stcs)):

    # let's average Hilbert amplitudes across trials:
    mean_ts = np.mean([stc.data for stc in stc_freq], axis=0)

    # get maximum voxel and maximum time point:
    vox, timep = np.unravel_index(mean_ts.argmax(), mean_ts.shape)

    # plot the source data: we want to plot the average across trials
    stc_plotting = stc.copy()
    stc_plotting.data[:, :] = mean_ts
    stc_plotting.plot(src=fwd['src'], subjects_dir=freesurf_dir)


###############################################################################
# As we can see, we have typical late beta activity in somatosensory cortex.
# Furthermore, we see some sharp early gamma activity, peaking at 30 to 43 ms
# after the stimulation.

###############################################################################
# Plot intertrial coherence in source space and compare to amplitude
# ------------------------------------------------------------------
# Let's plot intertrial coherence:

# setup plotting:
colors = ['midnightblue', 'orangered', 'lightseagreen']

# we loop again over our 3 frequency bands:
for ii, (color, itc_freq) in enumerate(zip(colors, itcs)):

    # plot time course of maximum voxel:
    vox, timep = np.unravel_index(itc_freq.argmax(), itc_freq.shape)
    plt.figure(0);
    plt.plot(x_ax, itc_freq[vox, :], color=color)


plt.ylabel('ITC')
plt.xlabel('time [ms]')
plt.xlim(-150, 2000)
plt.title('Intertrial coherence in maximum voxels')
plt.legend([freq[0] for freq in iter_freqs])
plt.show();

###############################################################################
# Apparently, the late beta activity is not phase coherent across trials.
# Earlier activity is phase coherent and probably reflecting the event-related
# field.
