# Multivariate_response_function_Machine_Learning
Machine_Learning_Ridge_regression
import numpy as np 
from pymtrf.mtrf import mtrf_predict, mtrf_crossval, mtrf_train, lag_gen
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
%matplotlib inline
import os
PATH = 'data'
# This is an example overview for the python translation of the mtrf toolbox: 
Examples described in the MATLAB README
See the documentation for the toolbox here: https://www.frontiersin.org/articles/10.3389/fnhum.2016.00604/full And the matlab version of the toolbox here: https://sourceforge.net/projects/aespa/

See http://www.mee.tcd.ie/lalorlab/resources.html for an example of temporal coherence stimuli.

Contrast: Forward Model (TRF/VESPA)
The data (taken from the original README):
contrast_data.mat
This MATLAB file contains 3 variables. The first is a matrix consisting of 120 seconds of 128-channel EEG data. The second is a vector consisting of a normalised sequence of numbers that indicate the contrast of a checkerboard that was presented during the EEG at a rate of 60 Hz. The third is a scaler which represents the sample rate of the contrast signal and EEG data (128 Hz). See Lalor et al. (2006) for further details.

Lalor EC, Pearlmutter BA, Reilly RB, McDarby G, Foxe JJ (2006) The VESPA: a method for the rapid estimation of a visual evoked potential. https://doi.org/10.1016/j.neuroimage.2006.05.054

data = loadmat(f'{PATH}{os.sep}contrast_data.mat')
print(data.keys())
dict_keys(['__header__', '__version__', '__globals__', 'Fs', 'EEG', 'contrastLevel'])
EEG = data['EEG']
Fs = data['Fs']
contrast_level = data['contrastLevel']
A quick look at the data
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.plot(contrast_level[: 10 * Fs.item()])
plt.title('10 Seconds of contrast data')
plt.subplot(122)
plt.plot(EEG[: 10 * Fs.item(), [10, 30, 40]])
plt.title('10 Seconds of EEG data')
Text(0.5, 1.0, '10 Seconds of EEG data')

Calculating a TRF
As described in the introduction our goal is now in the forward model, find the function, that when convolved with the contrast data creates the EEG.

Using the toolbox this is fairly simple:

[w, t, i] = mtrf_train(contrast_level, EEG, 128, 1, -150, 450, 1)
print(f'There are three outputs, the model coeffecients w, shape={w.shape}')
print(f'The model time lags in use t, shape={t.shape}')
print(f'And a bias term i, shape={i.shape}')
There are three outputs, the model coeffecients w, shape=(1, 79, 128)
The model time lags in use t, shape=(79,)
And a bias term i, shape=(1, 128)
We see that we have a matrix of coefficients for each channel in the EEG and for each time lag created by the function (i.e. between -150, 450). Note, that if we would have used more complex data as an input (for example several contrast levels), the first dimension of w would change accordingly.

plt.figure(figsize=(15,5))
ax = plt.subplot(121)
ax = sns.heatmap(w[0,8:72].T, xticklabels=12, ax=ax, cmap='coolwarm')
ax.set_xticklabels(np.round(t[9:71:8]))
plt.ylabel('Channel')
plt.xlabel('Time lag (ms)')
plt.title('Amplitude over Channels')

plt.subplot(122)
plt.plot(t, w[0][:, 22])
plt.ylabel('Amplitude (a.u)')
plt.xlabel('Time lag (ms)')
plt.xlim([-100, 400])
plt.title('Amplitude for Channel 23')
Text(0.5, 1.0, 'Amplitude for Channel 23')

On the left, we see the a heatmap of the fitted weights. On the right, there is the TRF that was estimates for channel 23.

Motion: Forward model (TRF/VESPA)
The data (taken from the original README):
coherentmotion_data.mat
This MATLAB file contains 3 variables. The first is a matrix consisting of 200 seconds of 128-channel EEG data. The second is a vector consisting of a normalised sequence of numbers that indicate the motion coherence of a dot field that was presented during the EEG at a rate of 60 Hz. The third is a scaler which represents the sample rate of the motion signal and EEG data (128 Hz). See Gonçalves et al. (2014) for further details.

Gonçalves NR, Whelan R, Foxe JJ, Lalor EC (2014) Towards obtaining spatiotemporally precise responses to continuous sensory stimuli in humans: a general linear modeling approach to EEG. NeuroImage 97(2014):196-205. DOI: https://doi.org/10.1016/j.neuroimage.2014.04.012

data = loadmat(f'{PATH}{os.sep}coherentmotion_data.mat')
print(data.keys())
dict_keys(['__header__', '__version__', '__globals__', 'Fs', 'EEG', 'coherentMotionLevel'])
EEG = data['EEG']
Fs = data['Fs']
coherent_motion_level = data['coherentMotionLevel']
[w, t, i] = mtrf_train(coherent_motion_level, EEG, 128, 1, -150, 450, 1)
plt.figure(figsize=(15,5))
ax = plt.subplot(121)
ax = sns.heatmap(w[0,8:72].T, xticklabels=12, ax=ax, cmap='coolwarm')
ax.set_xticklabels(np.round(t[9:71:8]))
plt.ylabel('Channel')
plt.xlabel('Time lag (ms)')
plt.title('Amplitude over Channels')

plt.subplot(122)
plt.plot(t, w[0][:, 20])
plt.ylabel('Amplitude (a.u)')
plt.xlabel('Time lag (ms)')
plt.xlim([-100, 400])
plt.title('Amplitude for Channel 23')
Text(0.5, 1.0, 'Amplitude for Channel 23')

Speech Approaches
The data (taken from the original README):
speech_data.mat
This MATLAB file contains 4 variables. The first is a matrix consisting of 120 seconds of 128-channel EEG data. The second is a matrix consisting of a speech spectrogram. This was calculated by band-pass filtering the speech signal into 128 logarithmically-spaced frequency bands between 100 and 4000 Hz and taking the Hilbert transform at each frequency band. The spectrogram was then downsampled to 16 frequency bands by averaging across every 8 neighbouring frequency bands. The third variable is the broadband envelope, obtained by taking the mean across the 16 narrowband envelopes. The fourth variable is a scaler which represents the sample rate of the envelope, spectrogram and EEG data (128 Hz). See Lalor & Foxe (2010) for further details.

Lalor, EC, & Foxe, JJ (2010) Neural responses to uninterrupted natural speech can be extracted with precise temporal resolution. Eur J Neurosci 31(1):189-193. DOI: https://doi.org/10.1111/j.1460-9568.2009.07055.x

data = loadmat(f'{PATH}{os.sep}speech_data.mat')
print(data.keys())
dict_keys(['__header__', '__version__', '__globals__', 'Fs', 'spectrogram', 'envelope', 'EEG'])
EEG = data['EEG']
Fs = data['Fs']
spectrogram = data['spectrogram']
envelope = data['envelope']
Speech: Forward model (TRF/AESPA)
[w, t, _] = mtrf_train(envelope, EEG, 128, 1, -150, 450, 0.1)
plt.figure(figsize=(15,5))
ax = plt.subplot(121)
ax = sns.heatmap(w[0,8:72].T, xticklabels=12, ax=ax, cmap='coolwarm')
ax.set_xticklabels(np.round(t[9:71:8]))
plt.ylabel('Channel')
plt.xlabel('Time lag (ms)')
plt.title('Amplitude over Channels')

plt.subplot(122)
plt.plot(t, w[0][:, 84])
plt.ylabel('Amplitude (a.u)')
plt.xlabel('Time lag (ms)')
plt.xlim([-100, 400])
plt.title('Amplitude for Channel 85')
Text(0.5, 1.0, 'Amplitude for Channel 85')

Speech: Spectrotemporal forward model (TRF/AESPA)
[w, t, _] = mtrf_train(spectrogram, EEG, 128, 1, -150, 450, 100)
plt.figure(figsize=(15,5))
ax = plt.subplot(121)
ax = sns.heatmap(w[:,8:73, 84], xticklabels=8, ax=ax, cmap='coolwarm')
ax.set_xticklabels(np.round(t[8:73:8]))
plt.ylabel('Frequency Band')
plt.xlabel('Time lag (ms)')
plt.title('Frequency Amplitude for Channel 85')
Text(0.5, 1.0, 'Frequency Amplitude for Channel 85')

Speech: Backward model (stimulus reconstruction)
from scipy.signal import resample
envelope = resample(envelope, np.int(envelope.shape[0]/2))
EEG = resample(EEG, np.int(EEG.shape[0]/2))
D:\Miniconda3\envs\mypy\lib\site-packages\scipy\signal\signaltools.py:2223: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  Y[sl] = X[sl]
D:\Miniconda3\envs\mypy\lib\site-packages\scipy\signal\signaltools.py:2225: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  Y[sl] = X[sl]
stim_train = envelope[:64 * 60]
eeg_train = EEG[:64 * 60]
stim_test = envelope[64 * 60:]
eeg_test = EEG[64 * 60:]
[g, t, con] = mtrf_train(stim_train, eeg_train, 64, -1, 0, 500, 1e5)
D:\Miniconda3\envs\mypy\lib\site-packages\pymtrf\helper.py:67: UserWarning: X: more features 4352 than samples 3840, check input dimensions!
  f' than samples {X.shape[0]}, check input dimensions!')
[recon, r, p, mse] = mtrf_predict(stim_test, eeg_test, g, 64, -1, 0, 500, con) 
plt.plot(stim_test)
plt.plot(recon)
plt.legend(['Original', 'Predicted'])
plt.title(f'Reconstruction with r={r.item():.3f} \n and p={p.item():.3f}')
Text(0.5, 1.0, 'Reconstruction with r=0.155 \n and p=0.000')
