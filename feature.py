# import
import numpy as np
from scipy import signal
import librosa
import sys

def feature1_LogMelEnergies(window,
                    sr = 48000,
                    n_mels=64,
                    n_frames=1,
                    n_fft=1024,
                    hop_length=512,
                    power=2.0):
    """
    Get LogMelEnergies from a window

    Args:
        window (np.array(sampleCount,))

    Returns:
        vectors (np.array(frameCount, melCount)) 

    Reference: https://github.com/Kota-Dohi/dcase2022_task2_baseline_ae/blob/main/common.py
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # generate melspectrogram using librosa
    y = window
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    # convert melspectrogram to log mel energies
    log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))

    # calculate total vector size
    n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

    # skip too short clips
    if n_vectors < 1:
        return np.empty((0, dims))

    # generate feature vectors by concatenating multiframes
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[:, t : t + n_vectors].T

    return vectors 


# Add other feature extractors here
# Note: Currently, we onlyextract features from each window. Later, we can extract features from multiple windows.
# Input: 
# window: 
#        window of one sensor
#        shape(sampleCountOfOneWindow, ) 

# Output:
# vectors:
#        feature of this window
#        shape(frameCount, featureCount)