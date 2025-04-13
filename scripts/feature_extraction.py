import numpy as np
import librosa
import scipy.stats as st
import math
import librosa.display
import matplotlib.pyplot as plt

def compute_statistics(feature_array):
    """
    Given a 1D array of feature values, compute summary statistics.
    
    Returns:
      A dictionary with mean, standard deviation, skewness, and kurtosis.
    """
    stats = {}
    stats['mean'] = np.mean(feature_array)
    stats['std'] = np.std(feature_array)
    stats['skew'] = st.skew(feature_array)
    stats['kurtosis'] = st.kurtosis(feature_array)
    return stats

def spectral_entropy(S, eps=1e-10):
    """
    Compute spectral entropy from a power spectrum S (for a given frame).

    Parameters:
      S: 1D numpy array (power spectrum or squared magnitude spectrum)
      eps: Small value to avoid log(0)
    
    Returns:
      Spectral entropy (normalized)
    """
    # Normalize the spectrum to sum to 1 (i.e. convert it to a probability distribution)
    S_norm = S / (np.sum(S) + eps)
    entropy = -np.sum(S_norm * np.log2(S_norm + eps))
    # Normalization: maximum entropy equals log2(N) if N frequency bins
    entropy_norm = entropy / (np.log2(len(S_norm)) + eps)
    return entropy_norm

def extract_features_from_audio(audio, sr):
    """
    Extracts a set of features from an audio signal and summarizes them.

    Parameters:
      audio: 1D numpy array, preprocessed audio signal.
      sr: sampling rate of the audio signal.
    
    Returns:
      features: Dictionary with aggregated feature statistics.
    """
    features = {}

    # --- Temporal Features ---
    # RMS Energy (use frame length of 2048 and hop length of 512 samples)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    rms_stats = compute_statistics(rms)
    for key, value in rms_stats.items():
        features[f'rms_{key}'] = value

    # Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=2048, hop_length=512)[0]
    zcr_stats = compute_statistics(zcr)
    for key, value in zcr_stats.items():
        features[f'zcr_{key}'] = value

    # --- Spectral Features ---
    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=2048, hop_length=512)[0]
    centroid_stats = compute_statistics(spec_centroid)
    for key, value in centroid_stats.items():
        features[f'spectral_centroid_{key}'] = value

    # Spectral Bandwidth (as a proxy for spectral spread)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=2048, hop_length=512)[0]
    bandwidth_stats = compute_statistics(spec_bandwidth)
    for key, value in bandwidth_stats.items():
        features[f'spectral_bandwidth_{key}'] = value

    # Spectral Roll-off (90% of energy)
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.90, n_fft=2048, hop_length=512)[0]
    rolloff_stats = compute_statistics(spec_rolloff)
    for key, value in rolloff_stats.items():
        features[f'spectral_rolloff_{key}'] = value

    # Spectral Entropy: We compute it frame by frame on the power spectrogram.
    # First, compute the STFT and then the power spectrum:
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))**2
    entropy_values = np.apply_along_axis(spectral_entropy, 0, stft)
    entropy_stats = compute_statistics(entropy_values)
    for key, value in entropy_stats.items():
        features[f'spectral_entropy_{key}'] = value

    # --- Spectrotemporal Features ---
    # MFCCs (commonly 13 coefficients are used)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    # For each MFCC coefficient, compute aggregated statistics:
    for i in range(mfcc.shape[0]):
        coeff = mfcc[i, :]
        coeff_stats = compute_statistics(coeff)
        for key, value in coeff_stats.items():
            features[f'mfcc_{i+1}_{key}'] = value

    return features


# if __name__ == '__main__':
#     import librosa.display
#     import matplotlib.pyplot as plt

#     # Path to a preprocessed audio file
#     audio_file_path = 'processed_audio/tb_neg (1).wav'
    
#     # Load the audio (assumes it's already preprocessed; otherwise use your earlier preprocessing pipeline)
#     audio, sr = librosa.load(audio_file_path, sr=None)
#     print(f"Loaded {audio_file_path} at {sr} Hz")

#     # Extract features
#     features = extract_features_from_audio(audio, sr)

#     # Print out the extracted features
#     print("Extracted Features:")
#     for key, value in features.items():
#         print(f"{key}: {value:.4f}")

#     # Visualize the MFCCs for inspection
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(mfcc, sr=sr, hop_length=512, x_axis='time')
#     plt.title('MFCC')
#     plt.colorbar()
#     plt.tight_layout()
#     plt.show()
