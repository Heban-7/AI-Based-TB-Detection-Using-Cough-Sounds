import os
import numpy as np
import logging
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Create a Butterworth bandpass filter.
    
    Parameters:
    - lowcut: Lower frequency cutoff (Hz).
    - highcut: Upper frequency cutoff (Hz).
    - fs: Sampling frequency (Hz).
    - order: The order of the filter.
    
    Returns:
    - b, a: Filter coefficients.
    """
    nyq = 0.5 * fs  # Nyquist frequency.
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0:
        high = 0.99
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to the data.
    
    Parameters:
    - data: The input audio signal.
    - lowcut: Lower frequency cutoff.
    - highcut: Upper frequency cutoff.
    - fs: Sampling frequency.
    - order: Filter order.
    
    Returns:
    - Filtered audio signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def normalize_audio(audio):
    """
    Normalize the audio signal so that its maximum absolute value is 1.
    
    Parameters:
    - audio: The input audio signal.
    
    Returns:
    - Normalized audio signal.
    """
    if np.max(np.abs(audio)) == 0:
        return audio
    return audio / np.max(np.abs(audio))

def preprocess_audio_file(file_path, target_sr=16000, lowcut=100, highcut=8000, trim_db=20):
    """
    Preprocess a single audio file:
      1. Load the audio.
      2. Resample it to the target sample rate if necessary.
      3. Apply a bandpass filter.
      4. Trim silence from the beginning and end.
      5. Normalize the audio signal.
    
    Parameters:
    - file_path: Path to the audio file.
    - target_sr: The target sampling rate (Hz).
    - lowcut: Lower cutoff for bandpass filtering (Hz).
    - highcut: Upper cutoff for bandpass filtering (Hz).
    - trim_db: The threshold (in dB) below reference to trim silence.
    
    Returns:
    - processed_audio: The preprocessed audio signal.
    - sr: The sampling rate of the processed audio.
    """
    # Load the audio file at its original sampling rate.
    audio, sr = librosa.load(file_path, sr=None)
    print(f"Loaded {file_path} with original sample rate: {sr} Hz")
    
    # Resample if the original sampling rate doesn't match the target.
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        print(f"Resampled to {target_sr} Hz")
    
    # Apply bandpass filter to reduce noise outside the frequency range of interest.
    audio = bandpass_filter(audio, lowcut, highcut, sr, order=5)
    print("Applied bandpass filtering")
    
    # Trim leading and trailing silence from the audio.
    audio, _ = librosa.effects.trim(audio, top_db=trim_db)
    
    # Normalize the audio to ensure consistent volume.
    audio = normalize_audio(audio)
    print("Normalized audio amplitude")
    
    return audio, sr

def preprocess_directory(input_dir, output_dir, target_sr=16000, lowcut=100, highcut=8000, trim_db=20):
    """
    Preprocess all .wav files in a specified input directory and save the processed files to the output directory.
    
    Parameters:
    - input_dir: Directory where raw audio files are stored.
    - output_dir: Directory to save preprocessed audio files.
    - target_sr: The target sampling rate.
    - lowcut: Lower frequency cutoff for filtering.
    - highcut: Upper frequency cutoff for filtering.
    - trim_db: dB threshold for silence trimming.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_dir, file_name)
            print(f"\nProcessing file: {file_name}")
            processed_audio, sr = preprocess_audio_file(file_path, target_sr, lowcut, highcut, trim_db)
            
            # Create the output file path and save the processed audio.
            output_file_path = os.path.join(output_dir, file_name)
            sf.write(output_file_path, processed_audio, sr)
            print(f"Saved processed audio to: {output_file_path}")

# if __name__ == '__main__':
#     # Define directories where raw and processed audio files are stored.
#     input_directory = '../data/raw_audio'        # Folder containing raw cough audio files (make sure these are .wav format)
#     output_directory = '../data/processed_audio'   # Folder where preprocessed audio files will be saved
    
#     # Run the preprocessing on the directory of audio files.
#     preprocess_directory(input_directory, output_directory, target_sr=16000, lowcut=100, highcut=8000, trim_db=20)
