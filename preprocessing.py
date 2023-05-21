import os, sys
sys.path.append(os.getcwd())
import numpy as np
from scipy.signal import butter, sosfiltfilt, lfilter
import librosa as lb
from brian2 import Hz, kHz
from brian2hears import Sound, erbspace, Gammatone, Filterbank
import globals


def load_eeg_data(filename):
    data = np.load(filename)
    return data

def bandpass_filter_eeg(eeg_data, fs, low_freq, high_freq):
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(1, [low, high], btype='band')
    filtered_eeg = lfilter(b, a, eeg_data, axis=0)
    return filtered_eeg

def downsample_eeg(filtered_eeg, fs, target_fs):
    downsample_ratio = fs // target_fs
    downsampled_eeg = lb.resample(filtered_eeg.T, orig_sr=fs, target_sr=target_fs).T
    return downsampled_eeg



def preprocessing_audio(filepath,downsampled_eeg_array):

    # ----- Audio data -----

    # Step 1 : Read audio filename.wav to the variable A.
    audio_signal, sr = lb.load(filepath, sr=44100)

    #print(f"Audio signal length: {len(audio_signal)}")

    # Convert the numpy array to a Sound object
    audio_sound = Sound(audio_signal, samplerate=sr*Hz)
    num_filters = 28

    # Step 2 : Decompose the audio in 28 frequency subbands
    center_freqs = np.linspace(50*Hz, 5*kHz, num_filters)
    gammatone_filterbank = Gammatone(audio_sound, center_freqs)

    # Step 3 & 4 : Perform a power-law compression on each subband signals and linearly
    # combine the bandpass filtered envelopes to a single envelope (See EnvelopeFromGammatoneFilterbank() class)
    envelope_calculation = EnvelopeFromGammatoneFilterbank(gammatone_filterbank)
    combined_envelope = envelope_calculation.process()

    # Step 5 : Filter the audio using a BPF (butterworth)
    
    # (5.1) Choose the frequency range based on the method
    low_freq = 1
    high_freq = 9

    # (5.2) Design a Butterworth bandpass filter
    nyquist = 0.5 * sr
    low = low_freq / nyquist
    high = high_freq / nyquist

    sos = butter(globals.butter_order, [low, high], btype='band', output='sos')

    normalized_envelope = combined_envelope / np.max(combined_envelope)
    filtered_envelope = sosfiltfilt(sos, normalized_envelope[:, 0])

    # Step 6 : Downsample the resulting signals
    # Choose the downsampling rate based on the method
    downsampling_rate = 20

    resampled_envelope = lb.resample(filtered_envelope, orig_sr=sr, target_sr=downsampling_rate)

    return resampled_envelope[:downsampled_eeg_array.shape[0]]


def preprocessing_eeg(eeg_path):
    
    # ----- EEG data -----

    # Step 1 : Load the EEG data
    
    eeg_data = load_eeg_data(eeg_path)
    eeg = eeg_data['eeg']
    fs = eeg_data['fs']

    #print("Speech attended: ", eeg_data['stimulus_attended'])
    #print("Speech unattended: ",eeg_data['stimulus_unattended'])

    # Step 2 : Filter the EEG signals using a bandpass filter
    low_freq, high_freq = 1, 9  # For linear regression
    filtered_eeg = bandpass_filter_eeg(eeg, fs, low_freq, high_freq)

    # Step 3 : Downsample the EEG signals
    target_fs = 20  # For linear regression
    downsampled_eeg_array = downsample_eeg(filtered_eeg, fs, target_fs)

    return downsampled_eeg_array, eeg_data['stimulus_attended'], eeg_data['stimulus_unattended']




class EnvelopeFromGammatoneFilterbank(Filterbank):
    """Converts the output of a GammatoneFilterbank to an envelope."""

    def __init__(self, source):
        """Initialize the envelope transformation.

        Parameters
        ----------
        source : Gammatone
            Gammatone filterbank output to convert to envelope
        """
        super().__init__(source)

        self.nchannels = 1

    def buffer_apply(self, input_):

        # 6. take absolute value of the input_
        compressed_subbands = np.abs(input_)**0.6

        combined_envelope = np.sum(compressed_subbands, axis=1)

        return  combined_envelope.reshape(combined_envelope.shape[0], 1)
    

