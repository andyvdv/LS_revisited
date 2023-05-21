import os, sys
sys.path.append(os.getcwd())
import numpy as np
from scipy.signal import butter, sosfiltfilt, lfilter
import librosa as lb
from brian2 import Hz, kHz
from brian2hears import Sound, erbspace, Gammatone, Filterbank
import globals
from preprocessing import *


def get_speech_resampled(sa_path,eeg_array):
    """
    The function first constructs the file path for the processed SA/SU and checks if the file already exists.
    If the file exists, it loads the SA/SU from the .npy file and returns it. 
    If the file does not exist, the function calls the preprocessing function to process the resampled SA and the corresponding EEG data. 
    The processed SA/SU is then saved as a .npy file in the audioprocessed folder and returned.

    Parameters
    ----------
    resampled_path : str
        The path to the resampled attended/unattended speech signal file.

    Returns
    -------
    s : ndarray
        The resampled attended/unattended speech signal.
    """

    last_path = sa_path.split('/')[-1].split('.')[0] + '.npy'
    
    folder_audioprocessed =  globals.folder_audio_processed # audio
    npy_path = os.path.join(folder_audioprocessed, last_path)

    if os.path.exists(npy_path) == False:
        
        s =  preprocessing_audio(sa_path, eeg_array)
        np.save(npy_path, s)
        return s
    
    
    return np.load(npy_path)



def get_sa_su_eeg(eeg_name):
    folder_eeg = globals.folder_eeg # audio
    eeg_path = os.path.join(folder_eeg,eeg_name)
    
    eeg, sa_name, su_name = preprocessing_eeg(eeg_path) # get the processed eeg and the attended speech name of that EEG

    # ---- GET ATTENDED SPEECH ARRAY
    folder_audio = globals.folder_audio # audio
    resampled_sa_path = os.path.join(folder_audio,str(sa_name))
    sa = get_speech_resampled(resampled_sa_path,eeg_path)

    # ---- GET UNATTENDED SPEECH ARRAY
    resampled_su_path = os.path.join(folder_audio,str(su_name))
    su = get_speech_resampled(resampled_su_path,eeg_path)

    sa = np.array(sa); su = np.array(su); eeg = np.array(eeg)

    return sa, su, eeg

