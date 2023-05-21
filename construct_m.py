import numpy as np
from sklearn.preprocessing import MinMaxScaler
import globals
from get_speeches import *

def build(eeg_name, chunk):
    sa, su, eeg = get_sa_su_eeg(eeg_name)

    Nl = globals.Nl
    t = globals.t
    window_size = globals.window_size

    max_length = len(sa)
    m = np.array(eeg[t:Nl]).ravel('F')

    # Determine the starting index of the chunk
    start_index = chunk * window_size

    # Ensure we do not exceed the array bounds
    if start_index >= max_length:
        raise IndexError("Chunk index exceeds data length.")
    
    # Determine the end index for the chunk
    end_index = start_index + window_size
    if end_index > max_length:
        end_index = max_length
    
    # Extract the chunk from the data
    sa_chunk = sa[start_index:end_index]
    su_chunk = su[start_index:end_index]
    eeg_chunk = eeg[start_index:end_index]


    # Prepare the m matrix and perform scaling
    m_t_chunk = np.zeros((window_size - Nl + 1, m.shape[0]))

    for i in range(window_size - Nl + 1):
        if eeg[start_index + i:start_index + i + Nl].shape[0] != Nl:
            m_t_chunk = m_t_chunk[:i]  # Adjust to only include rows that match EEG window size
            break
        m_t_chunk[i] = np.array(eeg[start_index + i:start_index + i + Nl]).ravel('F')

    m_chunk = m_t_chunk
    sa_chunk = sa_chunk[:m_chunk.shape[0]].reshape(-1,1)
    su_chunk = su_chunk[:m_chunk.shape[0]].reshape(-1,1)
    
 
    # Apply MinMaxScaler to the EEG data
    if globals.scale_input:
        scaler = MinMaxScaler()
        m_chunk = scaler.fit_transform(m_chunk)
        sa_chunk = scaler.fit_transform(sa_chunk).flatten()
        su_chunk = scaler.fit_transform(su_chunk).flatten()

    
    
    return m_chunk, sa_chunk, su_chunk
    


    









