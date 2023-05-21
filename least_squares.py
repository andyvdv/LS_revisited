from construct_m import *
from get_speeches import *
import globals




def least_squares(k, chunk):
    M = []
    SA = []

    all_files = globals.all_files
    K = globals.K
    max_length = globals.max_length
    window_size = globals.window_size

    # Prepare M & SA arrays from all instances but k
    #TODO
    for i in range(K):
        if i != k:
            m_i_chunk, sa_i_chunk, _ = build(all_files[i], chunk)
            
            m_i_chunk = m_i_chunk.T

            M.append(m_i_chunk)
            SA.append(sa_i_chunk)


    # Concatenate all M and SA
    M_stacked = np.concatenate(M, axis=1)
    SA_stacked = np.concatenate(SA)

    # Perform least squares using np.linalg.lstsq
    d_k, _, _, _ = np.linalg.lstsq(M_stacked.T, SA_stacked.T, rcond=None)

    return d_k



