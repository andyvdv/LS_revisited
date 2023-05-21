import os, sys
sys.path.append(os.getcwd())

# Preprocessing AUDIO
butter_order = 1
folder_audio = os.getcwd() + "/data/stimuli"
folder_audio_processed = os.getcwd() + "/data/audioprocessed"

# Preprocessing EEG
folder_eeg =  os.getcwd() + "/data/eeg/sub6"
all_files = os.listdir(folder_eeg); all_files.sort()#; all_files = all_files[1:]

# Construct m
Nl = int(250 * pow(10, -3) * 20)
t = 0
scale_input = True


# --- Parameters too choose ---
K = len(all_files) # Number of EEG trials to use (Choose 50, 100, 200, 300 or 484)
max_test = K # Number of EEG trials that we actually one to test and look for their average accuracy (Put K if you want to use all)
start_test = 0

w_secs = 5
window_size = 20*w_secs
max_length = 10000


max_chunk = 50 #max_length//window_size # Number of chunks of size window_size to use


# 12 for 60,
# 19  for 45
# 25 for 30
folder_dk = os.getcwd() + f"/data/d_k" # d_k file to use (Ex: replace d_k with d_k_K100_30s for precomputed d_k with K=100, w_length = 30s * 20Hz = 600)
# ------------ END ------------

