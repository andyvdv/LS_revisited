import globals
from timeit import default_timer as timer
import os, sys
sys.path.append(os.getcwd())
from least_squares import *
from accuracy import *
import struct
import sqlite3
import os
import pickle


if __name__ == "__main__":
    K = globals.K # Number of EEG trials to use (Choose 50, 100, 200, 300 or 484)
    window_size = globals.window_size # Window length: Number of samples to use (Choose 1200, 600, 300, 100) = (60s, 30s, 15s, 5s)
    max_length = globals.max_length # Maximum length of the speechs to take into account
    folder_dk = globals.folder_dk # d_k file to use (Ex: replace d_k with d_k_K100_30s for precomputed d_k with K=100, w_length = 30s * 20Hz = 600)
    max_chunk = globals.max_chunk
    all_files = globals.all_files
    overall_accuracy = []


    for chunk in range(max_chunk):
        print(f"--- Processing for chunk = {chunk}----")
        print("\n")
        sum_correct = 0
        for k in range(K):
            #print(f"Processing k = {k}")

            start = timer()
            d_k = least_squares(k, chunk)
            end = timer()

            print("Elapsed time for d_{0} : {1} seconds".format(k, round(end-start,4)))

            #print(" -> TEST FOR k =",k)
            m_test, sa_test, su_test = build(globals.all_files[k],chunk)
            sum_correct += test_accuracy(sa_test, su_test, d_k, m_test.T)
        
        accuracy = 100 * sum_correct/K
        print("Test accuracy with LOOCV = {} %".format(accuracy))
        overall_accuracy.append(accuracy)
        print(f'Overall Accuracy accross the whole data (at chunk {chunk}) = {np.mean(np.array(overall_accuracy))}')

    print(f"All the average test accuracy: {overall_accuracy}")
    print(f'Overall Accuracy accross the whole data = {np.mean(np.array(overall_accuracy))}')









    
