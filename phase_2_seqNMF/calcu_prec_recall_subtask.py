
# Load the g_estimate from C.mat file.
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy
import os
import argparse
import pdb

def smooth_option(C, window_size=3):
    # C: (T)
    # return: (T)
    C_smoothed = np.zeros_like(C)
    # detect change
    C_diff = C[1:] - C[:-1]
    C_diff = np.abs(C_diff)
    def is_constant_window(id_):
        if len(np.unique(C[id_:id_+window_size])) == 1:
            return True
        else:
            return False
    
    # smooth
    for id in range(len(C)):
        # If applicable, continue the previous smoothed value
        if id > 0:
            if C_diff[id-1] == 0:
                C_smoothed[id] = C_smoothed[id-1]
                continue
        # If the value has changes  and the current window is constant, keep it
        if is_constant_window(id_=id):
            C_smoothed[id] = C[id]
        else:# If the value has changes,  and the current window is NOT constant, find the nearest constant window
            t_shift = 0
            while not is_constant_window(id_=id+t_shift):
                t_shift += 1
            C_smoothed[id] = C[id+t_shift]
    return C_smoothed


parser = argparse.ArgumentParser()
parser.add_argument('--C_file_path', type=str, default='C_est_data/C_est_color_seed1.mat')
parser.add_argument('--L', type=int, default=3)
args = parser.parse_args()

# Load the g_estimate from C.mat file.
C_mat = scipy.io.loadmat(args.C_file_path)['C'] # 3, 300


# ground truth boundary is [3, 6, 9, ..., 297]
C_ground_truth = np.zeros(C_mat.shape[1])
C_ground_truth[np.arange(args.L, 300, args.L)] = 1

C_estimate_boundary = np.zeros(C_mat.shape[1]) # 300
option_seq = np.argmax(C_mat, axis=0)

if args.L > 3:
    option_seq = smooth_option(option_seq)

def is_already_done(option_seq, id, boundary, L=3):
    if id < 2:
        return False
    elif id == 2:
        return option_seq[id] == option_seq[id-1] and option_seq[id-1] == option_seq[id-2]
    else:
        return option_seq[id] == option_seq[id-1] and option_seq[id-1] == option_seq[id-2] and boundary[id-L] == 1

for i in range(C_mat.shape[1] - 1):
    C_estimate_boundary[i] = 1 if (option_seq[i] != option_seq[i+1] or is_already_done(option_seq, i, boundary=C_estimate_boundary, L=args.L)) else 0

# Calculate the precision
precision = np.sum(C_estimate_boundary * C_ground_truth) / np.sum(C_estimate_boundary)

# Calculate the recall
recall = np.sum(C_estimate_boundary * C_ground_truth) / np.sum(C_ground_truth)

# Calculate the F1 score
f1 = 2 * (precision * recall) / (precision + recall)
print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')