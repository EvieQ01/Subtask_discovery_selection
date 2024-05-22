import numpy as np
import torch
from scipy.io import savemat
import argparse
import matplotlib.pyplot as plt
import random

def get_g_pluscomplex_pattern():
    return random.choice([np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]),
                    np.array([2, 2, 2, 1, 1, 1, 0, 0, 0, 0])])

def get_g_plus(g_t):
    '''
    Generate a list of c markovianly
    '''
    p_0_0 = 0.8
    p_1_1 = 0.9
    p_2_2 = 0.9
    if g_t == 0:
        if random.random() < p_0_0:
            return 0
        else:
            return 1
    elif g_t == 1:
        if random.random() < p_1_1:
            return 1
        else:
            return 2
    elif g_t == 2:
        if random.random() < p_2_2:
            return 2
        else:
            return 0
    
def gen_selection_seq(length=300, lag=3):
    
    # complex pattern with lag == 10
    if lag == 10:
        g_seq = np.random.randint(0, 3, length)
        for i in range(length // lag):
            g_seq[i * lag:i * lag + lag] = get_g_pluscomplex_pattern()
        return g_seq
    else:
        # # 3-step markovian
        assert length % lag == 0
        single_c = np.random.randint(0, 3, length // lag) # (100), {0,1,2}
        return np.repeat(single_c, 3) # (300), {0, 0, 0, 1, 1, 1, 2, 2, ...}

    # Markovian generation
    # g_seq = np.random.randint(0, 3, length)
    # for i in range(length - 1):
    #     g_seq[i + 1] = get_g_plus(g_seq[i])
    # return g_seq

def gen_s_a_seq(g_seq, s0=0, noise_var=0.0):
    # length = len(g_seq)
        s_seq = np.concatenate((np.array([s0]), g_seq[:-1])) # state is always one step slower than c
        if noise_var != 0.:
            # Add noise
            noise = np.random.normal(0, np.sqrt(noise_var), size=s_seq.shape)  # Sample noise from the Gaussian distribution
            s_seq = s_seq + noise
            
        # select action
        a_seq = g_seq - s_seq

        s_plus_seq = np.zeros_like(s_seq)
        s_plus_seq[:-1] = s_seq[1:]
        if noise_var == 0.:
            assert (s_plus_seq == s_seq + a_seq).all()
        return s_seq, a_seq

def one_hot_encode(arr, num_classes):
    # Create an identity matrix with the number of classes
    identity_matrix = np.eye(num_classes)
    
    # Index the identity matrix with the input array
    one_hot_encoded = identity_matrix[arr]
    
    return one_hot_encoded.T

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic data for NMF')
    parser.add_argument('--data', type=str, help='data name', default='color_seq', choices=['color_seq'])
    parser.add_argument('--lag', type=int, help='length of pattern', default=3)
    parser.add_argument('--normalize', type=bool, help='normalize', default='True')
    args = parser.parse_args()

    if args.data == "color_seq":
        g_seq = gen_selection_seq(length=300, lag=args.lag)
        s_seq, a_seq = gen_s_a_seq(g_seq, s0=0)
        s_seq_noise, a_seq_noise = gen_s_a_seq(g_seq, s0=0, noise_var=0.01)
        # s_a_seq = np.stack((s_seq, a_seq))
        
        # Define the arrays to be saved
        arrays_to_save = {
            'state': s_seq,
            'action': a_seq,
            'g_seq': g_seq,
            'state_onehot': one_hot_encode(s_seq, num_classes=3 if args.lag == 10 else args.lag),
            'action_onehot': one_hot_encode(a_seq, num_classes=5 if args.lag == 10 else args.lag * 2 - 1)
        }
        # Save the NumPy array as a .mat file
        savemat(f'../seqNMF/color_{str(args.lag)}.mat', arrays_to_save)
        # Save the arrays to a .npz file (noised ones for CI test)
        np.savez(f'../seqNMF/color_{str(args.lag)}.npz', **arrays_to_save)
        np.savez(f'../seqNMF/color_{str(args.lag)}_noise.npz', **{
                    'state': s_seq_noise,
                    'action': a_seq_noise,
                    'sel': g_seq,
        })

