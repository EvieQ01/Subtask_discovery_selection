import argparse
from causallearn.utils.cit import CIT
import torch
import numpy as np
test_name = 'kci'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic data for NMF')
    parser.add_argument('--data', type=str, help='data name', default='color_seq', choices=['color_seq', 'driving_seq'])
    parser.add_argument('--lag', type=int, help='length of pattern', default=3)
    parser.add_argument('--count_of_demos', default=100, type=int,help='max trajectory count')
    
    args = parser.parse_args()

    if args.data == "color_seq":
        # Define the file path
        file_path = f'../seqNMF/color_{str(args.lag)}_noise.npz'

        # Load the arrays from the .npz file
        loaded_data = np.load(file_path)

        # Access the individual arrays by their names
        s_seq_loaded = loaded_data['state']
        a_seq_loaded = loaded_data['action']
        g_seq_loaded = loaded_data['sel']
        
        s_data = np.expand_dims(s_seq_loaded, axis=-1)
        a_data = np.expand_dims(a_seq_loaded, axis=-1)
        g_data = np.expand_dims(g_seq_loaded, axis=-1)
        
        s_plus_data = s_data[1:, :].copy()
        a_plus_data = a_data[1:, :].copy()
        g_plus_data = g_data[1:, :].copy()
        
        s_data = s_data[:-1, :]
        a_data = a_data[:-1, :]
        g_data = g_data[:-1, :]
        
        X = np.concatenate((s_data, a_data, s_plus_data, g_data, a_plus_data, g_plus_data), axis=1) # (1 + 1 + 1) * 999
        
        cit_obj = CIT(X, 'fisherz') # KCI for real world
        print('X sample (s, a, s_(t+1), c):\n', X[:5])
        p_value_s_a = cit_obj(X=2, Y=3, condition_set=[0, 1])
        print('p_value for CI test between s_(t+1) and g_t given (s_t, a_t): ', p_value_s_a)
        
        p_value_s_a = cit_obj(X=2, Y=3, condition_set=[1])
        print('p_value for CI test between s_(t+1) and g_t given (a_t): ', p_value_s_a)
        
        p_value_s_a = cit_obj(X=0, Y=1)
        print('p_value for CI test between s_t and a_t: ', p_value_s_a)
        
        p_value_s_a = cit_obj(X=0, Y=1, condition_set=[3])
        print('p_value for CI test between s_t and a_t (must) given g_t: ', p_value_s_a)
        
        p_value_s_a = cit_obj(X=3, Y=4, condition_set=[5])
        print('p_value for CI test between g_t and a_{t+1} (must) given g_{t+1}: ', p_value_s_a)

    if args.data == "driving_seq":
        fix_len = 100000
        single_len = 100
        n_samples = fix_len // single_len
        feature_dim = 2 # heading
        action_dim = 1
  
        # Define the file path
        file_path = f'../seqNMF/driving_{str(fix_len)}.npz'

        # Load the arrays from the .npz file
        loaded_data = np.load(file_path)

        # Access the individual arrays by their names
        s_seq_loaded = loaded_data['state']
        a_seq_loaded = loaded_data['action']
        g_seq_loaded = loaded_data['sel']
        
        def calculate_p_values(X, use_single_step=True):
            """Calculate p values for each pair of states and actions"""
            # X.shape: (n_trajs, 100, 6)
            # create list
            p_value_list_1 = [] # 0., dependent
            p_value_list_2 = []  # 0.99, independent
            p_value_list_3 = []  # 0, dependent
            p_value_list_4 = []  # 0, dependent
            if not use_single_step:
                starting_point = 35
                downsample = 1
                X = X[:, starting_point:]
                X = X[:, ::downsample]
                # pdb.set_trace()
                X = X.reshape((-1, 6))
                cit_obj = CIT(X, 'kci')
                p_value_1 = cit_obj(X=[0], Y=[1], condition_set=[3]) # ci test for s' and c given s
                p_value_2 = cit_obj(X=[0], Y=[1]) # ci test for s' and s,a
                p_value_3 = cit_obj(X=[3], Y=[4], condition_set=[5]) # ci test for s' and c given s, a
                p_value_4 = cit_obj(X=[2], Y=[3], condition_set=[0,1]) # ci test for s' and s,a
                # if p_value_s_plus_c > 0.1:
                #     # ID!!!
                #     pdb.set_trace()
                #     print(X[:10])
                p_value_list_1.append(p_value_1) # 0., dependent
                p_value_list_2.append(p_value_2) # 0. dependent
                p_value_list_3.append(p_value_3) # 0. dependent
                p_value_list_4.append(p_value_4) # 1, independent
            else:
                starting_point = 43
                sel_local_len = 2
                X = X[:, starting_point:starting_point+sel_local_len]

                # for t in range(single_len - 1):
                for t in range(sel_local_len):
                    X_test = X[:, t, :]

                    # Essencially, the kernelZ & kernelX should be 'Linear'
                    # print(X_teccst)
                    # cit_obj = CIT(X_test, test_name, kernelX='Linear', kernelY='Linear', kernelZ='Linear') # KCI for real world
                    # cit_obj = CIT(X_test, 'fisherz') # use for color; uncomment the above for kci for real world car.
                    cit_obj = CIT(X_test, 'kci')
                    p_value_1 = cit_obj(X=[0], Y=[1], condition_set=[3]) # ci test for s' and c given s
                    p_value_2 = cit_obj(X=[0], Y=[1]) # ci test for s' and s,a
                    p_value_3 = cit_obj(X=[3], Y=[4], condition_set=[5]) # ci test for s' and c given s, a
                    p_value_4 = cit_obj(X=[2], Y=[3], condition_set=[0,1]) # ci test for s' and s,a
                    p_value_list_1.append(p_value_1) # 0., dependent
                    p_value_list_2.append(p_value_2) # 0. dependent
                    p_value_list_3.append(p_value_3) # 0. dependent
                    p_value_list_4.append(p_value_4) # 1, independent
            
            return p_value_list_1, p_value_list_2, p_value_list_3, p_value_list_4

        # Initialize lists to store p_values
        p_values_1 = []
        p_values_2 = []
        p_values_3 = []
        p_values_4 = []

        s_data = np.expand_dims(s_seq_loaded, axis=-1)
        a_data = np.expand_dims(a_seq_loaded, axis=-1)
        g_data = np.expand_dims(g_seq_loaded, axis=-1)
        
        jitter = 1e-5
        g_data = g_data + jitter * np.random.rand(*g_data.shape) # add jitter to avoid division by zero
        
        s_plus_data = s_data[1:, feature_dim].copy()
        a_plus_data = a_data[1:, :].copy()
        g_plus_data = g_data[1:, :].copy()
        
        s_data = s_data[:-1, feature_dim]
        a_data = a_data[:-1, :]
        g_data = g_data[:-1, :]
        
        # shape: (100 * 100, 6)
        X = np.concatenate((s_data, a_data, s_plus_data, g_data, a_plus_data, g_plus_data), axis=1) # (1 + 1 + 1) * 999

        skip_index = single_len * (np.arange(n_samples - 1) + 1) - 1
        X = np.delete(X, skip_index, axis=0)

        X = np.reshape(X, [n_samples, single_len - 1, 6]) # (n_samples, single_len, 6)

        print('shape of s_data: ', X.shape)
        

        n_samples_test = 200
        use_single_step = False
        # Run the calculation 100 times
        for i in range(100):
            # Select 100 sequences randomly from the total 1000 sequences
            selected_indices = np.random.choice(range(1000), size=n_samples_test, replace=False)

            # Calculate p_values for the selected sequences
            p_value_list_1, p_value_list_2, p_value_list_3, p_value_list_4 = calculate_p_values(X[selected_indices], use_single_step=use_single_step)
            
            # Append the p_values to the lists
            p_values_1.append(p_value_list_1)
            p_values_2.append(p_value_list_2)
            p_values_3.append(p_value_list_3)
            p_values_4.append(p_value_list_4)

        # Calculate the mean and variance of the p_values
        mean_p_values_1 = np.mean(p_values_1, axis=0)
        mean_p_values_2 = np.mean(p_values_2, axis=0)
        mean_p_values_3 = np.mean(p_values_3, axis=0)
        mean_p_values_4 = np.mean(p_values_4, axis=0)

        var_p_values_1 = np.var(p_values_1, axis=0)
        var_p_values_2 = np.var(p_values_2, axis=0)
        var_p_values_3 = np.var(p_values_3, axis=0)
        var_p_values_4 = np.var(p_values_4, axis=0)

        # Plot the mean and shaded variance
        # plt.plot(mean_p_values_1, label='$s_t,a_t \mid g_t$')
        # plt.fill_between(range(single_len - 1), mean_p_values_1 - var_p_values_1, mean_p_values_1 + var_p_values_1, alpha=0.3)

        print("mean p_values 1: ", mean_p_values_2)
        print("mean p_values 2: ", mean_p_values_3)
        print("mean p_values 3: ", mean_p_values_4)

        print(var_p_values_2)
        print(var_p_values_3)
        print(var_p_values_4)
        
        print(f"Overall mean: (1){np.mean(mean_p_values_2)} (2){np.mean(mean_p_values_3)} (3){np.mean(mean_p_values_4)}")
        print(f"Overall var: (1){np.var(mean_p_values_2)} (2){np.var(mean_p_values_3)} (3){np.var(mean_p_values_4)}")
        
        