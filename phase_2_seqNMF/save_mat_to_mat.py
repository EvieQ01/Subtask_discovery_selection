import numpy as np
from task_config import *
import scipy.io
import random
import torch
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set the seed
set_seed(0)

def get_demo(train_path, test_path, n_traj, task_specific=False, selected_train_id=None, selected_test_id=None, option_gt=False):
    print(f"Demo Loaded from {train_path} and {test_path}")
    train_set = torch.load(train_path)
    test_set = torch.load(test_path)

    selected_train_id = list(train_set.keys()) if selected_train_id is None else selected_train_id
    selected_test_id = list(test_set.keys()) if selected_test_id is None else selected_test_id
    #  list1 and list2 are lists of integers from traning env and testing env, respectively
    selected_train_and_test_id = list(set(selected_test_id + selected_train_id))

    print(f"=> Selected train task id: {selected_train_id}, Selected target task id: {selected_test_id}")

    target_contexts = []
    for task_idx in selected_test_id:
        target_contexts.append(test_set[task_idx]['context'])

    train_and_test_contexts = []
    for task_idx in selected_train_and_test_id:
        train_and_test_contexts.append(test_set[task_idx]['context'])

    # the structure of the demonstration data for all the algorithms are kept the same for fairness
    if not task_specific:
        train_demos = []
        for task_idx in selected_train_id: # no need to shuffle the keys of the train set
            # pdb.set_trace()
            if option_gt:
                sar_demo = train_set[task_idx]['demos']
                sarc_demo = get_option_from_demo(sar_demo, task_idx)
                train_demos.extend(sarc_demo)
            else:
                train_demos.extend(train_set[task_idx]['demos'])
            if len(train_demos) >= n_traj:
                break
        # random.shuffle(train_demos) # the sort of the trajectories should not be correlated with the task variable

        return train_demos, train_and_test_contexts #, target_contexts

def get_option_from_demo(train_demo, task_id):
    # shape as [(s, a, r), ...]
    # return shape as [(s, a, r, c), ...]
    # r > 1000 means achieving a subgoal
    curr_option_list = TASK_SET[task_id]
    train_demo_sarc = []
    for traj in train_demo:
        curr_option_id = 0
        traj_c = ([], [], [], [])
        s_traj, a_traj, r_traj = traj    
        for t in range(len(s_traj)):
            if r_traj[t] > 1000:
                curr_option_id += 1
                print("current option id: ", curr_option_id)
                if curr_option_id >= len(curr_option_list):
                    curr_option_id -= 1 # keep this option id at last time step
            all_option_id = curr_option_list[curr_option_id]
            c = torch.as_tensor(all_option_id)
            
            traj_c[0].append(s_traj[t])
            traj_c[1].append(a_traj[t])
            traj_c[2].append(r_traj[t])
            traj_c[3].append(c)
        
        # pdb.set_trace()
        train_demo_sarc.append((torch.stack(traj_c[0]), torch.stack(traj_c[1]),
                                torch.stack(traj_c[2]), torch.stack(traj_c[3])))
    # ((T, dims), (T, dima), (T, 1), (T, 1))
    return train_demo_sarc


def save_data_to_mat(sel_demo, use_task, shuffle):
    evaluate_data = []
    traj_length_list = []
    traj_id_list = []

    # Create a list of indices
    indices = list(range(len(sel_demo)))   
    
    # Shuffle the indices
    for _ in range(shuffle ):
        random.shuffle(indices)
        # Use the shuffled indices to index into sel_demo
        shuffled_sel_demo = [sel_demo[i] for i in indices]

        for i in range(len(sel_demo)):
            traj_length_list.append(shuffled_sel_demo[i][0].shape[0])
            traj_id_list.append(indices[i])
            # pdb.set_trace()
            evaluate_data.append(np.concatenate((shuffled_sel_demo[i][0], shuffled_sel_demo[i][1], 
                                                shuffled_sel_demo[i][2], shuffled_sel_demo[i][3].unsqueeze(-1)), axis=-1))
    
    traj_id_list = np.array(traj_id_list)
    X = np.concatenate(evaluate_data, axis=0)
    states = X[:, :9]
    actions = X[:, -11:-2]
    X = np.concatenate([X[:, :9], X[:, -11:]], axis=1)
    rewards = X[:, -2:-1]
    options = X[:, -1:]

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    X_normed = (X - min_vals) / (max_vals - min_vals)
    states_normed = X_normed[:, :9]
    actions_normed  = X_normed[:, -10:-1]

    traj_length_list = np.array(traj_length_list)

    for interval in [1]:
        file_save_path = f'kitchen_data/kitchen_{use_task}_interval_{interval}_shuffle_{shuffle}.mat'

        traj_start_index = np.cumsum(np.floor(traj_length_list/interval)[:-1]) + 1
        traj_start_index = np.insert(traj_start_index, 0, 1)

        scipy.io.savemat(file_save_path, {'states': states.T[:, ::interval], 
                                            'actions': actions.T[:, ::interval],
                                            's_a_pairs': X.T[:, ::interval],
                                            's_a_pairs_normed': np.concatenate([states_normed, actions_normed], axis=1).T[:, ::interval],
                                            'rewards': rewards.T[:, ::interval],
                                            'traj_start_index': traj_start_index,
                                            'options': options.T[:, ::interval],
                                            'traj_id_list': traj_id_list,})
        print(f"Saving data to {file_save_path}")

shuffle = 1

for use_task_config in [0]:
    # configs
    use_task = SELECTED_TASK_COMP[use_task_config] #  1, 23

    train_set_name =  'mujoco_data/KitchenMetaEnv-v0_sample_train.torch'
    test_set_name =  'mujoco_data/KitchenMetaEnv-v0_sample_test.torch'

    # load
    demo, test_contexts = get_demo(train_set_name, test_set_name, n_traj=240, task_specific=False, option_gt=True)

    # concatenate
    evaluate_data = []
    traj_length_list = []
    # permute demo
    sel_demo = []
    for task in use_task:
        for i in range(10):
            task_id = task * 10 + i
            sel_demo.append(demo[task_id])
    
    # Initialize dictionaries to store states grouped by options
    save_data_to_mat(sel_demo, use_task, shuffle)


