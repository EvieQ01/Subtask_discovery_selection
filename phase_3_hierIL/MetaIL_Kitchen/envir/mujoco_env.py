import os
import random
import pdb
import torch
try:
    import pybullet_envs
except ImportError:
    print("Warning: pybullet not installed, bullet environments will be unavailable")
import gym
from envir import mujoco_maze, mujoco_manipulation, d4rl_env
from .d4rl_env.task_config import TASK_SET, ALL_INDX_TO_OPTION_MAP, ALL_INDX_TO_OPTION_MAP2
import numpy as np
from scipy.io import loadmat

class MujocoEnv(object):
    def __init__(self, task_name: str = "HalfCheetah-v2"):
        self.task_name = task_name
        self.env = None
        self.display = False

    def init(self, display=False):
        self.env = gym.make(self.task_name)
        self.display = display
        return self

    def get_context_info(self):
        if self.env is not None:
            return self.env.get_context_dim(), self.env.get_context_limit()
        else:
            env = gym.make(self.task_name)
            dim_cnt = env.get_context_dim()
            cnt_limit = env.get_context_limit()
            env.close()
        return dim_cnt, cnt_limit

    def sample_context(self):
        return self.env.sample_context()

    def reset(self, context, is_expert):
        self.env.apply_context(context, is_expert=is_expert)
        s = self.env.reset()
        return s

    def get_expert_act(self, obs):
        act = self.env.get_expert_action(obs)
        return act

    def render(self, mode="rgb_array"):
        self.env.render(mode)

    def seed(self, seed_idx):
        self.env.seed(seed_idx)

    def step(self, a):
        s, reward, terminate, info = self.env.step(a)
        if self.display:
            self.env.render()
        return s, reward, terminate
    
    def get_gt_option(self):
        return self.env.get_gt_option()
    
    def render_my(self, mode="rgb_array"):
        return self.env.render_my(mode)
    
    def state_action_size(self, is_expert=False):
        if self.env is not None:
            s = self.reset(self.sample_context(), is_expert=is_expert)
            s_dim = s.shape[0]
            a_dim = self.env.action_space.shape[0]
        else:
            env = gym.make(self.task_name)
            env.apply_context(env.sample_context(), is_expert=is_expert)
            s = env.reset()
            s_dim = s.shape[0]
            a_dim = env.action_space.shape[0]
            env.close()
        return s_dim, a_dim


def preprocess(data_set):
    # we need to replace the true task info (e.g., goal location) in the expert data with the context
    # since the learned policy will not be provided the true task info
    train_demos = []
    for task_idx in data_set:
        temp_context = torch.tensor(data_set[task_idx]['context'], dtype=torch.float32).unsqueeze(0)
        context_dim = len(data_set[task_idx]['context'])
        temp_traj_list = data_set[task_idx]['demos']
        for traj_idx in range(len(temp_traj_list)):
            s, a, r = temp_traj_list[traj_idx]
            s = s[:, :-context_dim] # eliminate the true task info # TODO: only do this elimination
            context_sup = temp_context.repeat(s.shape[0], 1)
            s = torch.cat([s, context_sup], dim=1)
            train_demos.append((s, a, r))

    return train_demos

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

        return train_demos, train_and_test_contexts, target_contexts

    train_demos = {}
    cur_traj = 0
    for task_idx in train_set:
        train_demos[task_idx] = train_set[task_idx]
        cur_traj += len(train_set[task_idx]['demos'])
        if cur_traj >= n_traj:
            break

    train_demos = preprocess(train_demos)
    random.shuffle(train_demos)
    # print("1: ", train_demos)
    return train_demos, target_contexts

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
            c = torch.as_tensor([ALL_INDX_TO_OPTION_MAP2[all_option_id]])
            
            traj_c[0].append(s_traj[t])
            traj_c[1].append(a_traj[t])
            traj_c[2].append(r_traj[t])
            traj_c[3].append(c)
        
        # pdb.set_trace()
        train_demo_sarc.append((torch.stack(traj_c[0]), torch.stack(traj_c[1]),
                                torch.stack(traj_c[2]), torch.stack(traj_c[3])))
    # ((T, dims), (T, dima), (T, 1), (T, 1))
    return train_demo_sarc


def get_option_from_NMF(nmf_result_path, demo_s_a, indices):
    # Load the .mat file
    mat = loadmat(nmf_result_path)

    # Extract the 'C' and 'traj_id_list' matrices
    C = mat['C']
    traj_id_list = mat['traj_id_list'][0]

    assert len(demo_s_a) == len(traj_id_list)

    # Create a dictionary to map indices in 'traj_id_list' to sequences in 'C'
    traj_length_dict = {idx: demo[0].shape[0] for idx, demo in zip(indices, demo_s_a)}

    traj_length_rearranged_in_C = [traj_length_dict[id] for id in traj_id_list]
    traj_length_rearranged_in_sa = [traj_length_dict[id] for id in indices]

    # Convert 'C' to a list of arrays
    C = np.argmax(C,0)
    # C = smooth_option(C)
    C_list = np.split(C, np.cumsum(traj_length_rearranged_in_C)[:-1])
    C_list = [smooth_option(c) for c in C_list]
    C_list_sa = [C_list[np.where(traj_id_list==i)[0].item()] for i in indices]
    # pdb.set_trace()
    # Match the sequences in 'C' with 'demo_s_a'
    matched_demo_s_c_a = tuple((s_a_seq[0], torch.from_numpy(c_seq).to(s_a_seq[0].device).unsqueeze(-1), s_a_seq[1] ) for s_a_seq, c_seq in zip(demo_s_a, C_list_sa))

    return matched_demo_s_c_a

def smooth_option(C, window_size=5):
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

def collect_demo(config, n_task=1000, demo_per_task=10, data_type='train',
                 display=False, is_manual=False, env_name=None, expert_path=None):
    from model.option_policy import Policy, OptionPolicy
    # you must have an expert model first, by running 'run_ppo_expert.py'.
    if not is_manual:
        env = MujocoEnv(config.env_name)
        path = f"./{config.env_name}_sample_" + data_type + '.torch'
    else:
        env = MujocoEnv(env_name)
        path = f"./{env_name}_sample_" + data_type + '.torch'
    dim_s, dim_a = env.state_action_size()
    env.init(display=display)

    if not is_manual:
        config.device = 'cpu'
        policy_state = torch.load(expert_path, map_location='cuda:0')
        policy = Policy(config, dim_s, dim_a)
        # policy = OptionPolicy(config, dim_s, dim_a)
        policy.load_state_dict(policy_state)

    demo_set = {}
    for task_idx in range(n_task):
        context = env.sample_context()
        demo_set[task_idx] = {'context': context}
        print("1: ", context)
        trajs = []
        while len(trajs) < demo_per_task:
            with torch.no_grad():
                s_array = []
                a_array = []
                r_array = []
                s, done = env.reset(context, is_expert=True), False

                while not done:
                    st = torch.as_tensor(s, dtype=torch.float32).unsqueeze(dim=0)
                    s_array.append(st.clone())
                    if not is_manual:
                        at = policy.sample_action(st, fixed=True)  # eliminate the randomness of the expert policy
                    else:
                        at = env.get_expert_act(obs=st.clone().numpy()[0])
                        at = torch.tensor(at, dtype=torch.float32, device=st.device).unsqueeze(dim=0)
                    a_array.append(at.clone())
                    s, r, done = env.step(at.squeeze(dim=0).cpu().detach().clone().numpy())
                    r_array.append(r)
                a_array = torch.cat(a_array, dim=0)
                s_array = torch.cat(s_array, dim=0)
                r_array = torch.as_tensor(r_array, dtype=torch.float32).unsqueeze(dim=1)

                print(f"R-Sum={r_array.sum()}, L={r_array.size(0)}")
                if r_array.sum().item() > 1200: # or 300
                    print("Keep it!")
                    trajs.append((s_array, a_array, r_array))

        demo_set[task_idx]['demos'] = trajs

    torch.save(demo_set, path)


def get_demo_stat(path=""):
    if os.path.isfile(path):
        print(f"Demo Loaded from {path}")
        samples = torch.load(path) # TODO
        # print(samples)
        aver_r = 0.0
        n_traj = 0
        n_tran = 0
        for task_idx in samples:
            temp_list = samples[task_idx]['demos']
            for traj in temp_list:
                s, a, r = traj
                print(s.shape, a.shape, r.shape, r.sum())
                aver_r += r.sum()
                n_traj += 1
                n_tran += r.shape[0]

        print(aver_r/n_traj, n_traj, n_tran)


if __name__ == '__main__':

    # collect_demo(config=None, n_demo=10000, is_manual=True, env_name='PointCell-v0')
    #
    import torch.multiprocessing as multiprocessing
    from utils.config import Config, ARGConfig
    from default_config import mujoco_config

    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, ...]")
    arg.add_arg("env_name", "KitchenMetaEnv-v0", "Environment name")
    arg.add_arg("algo", "option_ppo", "Environment type, can be [ppo, option_ppo]")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("seed", 0, "Random seed")
    arg.parser()

    config = mujoco_config
    config.update(arg)
    if config.env_name.startswith("Ant") or config.env_name.startswith("Walker"):
        config.hidden_policy = (128, 128)
        config.hidden_critic = (128, 128)
        print(f"Training this env with larger policy network size :{config.hidden_policy}")

    elif config.env_name.startswith("Kitchen"):
        # config.n_sample = 512
        config.hidden_option = (256, 256)
        config.hidden_policy = (256, 256)
        config.hidden_critic = (256, 256)

    print(config.algo)
    config.use_option = True
    config.use_c_in_discriminator = False  # in fact, there are no discriminators
    config.use_d_info_gail = False
    config.use_vae = False
    config.train_option = True
    if config.algo == 'ppo':
        config.use_option = False
        config.train_option = False

    config.is_airl = True
    config.use_option_posterior = True
    config.use_c_in_discriminator = True  # c actually corresponds to the option choice in the paper

    # collect_demo(config, n_task=100, demo_per_task=10, data_type='train', expert_path='../model_saved/MHIL/1399.torch')
    # collect_demo(config, n_task=50, demo_per_task=10, data_type='test', expert_path='./exp_model/PointCell/899.torch')

    # run_model(config, n_task=100, demo_per_task=10, expert_path='../model_saved/MHIL/1399.torch')

    # get_demo_stat('KitchenMetaEnv-v0_sample_train.torch')
    # get_demo_stat('KitchenMetaEnv-v0_sample_test.torch')

    # train, test = get_demo('PointCell-v0_sample_train.torch', 'PointCell-v0_sample_test.torch', 10, task_specific=True)
    # print(train)
    # print(len(test))