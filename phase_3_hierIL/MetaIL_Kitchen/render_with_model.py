#!/usr/bin/env python3
import os, time
import copy
import torch
from typing import Union
import torch.multiprocessing as multiprocessing
from model.option_ppo import OptionPPO, PPO
from model.option_gail import OptionGAIL, GAIL
from model.MHA_option_ppo import MHAOptionPPO
from model.MHA_option_il import MHAOptionAIRL
from utils.common_utils import reward_validate, get_dirs, set_seed
from sampler import Sampler
from vec_sampler import VecSampler
from utils.logger import Logger
from utils.config import ARGConfig, Config
from default_config import mujoco_config
from vae_pretrain import pretrain
import datetime
import wandb
import pdb
import torch.nn.functional as F
from tqdm import trange
import random
import argparse

def make_il(config: Config, dim_s, dim_a):
    use_option = config.use_option
    if use_option:
        il = OptionGAIL(config, dim_s=dim_s, dim_a=dim_a)
        ppo = OptionPPO(config, il.policy)
    else:
        il = GAIL(config, dim_s=dim_s, dim_a=dim_a)
        ppo = PPO(config, il.policy)
    return il, ppo


def sample_batch(il: Union[OptionGAIL, GAIL], agent, n_sample, render=False):
    sample_sxar_in = agent.collect(il.policy.state_dict(), n_sample, fixed=False, render=render)
    sample_sxar, sample_rsum, sample_rmax = il.convert_sample(sample_sxar_in) # replace the real environment reward with the one generated with IL
    # demo_sxar, demo_rsum = il.convert_demo(demo_sa_in)
    return sample_sxar, sample_rsum, sample_rmax


def learn(config: Config, msg="default"):
    ## prepare
    from envir.mujoco_env import MujocoEnv as Env, get_demo, get_option_from_NMF

    n_traj = config.n_traj
    n_sample = config.n_sample
    n_epoch = config.n_epoch
    seed = config.seed
    env_name = config.env_name

    set_seed(seed)
    log_dir, save_dir, train_set_name, test_set_name, pretrain_name = get_dirs(seed, config.algo, env_type, env_name, msg)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config)) # important for reproducing and visualisaton
    wandb.init(entity="", project="Multi-option-render", dir=log_dir, name=log_dir.split("/")[-2])  # wandb initialization        
    wandb.config.update(config)  # wandb logging
    
    env = Env(env_name)
    dim_s, dim_a = env.state_action_size()
    dim_cnt, cnt_limit = env.get_context_info()


    if not config.context_avail:
        dim_s = dim_s - dim_cnt
    print("The dimension info of the environment: name:{}, dim_s:{}, dim_a:{}, context_dim:{}, context_limit:{}.".format(
            env_name, dim_s, dim_a, dim_cnt, cnt_limit))

    # the demonstration does not contain task labels
    demo, train_and_test_contexts, target_contexts = get_demo(train_set_name, test_set_name, n_traj=n_traj, task_specific=config.context_avail, 
                                                              selected_train_id=config.selected_train_id, selected_test_id=config.selected_test_id,
                                                              option_gt=config.option_gt)

    # shuffle the trajectories and record the order
    indices = list(range(len(demo)))
    random.shuffle(indices) # the sort of the trajectories should not be correlated with the task variable
    demo = [demo[i] for i in indices]

    il, ppo = make_il(config, dim_s=dim_s, dim_a=dim_a) # does not contain context in the state
    
    demo_scar_array = None

    # for render
    if os.path.isfile(config.il_path):
        print(f"Loading pre-train il model from {config.il_path}")
        param = torch.load(config.il_path)
        il.load_state_dict(param)

    if os.path.isfile(config.ppo_path):
        print(f"Loading pre-train ppo model from {config.ppo_path}")
        param = torch.load(config.ppo_path)
        ppo.load_state_dict(param)

    sampling_agent = VecSampler(seed, env_name, config.n_thread, il.policy,
                            is_expert=False, task_list=train_and_test_contexts, 
                            contain_context=config.context_avail, option_gt=config.option_gt, 
                            task_list_target=target_contexts, option_nmf=config.option_nmf, option_nmf_generator_base_array=demo_scar_array)

    for i in trange(n_epoch):
        print("Starting collecting samples......")
        st = time.time()
        sample_sxar, sample_r, sample_r_max = sample_batch(il, sampling_agent, n_sample, render=config.render) # n_sample is too big
        et = time.time()
        print("time required: ", et - st)
        info_dict, cs_sample = reward_validate(sampling_agent, il.policy, do_print=True, render=config.render)
        
        print(f"{i}: r-sample-avg={sample_r}, r-sample-max={sample_r_max}; {msg}")
        wandb.log({"r-sample-avg": sample_r, "r-sample-max": sample_r_max}, step=i+1)  # wandb logging

    wandb.finish()  # wandb finish

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="render_with_model.py")
    parser.add_argument("--ppo_path", type=str, help="Environment type, can be [mujoco, ...]",
                      default="./result/KitchenMetaEnv-v0_2024_5_15_1_4_40_option_gail_seed_2/model/1999_critic.torch")
    parser.add_argument("--il_path", type=str, help="Environment type, can be [mujoco, ...]",
                      default="./result/KitchenMetaEnv-v0_2024_5_15_1_4_40_option_gail_seed_2/model/1999.torch")
    parser.add_argument("--device", type=str, default="cuda", help="Computing device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n_thread", type=int, default=1, help="Number of env threads")
    parser.add_argument("--n_traj", type=int, default=1000, help="Number of demonstration s-a")
    parser.add_argument("--n_epoch", type=int, default=100, help="Number of rendering")
    parser.add_argument("--n_sample", type=int, default=500, help="Number of sampling")
    parser.add_argument("--selected_train_id", type=list, default=[6, 14], help="Choose the train task") 
    parser.add_argument("--selected_test_id", type=list, default=[13], help="Choose the test task")
    parser.add_argument("--env_name", type=str, default="KitchenMetaEnv-v0", help="Environment name") # AntCell-v1

    args = parser.parse_args()

    config = args
    print(f">>>> Rendering ")

    learn(config)
