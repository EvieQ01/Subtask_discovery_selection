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

def make_il(config: Config, dim_s, dim_a):
    if config.algo == 'hier_airl':
        il = MHAOptionAIRL(config, dim_s=dim_s, dim_a=dim_a)
        ppo = MHAOptionPPO(config, il.policy)

        return il, ppo

    use_option = config.use_option
    if use_option:
        il = OptionGAIL(config, dim_s=dim_s, dim_a=dim_a)
        ppo = OptionPPO(config, il.policy)
    else:
        il = GAIL(config, dim_s=dim_s, dim_a=dim_a)
        ppo = PPO(config, il.policy)
    return il, ppo


def train_g(ppo: Union[OptionPPO, PPO], sample_sxar, factor_lr):

    ppo.step(sample_sxar, lr_mult=factor_lr)


def train_d(il: Union[OptionGAIL, GAIL, MHAOptionAIRL], sample_sxar, demo_sxar, n_step=10, demo_scar=None):
    il.step(sample_sxar, demo_sxar, n_step=n_step, demo_scar=demo_scar)


def sample_batch(il: Union[OptionGAIL, GAIL], agent, n_sample, demo_sa_array, render=False):
    demo_sa_in = agent.filter_demo(demo_sa_array)
    sample_sxar_in = agent.collect(il.policy.state_dict(), n_sample, fixed=False, render=render)
    sample_sxar, sample_rsum, sample_rmax = il.convert_sample(sample_sxar_in) # replace the real environment reward with the one generated with IL
    # demo_sxar, demo_rsum = il.convert_demo(demo_sa_in)
    return sample_sxar, sample_rsum, sample_rmax


def learn(config: Config, msg="default"):
    ## prepare
    env_type = config.env_type
    if env_type == "mujoco":
        from envir.mujoco_env import MujocoEnv as Env, get_demo, get_option_from_NMF
    else:
        raise ValueError(f"Unknown env type {env_type}")

    n_traj = config.n_traj
    n_sample = config.n_sample
    n_pre_epoch = config.n_pretrain_epoch
    n_epoch = config.n_epoch
    seed = config.seed
    pre_log_interval = config.pretrain_log_interval
    env_name = config.env_name
    use_d_info_gail = config.use_d_info_gail

    set_seed(seed)
    log_dir, save_dir, train_set_name, test_set_name, pretrain_name = get_dirs(seed, config.algo, env_type, env_name, msg)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config)) # important for reproducing and visualisaton
    wandb.init(entity="", project="Multi-option-IL", dir=log_dir, name=log_dir.split("/")[-2])  # wandb initialization        
    wandb.config.update(config)  # wandb logging
    
    save_name_pre_f = lambda i: os.path.join(save_dir, f"pre_{i}.torch")
    save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")
    save_name_ppo_f = lambda i: os.path.join(save_dir, f"{i}_critic.torch")

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
    if not config.context_avail:
        if config.option_gt:
            demo_sa_array = tuple((s[:, :-dim_cnt].to(il.device), a.to(il.device)) for s, a, r, _ in demo)
            demo_scar_array = tuple((s[:, :-dim_cnt].to(il.device), c.to(il.device), a.to(il.device)) for s, a, r, c in demo)
            # one_hot encoding options
            # demo_scar_array = tuple((s[:, :-dim_cnt].to(il.device), F.one_hot(c.to(il.device).squeeze(-1), num_classes=config.dim_c), a.to(il.device)) for s, a, r, c in demo)
        else:
            demo_sa_array = tuple((s[:, :-dim_cnt].to(il.device), a.to(il.device)) for s, a, r in demo)
        
        if config.option_nmf:
            # pdb.set_trace()
            demo_scar_array = get_option_from_NMF(config.nmf_result_path, demo_sa_array, indices)
    else:
        demo_sa_array = tuple((s.to(il.device), a.to(il.device)) for s, a, r in demo)

    if use_d_info_gail: # only for DI-GAIL
        if os.path.isfile(pretrain_name):
            print(f"Loading pre-train model from {pretrain_name}")
            param = torch.load(pretrain_name)
            il.policy.load_state_dict(param)
        else:
            pretrain(il.policy, demo_sa_array, save_name_pre_f,
                      msg, n_pre_epoch, pre_log_interval)

        temp_state_dict = copy.deepcopy(il.policy.option_policy.state_dict())
        config.use_vae = False
        il, ppo = make_il(config, dim_s=dim_s, dim_a=dim_a)

        il.policy.option_policy.load_state_dict(temp_state_dict) # only pretrain the high-level policy, so the low-level policy should be recovered

    if config.n_thread == 1:
        sampling_agent = Sampler(seed, env, il.policy, is_expert=False, task_list=train_and_test_contexts, contain_context=config.context_avail)
    else:
        sampling_agent = VecSampler(seed, env_name, config.n_thread, il.policy,
                                    is_expert=False, task_list=train_and_test_contexts, 
                                    contain_context=config.context_avail, option_gt=config.option_gt, 
                                    task_list_target=target_contexts, option_nmf=config.option_nmf, option_nmf_generator_base_array=demo_scar_array)

    st = time.time()
    sample_sxar, sample_r, sample_r_max = sample_batch(il, sampling_agent, n_sample, demo_sa_array)
    et = time.time()
    print("time required: ", et - st)

    info_dict, cs_sample = reward_validate(sampling_agent, il.policy, do_print=True)

    wandb.log(info_dict, step=0)  # wandb logging
    print(f"init: r-sample-avg={sample_r}, r-sample-max={sample_r_max}; {msg}")

    for i in trange(n_epoch):
        print("Starting collecting samples......")
        st = time.time()
        temp_render = config.render if i % config.log_interval == 0 else False
        # temp_render = False
        sample_sxar, sample_r, sample_r_max = sample_batch(il, sampling_agent, n_sample, demo_sa_array, render=temp_render) # n_sample is too big
        et = time.time()
        print("time required: ", et - st)

        if i % 3 == 0:
            # provide option variable to discriminator if available
            # pdb.set_trace()
            train_d(il, sample_sxar, demo_sa_array, demo_scar=demo_scar_array) 
        # factor_lr = lr_factor_func(i, 1000., 1., 0.0001) # not commented by me
        sample_sxar = il.get_il_reward(sample_sxar)
        train_g(ppo, sample_sxar, factor_lr=1.)

        if (i + 1) % config.log_interval == 0:
            info_dict, cs_sample = reward_validate(sampling_agent, il.policy, do_print=True, render=config.render)

            if (i + 1) % (100) == 0:
                torch.save(il.state_dict(), save_name_f(i))
                torch.save(ppo.state_dict(), save_name_ppo_f(i))
                print(f"{i}: Save il model to {save_name_f(i)}")
                print(f"{i}: Save ppo model to {save_name_ppo_f(i)}")
            wandb.log(info_dict, step=i+1)  # wandb logging
        
        print(f"{i}: r-sample-avg={sample_r}, r-sample-max={sample_r_max}; {msg}")
        wandb.log({"r-sample-avg": sample_r, "r-sample-max": sample_r_max}, step=i+1)  # wandb logging

    wandb.finish()  # wandb finish

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type")
    arg.add_arg("env_name", "KitchenMetaEnv-v0", "Environment name") # AntCell-v1
    arg.add_arg("algo", "DI_gail", "which algorithm to use, can be [gail, option_gail, DI_gail, hier_airl]")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("seed", 0, "Random seed")
    arg.add_arg("n_pretrain_epoch", 100, "Pre-training epochs")
    arg.add_arg("n_traj", 1000, "Number of demonstration s-a")
    arg.add_arg("selected_train_id", [4, 11], "Choose the train task") 
    arg.add_arg("selected_test_id", [3], "Choose the test task")
    arg.add_arg("option_gt", 0, "Use option ground truth or not")
    arg.add_arg("option_nmf", 0, "Use option extracted by NMF")
    arg.add_arg("render", 0, "Render video or not")
    arg.add_arg("nmf_result_path", "../NMF_result/C_6_14.mat", )
    arg.parser()

    if arg.env_type == "mujoco":
        config = mujoco_config
    else:
        raise NotImplementedError

    config.update(arg)
    if config.env_name.startswith("Ant") or config.env_name.startswith("Walker"):
        config.hidden_option = (128, 128)
        config.hidden_policy = (128, 128)
        config.hidden_critic = (128, 128)

    elif config.env_name.startswith("Kitchen"):
        # config.n_sample = 512
        config.hidden_option = (256, 256)
        config.hidden_policy = (256, 256)
        config.hidden_critic = (256, 256)

    ## set up the corresponding algorithm
    if config.algo == 'gail':
        config.use_option = False
        config.use_c_in_discriminator = False
        config.use_d_info_gail = False
        config.use_vae = False
        config.train_option = False
    elif config.algo == 'option_gail':
        config.use_option = True
        config.use_c_in_discriminator = True
        config.use_d_info_gail = False
        config.use_vae = False
        config.train_option = True
    elif config.algo == 'DI_gail':
        config.use_option = True
        config.use_c_in_discriminator = False
        config.use_d_info_gail = True
        config.use_vae = True
        config.train_option = False # for DI-GAIL, the high-level policy is only trained in the pre_train stage
    elif config.algo == 'hier_airl':
        config.use_option = True
        config.use_c_in_discriminator = True
        config.use_d_info_gail = False
        config.use_vae = False
        config.train_option = True
        config.is_airl = True
        config.use_posterior = True

        # config.hidden_policy = (128, 128)
        # config.hidden_critic = (128, 128)
        # print("Using AIRL: change policy")

        # config.mini_batch_size = 256
        # print("Using AIRL: change minibatch: ", config.mini_batch_size)
    else:
        raise NotImplementedError

        
    if config.option_gt:
        config.train_option = False

    print(f">>>> Training {config.algo} using {config.env_name} environment on {config.device}")

    learn(config, msg=config.tag)
