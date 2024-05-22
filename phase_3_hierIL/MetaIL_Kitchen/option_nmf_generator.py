import random
import numpy as np
import pdb
class OptionNMFGenerator():
    def __init__(self, demo_sca_array, option_num=5, env_num=5):

        self.demo_sca_array = demo_sca_array
        # get all the possible init option
        c_init_array = [traj[1][0].item() for traj in self.demo_sca_array]
        self.init_option = np.unique(c_init_array) # nparray
        self.current_option = [None for _ in range(env_num)] # list of length env_num 
        self.eps = 1.0
        self.option_num = option_num
        self.start_state_dict = {}
        self.end_state_dict = {}

        for traj in self.demo_sca_array:
            c_diff = traj[1][:-1] - traj[1][1:]
            c_diff = c_diff.cpu().numpy()
            start_option_idx = np.where(c_diff != 0)[0] + 1
            end_option_idx = start_option_idx - 1
            # add idx 0 to start_option_idx
            start_option_idx = np.concatenate([np.array([0]), start_option_idx])
            end_option_idx =  np.concatenate([end_option_idx, np.array([traj[1].shape[0] - 1])])
            # pdb.set_trace()
            # for start option (4 in each traj)
            for id in start_option_idx:
                op = traj[1][id].item()
                # If the key does not exist in the dictionary, initialize it with an empty list
                if op not in self.start_state_dict.keys():
                    self.start_state_dict[op] = []

                # Append the numpy array to the list corresponding to the key
                self.start_state_dict[op].append(traj[0][id].cpu().numpy())
                
            # for end option  (3 in each traj)
            for id in end_option_idx:
                op = traj[1][id].item()
                # If the key does not exist in the dictionary, initialize it with an empty list
                if op not in self.end_state_dict.keys():
                    self.end_state_dict[op] = []

                # Append the numpy array to the list corresponding to the key
                self.end_state_dict[op].append(traj[0][id].cpu().numpy())
        
        # pdb.set_trace()


    def reset(self, state=None):
        self.current_option = [None for _ in range(len(self.current_option))]
        # if state is not None:
        #     self.current_option = self.get_current_option(state)
        # else:
        #     init_option_idx = random.randint(0, len(self.init_option) - 1)
        #     self.current_option = self.init_option[init_option_idx]
        

    def get_current_option(self, state):
        # state = (env_num, s_dim)
        # pdb.set_trace()
        if len(state.shape) > 1:
            for env in range(state.shape[0]):
                if self.is_end_option(state[env], env) or self.current_option[env] is None:
                    self.current_option[env] = self.query_nearst_start_option(state[env])
                    # tuning
                    if env == 0:
                        print(f"=> Env [{env}] Change to current option: {self.current_option[env]}" )
            return self.current_option
        else:
            if self.is_end_option(state) or self.current_option is None:
                self.current_option = self.query_nearst_start_option(state)
            return self.current_option
    
    def is_end_option(self, state, env=None)-> bool:
        if self.current_option[env] is None:
            return True
        
        if env is not None:
            dist = np.min(np.sum(np.abs(state - self.end_state_dict[self.current_option[env]]), axis=1), axis=-1)
            # dist = np.min(np.linalg.norm(state - self.end_state_dict[self.current_option[env]], axis=-1))
            # tuning
            # if env == 0:
            #     pdb.set_trace()
            #     print(f"=> Env [{env}] distance to end option: {self.current_option[env], dist}" )
            if dist < self.eps:
                return True
            return False
        else:
            if np.min(state - self.end_state_dict[self.current_option]) < self.eps:
                return True
            return False

    def query_nearst_start_option(self, state)-> int:
        # pdb.set_trace()
        dists = np.array([np.min(np.sum(np.abs(state - self.start_state_dict[op]), axis=1), axis=-1) for op in self.start_state_dict.keys()])
        # print(np.array([np.min(np.sum(np.abs(state - self.end_state_dict[op]), axis=1), axis=-1) for op in self.end_state_dict.keys()]))
        argmin_idx = np.argmin(dists)
        return list(self.start_state_dict.keys())[argmin_idx]

