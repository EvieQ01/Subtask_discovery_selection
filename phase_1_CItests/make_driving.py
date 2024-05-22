import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy.io import savemat
import pdb
# Set random seed for reproducibility
np.random.seed(0)
class DrivingEnv(gym.Env):
    def __init__(self, verbose=False):
        super(DrivingEnv, self).__init__()
        self.world_width = 60
        self.world_height = 20
        self.eps_done = 1.5
        self.end_point = np.array([52.5, 10.0])
        self.state = np.zeros(3)  # (x, y, theta)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, -np.pi]),
                                            high=np.array([self.world_width, self.world_height, np.pi]),
                                            dtype=np.float32)
        self.verbose = verbose
    def reset(self):
        self.state = np.zeros(3)
        self.state[0] = 5 + 2.5 + np.random.uniform(-0.5, 0.5)
        self.state[1] = 10
        self.state[2] = np.random.choice( [0.7 *np.pi/2, -0.7 *np.pi/2])  # Randomly choose initial orientation
        self.state[1] += 1.5 * self.state[2] + np.random.uniform(-0.1, 0.1)
        return self.state

    def step(self, action):
        dt = 0.5
        theta = self.state[2] + action[0] * dt
        
        # Ensure theta is within [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Update x, y based on the new theta
        dx = np.cos(theta) * dt
        dy = np.sin(theta) * dt

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] = theta

        # Check for out of bounds
        done = self.state[0] >= self.world_width
        done = done or self.state[0] < 0 or self.state[1] < 0 or self.state[1] > self.world_height
        reward = -1  # Basic reward structure
        
        if np.linalg.norm(self.state[0:2] - self.end_point) <= self.eps_done:
            done = True
            reward += 1000
        if self.verbose:
            print("Self.state, done: ", self.state, done)

        return self.state, reward, done, {}

    def render(self, mode='human'):
        plt.figure(figsize=(12, 4))
        plt.xlim(0, self.world_width)
        plt.ylim(0, self.world_height)
        
        # Draw the road
        road1 = plt.Circle((15, 10), 5, color='gray', fill=False)
        road2 = plt.Circle((45, 10), 5, color='gray', fill=False)
        plt.gca().add_artist(road1)
        plt.gca().add_artist(road2)
        plt.plot([15, 45], [5, 5], color='gray')
        plt.plot([15, 45], [15, 15], color='gray')

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Driving Path')
        plt.show()

# Define two expert policies
def policy_blue(state, noise=0.02):
    policy_noise = np.random.uniform(-noise, noise, 1)
    if state[1] < 10 - 2.8:
        return np.array([0.13]) + policy_noise
    elif state[1] >= 10 - 2.8 and state[0] <= 33:
        # return np.array([0.0])
        return np.clip(-state[2], np.array([-0.2]), np.array([0.0]))  + policy_noise
    elif state[2] < np.pi / 3 and state[0] < 38:
        # return np.array([0.13])
        return np.clip(np.pi / 2 - state[2], np.array([0.0]), np.array([0.2]))# + policy_noise
    else:
        return np.array([-0.13]) + policy_noise

def policy_yellow(state, noise=0.02):
    policy_noise = np.random.uniform(-noise, noise, 1)
    if state[1] > 10 + 2.8:
        return np.array([-0.13]) + policy_noise
    elif state[1] <= 10 + 2.8 and state[0] <= 33:
        # return np.array([0.0])
        return np.clip(-state[2], np.array([0.0]), np.array([0.2])) + policy_noise
    elif state[2] > -np.pi / 3 and state[0] < 38:
        # return np.array([-0.13])
        return np.clip(-np.pi / 2 - state[2], np.array([-0.2]), np.array([0.0])) #+ policy_noise
    else:
        return np.array([0.13]) + policy_noise

# Create environment
env = DrivingEnv(verbose=False)

# Function to run a single policy
def run_policy(env, policy, state):
    # state = env.reset()
    done = False
    trajectory = []
    while not done:
        action = policy(state)
        trajectory.append(np.concatenate([state, action]))
        state, reward, done, _ = env.step(action)
    return np.array(trajectory)

def add_subgoal_to_trajectory(trajectory):
    subgoal_list = np.array([1, 3, 5]) if trajectory[0, 2] > 0. else np.array([2, 3, 4])
    subgoal_idx = 0
    g_seq = np.zeros(trajectory.shape[0])
    for i in range(trajectory.shape[0]):
        if trajectory[i, 0] >= 25 and subgoal_idx < 1:
            subgoal_idx += 1
        if trajectory[i, 0] >= 35 and subgoal_idx < 2:
            subgoal_idx += 1
        g_seq[i] = subgoal_list[subgoal_idx]
    return np.concatenate([trajectory, g_seq.reshape(-1, 1)], axis=1)

# Render function that draws the paths
def render_paths(trajectories, env):
    plt.figure(figsize=(12, 4))
    plt.xlim(0, env.world_width)
    plt.ylim(0, env.world_height)
    
    # Draw the road
    road1 = plt.Circle((15, 10), 9, color='gray', fill=True, linewidth=2)
    road2 = plt.Circle((45, 10), 9, color='gray', fill=True, linewidth=2)
    road3 = plt.Circle((15, 10), 6, color='white', fill=True, linewidth=2)
    road4 = plt.Circle((45, 10), 6, color='white', fill=True, linewidth=2)
    end = plt.Circle((env.end_point[0], env.end_point[1]), 1.5, color='orange', fill=True, linewidth=2)
    plt.gca().add_artist(road1)
    plt.gca().add_artist(road2)
    plt.gca().add_artist(road3)
    plt.gca().add_artist(road4)
    plt.gca().add_artist(end)
    plt.plot([23, 37], [10, 10], color='gray', linewidth=30)
    # plt.plot([15, 45], [5, 5], color='gray', linewidth=2)
    # plt.plot([15, 45], [15, 15], color='gray', linewidth=2)
    
    # Separate and plot paths
    for traj in trajectories:
        states = traj[:, :3]
        actions = traj[:, 3]
        if states[0, 2] > 0:
            plt.plot(states[:, 0], states[:, 1], color='yellow', alpha=0.5)
        else:
            plt.plot(states[:, 0], states[:, 1], color='blue', alpha=0.5)
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Driving Paths')
    plt.show()


# Generate 100 trajectories
trajectories = []
trajectories_clipped = []
max_length = 100
num_trajs = 20
for _ in range(num_trajs):
    state_ini = env.reset()
    if state_ini[2] > 0.:
        trajectory = run_policy(env, policy_yellow, state=state_ini)
    else:
        trajectory = run_policy(env, policy_blue, state=state_ini)
    
    trajectory = add_subgoal_to_trajectory(trajectory)
    print("Traj length: ", trajectory.shape[0])

    trajectories.append(trajectory)
    trajectories_clipped.append(trajectory[:max_length])
    # Traj length

# Concatenate all trajectories into a single array
traj_lengths = np.array([traj.shape[0] for traj in trajectories])
start_indices = np.cumsum(traj_lengths) + 1
start_indices = np.concatenate([[1], start_indices])
all_trajectories = np.concatenate(trajectories, axis=0)
all_trajectories_clipped = np.concatenate(trajectories_clipped, axis=0)
print("Shape of all trajectories: ", all_trajectories.shape)
print("Shape of all trajectories (clipped): ", all_trajectories_clipped.shape)
# Save the trajectories
np.save('trajectories.npy', all_trajectories)
np.save('trajectories_clipped.npy', all_trajectories_clipped)

# Test render paths
render_paths(trajectories, env)

def save_files(all_trajectories, start_indices):
    # Save the array to a .mat file
    s_seq = all_trajectories[:, :3].T
    a_seq = all_trajectories[:, 3].T
    g_seq = all_trajectories[:, 4].T
    
    # pdb.set_trace()
    s_seq_normed_expand = np.repeat(s_seq[-1][np.newaxis, :], 2, axis=0)
    s_seq_normed_expand[0][s_seq_normed_expand[0] < 0] = 0
    s_seq_normed_expand[1][s_seq_normed_expand[1] >= 0] = 0
    s_seq = np.concatenate([s_seq[0:2], np.abs(s_seq_normed_expand)], axis=0)

    a_seq_normed_expand = np.repeat(a_seq[np.newaxis, :], 2, axis=0)
    a_seq_normed_expand[0][a_seq_normed_expand[0] < 0] = 0
    a_seq_normed_expand[1][a_seq_normed_expand[1] >= 0] = 0
    a_seq_normed_expand = np.abs(a_seq_normed_expand)

    arrays_to_save = {
        'state': s_seq,
        'action': a_seq,
        'sel': g_seq,
        'start_indices': start_indices,
        'state_normed': s_seq / np.expand_dims(np.max(s_seq, axis=1),axis=-1),
        'action_normed': a_seq_normed_expand / np.expand_dims(np.max(a_seq_normed_expand, axis=1),axis=-1),
                    }

    # Save the NumPy array as a .mat file
    savemat(f'../seqNMF/driving_{str(all_trajectories.shape[0])}.mat', arrays_to_save)
    # Save the arrays to a .npz file (noised ones for CI test)
    np.savez(f'../seqNMF/driving_{str(all_trajectories.shape[0])}.npz', **arrays_to_save)

save_files(all_trajectories, start_indices)
save_files(all_trajectories_clipped, np.arange(1, all_trajectories_clipped.shape[0]) * 100)