import numpy as np
import gymnasium as gym
import icu_sepsis
import torch
import torch.nn as nn
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'action_mask','reward', 'next_state', 'next_action_mask', 'done'))
TransitionSARSA = namedtuple('TransitionSARSA', ('state', 'action', 'action_mask','reward', 'next_state', 'next_action','next_action_mask', 'done'))

NUM_ACTIONS = 25 
NUM_STATES = 722
NUM_ENVS = 4



def make_env(seed, env_type = None):
    def thunk():
        env = gym.make('Sepsis/ICU-Sepsis-v2')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def many_hot(indices, vector_length):
    """
    Create a many-hot encoded vector in PyTorch.
    
    :param indices: Tensor of indices to set to 1.
    :param vector_length: Total length of the output vector.
    :return: Many-hot encoded vector.
    """
    vector = torch.zeros(vector_length)
    vector[indices] = 1
    return vector

def get_mask(info, num_envs = 1, n_actions = NUM_ACTIONS):
    allowed_actions = info['admissible_actions']
    masks = np.zeros((num_envs, n_actions))
    for i, allowed_action in enumerate(allowed_actions):
        masks[i] = many_hot(allowed_action, n_actions)
    return torch.Tensor(masks)

def encode_state(obs, n_states = NUM_STATES):
    obs = torch.Tensor(obs)
    return nn.functional.one_hot(obs.long(), n_states).float()

def encode_state_cont(obs, n_states):
    obs = torch.Tensor(obs).float().reshape(-1)
    return obs
    
def layer_init(layer):
    # torch.nn.init.orthogonal_(layer.weight, std)
    # use a constant value to intialize stuff. 
    torch.nn.init.constant_(layer.weight, 0.0)
    # torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0)
    # we are not using a bias unit as of yet. 
    # torch.nn.init.constant_(layer.bias, bias_const)
    return layer



def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Samples a batch of experiences."""
        transitions = random.sample(self.buffer, batch_size)
        
        # Transpose the batch from Transition of batch_size
        # to batch_size of Transitions
        batch = Transition(*zip(*transitions))

        # Convert to PyTorch tensors and return
        # Convert to PyTorch tensors and return
        states = torch.stack(batch.state).reshape((batch_size, -1))
        actions = torch.tensor(batch.action, dtype=torch.int64).reshape((batch_size, -1))
        action_masks = torch.stack(batch.action_mask).reshape((batch_size, -1))
        rewards = torch.tensor(batch.reward, dtype=torch.float32).reshape((batch_size, -1))
        next_states = torch.stack(batch.next_state).reshape((batch_size, -1))
        next_action_masks = torch.stack(batch.next_action_mask).reshape((batch_size, -1))
        dones = torch.tensor(batch.done, dtype=torch.float32).reshape((batch_size, -1))
        
        return Transition(states, actions, action_masks, rewards, next_states, next_action_masks, dones)
    


    def __len__(self):
        return len(self.buffer)
    

def calculate_discounted_return(reward_list, gamma = 0.99):
    discounted_return = 0
    for reward in reversed(reward_list):
        discounted_return = reward + gamma * discounted_return
    return discounted_return
