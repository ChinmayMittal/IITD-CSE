import itertools
import torch
import random
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
import torch.distributions as distributions

from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
from policies.experts import load_expert_policy



class ImitationAgent(BaseAgent):
    '''
    Please implement an Imitation Learning agent. Read scripts/train_agent.py to see how the class is used. 
    
    
    Note: 1) You may explore the files in utils to see what helper functions are available for you.
          2)You can add extra functions or modify existing functions. Dont modify the function signature of __init__ and train_iteration.  
          3) The hyperparameters dictionary contains all the parameters you have set for your agent. You can find the details of parameters in config.py.  
    
    Usage of Expert policy:
        Use self.expert_policy.get_action(observation:torch.Tensor) to get expert action for any given observation. 
        Expert policy expects a CPU tensors. If your input observations are in GPU, then 
        You can explore policies/experts.py to see how this function is implemented.
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.replay_buffer = ReplayBuffer(5000) #you can set the max size of replay buffer if you want
        

        #initialize your model and optimizer and other variables you may need
        self.optimizer = None 

        

    def forward(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        action = None #change this to your action
        return action


    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        
        action = None #change this to your action
        return action 

    
    
    def update(self, observations, actions):
        #*********YOUR CODE HERE******************
        
        pass
    


    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        if not hasattr(self, "expert_policy"):
            self.expert_policy, initial_expert_data = load_expert_policy(env, self.args.env_name)
            self.replay_buffer.add_rollouts(initial_expert_data)
            
            #to sample from replay buffer use self.replay_buffer.sample_batch(batch_size, required = <list of required keys>)
            # for example: sample = self.replay_buffer.sample_batch(32)
        
        #*********YOUR CODE HERE******************
        
        return {'episode_loss': None, 'trajectories': None, 'current_train_envsteps': None} #you can return more metadata if you want to









class RLAgent(BaseAgent):

    '''
    Please implement an policy gradient agent. Read scripts/train_agent.py to see how the class is used. 
    
    
    Note: Please read the note (1), (2), (3) in ImitationAgent class. 
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        #initialize your model and optimizer and other variables you may need
        

    
    def forward(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        pass


    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        pass


    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        #*********YOUR CODE HERE******************
        self.train()
        
        return {'episode_loss': None, 'trajectories': None, 'current_train_envsteps': None} #you can return more metadata if you want to





class SBAgent(BaseAgent):
    def __init__(self, env_name, **hyperparameters):
        #implement your init function
        from stable_baselines3.common.env_util import make_vec_env
        import gymnasium as gym
        self.env_name = env_name
        self.env = None #initialize your environment. This variable will be used for evaluation. See train_sb.py
        pass
    
    def learn(self):
        #implement your learn function. You should save the checkpoint periodically to <env_name>_sb3.zip
        pass
    
    def load_checkpoint(self, checkpoint_path):
        #implement your load checkpoint function
        pass
    
    def get_action(self, observation):
        #implement your get action function
        pass
    