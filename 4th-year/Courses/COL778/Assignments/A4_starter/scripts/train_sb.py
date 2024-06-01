#This file is a part of COL778 A4
import os
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import gymnasium as gym
import numpy as np
import torch

import config as exp_config
import utils.utils as utils
import utils.pytorch_util as ptu
from utils.logger import Logger


def setup_agent(args, configs):
    
    from agents.mujoco_agents import SBAgent as Agent
    global agent, env
    agent = Agent(args.env_name, **configs['hyperparameters'])
    env = agent.env
    if args.load_checkpoint is not None:
        agent.load_checkpoint(args.load_checkpoint)
    
def train_agent():
    agent.learn() #A timeout will be imposed on this function during evaluation. Make sure you save the model periodically to avoid losing progress.
    
    
    
def test_agent(args):
    
    obs, _ = env.reset()
    rewards = []
    image_obs =  []
    for _ in range(500):
        action, _ = agent.get_action(obs)
        if args.env_name == "PandaPush-v3":
            obs, rew, terminated, truncated, _ = env.step(action)
        else:
            obs, rew, done, info = env.step(action)
        if truncated or terminated:
            obs, _ = env.reset()
            continue
        image_obs.append(env.render())
        rewards.append(rew)
    
    print(f"Mean reward: {np.mean(rewards)}")
    
        
    # Save the images as video
    import cv2
    import numpy as np
    height, width, layers = image_obs[0].shape
    size = (width,height)
    
    out = cv2.VideoWriter('{n}.mp4'.format(n=args.env_name),cv2.VideoWriter_fourcc(*'mp4v'),10, (size[0], size[1]))
    for i in range(len(image_obs)):
        # rgb_img = cv2.cvtColor(traj["image_obs"][i], cv2.COLOR_RGB2BGR)
        out.write(image_obs[i])
    out.release()
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--load_checkpoint', type=str)
    
    args = parser.parse_args()
    
    configs = exp_config.configs[args.env_name]['SB3']

    setup_agent(args, configs)
    
    if args.test:
        test_agent(args)
    else:
        train_agent()
            
        
    
    















