#This file is a part of COL778 A4
import os
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import  gym
import numpy as np
import torch


import config as exp_config
import utils.utils as utils
import utils.pytorch_util as ptu
from utils.logger import Logger

MAX_NVIDEO = 2

def setup_agent(args, configs):
    global env, agent
    
    env = gym.make(args.env_name,render_mode=None)
    env.action_space.seed()
    env.reset()
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    if args.exp_name == "imitation":
        from agents.mujoco_agents import ImitationAgent as Agent    
    elif args.exp_name == "RL":
        from agents.mujoco_agents import RLAgent as Agent
    else:
        raise ValueError(f"Invalid experiment name {args.exp_name}")

    agent = Agent(ob_dim, ac_dim, args, **configs['hyperparameters'])
    if args.load_checkpoint is not None:
        agent.load_state_dict(torch.load(args.load_checkpoint))




def train_agent(args, configs):
    logger = Logger(args.logdir)
    max_ep_len = configs.get("episode_len", None) or env.spec.max_episode_steps
    # set random seeds
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    
    total_envsteps = 0
    start_time = time.time()

    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    agent.to(ptu.device)

    for itr in range(configs['num_iteration']):
        print(f"\n********** Iteration {itr} ************")
        
        #train one iteration of the agent and return the loss and the training trajectory
        train_info = agent.train_iteration(env, envsteps_so_far = total_envsteps, render=False, itr_num = itr)

        total_envsteps += train_info['current_train_envsteps']
        
        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.get_action, 15*max_ep_len, max_ep_len
            )
            # train_info = {'episode_loss':0, 'trajectories':eval_trajs} # dummy values used for testing

            logs = utils.compute_metrics(train_info['trajectories'], eval_trajs)
            # compute additional metrics
            logs.update({'train_loss':train_info['episode_loss']})

            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            logger.flush()


        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.get_action, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, choices = ["imitation", "RL"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    args = parser.parse_args()

    configs = exp_config.configs[args.env_name][args.exp_name]

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")
    model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models")
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = (
        args.exp_name
        + "_"
        + args.env_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    args.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    setup_agent(args, configs)
    train_agent(args, configs)
    torch.save(agent.state_dict(), os.path.join(model_save_path, "model_"+ args.env_name + "_"+ args.exp_name+".pth"))


if __name__ == "__main__":
    main()
