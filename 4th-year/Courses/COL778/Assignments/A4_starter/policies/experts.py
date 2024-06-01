
import torch
import torch.nn as nn
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu



class ExpertPolicy(nn.Module):
    def __init__(self, observation_dim, action_dim, num_layers, hidden_size) -> None:
        super().__init__()
        self.policy = ptu.build_mlp(observation_dim, action_dim, num_layers, hidden_size)

    def forward(self, observation):
        return self.policy(observation)
    
    @torch.no_grad()
    def get_action(self, observation):
        action  = self.forward(observation)
        action = action.detach().cpu().numpy()
        return action
    


class ExpertPolicyJIT(object):
    def __init__(self, policy):
        self.policy = policy
    
    def forward(self, observation):
        return self.policy(observation)

    @torch.no_grad()
    def get_action(self, observation):
        return self.policy(observation).detach().cpu().numpy()

def load_expert_policy(env, env_name, jit = False):
    import os
    if not jit:
        state_dict =  torch.load(os.path.join("./policies/experts", env_name + ".pth"))
        state_dict = {"policy."+k :v for k,v in state_dict.items()}
        expert_policy = ExpertPolicy(env.observation_space.shape[0], env.action_space.shape[0], len(state_dict)//2 -1, state_dict['policy.0.weight'].shape[0])

        expert_policy.load_state_dict(state_dict)
    else:
        expert_policy = torch.jit.load(os.path.join("./policies/experts", env_name + "_jit.pth"))
        expert_policy = ExpertPolicyJIT(expert_policy)
    expert_data, _ = utils.sample_trajectories(env,
                                        expert_policy.forward,
                                        10000,
                                        env.spec.max_episode_steps,)
    expert_policy.to(ptu.device)
    return expert_policy, expert_data



