from utils import SMALL_ENV, LARGE_ENV
from utils import plot_rewards
import numpy as np
from collections import namedtuple, deque
import numpy as np
import torch.nn as nn
import torch
import random
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))
from tqdm.auto import tqdm
import wandb 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNetwork(nn.Module):

    def __init__(self,layer_arch, num_actions):
        super(DQNetwork,self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_arch) - 1):
            self.layers.append(nn.Linear(layer_arch[i],layer_arch[i+1]))
        self.output = nn.Linear(layer_arch[-1],num_actions)
        self.activation = nn.ReLU()
        self.num_actions = num_actions
    
    def forward(self,x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return (self.output(x))
    
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PrioritizeReplayMemory:

    def __init__(self, buffer_size, alpha=0.0, random_state=None):
        self._buffer_size = buffer_size
        self._alpha = alpha # alpha is the probability parameter
        self._buffer_length = 0
        self._buffer = np.empty(self._buffer_size, dtype=[("priority", np.float32), ("transition", Transition)])
        self._random_state = np.random.RandomState() if random_state is None else random_state

    def __len__(self):
        return self._buffer_length 
    
    def alpha(self):
        return self._alpha 
    
    
    def buffer_size(self):
        return self._buffer_size

    def push(self, *args):
        transition = Transition(*args)
        priority = 1.0 if self.is_empty() else self._buffer["priority"].max()
        if self.is_full():
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, transition)
            else:
                pass # low priority transition should not be included in the buffer 
        else:
            self._buffer[self._buffer_length] = (priority, transition)
            self._buffer_length += 1
    
    def is_empty(self):
        return self._buffer_length == 0
    
    def is_full(self):
        return self._buffer_length == self._buffer_size
    
    def sample(self, beta, batch_size):
        # sample using computed probabilities
        ps = self._buffer[:self._buffer_length]["priority"]
        sampling_probs = ps**self._alpha/np.sum(ps**self._alpha)
        idxs = self._random_state.choice(np.arange(ps.size), size=batch_size, replace=True, p=sampling_probs)

        # select the experiences and compute their respective sampling weights
        experiences = self._buffer["transition"][idxs]
        weights = (self._buffer_length * sampling_probs[idxs])**(-beta)
        normalized_weights = weights / weights.max()
        return idxs, experiences, normalized_weights 

    def update_priorities(self, idxs, priorities):
        self._buffer["priority"][idxs] = priorities 

class DQNAgent:

    def __init__(self, env, model_arch, double = False, gamma = 0.99, memory_limit = 5000, model_path=None):
        self.env = env
        self.large = (env == LARGE_ENV)
        self.num_rows, self.num_cols = env.nrow, env.ncol
        self.n_states = env.observation_space.n
        self.policy_dqn = DQNetwork(model_arch,env.action_space.n).to(device)
        if double:
            self.target_dqn = DQNetwork(model_arch,env.action_space.n).to(device)

        self.memory = ReplayMemory(memory_limit)
        self.gamma = gamma
        self.optimizer = None
        self.criterion = None
        self.all_training_rewards = []
        self.last_training_rewards = None
        self.double = double
        self.model_path = model_path

    def synchronize_network(self):
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())


    def select_action(self, state, epsilon):
        sample = np.random.random()
        state = state.to(device)
        if sample > epsilon:
            with torch.no_grad():
                return self.policy_dqn(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=state.device, dtype=torch.long)

    def train(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)
        mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=state_batch.device, dtype=torch.bool)
        next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        state_action_values = self.policy_dqn(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.batch_size, device=state_batch.device)
        with torch.no_grad():
            if self.double:
                next_state_values[mask] = self.target_dqn(next_states).max(1).values
            else:
                next_state_values[mask] = self.policy_dqn(next_states).max(1).values
                
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def one_hot_tensor(self, index):
        tensor = torch.zeros(self.n_states, dtype=torch.float32).to(device)
        tensor[index] = 1
        return tensor

    def evaluate(self):
        state, info = self.env.reset()
        state = self.one_hot_tensor(state).unsqueeze(0)
        terminated = False
        total_reward = 0
        if self.large:
            max_step = 1700
        else:
            max_step = 500
        step = 0
        while step < max_step:
            action = self.select_action(state, 0)
            next_state, reward, terminated, _, _ = self.env.step(action.item())
            if terminated:
                next_state = None
            else:
                next_state = self.one_hot_tensor(next_state).unsqueeze(0)
            state = next_state
            total_reward += ((self.gamma**step) * reward)
            step += 1
            if terminated:
                break
        return total_reward

    def get_initial_state_value(self):
        # run the policy network, get q(s,a) and then take max
        state_batch = self.one_hot_tensor(0).unsqueeze(0)
        with torch.no_grad():
            state_action_values = self.policy_dqn(state_batch)
            max_value, max_index = torch.max(state_action_values, dim=1)
            return max_value.item()


    def dqn_learning(self, optimizer, criterion, num_episodes = 1000,batch_size = 512, exp_start = 1,  exp_end = 0.1, exp_decay = 2000, MODEL_MIX = 0.0005, device = 'cuda'):
        self.last_training_rewards = []
        steps_done = 0
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        report_dict = dict()
        max_val_reward = -200
        for episode in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            state = self.one_hot_tensor(state).unsqueeze(0)
            train_reward = 0
            terminated = False
            episode_step = 0
            while not terminated:
                epsilon = exp_end + (exp_start - exp_end) * np.exp(-1. * steps_done / exp_decay)
                action = self.select_action(state, epsilon=epsilon)
                steps_done += 1
                next_state, reward, terminated, _, _ = self.env.step(action.item())
                train_reward += ((self.gamma**episode_step) * reward)
                episode_step += 1
                reward = torch.tensor([reward], device=device)
                if terminated:
                    next_state = None
                else:
                    next_state = self.one_hot_tensor(next_state).unsqueeze(0)

                self.memory.push(state, action, next_state, reward, terminated)
                state = next_state

                if len(self.memory) >= self.batch_size:
                    self.train()

                if self.double:
                    self.synchronize_network()

            val_reward = self.evaluate()

            initial_state_value = self.get_initial_state_value()
            print(f"Training, Episode {episode}, reward: {train_reward}")
            print(f"Validation, Episode {episode}, reward: {val_reward}")
            report_dict['train_reward'] = train_reward
            report_dict['val_reward'] = val_reward 
            report_dict["initial_state_value"] = initial_state_value
            report_dict["epsilon"] = epsilon
            report_dict['episode'] = episode 
            if episode%200 == 0:
                with open(os.path.join(self.model_path, "agent_{}.obj".format(episode)), "wb") as f:
                    pickle.dump(self, f)

            elif val_reward > max_val_reward:
                max_val_reward = val_reward
                with open(os.path.join(self.model_path, "agent_{}.obj".format(episode)), "wb") as f:
                    pickle.dump(self, f)
            
            wandb.log(report_dict) 
        print('Complete')

if __name__ == "__main__":
    model_name = "double_dqn_small_final"
    model_path = os.path.join("/home/cse/dual/cs5190448/scratch/COL778/A3/models", model_name)
    os.makedirs(model_path, exist_ok=True)
    wandb.init(project = "COL778_A3", name = f"{model_name}")
    wandb.run.log_code(".")
    env = SMALL_ENV
    if "double" in model_name:
        double=True
    else:
        double=False
    action_map = { 0: "\u25C4", 1: "\u25BC", 2: "\u25BA",3: "\u25B2"}
    architecture = [env.observation_space.n, 128, 128]
    agentDeep = DQNAgent(env, architecture, double = double, memory_limit=10000, model_path=model_path)
    print('Model Loaded\nStarting')
    optimizer = torch.optim.AdamW(agentDeep.policy_dqn.parameters(), lr = 1e-4)
    criterion = torch.nn.SmoothL1Loss()
    if env == SMALL_ENV:
        agentDeep.dqn_learning(optimizer, criterion, 2000, device = 'cuda')
    else:
        agentDeep.dqn_learning(optimizer, criterion, 10000, device = 'cuda')