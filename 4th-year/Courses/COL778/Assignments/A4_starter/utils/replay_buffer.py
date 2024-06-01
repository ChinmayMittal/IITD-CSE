from utils.utils import *
import numpy as np

class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        if hasattr(self, 'obs') and self.obs is not None:
            assert isinstance(self.obs, np.ndarray), "obs should be numpy array. Check the add_rollouts function"
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths, concat_rew=True):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, next_observations, terminals, concat_rewards, rewards = (
            convert_listofrollouts(paths))
        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = concat_rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, concat_rewards], axis = 0
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]


    def sample_batch(self,size, required = ['obs', 'acs', 'next_obs', 'rews', 'terminals']):
        random_indices = np.random.randint(0, len(self), size)
        batch_dict = {k: self.__dict__[k][random_indices] for k in required}
        return batch_dict
        