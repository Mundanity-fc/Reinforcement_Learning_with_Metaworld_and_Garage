import numpy as np
import torch
import torch.nn.functional as F

"""
The code is adapted from 
https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/episode.py
"""

class BatchEpisodes(object):
    def __init__(self, batch_size, gamma=0.95, device='cpu'):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self._observations_list = [[] for _ in range(batch_size)]
        self._actions_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)]
        self._envinfos_list = [[] for _ in range(batch_size)]
        self._imageobs_list = [[] for _ in range(batch_size)]
        self._mask_list = []

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._success_scores = None
        self._mask = None
        self._imageobs = None

    @property
    def observations(self):
        if self._observations is None:
            observation_shape = self._observations_list[0][0].shape
            observations = np.zeros((len(self), self.batch_size)
                + observation_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._observations_list[i])
                observations[:length, i] = np.stack(self._observations_list[i], axis=0)
            self._observations = torch.from_numpy(observations).float().to(self.device)
        return self._observations

    @property
    def actions(self):
        if self._actions is None:
            action_shape = self._actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size)
                + action_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                actions[:length, i] = np.stack(self._actions_list[i], axis=0)
            self._actions = torch.from_numpy(actions).float().to(self.device)
        return self._actions

    @property
    def rewards(self):
        if self._rewards is None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._rewards_list[i])
                rewards[:length, i] = np.stack(self._rewards_list[i], axis=0)
            self._rewards = torch.from_numpy(rewards).float().to(self.device)
        return self._rewards

    @property
    def returns(self):
        if self._returns is None:
            return_ = np.zeros(self.batch_size, dtype=np.float32)
            returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
            rewards = self.rewards.cpu().numpy()
            mask = self.mask.cpu().numpy()
            for i in range(len(self) - 1, -1, -1):
                return_ = self.gamma * return_ + rewards[i] * mask[i]
                returns[i] = return_
            self._returns = torch.from_numpy(returns).float().to(self.device)
        return self._returns

    @property
    def mask(self):
        if self._mask is None:
            mask = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                mask[:length, i] = 1.0
            self._mask = torch.from_numpy(mask).float().to(self.device)
        return self._mask

    @property
    def success_scores(self):
        if self._success_scores is None:
            success_scores = np.zeros(self.batch_size, dtype=np.float32)
            for i in range(self.batch_size):
                env_infos = self._envinfos_list[i]
                ep_success = float(any([info['success'] for info in env_infos]))
                success_scores[i] = ep_success      
            self._success_scores = success_scores
        return self._success_scores

    @property
    def image_obses(self):
        if self._imageobs is None:
            imageobs = []
            for i in range(self.batch_size):
                if self._imageobs_list[i]:
                    imageobs.append(np.stack(self._imageobs_list[i], axis=0))
            self._imageobs = imageobs
        return self._imageobs

    def gae(self, values, gae_lam=1.0):
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        values = values.squeeze(2).detach()
        values = F.pad(values * self.mask, (0, 0, 0, 1))

        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        advantages = torch.zeros_like(deltas).float()
        gae = torch.zeros_like(deltas[0]).float()
        for i in range(len(self) - 1, -1, -1):
            gae = gae * self.gamma * gae_lam + deltas[i]
            advantages[i] = gae

        return advantages.to(self.device)

    def append(self, observations, actions, rewards, batch_ids, image_obses, env_infos):

        for observation, action, reward, batch_id, image_obs, env_info in zip(
            observations, actions, rewards, batch_ids, image_obses, env_infos):

            if batch_id is None:
                continue
            self._observations_list[batch_id].append(observation.astype(np.float32))
            self._actions_list[batch_id].append(action.astype(np.float32))
            self._rewards_list[batch_id].append(reward.astype(np.float32))
            if image_obs is not None:
                self._imageobs_list[batch_id].append(image_obs.astype(np.uint8))
            self._envinfos_list[batch_id].append(env_info)

    def __len__(self):
        # the lengthe of the longest episode
        return max(map(len, self._rewards_list))
