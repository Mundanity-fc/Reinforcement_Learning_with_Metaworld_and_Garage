import gym
import torch
import numpy as np
import multiprocessing as mp

from maml.samplers.episode import BatchEpisodes

class Sampler:
    """
    Samples batch of Episodes for a task using parallel workers

    """

    def __init__(self, env, num_workers):

        self.env = env 
        self.batch_size = num_workers
        self.num_workers = num_workers 

    def update_task(self, env_cls, task):

        _ = self.env.update_env(env_fns = [env_cls for _ in range(self.num_workers)])
        _ = self.env.set_task(task)

    def sample(self, policy, eps_per_task=None, params=None, gamma=0.99, device='cpu', render=False):

        episodes = BatchEpisodes(batch_size= eps_per_task or self.batch_size, gamma=gamma, device=device)
        observations, batch_ids = self.env.reset(eps_per_task) #AsyncVectorEnv reset

        dones=[False]
        while ((not all(dones)) or (not self.env.queue.empty())):
            # Rendering done for evaluation step only with single worker 
            if render and self.num_workers==1:
                image_obs = self.env.render(offscreen=True, resolution=(640,480))
                image_obs = np.expand_dims(image_obs, axis=0)
            else:
                image_obs = None
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).float().to(device=device)
                actions_tensor = policy.sample(observations_tensor)
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, batch_ids, env_infos= self.env.step(actions)
            if image_obs is None:
                image_obs=[None]*rewards.shape[0]
            episodes.append(observations, actions, rewards, batch_ids, image_obs, env_infos)
            observations = new_observations

        return episodes

