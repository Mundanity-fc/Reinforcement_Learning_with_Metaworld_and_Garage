import os
import time
import random

import numpy as np
import torch
import gym

from maml_trpo.envs.asyn_vector_env import AsynVectorEnv

def make_mujoco_env(env_name, seed, num_workers):
    # Set a specific task or left empty to train on all available tasks
   
    def _make_env():
        return gym.make(env_name)
    env_fn =  _make_env

    env = AsyncVectorEnv([env_fn for _ in range(num_workers)])
    env.seed(seed)
    env.set_task(env.sample_tasks(1)[0])

    return env

def make_env(env_name):

	def _make_env(env_name):
		env = gym.make(env_name)
		return env

	return _make_env

class MujocoEnv(AsynVectorEnv):

	def __init__(self, env_name, num_workers, seed=None):

		self.env_name = env_name
		self.num_workers = workers
		env_fns = [make_env(env_name) for _ in range(self.num_workers)]
		super(AsyncVectorEnv).__init__(env_fns, seed)

	def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks
