import multiprocessing as mp

from .subproc_vec_env import SubprocVecEnv

"""
Code adpated from learn2learn library

"""

class AsyncVectorEnv(SubprocVecEnv):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/async_vec_env.py)
    **Description**
    Asynchronous vectorized environment for working with MetaEnvs.
    Allows multiple environments to be run as separate processes.
    **Credit**
    Adapted from OpenAI and Tristan Deleu's implementations.
    """
    def __init__(self, env_fns, seeds=None, env=None):
        self.num_envs = len(env_fns)
        self.queue = mp.Queue()
        super(AsyncVectorEnv, self).__init__(env_fns, queue=self.queue, seeds=seeds)
        if env is None:
            env = env_fns[0]()
        self._env = env
        self._env.seed(seeds[0])
        #self.reset()

    def set_task(self, task):
        tasks = [task for _ in range(self.num_envs)]
        reset = super(AsyncVectorEnv, self).set_task(tasks)
        return all(reset)

    def update_env(self, env_fns):
        updated =  super(AsyncVectorEnv, self).update_env(env_fns)
        return all(updated)
        
    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks

    def step(self, actions):
        obs, rews, dones, ids, infos = super(AsyncVectorEnv, self).step(actions)
        return obs, rews, dones, ids, infos

    def reset(self, batch_size):
        for i in range(batch_size):
            self.queue.put(i)
        for i in range(self.num_envs):
            self.queue.put(None)
        obs, ids = super(AsyncVectorEnv, self).reset()
        return obs, ids

    def get_spaces(self):

        obs_space = self._env.observation_space
        ac_space = self._env.action_space

        return obs_space, ac_space

    def render(self, offscreen, resolution):
        image_obs = super(AsyncVectorEnv, self).render(offscreen=offscreen, resolution=resolution)
        return image_obs
