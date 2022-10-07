import multiprocessing as mp
import sys

import gym
import numpy as np

is_py2 = (sys.version[0] == '2')
if is_py2:
    pass
else:
    pass

"""
Code adapted and modified from lear2learn library

"""

class EnvWorker(mp.Process):
    def __init__(self, remote, env_fn, queue, lock, seed):
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.env = env_fn()
        self.queue = queue
        self.lock = lock
        self.task_id = None
        self.done = False
        self.env.seed(seed)
        self.step=0

    def empty_step(self):
        observation = np.zeros(self.env.observation_space.shape,
                               dtype=np.float32)
        reward, done = 0.0, True
        return observation, reward, done, {}

    def try_reset(self):

        self.step=0
        with self.lock:
            try:
                self.task_id = self.queue.get(True)
                self.done = (self.task_id is None)
            except queue.Empty:
                self.done = True
        observation = (np.zeros(self.env.observation_space.shape,
            dtype=np.float32) if self.done else self.env.reset())
        return observation

    def update_env(self, env_fn):

        old_env = self.env
        self.env = env_fn()
        old_env.close()

    def run(self):
        while True:
            command, data = self.remote.recv()
            if command == 'step':
                observation, reward, done, info = (self.env.step(data) 
                    if self.step<self.env.max_path_length and not self.done
                    else self.empty_step())
                self.step+=1

                # empty step checking
                if info:
                    task_id = self.task_id
                else:
                    task_id = None
                    
                if done and (not self.done):
                    observation = self.try_reset()

                self.remote.send((observation, reward, done, task_id, info))
            elif command == 'reset':
                observation = self.try_reset()
                self.remote.send((observation, self.task_id))
            elif command == 'set_task':
                self.env.unwrapped.set_task(data)
                self.remote.send(True)
            elif command == 'close':
                self.remote.close()
                break
            elif command == 'get_spaces':
                self.remote.send((self.env.observation_space,
                                  self.env.action_space))
            elif command == 'update_env':
                self.update_env(data)
                self.remote.send(True)
            elif command == 'render':
                image_obs = self.env.render(offscreen = data['offscreen'], resolution=data['resolution'])
                self.remote.send((image_obs))
            else:
                raise NotImplementedError()


class SubprocVecEnv(gym.Env):
    def __init__(self, env_factory, queue, seeds):
        self.lock = mp.Lock()
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_factory])
        self.workers = [EnvWorker(remote, env_fn, queue, self.lock, seed)
                        for (remote, env_fn, seed) in zip(self.work_remotes, env_factory, seeds)]
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()
        self.waiting = False
        self.closed = False

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, task_ids, infos = zip(*results)
        return np.stack(observations), np.stack(rewards), np.stack(dones), task_ids, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        observations, task_ids = zip(*results)
        return np.stack(observations), task_ids

    def set_task(self, tasks):
        for remote, task in zip(self.remotes, tasks):
            remote.send(('set_task', task))
        return np.stack([remote.recv() for remote in self.remotes])

    def update_env(self, env_fns):
        for remote, env_fn in zip(self.remotes, env_fns):
            _ = remote.send(('update_env', env_fn))
        return np.stack([remote.recv() for remote in self.remotes])

    # only if num_workers ==1 
    def render(self, **kwargs):
        for remote in self.remotes:
            remote.send(('render', kwargs))
        return np.stack([remote.recv() for remote in self.remotes])[0]
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True