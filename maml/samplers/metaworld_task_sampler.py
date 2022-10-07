import os
import time
import random

import numpy as np
import torch
import metaworld

MW_TASKS_PER_ENV = 50


class MetaWorldTaskSampler:

    """Task Sampler for MetaWorld Benchmarks"""

    def __init__(self, benchmark, mode='train'):

        self.benchmark = benchmark
        self.mode = mode

        self.train_env_classes = benchmark.train_classes
        self.test_env_classes = benchmark.test_classes
        self.train_tasks = benchmark.train_tasks
        self.test_tasks = benchmark.test_tasks

        self.num_train_envs = len(self.train_env_classes)
        self.num_test_envs = len(self.test_env_classes)

        if self.mode == 'train':
            self._classes = self.train_env_classes
            self._tasks = self.train_tasks
        elif self.mode == 'test':
            self._classes = self.test_env_classes
            self._tasks = self.test_tasks

        self._task_map = {
            env_name:
            [task for task in self._tasks if task.env_name == env_name]
            for env_name in self._classes.keys()
        }
        for tasks in self._task_map.values():
            assert len(tasks) == MW_TASKS_PER_ENV
        self._task_orders = {
            env_name: np.arange(50)
            for env_name in self._task_map.keys()
        }

        self._next_order_index = 0

    def _shuffle_tasks(self):
        for tasks in self._task_orders.values():
            np.random.shuffle(tasks)
 
    def sample_tasks(self, meta_batch_size, shuffle=False, with_replacement=False):

        tasks = []

        assert meta_batch_size % len(self._classes) == 0
        tasks_per_class = int(meta_batch_size/len(self._classes))

        for env_name, env_cls in self._classes.items():

            order_index = self._next_order_index
            for _ in range(tasks_per_class):
                task_index = self._task_orders[env_name][order_index]
                task = self._task_map[env_name][task_index]
                tasks.append((env_name, env_cls, task))
                if with_replacement:
                    order_index = np.random.randint(0, MW_TASKS_PER_ENV)
                else:
                    order_index += 1
                    order_index %= MW_TASKS_PER_ENV

        self._next_order_index += tasks_per_class
        if self._next_order_index >= MW_TASKS_PER_ENV:
            self._next_order_index %= MW_TASKS_PER_ENV
            if shuffle:
                self._shuffle_tasks()
        return tasks
