import os
import time
import random
import numpy as np
import torch
import metaworld
from collections import OrderedDict
import utils.pytorch_utils as ptu
from maml.envs import AsyncVectorEnv
from maml.policies import GaussianPolicy, CategoricalPolicy
from maml.critics import ValueFun
from maml.samplers import Sampler, MetaWorldTaskSampler
from maml.utils import Logger
from maml.meta_learner import MetaLearner


def init_metaworld_env(benchmark, seed, num_workers):
    """
    initializes metaworld env across num_workers

    Args:
    benchmark : Metworld Benchmark(one of ML1, ML10, ML45 from metaworld)
    seed : random seed
    num_workers : number of parallel workers

    Returns
    MetaWorld Env distributed across workers
    """
    _, env_cls = list(benchmark.train_classes.items())[0]
    seeds = [ seed * (i+1) for i in range(num_workers)]
    env = AsyncVectorEnv([env_cls for _ in range(num_workers)], seeds)
    return env

def set_random_seed(seed):
    """
    Sets random seed for reproducibility of experiments
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_benchmark(benchmark_type):
    """
    initializes metaworld benchmark
    Args:
    benchmark_type : str input one of ('ml1', 'ml10', 'ml45')

    Returns 
    MetaWorld Benchmark
    """
    if benchmark_type == 'ml1':
        # pick-place-v2 is chosen arbitarily, can choose any other env
        return metaworld.ML1('pick-place-v2') 
    elif benchmark_type == 'ml10':
        return metaworld.ML10()
    elif benchmark_type == 'ml45':
        return metaworld.ML45()

class Trainer:
    """
    Trainer class to run train and evaluation experiments
    """

    def __init__(self, params):

        self.params = params
        # self.benchmark = set_benchmark(self.params['benchmark_type'])
        self.benchmark = metaworld.ML1('assembly-v2')

        set_random_seed(self.params['seed'])

        self.env = init_metaworld_env(self.benchmark, self.params['seed'], self.params['num_workers']) 
        self.eval_env = init_metaworld_env(self.benchmark, self.params['seed'], 1)   

        self.task_sampler = MetaWorldTaskSampler(self.benchmark, mode='train')
        self.test_task_sampler = MetaWorldTaskSampler(self.benchmark, mode='test')

        self.sampler = Sampler(self.env, num_workers=self.params['num_workers'])
        self.render_sampler = Sampler(self.eval_env, num_workers=1)

        obs_space, ac_space = self.env.get_spaces()

        self.device = ptu.init_gpu(self.params['use_gpu'], self.params['which_gpu'])

        self.policy = GaussianPolicy(
            in_size = int(np.prod(obs_space.shape)),
            out_size = int(np.prod(ac_space.shape)),
            hidden_units=[100, 100], 
            activation='tanh', 
            device=self.device)

        self.value_fn = ValueFun(
            in_size = int(np.prod(obs_space.shape)),
            out_size=1, 
            hidden_units=[32, 32], 
            activation='tanh', 
            device=self.device)      
        
        self.logger = Logger(self.params['logdir'])

        self.meta_train_itrs = self.params['n_iters']
        self.meta_train_batch_size = self.params['meta_batch_size']

        # meta-train and meta-test envs have same number of task variations in Metaworld benchmark
        # Eval-itrs is to evaluate for each task variation across meta-train and meta-test envs
        self.eval_itrs =  int(len(self.benchmark.test_tasks)/len(self.benchmark.test_classes))

        self.metalearner = MetaLearner(
            policy = self.policy,
            value_fn = self.value_fn,
            inner_lr = self.params['inner_lr'],
            num_adapt_steps = self.params['num_adapt_steps'],
            episodes_per_task = self.params['episodes_per_task'],
            gamma = self.params['gamma'],
            gae_lam = self.params['gae_lam'],
            max_kl = self.params['max_kl_increment'],
            max_backtracks = self.params['max_backtracks'],
            cg_steps = self.params['cg_steps'],
            cg_damping =self.params['cg_damping'],
            device = self.device,
            restore = self.params['restore'],
            ckpt_path = self.params['ckpt_path'])  

        self.start_itr = self.metalearner.itr

        self.meta_eval_freq =  params['eval_freq']

    def meta_train(self):

        if self.start_itr!=0:
            self.start_itr+= 1

        for itr in range(self.start_itr, self.meta_train_itrs):

            print(f"At Iteration {itr}")

            logs = OrderedDict()

            tasks = self.task_sampler.sample_tasks(self.meta_train_batch_size)
            train_logs = self.metalearner.train_step(tasks, self.sampler)
            logs.update(train_logs) 

            if itr==0 or ((itr+1)% self.meta_eval_freq)==0:
                meta_batch_size = len(self.benchmark.test_classes)
                test_tasks = self.test_task_sampler.sample_tasks(meta_batch_size)
                test_logs = self.metalearner.evaluate_step(test_tasks, self.sampler)
                logs.update(test_logs)

                self.metalearner.save('{}/maml_ckpt_itr_{}.pt'.format(self.params['logdir'], itr))

            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)

            self.logger.flush()

    def meta_evaluate(self, eval_on='train_tasks', render=False):

        if eval_on == 'train_tasks':
            task_sampler = self.task_sampler
            meta_batch_size = len(self.benchmark.train_classes)
            prefix ='mt_tr'
            
        else:
            task_sampler = self.test_task_sampler
            meta_batch_size = len(self.benchmark.test_classes)
            prefix ='mt_ts'

        if render==True:
            sampler = self.render_sampler
        else:
            sampler = self.sampler

        for itr in range(self.eval_itrs):
            tasks = task_sampler.sample_tasks(meta_batch_size, shuffle=False)
            logs = self.metalearner.evaluate_step(tasks, sampler, log_videofn=self.logger.log_videos,
                                                                render=render, prefix=prefix,video_itr=itr)
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)

            self.logger.flush()

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_on', type=str, choices=['train_tasks', 'test_tasks'], default='meta_train',
                                    help='test meta-trained policy on either meta_train/meta_test tasks')
    parser.add_argument('--render', action='store_true')

    parser.add_argument('--benchmark_type', type=str, default='ml10', choices=['ml1', 'ml10', 'ml45'])
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iters', '-n', type=int, default=300)
    parser.add_argument('--num_workers', '-n_w', type=int, default=1)

    parser.add_argument('--use_gae', action='store_true')
    parser.add_argument('--gae_lam', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--meta_batch_size', '-mb', type=int, default=20) # tasks collected per meta train iteration
    parser.add_argument('--episodes_per_task', type=int, default=10) # rollouts collected per a single task

    parser.add_argument('--num_adapt_steps', type =int, default=1)
    
    parser.add_argument('--inner_lr', type=float, default=1e-4, help='learning_rate for inner-loop/adaptation steps')

    # TRPO update params
    parser.add_argument('--cg_steps', type=int, default=10, help='num cg steps')
    parser.add_argument('--cg_damping', type=float, default=1e-1, help='damping to use in cg')
    parser.add_argument('--max_backtracks', type=int, default=10, help='no of backtracks for line search')
    parser.add_argument('--max_kl_increment', type=float, default=1e-2, 
                                           help='max_kl divergence between old and new policy')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--eval_freq', type=int, default=10)

    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--ckpt_path', type=str, help='checkpoint path to restore from')
    
    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.benchmark_type + '_' + args.exp_name + '_' + time.strftime("%d-%m-%Y-%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    trainer = Trainer(params)

    if args.train:
        trainer.meta_train()

    elif args.evaluate:
        trainer.meta_evaluate(params['eval_on'], params['render'])

if __name__ == "__main__":
    main()
