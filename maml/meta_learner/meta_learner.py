import math
import copy
import random
import numpy as np
from collections import OrderedDict

import torch
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

from maml.utils import DifferentiableSGD
import maml.utils.pytorch_utils as ptu


class MetaLearner(object):

    def __init__(self, 
        policy, 
        value_fn, 
        inner_lr, 
        num_adapt_steps=1,
        episodes_per_task=10,
        gamma =0.99, 
        gae_lam = 0.95, 
        max_kl = 0.01 ,
        max_backtracks = 10 ,
        cg_steps = 10,
        cg_damping = 0.01,
        device='cpu',
        restore=False,
        ckpt_path=None):
        
        self.policy = policy
        self.old_policy = copy.deepcopy(policy)
        self.value_fn = value_fn
        self.value_fn_optimizer = torch.optim.Adam(self.value_fn.parameters(), lr=inner_lr)
        self.inner_optimizer = DifferentiableSGD(self.policy, lr=inner_lr)
        self.gamma = gamma
        self.num_adapt_steps = num_adapt_steps
        self.episodes_per_task = episodes_per_task
        self.gamma = gamma
        self.gae_lam = gae_lam

        # trpo and conjugate gradient params
        self.max_kl = max_kl
        self.max_backtracks = max_backtracks
        self.cg_steps = cg_steps
        self.cg_damping = cg_damping

        self.to(device)

        self.itr = 0
        if restore:
            self.restore(ckpt_path)

    def sample(self, tasks, sampler):
        """
        """

        old_params = dict(self.old_policy.named_parameters())
        
        self.all_episodes = [[] for _ in range(len(tasks))]
        self.all_params = []

        logs = OrderedDict()

        for i, (env_name, env_cls, task) in enumerate(tasks):

            sampler.update_task(env_cls, task)
            for j in range(self.num_adapt_steps):
                train_episodes = sampler.sample(self.policy, eps_per_task = self.episodes_per_task, 
                                                gamma=self.gamma, device=self.device)
                self.train_value_function(train_episodes)
                require_grad = j < self.num_adapt_steps -1
                self.adapt(train_episodes, set_grad=require_grad)
                self.all_episodes[i].append(train_episodes)

                if j==0:
                    logs.update(self.log_performance(train_episodes, metric_prefix=f"mt_tr_{env_name}_pre_adapt"))

            self.all_params.append(dict(self.policy.named_parameters()))
            valid_episodes = sampler.sample(self.policy, eps_per_task = self.episodes_per_task,
                                            gamma=self.gamma, device=self.device)

            logs.update(self.log_performance(valid_episodes, metric_prefix=f'mt_tr_{env_name}_post_adapt'))
            self.train_value_function(valid_episodes)
            self.all_episodes[i].append(valid_episodes)

            ptu.update_module_params(self.policy, old_params)

        return logs

    def train_value_function(self, episodes):

        value_loss = self.value_fn.value_loss(episodes.observations, episodes.returns)

        self.value_fn_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_fn_optimizer.step()

    def inner_loss(self, episodes):
        
        values = self.value_fn(episodes.observations)
        advantages = episodes.gae(values, gae_lam=self.gae_lam)
        advantages = ptu.weighted_normalize(advantages, weights=episodes.mask)

        logprobs = self.policy.logprobs(episodes.observations, episodes.actions)
        if logprobs.dim() > 2:
            logprobs = torch.sum(logprobs, dim=2)
        loss = -ptu.weighted_mean(logprobs * advantages, dim=0,
            weights=episodes.mask)
        return loss.mean()

    def adapt(self, episodes, set_grad=True):

        inner_loss = self.inner_loss(episodes)
        self.inner_optimizer.set_grads_none()
        inner_loss.backward(create_graph=set_grad)
        with torch.set_grad_enabled(set_grad):
            self.inner_optimizer.step()

    def train_step(self, tasks, sampler):

        metric_logs = OrderedDict()

        metric_logs.update(self.sample(tasks, sampler))

        kl_before = self.compute_kl_divergence(set_grad=False)
        meta_loss = self.meta_loss()
        grads = torch.autograd.grad(meta_loss, self.policy.parameters())
        policy_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        step_dir = self.conjugate_gradient(-policy_grad)

        max_step = torch.sqrt(2 *self.max_kl/torch.dot(step_dir, self.fisher_vector_product(step_dir)))
        full_step = max_step * step_dir
        expected_improve = torch.dot(-policy_grad, full_step)

        prev_params = parameters_to_vector(self.policy.parameters()).clone()
        success, new_params = self.line_search(prev_params, full_step, expected_improve)
        vector_to_parameters(new_params, self.policy.parameters())

        meta_loss_after = self.meta_loss(set_grad=False)
        kl_after = self.compute_kl_divergence(set_grad=False)

        metric_logs.update({

            'pre_adapt_kl':kl_before,
            'pre_adapt_meta_loss':meta_loss,
            'post_adapt_kl':kl_after,
            'post_adapt_meta_loss':meta_loss_after
            })

        self.itr+=1

        return metric_logs


    def evaluate_step(self, eval_tasks, eval_sampler, log_videofn=None, prefix='mt_ts', render=False, video_itr=None):

        # eval meta batch size  == benchmark test classes

        #eval_policy = copy.deepcopy(self.policy)
        theta = dict(self.policy.named_parameters())
        value_theta = dict(self.value_fn.named_parameters())

        logs = OrderedDict()

        for (env_name, env_cls, task) in eval_tasks:
            eval_sampler.update_task(env_cls, task)

            for j in range(self.num_adapt_steps):
                adapt_episodes = eval_sampler.sample(self.policy, eps_per_task=self.episodes_per_task,
                                                     gamma=self.gamma, device=self.device, render=render)

                if j==0:
                    logs.update(self.log_performance(adapt_episodes, f'{prefix}_{env_name}_pre_adapt'))
                    if render:
                        pre_imageobs = adapt_episodes.image_obses
                        log_videofn(pre_imageobs, video_itr, video_title=f'{env_name}_pre_adapt')
                       
                self.train_value_function(adapt_episodes)
                require_grad = j < self.num_adapt_steps - 1
                self.adapt(adapt_episodes, set_grad=require_grad)

            valid_episodes = eval_sampler.sample(self.policy, eps_per_task =10, gamma=self.gamma, device=self.device, render=render)
            if render:
                post_imageobs = valid_episodes.image_obses
                log_videofn(post_imageobs, video_itr, video_title=f'{env_name}_post_adapt')
            
            logs.update(self.log_performance(valid_episodes, f'{prefix}_{env_name}_post_adapt'))

            ptu.update_module_params(self.policy, theta)
            ptu.update_module_params(self.value_fn, value_theta)

        return logs

    def meta_loss(self, set_grad=True):

        old_params = dict(self.old_policy.named_parameters())
        params = dict(self.policy.named_parameters())

        task_losses = []

        for task_episodes, task_params in zip(self.all_episodes, self.all_params):

            train_episodes = task_episodes[:-1]
            valid_episodes = task_episodes[-1]

            for i in range(self.num_adapt_steps):
                require_grad = i < self.num_adapt_steps-1 or set_grad
                self.adapt(train_episodes[i], set_grad = require_grad)

            ptu.update_module_params(self.old_policy, task_params)

            with torch.set_grad_enabled(set_grad):

                oldlogprobs = self.old_policy.logprobs(valid_episodes.observations, valid_episodes.actions).detach()

                logprobs =self.policy.logprobs(valid_episodes.observations, valid_episodes.actions)
                values = self.value_fn(valid_episodes.observations)

                advantages = valid_episodes.gae(values, gae_lam=self.gae_lam)
                advantages = ptu.weighted_normalize(advantages,
                    weights=valid_episodes.mask)

                log_ratio = logprobs-oldlogprobs
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -ptu.weighted_mean(ratio * advantages, dim=0,
                    weights=valid_episodes.mask)
                task_losses.append(loss)

    
            ptu.update_module_params(self.policy, params)
            ptu.update_module_params(self.old_policy, old_params)

        return torch.mean(torch.stack(task_losses, dim=0))

    def compute_kl_divergence(self, set_grad=True):

        #old_pi = copy.deepcopy(self.policy)

        old_params = dict(self.old_policy.named_parameters())
        params = dict(self.policy.named_parameters())
        kls = []

        for task_episodes, task_params in zip(self.all_episodes, self.all_params):

            train_episodes = task_episodes[:-1]
            valid_episodes = task_episodes[-1]
    
            for i in range(self.num_adapt_steps):
                require_grad = i < self.num_adapt_steps-1 or set_grad
                self.adapt(train_episodes[i], set_grad=require_grad)

            ptu.update_module_params(self.old_policy, task_params)

            with torch.set_grad_enabled(set_grad):

                old_pi = ptu.detach_distribution(self.old_policy(valid_episodes.observations))
                new_pi  = self.policy(valid_episodes.observations)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)

                kl_loss = ptu.weighted_mean(kl_divergence(new_pi, old_pi), dim=0, 
                                                         weights=mask)
                kls.append(kl_loss)

            ptu.update_module_params(self.policy, params)
            ptu.update_module_params(self.old_policy, old_params)

        return torch.mean(torch.stack(kls, dim=0))

    def conjugate_gradient(self, b, residual_tol=1e-10):
        """
           Conjugate gradient descent algorithm

           For Conjugate gradient descent algorithm and derivation refer below link

           http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/conjugate_direction_methods.pdf
        """
        x_k = torch.zeros(b.size())
        d_k = b.clone().detach()
        g_k = b.clone().detach()
        g_dot_g = torch.dot(g_k, g_k)

        for _ in range(self.cg_steps):

            fvp = self.fisher_vector_product(d_k)
            alpha = g_dot_g / torch.dot(d_k, fvp)
            x_k += alpha * d_k

            g_k -= alpha * fvp 
            new_g_dot_g = torch.dot(g_k, g_k)

            beta = new_g_dot_g / g_dot_g
            d_k = g_k + beta * d_k
            g_dot_g = new_g_dot_g

            if g_dot_g < residual_tol:
                break

        return x_k.detach()

    def line_search(self, prev_params, fullstep, expected_improve, accept_ratio=0.1):
        """
           line search to find optimal parameters in trust region
        """

        prev_loss  = self.meta_loss()

        for stepfrac in [.5**x for x in range(self.max_backtracks)]:
            new_params = prev_params + stepfrac * fullstep
            vector_to_parameters(new_params, self.policy.parameters())
            loss = self.meta_loss(set_grad=False)
            kl = self.compute_kl_divergence(set_grad=False)
            improved = prev_loss - loss
            #expected_improve = expected_improve * stepfrac
            #ratio = improved/expected_improve

            if improved.item() > 0.0 and kl.item() < self.max_kl:
                return True, new_params

        return False, prev_params

    def fisher_vector_product(self, vector):
        """
           Helper_fn to compute Hessian vector product to be used in cg algorithm
        """
        kl_loss = self.compute_kl_divergence(set_grad=True)

        grads = torch.autograd.grad(kl_loss, self.policy.parameters(), create_graph=True)
        grad_vector = torch.cat([grad.view(-1) for grad in grads])
        grad_vector_product = torch.sum(grad_vector * vector)
        grad_grads = torch.autograd.grad(grad_vector_product, self.policy.parameters())
        fisher_vector_product = torch.cat([grad.contiguous().view(-1) for grad in grad_grads]).detach()

        return fisher_vector_product + self.cg_damping * vector

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.old_policy.to(device, **kwargs)
        self.value_fn.to(device, **kwargs)
        self.device = device

    def save(self, path):
        torch.save({
            "meta_policy" : self.policy.state_dict(),
            "value_function" : self.value_fn.state_dict(),
            "value_fn_optimizer": self.value_fn_optimizer.state_dict(),
            "iteration": self.itr,
            }, path)

    def restore(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['meta_policy'])
        self.value_fn.load_state_dict(checkpoint['value_function'])
        self.value_fn_optimizer.load_state_dict(checkpoint['value_fn_optimizer'])
        self.itr = checkpoint['iteration']

    def log_performance(self, batch_episodes, metric_prefix='mt_tr'):

        rewards = batch_episodes.rewards.to('cpu').detach().numpy()
        success_scores = batch_episodes.success_scores
        return_per_episode = np.sum(rewards, axis=0)

        avg_return = np.mean(return_per_episode)
        std_return = np.std(return_per_episode)
        max_return = np.max(return_per_episode)
        min_return = np.min(return_per_episode)
        avg_success = np.mean(success_scores)

        metric_log = {

        f"{metric_prefix}_AvgReturn": avg_return,
        f"{metric_prefix}_StdReturn" : std_return,
        f"{metric_prefix}_MaxReturn" : max_return,
        f"{metric_prefix}_MinReturn" : min_return,
        f"{metric_prefix}_AvgSuccess" : avg_success,

        }

        return metric_log
