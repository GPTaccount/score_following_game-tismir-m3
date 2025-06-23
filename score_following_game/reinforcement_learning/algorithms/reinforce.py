import torch

import numpy as np

from collections import OrderedDict
from score_following_game.reinforcement_learning.algorithms.agent import Agent
from score_following_game.reinforcement_learning.torch_extentions.distributions.adapted_categorical import AdaptedCategorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class ReinforceAgent(Agent):

    def __init__(self, observation_space, model, n_actions=1, no_baseline=False, batch_size=1, gamma=0.99, distribution=AdaptedCategorical,
                 use_cuda=torch.cuda.is_available(), max_steps=1e8, log_writer=None, log_interval=10,  evaluator=None,
                 eval_interval=5000, lr_scheduler=None, score_name=None, high_is_better=False, dump_interval=100000, dump_dir=None):

        Agent.__init__(self, observation_space=observation_space, model=model, n_actions=n_actions, gamma=gamma,
                       distribution=distribution, use_cuda=use_cuda, log_writer=log_writer,
                       log_interval=log_interval, evaluator=evaluator, eval_interval=eval_interval,
                       lr_scheduler=lr_scheduler, score_name=score_name, high_is_better=high_is_better,
                       dump_interval=dump_interval, dump_dir=dump_dir)

        self.no_baseline = no_baseline
        # - : 使用已棄用的 NumPy 型別 np.long
        # self.action_dtype = np.long
        self.action_dtype = np.int64
        # + : 將 np.long 替換為建議的 np.int64 型別
        self.max_steps = max_steps

        self.observations = OrderedDict()
        for obs_key in self.observation_space:

            self.observations[obs_key] = []

        self.rewards = []
        self.actions = []
        self.episode_reward = 0
        self.n_worker = 1
        self.batch_size = batch_size

    def prepare_model_input(self, observation):

        model_in = OrderedDict()

        for obs_key in observation:
            model_in[obs_key] = torch.from_numpy(observation[obs_key]).float().unsqueeze(0).to(self.device)

        return model_in

    def select_action(self, state, train=True):
        super().select_action(state, train)

        self.model.set_train_mode()

        # - : 舊版 gym env.step() 回傳四元組 (obs, reward, done, info)。此處解析該四元組或將 state 視為初始觀測值。
        # # if len(state) == 4:
        # #     observation, reward, done, _ = state
        # #     self.rewards.append(reward)
        # #     self.episode_reward += reward
        # # else:
        # #     observation = state
        # #     done = False
        
        # + : 新版 Gymnasium env.step() 回傳五元組 (obs, reward, terminated, truncated, info)。
        # + : 此處根據 state 是否為五元組來解析環境回傳值或將 state 視為初始觀測值。
        if isinstance(state, tuple) and len(state) == 5:
            observation, reward, terminated, truncated, info = state
            # + : 解構 Gymnasium 的五元組回傳值
            self.rewards.append(reward)
            self.episode_reward += reward
            done = terminated or truncated
            # + : done 狀態由 terminated 與 truncated 共同決定，符合 Gymnasium API
        else:  # state is an initial observation
            observation = state
            done = False
            # + : 若 state 非五元組 (例如初始觀測)，則 done 設為 False

        if not done:

            for obs_key in observation:
                    self.observations[obs_key].append(observation[obs_key])

            model_returns = self.model(self.prepare_model_input(observation))

            policy = model_returns['policy']

            actions, np_actions = self.model.sample_action(policy)

            self.actions.append(np_actions[0])
        else:
            np_actions = [None]

        action = np_actions[0]

        if done:
            self.perform_update()

            self.observations = OrderedDict()
            for obs_key in self.observation_space:
                self.observations[obs_key] = []

            self.rewards = []
            self.actions = []
            self.episode_reward = 0

        return action, done

    def perform_update(self):
        super().perform_update()

        rewards = np.asarray(self.rewards, dtype=np.float32)

        self.actions = np.asarray(self.actions)
        T = len(self.actions)
        mean_policy_loss = 0
        mean_bl_loss = 0

        Gts = np.zeros((T, 1), dtype=np.float32)

        # iterate collected transitions
        for t in range(T):
            Gts[t] = np.sum(rewards[t:] * [self.gamma ** (i - t) for i in range(t, T)])

        sampler = BatchSampler(SubsetRandomSampler(list(range(T))), self.batch_size, drop_last=False)

        for obs_key in self.observations:
            self.observations[obs_key] = np.asarray(self.observations[obs_key])

        i = 0
        for indices in sampler:

            Gt = torch.from_numpy(Gts[indices]).to(self.device)
            At = self.action_tensor(self.actions[indices]).to(self.device)

            St = OrderedDict()

            for obs_key in self.observations:
                St[obs_key] = torch.from_numpy(self.observations[obs_key][indices]).to(self.device)

            model_returns = self.model(St)

            policy = model_returns['policy']

            if self.no_baseline:
                baseline = 0
            else:
                baseline = model_returns['value']

            delta = Gt - baseline

            eligibility = self.model.get_log_probs(policy, At) * delta.data

            policy_loss = -eligibility.mean(dim=0)
            bl_loss = (delta ** 2).mean(dim=0)

            if self.no_baseline:
                bl_loss = 0

            self.model.update({'policy_loss': policy_loss, 'value_loss': bl_loss, 'dist_entropy': 0})

            mean_bl_loss += bl_loss
            mean_policy_loss += policy_loss

            i += 1

        mean_policy_loss /= i
        mean_bl_loss /= i

        # logging
        self.log_dict = {
            'policy_loss': mean_policy_loss.detach(),
            'avg_reward': self.episode_reward
        }

        if not self.no_baseline:
            self.log_dict['value_loss'] = mean_bl_loss.detach()