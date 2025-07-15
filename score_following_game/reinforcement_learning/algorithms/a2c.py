
import torch

import numpy as np

from collections import OrderedDict
from score_following_game.reinforcement_learning.algorithms.agent import Agent
from score_following_game.reinforcement_learning.torch_extentions.distributions.adapted_categorical import AdaptedCategorical


class A2CAgent(Agent):

    def __init__(self, observation_space, model, n_actions=1, t_max=5, n_worker=1, gamma=0.99, distribution=AdaptedCategorical,
                 use_cuda=torch.cuda.is_available(), gae=False, gae_lambda=0.95, log_writer=None, log_interval=10,
                 evaluator=None, eval_interval=5000, lr_scheduler=None, score_name=None, high_is_better=False,
                 dump_interval=100000, dump_dir=None, buffer=None):

        Agent.__init__(self, observation_space=observation_space, model=model,  n_actions=n_actions, gamma=gamma,
                       distribution=distribution, use_cuda=use_cuda, log_writer=log_writer, log_interval=log_interval,
                       evaluator=evaluator, eval_interval=eval_interval, lr_scheduler=lr_scheduler, score_name=score_name,
                       high_is_better=high_is_better, dump_interval=dump_interval, dump_dir=dump_dir, buffer=buffer)

        self.t_max = t_max
        self.n_worker = n_worker

        self.observations = OrderedDict()
        for obs_key in self.observation_space:

            obs_shape = self.observation_space[obs_key].shape
            self.observations[obs_key] = torch.zeros(self.t_max + 1, self.n_worker,
                                                     *[int(x) for x in list(obs_shape)]).to(self.device)

        self.rewards = []
        self.value_predictions = torch.zeros(self.t_max + 1, self.n_worker, 1).to(self.device)
        self.returns = torch.zeros(self.t_max + 1, self.n_worker, 1).to(self.device)

        self.actions = self.action_tensor(self.t_max, self.n_worker, self.n_actions).to(self.device)
        self.masks = torch.ones(self.t_max, self.n_worker, 1).to(self.device)

        # we will only store the log probs of the chose actions
        self.old_log_probs = torch.zeros(self.t_max, self.n_worker, 1).to(self.device)

        # reward bookkeeping
        self.episode_rewards = torch.zeros([self.n_worker, 1]).to(self.device)
        self.final_rewards = torch.zeros([self.n_worker, 1]).to(self.device)

        self.step = 0
        self.first_obs = False
        self.gae = gae
        self.gae_lambda = gae_lambda

    def on_done(self, state_tensor_list, np_masks):

        # If done then clean the current observation
        for key in state_tensor_list:
            pt_masks = torch.from_numpy(np_masks.reshape(np_masks.shape[0], *[1 for _ in range(
                len(state_tensor_list[key].shape[1:]))])).to(self.device)
            state_tensor_list[key] *= pt_masks

    def prepare_model_input(self, step):

        model_in = OrderedDict()

        for obs_key in self.observations:
            model_in[obs_key] = self.observations[obs_key][step]

        return model_in

    def store_step_states(self):
        # save latest state
        for obs_key in self.observations:
            self.observations[obs_key][0].copy_(self.observations[obs_key][-1])

    def select_action(self, state, train=True):
        super().select_action(state, train)
        self.model.set_train_mode()

        # 初始的 select_action 呼叫，state 是一個觀測值，不是元組
        # 這個 else 區塊處理第一次呼叫，將觀測值存到正確的初始位置
        if not isinstance(state, tuple) or len(state) != 4:
            observation = state
            for key in observation:
                self.observations[key][0].copy_(torch.from_numpy(observation[key]))
        else:
            # 處理後續的呼叫，state 是一個 (obs, reward, done, info) 元組
            observation, reward, done, _ = state

            # 1. [已修正] 將純量 reward 轉換為 [1, 1] 形狀的張量
            reward_tensor = torch.tensor([[reward]], dtype=torch.float32).to(self.device)
            self.episode_rewards += reward_tensor
            self.rewards.append(reward_tensor)

            # 2. [已修正] 將純量 done 轉換為對應的 NumPy mask 陣列
            np_masks = np.array([0.0 if done else 1.0], dtype=np.float32)

            state_tensor_list = OrderedDict()
            for obs_key in observation:
                # 3. [已修正] 這裡的 view 操作現在是安全的，因為我們在 experiment.py 中
                #    已經將 n_worker 強制設定為 1。
                state_tensor_list[obs_key] = torch.from_numpy(observation[obs_key]).view(self.observations[obs_key].shape[1:]).to(self.device)

            self.on_done(state_tensor_list, np_masks)
            for obs_key in state_tensor_list:
                self.observations[obs_key][self.step].copy_(state_tensor_list[obs_key])

            self.masks[self.step].copy_(torch.from_numpy(np_masks).unsqueeze(1))
            self.final_rewards *= self.masks[self.step]
            self.final_rewards += (1 - self.masks[self.step]) * self.episode_rewards
            self.episode_rewards *= self.masks[self.step]

        # 4. 進行神經網路的正向傳播，獲取動作
        with torch.no_grad():
            model_returns = self.model(self.prepare_model_input(self.step))

        policy = model_returns['policy']
        value = model_returns['value']
        action_tensor, np_actions = self.model.sample_action(policy)
        log_probs = self.model.get_log_probs(policy, action_tensor).data

        if self.n_actions == 1 and action_tensor.dim() < 2:
            action_tensor = action_tensor.unsqueeze(-1)

        self.actions[self.step].copy_(action_tensor)
        self.value_predictions[self.step].copy_(value.data)
        self.old_log_probs[self.step].copy_(log_probs)

        # 5. 更新計數器
        if train:
            self.step += 1

        # 6. [已修正] 在收集了 t_max 步的經驗後，觸發更新
        if train and self.step == self.t_max:
            self.perform_update()

        # 7. 返回動作。agent_decided_done 對於 A2C 來說永遠是 False，
        #    因為環境的結束由 train 迴圈自己處理。
        return np_actions, False

    def prepare_single_forward_pass(self):

        model_in = OrderedDict()

        for obs_key in self.observations:
            obs = self.observations[obs_key]
            model_in[obs_key] = obs[:-1].view(-1, *obs.size()[2:])

        return model_in

    def perform_update(self):
        super().perform_update()

        model_returns = self.model(self.prepare_single_forward_pass())
        policy = model_returns['policy']
        values = model_returns['value']

        if self.gae:
            with torch.no_grad():
                self.value_predictions[-1] = self.model.forward_value(self.prepare_model_input(-1))
            gae = 0
            for step in reversed(range(self.t_max)):
                delta = self.rewards[step] + self.gamma * self.value_predictions[step + 1] * self.masks[step] - self.value_predictions[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step] * gae
                self.returns[step] = gae + self.value_predictions[step]
        else:
            with torch.no_grad():
                self.returns[-1] = self.model.forward_value(self.prepare_model_input(-1))
            for step in reversed(range(self.t_max)):
                self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step] + self.rewards[step]

        advantages = self.returns[:-1].view(-1).unsqueeze(1) - values
        log_probs = self.model.get_log_probs(policy, self.actions.view(self.n_worker * self.t_max, -1)).view(-1, 1)
        dist_entropy = self.model.calc_entropy(policy)
        value_loss = advantages.pow(2).mean(dim=0)
        policy_loss = -(advantages.data * log_probs).mean(dim=0)

        losses = dict(policy_loss=policy_loss, value_loss=value_loss, dist_entropy=dist_entropy)
        self.model.update(losses)

        self.log_dict = {
            'policy_loss': policy_loss.detach(),
            'value_loss': value_loss.detach(),
            'entropy': dist_entropy,
            'avg_reward': self.final_rewards.mean(),
            'median_reward': self.final_rewards.median()
        }

        # 8. [已修正] 在更新結束後，重設計數器並清理經驗，為下一批做準備
        self.step = 0
        self.rewards = []
        self.store_step_states()

