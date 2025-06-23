import os
import sys
import time
import torch

import numpy as np

from collections import OrderedDict
from score_following_game.reinforcement_learning.torch_extentions.distributions.adapted_categorical import AdaptedCategorical


class Agent(object):

    def __init__(self, observation_space, model, n_actions=1, gamma=0.99, distribution=AdaptedCategorical,
                 use_cuda=torch.cuda.is_available(), log_writer=None, log_interval=10, evaluator=None, eval_interval=5000,
                 lr_scheduler=None, score_name=None, high_is_better=False, dump_interval=100000, dump_dir=None, buffer=None):

        self.observation_space = observation_space
        self.model = model
        self.n_actions = n_actions

        self.log_writer = log_writer
        self.log_interval = log_interval

        self.evaluator = evaluator
        self.eval_interval = eval_interval

        self.lr_scheduler = lr_scheduler
        self.score_name = score_name
        self.high_is_better = high_is_better
        self.best_score = -np.inf if self.high_is_better else np.inf

        self.dump_interval = dump_interval
        self.dump_dir = dump_dir

        self.gamma = gamma
        self.use_cuda = use_cuda

        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.log_dict = dict()

        self.trained_agent = TrainedAgent

        self.n_worker = 1
        self.update_cnt = 0
        self.step_cnt = 0
        self.now = self.after = None
        # Note: np.float32 is a specific type, not a deprecated alias like np.float. No change needed here for NumPy alias replacement.
        self.step_times = np.ones(11, dtype=np.float32) 

        self.distribution = distribution
        self.action_tensor = torch.LongTensor if self.distribution == AdaptedCategorical else torch.FloatTensor

        self.buffer = buffer

    def select_action(self, state, train=True):
        self.step_cnt += 1*self.n_worker
        if self.now is None:
            self.now = self.after = time.time()

        # return dummy values
        return None, None

    def perform_update(self):

        # logging
        if self.update_cnt % self.log_interval == 0 and self.update_cnt > 0:
            self.log()

        # evaluation
        if self.evaluator is not None and self.update_cnt % self.eval_interval == 0 and self.update_cnt > 0:
            self.evaluate()

        # dump model regularly
        if self.update_cnt % self.dump_interval == 0 and self.update_cnt > 0:
            print('Saved model at update {}'.format(self.update_cnt))
            self.store_model('model_update_{}'.format(self.update_cnt), self.dump_dir)

        self.update_cnt += 1

        # estimate updates per second (running avg)
        self.step_times[0:-1] = self.step_times[1::]
        self.step_times[-1] = time.time() - self.after
        ups = 1.0 / self.step_times.mean()
        self.after = time.time()
        print("update %d @ %.1fups" % (np.mod(self.update_cnt, self.log_interval), ups), end="\r")
        sys.stdout.flush()

    def store_model(self, name, store_dir=None):

        if store_dir is not None:
            model_path = os.path.join(store_dir, name)
        else:
            model_path = name

        self.model.save_network(model_path)

    def log(self):

        self.log_dict['steps'] = self.step_cnt
        self.log_dict['learn_rate'] = self.model.get_learn_rate()

        print('-' * 32)
        print('| {:<15} {: 12d} |'.format('update', self.update_cnt))
        print('| {:<15} {: 12.1f} |'.format('duration(s)', time.time() - self.now))
        for log_key in self.log_dict:

            log_var = self.log_dict[log_key].cpu().item() if type(self.log_dict[log_key]) == torch.Tensor \
                else self.log_dict[log_key]

            if type(self.log_dict[log_key]) == int:
                print('| {:<15} {: 12d} |'.format(log_key, log_var))
            else:
                if abs(log_var) < 1e-5:
                    print('| {:<15} {: 12.0e} |'.format(log_key, log_var))
                else:
                    print('| {:<15} {: 12.5f} |'.format(log_key, log_var))

            if self.log_writer is not None:
                self.log_writer.add_scalar('training/{}'.format(log_key), log_var,
                                           int(self.update_cnt / self.log_interval))

        if self.log_writer is not None and self.log_writer.log_gradients:

            for tag, value in self.model.net.named_parameters():
                tag = tag.replace('.', '/')
                if value.grad is not None:
                    self.log_writer.add_histogram(tag + '/grad', value.grad.data.cpu().norm(2).item(),  int(self.update_cnt / self.log_interval))

        print('-' * 32)
        self.now = time.time()

    def evaluate(self):
        self.model.set_eval_mode()
        stats = self.evaluator.evaluate(self.trained_agent(self.model, self.use_cuda, distribution=self.distribution),
                                        self.log_writer, int(self.update_cnt / self.eval_interval))

        self.model.set_train_mode()

        if self.score_name is not None:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(stats[self.score_name])
                if self.model.get_learn_rate() == 0:
                    print('Training stopped')

            improvement = (self.high_is_better and stats[self.score_name] >= self.best_score) or \
                          (not self.high_is_better and stats[self.score_name] <= self.best_score)

            if improvement:
                print('New best model at update {}'.format(self.update_cnt))
                self.store_model('best_model', self.dump_dir)
                self.best_score = stats[self.score_name]

    def train(self, env, max_steps):
        # - : gym.Env.reset() 僅回傳觀測值 (observation)
        # state = env.reset()
        obs, info = env.reset()
        # + : gymnasium.Env.reset() 現在回傳 (obs, info) 二元組
        state = obs
        # + : 將觀測值 (obs) 指派給 state 變數，作為代理程式的狀態輸入

        step_cnt = 0

        while step_cnt < max_steps:
            # 假設 self.select_action(state) 可能回傳 (action, agent_decided_done)
            # 根據「不更動原始邏輯」的原則，保留此部分
            action, agent_decided_done = self.select_action(state)

            if agent_decided_done:
                # - : gym.Env.reset() 僅回傳觀測值 (observation)
                # state = env.reset()
                obs, info_reset = env.reset() # Gymnasium returns obs, info
                # + : gymnasium.Env.reset() 現在回傳 (obs, info) 二元組
                state = obs # 代理程式的狀態是觀測值
                # + : 將新的觀測值 (obs) 指派給 state 變數
            else:
                # - : gym.Env.step() 原回傳四元組 (obs, reward, done, info)，且假設 state 被指派為其中的 obs
                # prev_obs, prev_reward, prev_done, prev_info = env.step(action) # 概念上的舊呼叫
                # state = prev_obs 
                current_obs, reward, terminated, truncated, info_step = env.step(action) # Gymnasium step API
                # + : gymnasium.Env.step() 現在回傳五元組 (obs, reward, terminated, truncated, info)
                state = current_obs # 代理程式的狀態是新的觀測值
                # + : 將新的觀測值 (current_obs) 指派給 state 變數
                # 注意：環境的 done 旗標 (terminated 或 truncated) 在此處未直接用於重置環境，
                # 因為 agent_decided_done 旗標負責處理重置。這是原始邏輯的一部分。

            step_cnt += 1

            # stop training if learn rate scheduler stopped
            if self.lr_scheduler is not None and self.lr_scheduler.learning_stopped():
                    break


class TrainedAgent(Agent):

    def __init__(self, model, use_cuda=torch.cuda.is_available(), deterministic=False, distribution=AdaptedCategorical):

        Agent.__init__(self, observation_space=None, model=None, use_cuda=use_cuda, distribution=distribution)
        self.model = model
        self.use_cuda = use_cuda
        self.deterministic = deterministic
        self.policy = None

        if self.use_cuda:
            self.model.cuda()

    def prepare_state(self, observation):

        state = OrderedDict()

        for obs_key in observation.keys():
            state[obs_key] = torch.from_numpy(observation[obs_key]).unsqueeze(0).to(self.device)

        return state

    def select_action(self, state, train=False):

        with torch.no_grad():
            model_returns = self.model(self.prepare_state(state))

        self.policy = model_returns['policy']

        return self.model.sample_action(self.policy, deterministic=self.deterministic)[1][0]

    def get_policy(self):
        return self.distribution(**self.policy)

    def predict_value(self, state):

        value = self.model(self.prepare_state(state))['value']

        return value.data.cpu().numpy()[0, 0]


def get_agent(agent, **params):
    from score_following_game.reinforcement_learning.algorithms.a2c import A2CAgent
    from score_following_game.reinforcement_learning.algorithms.ppo import PPOAgent
    from score_following_game.reinforcement_learning.algorithms.reinforce import ReinforceAgent

    if agent == 'reinforce':
        agent_type = ReinforceAgent
    elif agent == 'a2c':
        agent_type = A2CAgent
    elif agent == 'ppo':
        agent_type = PPOAgent
    else:
        raise NotImplementedError('Invalid Algorithm')

    return agent_type(**params)

# 📝 NumPy 廢棄別名替換說明：
# 經過檢查，此檔案 (agent.py) 的提供片段中，並未發現 np.int, np.float, np.bool, np.object 等 NumPy 廢棄別名。
# 例如，self.step_times 的 dtype 已明確指定為 np.float32，此為具體型別，非廢棄別名。
# 因此，針對此檔案的 NumPy 廢棄別名替換任務，在此片段中無適用變更。

# 📝 Librosa ➜ soundfile 重構說明：
# 此檔案 (agent.py) 未使用 librosa 相關功能，故無需進行 Librosa 至 soundfile 的重構。