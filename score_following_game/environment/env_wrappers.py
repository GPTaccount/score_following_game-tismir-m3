import cv2
# - : gym 舊版匯入
# import gym
import gymnasium as gym
# + : 改為 gymnasium 匯入，符合新版 API 標準

import numpy as np


class ConvertToFloatWrapper(gym.ObservationWrapper):

    def __init__(self, env, key):
        """
        Convert image observation (determined by key) to float and scale it to range (0, 1)
        """
        # - : gym.ObservationWrapper 舊版初始化方式
        # gym.ObservationWrapper.__init__(self, env)
        super().__init__(env)
        # + : 改為使用 super() 初始化父類別，符合 Gymnasium API
        self.key = key

    def observation(self, observation):

        observation[self.key] = observation[self.key].astype(np.float32)/255.

        return observation


class InvertWrapper(gym.ObservationWrapper):

    def __init__(self, env, key):
        """
        Invert color of an image that is already in range (0, 1)
        """
        # - : gym.ObservationWrapper 舊版初始化方式
        # gym.ObservationWrapper.__init__(self, env)
        super().__init__(env)
        # + : 改為使用 super() 初始化父類別，符合 Gymnasium API
        self.key = key

    def observation(self, observation):

        # unfold observation vector
        observation[self.key] = 1.0 - observation[self.key]

        return observation


class DifferenceWrapper(gym.ObservationWrapper):
    def __init__(self, env, key, keep_raw=True):
        """
        Returns original and delta observation if keep_raw=True else only the delta observation
        """
        # - : gym.ObservationWrapper 舊版初始化方式
        # gym.ObservationWrapper.__init__(self, env)
        super().__init__(env)
        # + : 改為使用 super() 初始化父類別，符合 Gymnasium API
        self.prev = None
        self.keep_raw = keep_raw
        self.key = key

        space = self.observation_space.spaces[self.key]
        shape = list(space.shape)

        # adapt the observation space
        if self.keep_raw:
            shape[0] = shape[0]*2

        self.observation_space.spaces[self.key] = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.float32)
        # + : gym.spaces.Box 已透過 `import gymnasium as gym` 更新 (註: np.float32 為正確 NumPy 資料型態用法，low/high 為 Gymnasium 建議參數名)

    def observation(self, observation):

        act = observation[self.key]

        if self.prev is None:
            self.prev = act

        act_diff = act - self.prev
        self.prev = act

        if self.keep_raw:
            act_diff = np.vstack((act, act_diff))

        observation[self.key] = act_diff

        return observation


class ResizeSizeWrapper(gym.ObservationWrapper):

    def __init__(self, env, key, factor, dim):
        """
        Scale size of observation image (sheet, spec) by certain factor
        """
        # - : gym.ObservationWrapper 舊版初始化方式
        # gym.ObservationWrapper.__init__(self, env)
        super().__init__(env)
        # + : 改為使用 super() 初始化父類別，符合 Gymnasium API

        self.key = key
        self.factor = factor
        self.dim = dim

        space = self.observation_space.spaces[self.key]
        shape = list(space.shape)

        for d in dim:
            shape[d] *= self.factor

        # - : 使用已棄用的 np.int 型別
        # shape = np.asarray(shape, dtype=np.int)
        shape = np.asarray(shape, dtype=int)
        # + : 將 np.int 替換為 int，以符合 NumPy 更新並避免廢棄警告

        self.observation_space.spaces[self.key] = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.float32)
        # + : gym.spaces.Box 已透過 `import gymnasium as gym` 更新 (註: np.float32 為正確 NumPy 資料型態用法，low/high 為 Gymnasium 建議參數名)

    def observation(self, observation):

        # get observation vector
        ob = observation[self.key]

        # resize observation if factor is not 1
        if self.factor != 1.0:
            s = list(ob.shape)
            for d in self.dim:
                s[d] = int(s[d]*self.factor)

            ob = cv2.resize(ob.transpose(1, 2, 0), (s[2], s[1]))

            ob = np.expand_dims(ob, 0)
            observation[self.key] = ob

        return observation


class TanhActionWrapper(gym.ActionWrapper):

    def __init__(self, env):
        # - : gym.ActionWrapper 舊版初始化方式
        # gym.ActionWrapper.__init__(self, env)
        super().__init__(env)
        # + : 改為使用 super() 初始化父類別，符合 Gymnasium API

    def action(self, action):
        return action*128

    def reverse_action(self, action):

        return action/128