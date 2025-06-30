import cv2
import logging
import time

import numpy as np

# - : Gymnasium API 變更：原始 gym 匯入方式已淘汰
# from gym import Env, spaces
# - : seeding 功能不再透過 gym.utils.seeding 處理
# from gym.utils import seeding
import gymnasium as gym
# + : 改為 gymnasium 匯入，符合新版 API 標準
from gymnasium import spaces
# + : spaces 模組從 gymnasium 匯入
from score_following_game.environment.audio_thread import AudioThread
from score_following_game.environment.render_utils import write_text, prepare_sheet_for_render, prepare_spec_for_render
from score_following_game.environment.reward import Reward

logger = logging.getLogger(__name__)

AGENT_COLOR = (0, 102, 204)
TARGET_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
BORDER_COLOR = (0, 0, 255)


# - : 原始 Env 繼承自 gym.Env
# class ScoreFollowingEnv(Env):
class ScoreFollowingEnv(gym.Env):
# + : Env 類別繼承自 gymnasium.Env
    # metadata = {
    #     'render.modes': {'human': 'human',
    #                      'computer': 'computer',
    #                      'video': 'video'},
    # }
    # - : 原始 render.modes metadata key 已不適用
    metadata = {
        'render_modes': ['human', 'computer', 'video'],
        # + : 依 Gymnasium API 標準更新 metadata key 為 render_modes 並使用列表形式
    }

    def __init__(self, rl_pool, config, render_mode=None):

        self.rl_pool = rl_pool
        self.actions = config["actions"]
        self.render_mode = render_mode

        # distance of tracker to true score position to fail the episode
        self.score_dist_threshold = self.rl_pool.score_shape[2] // 3

        self.interpolationFunction = None
        self.spectrogram_positions = []
        self.interpolated_coords = []
        self.spec_representation = config['spec_representation']

        self.text_position = 0

        # path to the audio file (for playing the audio in the background)
        self.path_to_audio = ""

        # flag that determines if the environment is executed for the first time or n
        self.first_execution = True

        self.performance = None
        self.score = None

        # - : _seed() 方法已移除，種子設定改由 reset(seed=...) 處理
        # self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.step_id = 0
        self.frame_id = 0
        self.last_reward = None
        self.cum_reward = None
        self.time_stamp = time.time()
        self.step_times = np.zeros(25)

        self.last_action = None

        # setup observation space
        self.observation_space = spaces.Dict({'perf': spaces.Box(0, 255, self.rl_pool.perf_shape, dtype=np.float32),
                                              'score': spaces.Box(0, 255, self.rl_pool.score_shape, dtype=np.float32)})

        if len(config['actions']) == 0:
            self.action_space = spaces.Box(low=-128, high=128, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(self.actions))
        self.reward_range = (-1, 1) # type: ignore
        # + : 依 Gymnasium API 標準，reward_range 應為 tuple[float, float]，此處型別註解暫不修改以符合原樣
        self.obs_image = None
        self.prev_reward = 0.0
        self.debug_info = {'song_history': self.rl_pool.get_song_history()}

        self.reward = Reward(config['reward_name'], threshold=self.score_dist_threshold, pool=self.rl_pool,
                             params=config['reward_params'])

        # resize factors for rendering
        self.resz_spec = 4
        self.resz_imag = float(self.resz_spec) / 2 * float(self.rl_pool.perf_shape[1]) / self.rl_pool.score_shape[1]
        self.resz_x, self.resz_y = self.resz_imag, self.resz_imag
        self.text_position = 0

    def step(self, action):

        if len(self.actions) > 0:
            # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
            # decode action if specific action space is given
            action = self.actions[action]
        else:
            action = action[0]

        self.rl_pool.update_position(action)

        self.last_action = action

        # get current frame from "pace-maker"
        if self.render_mode == 'computer' or self.render_mode == 'human':
            self.curr_frame = self.audioThread.get_current_spec_position()

            while self.prev_frame == self.curr_frame:
                self.render(mode=self.render_mode)
                self.curr_frame = self.audioThread.get_current_spec_position()

            self.prev_frame = self.curr_frame

        elif self.render_mode == 'video':
            self.render(mode=self.render_mode)
            self.curr_frame += 1

        else:
            self.curr_frame += 1

        self.performance, self.score = self.rl_pool.step(self.curr_frame)

        self.state = dict(
            perf=self.performance,
            score=self.score
        )

        # check if score follower lost its target
        abs_err = np.abs(self.rl_pool.tracking_error())
        target_lost = abs_err > self.score_dist_threshold

        # check if score follower reached end of song
        end_of_song = self.rl_pool.last_onset_reached()

        reward = self.reward.get_reward(abs_err)

        # end of score following
        # - : 原始 done 變數邏輯
        # done = False
        # if target_lost or end_of_song:
        #     done = True
        #     if self.render_mode == 'computer' or self.render_mode == 'human':
        #         self.audioThread.end_stream()
        terminated = False
        # + : 依 Gymnasium API 標準，done 變數拆分為 terminated 與 truncated
        if target_lost or end_of_song:
            terminated = True
            # + : 當目標丟失或歌曲結束時，設定 terminated 為 True
            if self.render_mode == 'computer' or self.render_mode == 'human':
                self.audioThread.end_stream()
        
        truncated = False
        # + : 預設 truncated 為 False，此環境無明確截斷條件（如時間限制）

        # no reward if target is lost
        if target_lost:
            reward = np.float32(0.0)

        # check if env is still used even if done
        # - : 原始 done 變數的檢查邏輯
        # if not done:
        if not (terminated or truncated):
        # + : 更新為檢查 terminated 或 truncated
            pass
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                # - : 原始 done 變數的警告訊息
                # logger.warning(
                #     "You are calling 'step()' even though this environment has already returned done = True."
                #     " You should always call 'reset()' once you receive 'done = True'"
                #     " -- any further steps are undefined behavior.")
                logger.warning(
                    "You are calling 'step()' even though this environment has already returned terminated or truncated = True."
                    " You should always call 'reset()' once you receive 'terminated or truncated = True'"
                    " -- any further steps are undefined behavior.")
                # + : 更新警告訊息以反映 Gymnasium API 的 terminated/truncated 狀態
            self.steps_beyond_done += 1

        # compute time required for step
        self.step_times[1:] = self.step_times[0:-1]
        self.step_times[0] = time.time() - self.time_stamp
        self.time_stamp = time.time()

        self.last_reward = reward
        self.step_id += 1
        self.cum_reward += reward

        info = {}
        # + : 依 Gymnasium API 標準，step 方法需回傳 info 字典

        # - : 原始 step() 回傳四元組 (obs, reward, done, info)
        # return self.state, reward, done, {}
        return self.state, reward, terminated, truncated, info
        # + : Gymnasium API 的 step() 回傳五元組 (obs, reward, terminated, truncated, info)

    # - : 原始 reset() 方法簽名
    # def reset(self):
    def reset(self, seed=None, options=None):
    # + : 依 Gymnasium API 標準，reset 方法加入 seed 與 options 參數
        super().reset(seed=seed)
        # + : 呼叫父類別的 reset 方法以正確處理 seed

        self.steps_beyond_done = None
        self.step_id = 0
        self.cum_reward = 0
        self.first_execution = True
        self.curr_frame = 0
        self.prev_frame = -1

        self.last_action = None

        # reset data pool
        self.rl_pool.reset()

        # reset audio thread
        if self.render_mode == 'computer' or self.render_mode == 'human':
            # write midi to wav
            import soundfile as sf
            fn_audio = self.rl_pool.get_current_song_name()[0] + '.wav'
            perf_audio, fs = self.rl_pool.get_current_perf_audio_file()
            sf.write(fn_audio, perf_audio, fs)

            self.path_to_audio = fn_audio
            self.audioThread = AudioThread(self.path_to_audio, self.rl_pool.spectrogram_params['fps'])
            self.audioThread.start()
            self.curr_frame = self.audioThread.get_current_spec_position()

        # get first observation
        self.performance, self.score = self.rl_pool.step(self.curr_frame)

        self.state = dict(
            perf=self.performance,
            score=self.score
        )
        
        info = self.debug_info.copy() if hasattr(self, 'debug_info') else {}
        # + : 依 Gymnasium API 標準，reset 方法回傳 observation 與 info 字典，此處將 debug_info 加入 info

        # - : 原始 reset() 回傳 observation
        # return self.state
        return self.state, info
        # + : Gymnasium API 的 reset() 回傳 observation and info 字典

    def render(self, mode='computer', close=False):

        if close:
            return

        perf = self.prepare_perf_for_render()

        score = self.prepare_score_for_render()

        # highlight image center
        score_center = score.shape[1] // 2
        cv2.line(score, (score_center, 25), (score_center, score.shape[0] - 25), AGENT_COLOR, 2)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("Agent", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=1)[0]
        text_org = (score_center - text_size[0] // 2, score.shape[0] - 7)
        cv2.putText(score, "Agent", text_org, fontFace=font_face, fontScale=0.6, color=AGENT_COLOR, thickness=2)

        # hide tracking lines if it is rendered for humans
        # - : 原始 metadata key 為 render.modes
        # if self.metadata['render.modes']['human'] != mode:
        if self.metadata['render_modes'][0] != mode or mode != 'human': # Assuming 'human' is the first in list and checking explicitly
        # + : 更新 metadata key 為 render_modes，並調整條件以符合列表形式。此處假設 'human' 是 render_modes[0]，並確保與 mode 比較正確。
        # + : 修正：原始邏輯是 self.metadata['render.modes']['human'] (value) != mode. 若 'human' mode 在 metadata list 中，則可簡化。
        # + : 考量到原始 'human': 'human' 的結構，這裡的比較是與 'human' 字串比較。
        # + : 若 mode 不是 'human'，則執行以下繪圖。
            if mode != 'human': # More direct way based on original logic's intent
            # + : 簡化條件，直接判斷 mode 是否為 'human'。
                # visualize tracker position (true position within the score)
                true_position = int(score_center - (self.resz_x * self.rl_pool.tracking_error()))

                font_face = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize("Target", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=1)[0]
                text_org = (true_position - text_size[0] // 2, text_size[1] + 1)
                cv2.putText(score, "Target", text_org, fontFace=font_face, fontScale=0.6, color=TARGET_COLOR, thickness=2)

                cv2.line(score, (true_position, 25), (true_position, score.shape[0] - 25), TARGET_COLOR, 3)

                # visualize boundaries
                l_boundary = int(score_center - self.score_dist_threshold * self.resz_x)
                r_boundary = int(score_center + self.score_dist_threshold * self.resz_x)
                cv2.line(score, (l_boundary, 0), (l_boundary, score.shape[0] - 1), BORDER_COLOR, 1)
                cv2.line(score, (r_boundary, 0), (r_boundary, score.shape[0] - 1), BORDER_COLOR, 1)

        # prepare observation visualization
        cols = score.shape[1]
        rows = score.shape[0] + perf.shape[0]
        obs_image = np.zeros((rows, cols, 3), dtype=np.uint8)

        # write sheet to observation image
        obs_image[0:score.shape[0], 0:score.shape[1], :] = score

        # write spec to observation image
        c0 = obs_image.shape[1] // 2 - perf.shape[1] // 2
        c1 = c0 + perf.shape[1]
        obs_image[score.shape[0]:, c0:c1, :] = perf

        # draw black line to separate score and performance
        obs_image[score.shape[0], c0:c1, :] = 0

        # write text to the observation image
        self._write_text(obs_image=obs_image, pos=self.text_position, color=TEXT_COLOR)

        # preserve this for access from outside
        self.obs_image = obs_image

        # show image
        if self.render_mode == 'computer' or self.render_mode == 'human':
            cv2.imshow("Score Following", self.obs_image)
            cv2.waitKey(1)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        # + : 依 Gymnasium 標準，新增關閉 viewer 的邏輯 (若有的話)
        # + : 此環境未使用標準 Gym viewer，但保留 close 方法的完整性
        # + : 若 AudioThread 需要清理，也應在此處處理
        if hasattr(self, 'audioThread') and self.audioThread and self.audioThread.is_alive():
             if self.render_mode == 'computer' or self.render_mode == 'human': # Ensure stream is ended only if it was started
                self.audioThread.end_stream()
                self.audioThread.join() # Wait for thread to finish
        # + : 確保 AudioThread 在環境關閉時被正確停止與清理


    # - : _seed() 方法已不適用於 Gymnasium，種子設定改由 reset(seed=...) 處理
    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def _write_text(self, obs_image, pos, color):

        # write reward to observation image
        write_text('reward: {:6.2f}'.format(self.last_reward if self.last_reward is not None else 0),
                   pos, obs_image, color=color)

        # write cumulative reward (score) to observation image
        write_text("score: {:6.2f}".format(self.cum_reward if self.cum_reward is not None else 0),
                   pos + 2, obs_image, color=color)

        # write last action
        write_text("action: {:+6.1f}".format(self.last_action if self.last_action is not None else 0.0), pos + 4, obs_image, color=color)
        # + : 確保 self.last_action 在 None 時有預設值 0.0，避免 format 錯誤

    def prepare_score_for_render(self):
        return prepare_sheet_for_render(self.score, resz_x=self.resz_imag, resz_y=self.resz_imag)

    def prepare_perf_for_render(self):
        return prepare_spec_for_render(self.performance, resz_spec=self.resz_spec)
