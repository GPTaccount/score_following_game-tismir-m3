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
        'render_modes': ['human', 'computer', 'video', 'rgb_array'], # + : 增加 'rgb_array' 到 metadata，標準模式
        # + : 依 Gymnasium API 標準更新 metadata key 為 render_modes 並使用列表形式
    }

    def __init__(self, rl_pool, config, render_mode=None):

        self.rl_pool = rl_pool
        self.actions = config["actions"]
        self.render_mode = render_mode # + : 初始化時設定 render_mode

        # + : 檢查傳入的 render_mode 是否在 metadata 中，這是 Gymnasium 推薦的做法
        if self.render_mode is not None and self.render_mode not in self.metadata["render_modes"]:
            # + : 如果傳入的模式無效，發出警告或錯誤
            logger.warning(f"Invalid render_mode '{self.render_mode}' passed to environment. Available modes: {self.metadata['render_modes']}")
            # Optionally set to None or a default valid mode
            # self.render_mode = None # Or raise ValueError

        # distance of tracker to true score position to fail the episode
        self.score_dist_threshold = self.rl_pool.score_shape[2] // 3

        self.interpolationFunction = None
        self.spectrogram_positions = []
        self.interpolated_coords = []
        self.spec_representation = config['spec_representation']

        self.text_position = 0

        # path to the audio file (for playing the audio in the background)
        self.path_to_audio = ""

        self.first_execution = True

        self.performance = None
        self.score = None

        # - : _seed() 方法已移除，種子設定改由 reset(seed=...) 處理
        # from gym.utils import seeding # Re-import if needed, but usually not for reset/seed
        # self._seed()
        self.viewer = None # Note: This environment seems to use cv2, not a standard gym viewer.
        self.state = None

        self.steps_beyond_done = None

        self.step_id = 0
        self.frame_id = 0 # This seems related to step in render logic?
        self.last_reward = None
        self.cum_reward = None
        self.time_stamp = time.time()
        self.step_times = np.zeros(25)

        self.last_action = None

        # setup observation space
        # + : 確保 Box space 的 low/high 與實際數據範圍和類型匹配。0-255 用於圖像數據通常是 uint8，但這裡指定 float32，這需要網絡兼容。保持原樣，但要注意潛在的類型問題。
        self.observation_space = spaces.Dict({'perf': spaces.Box(0, 255, self.rl_pool.perf_shape, dtype=np.float32),
                                              'score': spaces.Box(0, 255, self.rl_pool.score_shape, dtype=np.float32)})


        if len(config['actions']) == 0:
            self.action_space = spaces.Box(low=-128, high=128, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(self.actions))
        self.reward_range = (-float('inf'), float('inf')) # + : 將 reward_range 設為標準的 (-inf, inf)
        # - : 原始 reward_range 註解不符標準類型
        # self.reward_range = (-1, 1) # type: ignore # Original comment was misleading, standard range is typically -inf to +inf unless specified.
        # + : 依 Gymnasium API 標準，reward_range 應為 tuple[float, float]

        self.obs_image = None # + : attribute to store the last rendered image
        self.prev_frame = -1 # Used in step for 'computer'/'human' modes to wait for audio frame
        self.curr_frame = 0 # Used in step for 'video'/'None' modes to increment frame counter


        self.prev_reward = 0.0 # Unused? Or maybe implicitly used in Reward class?
        self.debug_info = {'song_history': self.rl_pool.get_song_history()}

        self.reward = Reward(config['reward_name'], threshold=self.score_dist_threshold, pool=self.rl_pool,
                             params=config['reward_params'])

        # resize factors for rendering
        self.resz_spec = 4
        # + : 確保計算 resz_imag 時使用浮點數除法，避免整數除法問題
        self.resz_imag = float(self.resz_spec) / 2 * float(self.rl_pool.perf_shape[1]) / float(self.rl_pool.score_shape[1])
        # - : 原始計算可能因整數除法丟失精度
        # self.resz_imag = float(self.resz_spec) / 2 * float(self.rl_pool.perf_shape[1]) / self.rl_pool.score_shape[1]

        self.resz_x, self.resz_y = self.resz_imag, self.resz_imag
        # self.text_position = 0 # Redefined above, possibly redundant? Keeping for safety.


        # + : Initialize audioThread attribute to None
        self.audioThread = None


    def step(self, action):

        # + : Debug log for step call
        # print(f"DEBUG: ScoreFollowingEnv.step() called with action: {action}, current frame (before update): {self.curr_frame}")


        if len(self.actions) > 0:
            # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
            # decode action if specific action space is given
            # + : ensure action is within discrete space bounds if applicable
            if isinstance(self.action_space, spaces.Discrete):
                 if not self.action_space.contains(action):
                      logger.warning(f"Action {action} is outside the discrete action space bounds.")
                      # Decide how to handle invalid action - clip, ignore, etc.
                      # For now, proceed with the potentially out-of-bounds action for decoding.
            # + : Decode the discrete action index to its continuous value
            action_decoded = self.actions[action]
        else:
            # + : For continuous action space, action is already the value
            action_decoded = action[0] # Assuming action is a 1-element array/list for continuous space

        self.rl_pool.update_position(action_decoded) # + : Use decoded action

        self.last_action = action_decoded # + : Store decoded action


        # get current frame from "pace-maker"
        # + : Logic for frame advancement based on render_mode
        if self.render_mode == 'computer' or self.render_mode == 'human':
            # + : In these modes, audioThread drives the frame advancement
            if self.audioThread is None or not self.audioThread.is_alive():
                 # + : Handle case where audio thread wasn't started or died unexpectedly
                 logger.error("AudioThread is not running in render_mode 'computer' or 'human'. Cannot advance frame.")
                 # + : Force termination or fallback - check if pool is done instead?
                 # + : If audio thread is essential and not running, maybe the episode should terminate?
                 # + : For now, let's check if the pool itself is done as a fallback termination.
                 if self.rl_pool.last_onset_reached(): # Check if pool has reached the end anyway
                      terminated = True
                      truncated = False
                 else: # If pool is not done, but audio thread failed, log and maybe terminate?
                     logger.error("AudioThread failed but pool is not done. Forcing termination.")
                     terminated = True
                     truncated = False
                 info = {} # Add error info if needed
                 # Skip further processing in this step and return episode end
                 return self.state, np.float32(0.0), terminated, truncated, info


            # + : Get the latest frame from audio thread
            new_curr_frame = self.audioThread.get_current_spec_position()

            # + : Wait for a new frame only if rendering mode is enabled and frame hasn't advanced
            # + : This loop blocks until a new frame is available or stream ends
            # + : Use the new_curr_frame for comparison
            while self.prev_frame == new_curr_frame and (self.render_mode == 'computer' or self.render_mode == 'human'):
                 # + : Only render inside the waiting loop if the mode supports it
                 if self.render_mode in ['computer', 'human']:
                      # print(f"DEBUG: Step {self.step_id}: Waiting for audio frame. prev_frame: {self.prev_frame}, new_curr_frame: {new_curr_frame}") # + : DEBUG LOG
                      self.render(mode=self.render_mode) # + : Call render inside wait loop
                      # + : Add small sleep to prevent tight loop from consuming too much CPU
                      time.sleep(0.001) # sleep for 1ms
                 # + : Re-check frame position
                 new_curr_frame = self.audioThread.get_current_spec_position()
                 # + : Check if audio thread has ended while waiting
                 if not self.audioThread.is_alive() and self.prev_frame == new_curr_frame:
                     print("DEBUG: AudioThread ended while waiting for new frame. Exiting wait loop.") # + : DEBUG LOG
                     break # Exit waiting loop if audio thread ends

            self.curr_frame = new_curr_frame # + : Update self.curr_frame after the wait loop
            self.prev_frame = self.curr_frame # + : Update prev_frame after getting new curr_frame
            # print(f"DEBUG: Step {self.step_id}: Finished waiting loop. self.curr_frame: {self.curr_frame}") # + : DEBUG LOG


        # + : If render_mode is 'video' or 'rgb_array' or None, frame advances every step
        # + : This logic was implicitly handled in the original code's else block
        # + : Let's ensure frame advancement happens *once* per step, outside the audio wait loop
        else: # if self.render_mode not in ['computer', 'human']
             self.curr_frame += 1 # + : Increment frame counter in non-audio modes
             # print(f"DEBUG: Step {self.step_id}: Render mode '{self.render_mode}'. Incremented self.curr_frame to {self.curr_frame}") # + : DEBUG LOG


        # + : Check if curr_frame exceeds performance length (indicates song end in non-audio modes)
        # + : This check should use the *new* curr_frame value after potential increment
        # + : Also ensure we don't step beyond the pool's data length
        # - : 錯誤地使用了不存在的方法 get_current_perf_length
        # max_perf_frames = self.rl_pool.get_current_perf_length()
        # + : 使用正確的方法 get_current_song_timesteps 來獲取歌曲總長度
        max_perf_frames = self.rl_pool.get_current_song_timesteps() # + : 使用正確的方法獲取歌曲總時長/總幀數
        # print(f"DEBUG: Step {self.step_id}: Current frame: {self.curr_frame}, Max frames (song timesteps): {max_perf_frames}") # + : DEBUG LOG

        if self.curr_frame >= max_perf_frames:
             end_of_song_by_frame_count = True
             # print(f"DEBUG: Step {self.step_id}: Reached end of performance frames ({self.curr_frame}/{max_perf_frames}) based on frame count.") # + : DEBUG LOG
        else:
             end_of_song_by_frame_count = False

        # + : Ensure we don't request a frame index beyond what the pool can provide
        # + : If curr_frame is >= max_perf_frames, calling rl_pool.step(self.curr_frame) might fail or return empty data.
        # + : A common pattern is to return terminal state's last observation and info.
        # + : However, the original code calls rl_pool.step *before* checking termination via end_of_song.
        # + : Let's keep the original flow for now, assuming rl_pool.step() can handle index >= length
        # + : and the termination check happens after.


        # + : Perform the step in the data pool with the updated curr_frame
        # + : The pool gets the data for this specific frame index
        # print(f"DEBUG: Step {self.step_id}: Calling rl_pool.step() with frame {self.curr_frame}") # + : DEBUG LOG
        self.performance, self.score = self.rl_pool.step(self.curr_frame)
        # print(f"DEBUG: Step {self.step_id}: rl_pool.step() returned perf shape {self.performance.shape}, score shape {self.score.shape}") # + : DEBUG LOG


        self.state = dict(
            perf=self.performance,
            score=self.score
            # + : Note: DifferenceWrapper adds 'perf_diff'. The state returned by env.step will include it if wrapper is used.
        )

        # check if score follower lost its target
        abs_err = np.abs(self.rl_pool.tracking_error())
        target_lost = abs_err > self.score_dist_threshold
        # + : Debug log for tracking error and threshold
        # print(f"DEBUG: Step {self.step_id}: Tracking error: {abs_err:.2f}, Threshold: {self.score_dist_threshold}")


        # check if score follower reached end of song
        # + : Check song end based on pool logic OR frame count in non-audio modes
        end_of_song = self.rl_pool.last_onset_reached() or end_of_song_by_frame_count
        # print(f"DEBUG: Step {self.step_id}: rl_pool.last_onset_reached(): {self.rl_pool.last_onset_reached()}, end_of_song_by_frame_count: {end_of_song_by_frame_count}, end_of_song: {end_of_song}") # + : DEBUG LOG


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
        # + : Terminated if song ends naturally or target is lost
        if target_lost or end_of_song:
            terminated = True
            # + : 當目標丟失或歌曲結束時，設定 terminated 為 True
            if self.render_mode == 'computer' or self.render_mode == 'human':
                if self.audioThread and self.audioThread.is_alive(): # + : Ensure audioThread exists and is running before ending stream
                     print("DEBUG: Ending audio stream due to episode termination.") # + : DEBUG LOG
                     self.audioThread.end_stream()
                     # Note: AudioThread is joined in close()


        truncated = False
        # + : 預設 truncated 為 False，此環境無明確截斷條件（如時間限制）
        # + : Note: If there was a step limit, you would set truncated = self.step_id >= max_steps


        # no reward if target is lost (overrides calculated reward)
        if target_lost:
            reward = np.float32(0.0)
            # print(f"DEBUG: Step {self.step_id}: Target lost, setting reward to 0.") # + : DEBUG LOG


        # check if env is still used even if done
        # - : 原始 done 變數的檢查邏輯
        # if not done:
        if not (terminated or truncated):
        # + : 更新為檢查 terminated 或 truncated
            pass # Episode is ongoing
        elif self.steps_beyond_done is None:
            # First step after terminal state
            self.steps_beyond_done = 0
            # print(f"DEBUG: Step {self.step_id}: Episode reached terminal state (terminated={terminated}, truncated={truncated}).") # + : DEBUG LOG
        else:
            # Subsequent steps after terminal state - undefined behavior
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
            # + : In Gymnasium, stepping after termination/truncation should typically return the last observation and info,
            # + : with reward 0 and terminated/truncated True. The original code doesn't explicitly handle this.
            # + : Returning early with previous state might be safer if the simulation logic breaks.
            # return self.state, np.float32(0.0), terminated, truncated, info # Example of returning gracefully after termination


        # compute time required for step
        self.step_times[1:] = self.step_times[0:-1]
        self.step_times[0] = time.time() - self.time_stamp
        self.time_stamp = time.time()

        self.last_reward = reward
        self.step_id += 1
        self.cum_reward += reward

        info = self.debug_info.copy() if hasattr(self, 'debug_info') else {}
        # + : Add useful info to the info dictionary. Access these AFTER rl_pool.step() is called.
        # + : Let's add these logs to test_agent.py instead of here to avoid crashing reset.
        # info['song_timestep'] = self.rl.pool.current_song_timestep # Current position in song timesteps
        # info['tracking_error'] = abs_err # Current tracking error
        # info['song_speed'] = self.rl.pool.sheet_speed # Current speed

        # + : 依 Gymnasium API 標準，step 方法需回傳 info 字典

        # - : 原始 step() 回傳四元組 (obs, reward, done, info)
        # return self.state, reward, done, {}
        # print(f"DEBUG: Step {self.step_id}: ScoreFollowingEnv.step() returning state keys: {list(self.state.keys()) if isinstance(self.state, dict) else 'Not a dict'}, reward: {reward:.2f}, terminated: {terminated}, truncated: {truncated}, info keys: {list(info.keys())}") # + : DEBUG LOG
        return self.state, reward, terminated, truncated, info
        # + : Gymnasium API 的 step() 回傳五元組 (obs, reward, terminated, truncated, info)

    # - : 原始 reset() 方法簽名
    # def reset(self):
    def reset(self, seed=None, options=None):
    # + : 依 Gymnasium API 標準，reset 方法加入 seed 與 options 參數
        # + : Call parent reset first. This will handle the seeding.
        super().reset(seed=seed) # + : 調用父類別的 reset 方法以正確處理 seed，此方法通常不返回 info 或返回 None

        self.steps_beyond_done = None
        self.step_id = 0
        self.cum_reward = 0
        self.first_execution = True
        self.prev_frame = -1

        # + : Reset curr_frame to 0 at the start of the episode, as managed by the environment
        # - : 錯誤地嘗試從 rl_pool 獲取 initial frame
        # self.curr_frame = self.rl_pool.current_perf_frame # + : Get initial frame from pool
        self.curr_frame = 0 # + : Initialize frame counter to 0
        # print(f"DEBUG: Initialized self.curr_frame = {self.curr_frame}") # + : DEBUG LOG


        self.last_action = None

        # reset data pool
        self.rl_pool.reset()
        # + : The pool's reset should align its internal state to the beginning


        # reset audio thread if applicable
        if self.render_mode == 'computer' or self.render_mode == 'human':
            # + : If audio thread exists and is running, stop it before starting a new one
            if hasattr(self, 'audioThread') and self.audioThread and self.audioThread.is_alive():
                 print("DEBUG: Stopping existing AudioThread before reset.") # + : DEBUG LOG
                 self.audioThread.end_stream()
                 self.audioThread.join() # Wait for the old thread to finish


            # write midi to wav
            # + : Import soundfile inside the if block to avoid importing if not needed
            try:
                import soundfile as sf
                # + : Construct audio filename safely
                song_name = self.rl_pool.get_current_song_name()
                if isinstance(song_name, (list, tuple)): # Handle cases where get_current_song_name returns list/tuple
                    song_name = song_name[0] # Use the first element as base name
                elif not isinstance(song_name, str):
                    song_name = "temp_audio" # Fallback name if format is unexpected

                fn_audio = f"{song_name}.wav" # + : Use f-string for clearer formatting
                perf_audio, fs = self.rl_pool.get_current_perf_audio_file()
                # + : Add directory for audio file to avoid cluttering root if needed
                # audio_output_dir = "." # Current directory
                # fn_audio_path = os.path.join(audio_output_dir, fn_audio)
                # sf.write(fn_audio_path, perf_audio, fs)
                # self.path_to_audio = fn_audio_path
                # + : For simplicity, keeping it in current directory as original
                sf.write(fn_audio, perf_audio, fs)
                self.path_to_audio = fn_audio


                self.audioThread = AudioThread(self.path_to_audio, self.rl_pool.spectrogram_params['fps'])
                print(f"DEBUG: Starting AudioThread for {fn_audio}") # + : DEBUG LOG
                self.audioThread.start()
                # + : Get initial frame position after starting audio, only if thread is active
                # + : Otherwise, curr_frame remains 0 from initialization.
                if self.audioThread.is_alive():
                     self.curr_frame = self.audioThread.get_current_spec_position()
                     self.prev_frame = self.curr_frame # + : Initialize prev_frame


            except ImportError:
                logger.error("Soundfile not installed. Cannot write WAV for audio playback.")
                # + : Decide how to handle this - fail reset, disable audio, etc.
                # For now, proceed without audio playback
                self.audioThread = None
            except Exception as e:
                 logger.error(f"Error during audio file writing or AudioThread setup: {e}", exc_info=True)
                 self.audioThread = None # Ensure audioThread is None on error


        # get first observation using the current frame position (which is now correctly initialized to 0 or audio pos)
        # print(f"DEBUG: Getting initial observation with frame {self.curr_frame}") # + : DEBUG LOG
        self.performance, self.score = self.rl_pool.step(self.curr_frame)
        # print(f"DEBUG: Initial observation perf shape {self.performance.shape}, score shape {self.score.shape}") # + : DEBUG LOG


        self.state = dict(
            perf=self.performance,
            score=self.score
            # + : Note: Wrapper might add 'perf_diff' *after* this base environment reset returns.
            # + : The observation returned by the wrapped env will have 'perf_diff'.
        )

        # + : Prepare info dictionary for reset return
        info = self.debug_info.copy() if hasattr(self, 'debug_info') else {}
        # - : 原始在 reset() 中嘗試獲取狀態資訊，但 pool 的屬性可能尚未更新，導致 AttributeError
        # info['song_timestep'] = self.rl.pool.current_song_timestep # Current position in song timesteps
        # info['tracking_error'] = self.rl.pool.tracking_error() # Current tracking error
        # info['song_speed'] = self.rl.pool.sheet_speed # Current speed
        # + : reset() 的 info 通常只包含關於重置本身的資訊，如 seed
        # + : Episode狀態資訊 (timestep, error等) 在 step() 後才可靠
        # + : Gymnasium's base Env.reset() might return some seeding info, but it's not guaranteed to be a dict.
        # + : For now, just return our own debug_info as the info dict.
        # + : If specific seeding info is needed from super().reset(), it would need more careful handling.

        # print(f"DEBUG: ScoreFollowingEnv.reset() returning state keys: {list(self.state.keys()) if isinstance(self.state, dict) else 'Not a dict'}, info keys: {list(info.keys())}") # + : DEBUG LOG
        return self.state, info
        # + : Gymnasium API 的 reset() 回傳 observation and info 字典

    # - : _seed() 方法已不適用於 Gymnasium，種子設定改由 reset(seed=...) 處理
    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def render(self, mode=None, close=False):
        # + : DEBUG: Add debug log inside render method
        # print(f"DEBUG: ScoreFollowingEnv.render() called. mode param: {mode}, self.render_mode: {self.render_mode}")

        if close:
            # + : Debug log for close
            # print("DEBUG: render() called with close=True. Closing resources...")
            # Note: AudioThread cleanup moved to close() method
            if hasattr(self, 'viewer') and self.viewer: # Standard gym viewer cleanup
                 self.viewer.close()
                 self.viewer = None
            # Do NOT return here immediately in case there's cleanup below
            return None # As per Gymnasium standard, render should return None when closing


        # + : Check if render_mode was set during __init_. If not, rendering might be disabled.
        # + : Let's strictly enforce that render() only proceeds if self.render_mode is set and matches the requested mode,
        # + : or if the requested mode is one of the supported modes, especially for rgb_array.
        # + : The test script calls unwrapped_env.render() *without* a mode parameter, so the default 'computer' is used.
        # + : But the logic inside should depend on the 'rgb_array' mode set during init.
        # + : A common pattern is for render(mode=mode) to check if 'mode' is supported *and* if the environment was *initialized* for rendering.
        # + : Let's make the logic simpler: proceed if self.render_mode is NOT None, and return the frame if self.render_mode is 'rgb_array'.
        # + : The display logic happens if self.render_mode is 'computer' or 'human'.

        if self.render_mode is None:
             # print("DEBUG: render_mode is None. Skipping rendering.") # DEBUG
             return None # Cannot render if no mode was specified at init

        # + : No need to explicitly check the 'mode' parameter passed to render(), rely on self.render_mode for output type.


        perf = self.prepare_perf_for_render()
        score = self.prepare_score_for_render()
        # print(f"DEBUG: render(): perf shape {perf.shape}, score shape {score.shape}") # DEBUG


        # highlight image center
        score_center = score.shape[1] // 2
        cv2.line(score, (score_center, 25), (score_center, score.shape[0] - 25), AGENT_COLOR, 2)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("Agent", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=1)[0]
        text_org = (score_center - text_size[0] // 2, score.shape[0] - 7)
        cv2.putText(score, "Agent", text_org, fontFace=font_face, fontScale=0.6, color=AGENT_COLOR, thickness=2)

        # hide tracking lines if it is rendered for humans
        # - : 原始 metadata key 為 render.modes, 條件複雜
        # if self.metadata['render.modes']['human'] != mode:
        # + : 簡化條件，直接判斷 self.render_mode 是否不是 'human'
        if self.render_mode != 'human':
        # + : 更新 metadata key 為 render_modes，並調整條件以符合列表形式。此處假設 'human' 是 render_modes[0]，並確保與 mode 比較正確。
        # + : 考量到原始 'human': 'human' 的結構，這裡的比較是與 'human' 字串比較。
        # + : 若 self.render_mode 不是 'human'，則執行以下繪圖。
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
        # else: # + : If self.render_mode is 'human', do not draw tracking lines. (Implicit in the if block)


        # prepare observation visualization
        cols = score.shape[1]
        rows = score.shape[0] + perf.shape[0]
        # + : 確保創建 obs_image 使用 np.uint8，因為 cv2.imshow 和 video writer 通常需要這個類型
        obs_image = np.zeros((rows, cols, 3), dtype=np.uint8)
        # - : 原始 dtype=np.uint8 已經是 uint8，但強調一下

        # write sheet to observation image
        # + : 確保 score 在賦值前是 uint8 且有 3 通道 (BGR)
        if score.dtype == np.float32: # Check if it's float, implying [0,1] range
             # + : 核心修補: float32轉uint8時需乘以255
             score_u8_scaled = np.uint8(np.clip(score * 255, 0, 255))
        else: # Assume it's already uint8 or other type, just clip
             score_u8_scaled = np.uint8(np.clip(score, 0, 255))

        if score.ndim == 2: # If grayscale, convert to 3-channel BGR
             score_display = cv2.cvtColor(score_u8_scaled, cv2.COLOR_GRAY2BGR)
        else: # Assume it's already 3-channel
             score_display = score_u8_scaled
        obs_image[0:score_display.shape[0], 0:score_display.shape[1], :] = score_display # Use score_display

        # write spec to observation image
        # + : 假設 perf 從 prepare_perf_for_render() 返回時已經是 uint8
        perf_u8_scaled = perf # + : 直接使用 prepare_perf_for_render() 的結果

        if perf_u8_scaled.ndim == 2: # If grayscale, convert to 3-channel BGR
             perf_display = cv2.cvtColor(perf_u8_scaled, cv2.COLOR_GRAY2BGR)
        else: # Assume it's already 3-channel
             perf_display = perf_u8_scaled

        c0 = obs_image.shape[1] // 2 - perf_display.shape[1] // 2 # Use perf_display shape
        c1 = c0 + perf_display.shape[1] # Use perf_display shape
        # + : 檢查拼貼範圍是否有效
        if score_display.shape[0] + perf_display.shape[0] <= rows and c1 <= cols:
             obs_image[score_display.shape[0]:score_display.shape[0] + perf_display.shape[0], c0:c1, :] = perf_display # Use perf_display
        else:
             logger.error(f"Render stitching failed: perf shape {perf_display.shape} at position exceeds obs_image bounds {obs_image.shape}. Score shape: {score_display.shape}")
             # Handle this error case - maybe return None or the partial image?
             # For now, let it potentially fail or return partial.


        # draw black line to separate score and performance
        # + : 確保線段繪製在有效範圍內
        line_row = score_display.shape[0]
        if line_row < rows and c0 >= 0 and c1 <= cols:
            obs_image[line_row, c0:c1, :] = 0
        # else: logger.warning("Skipping separator line drawing due to invalid coordinates.") # Optional warning


        # write diagnostic text aligned with the spectrogram region
        # + : 計算文字的 y 座標，從樂譜區塊高度開始略微往下偏移
        text_y = score_display.shape[0] + 10
        self._write_text(obs_image=obs_image, pos_px=text_y, color=TEXT_COLOR)


        # preserve this for access from outside
        self.obs_image = obs_image # This attribute is still used by the plotting logic in test_agent.py

        # show image (only for specific modes)
        if self.render_mode == 'computer' or self.render_mode == 'human':
            # print(f"DEBUG: ScoreFollowingEnv.render(): Displaying with cv2.imshow for mode {self.render_mode}") # DEBUG
            cv2.imshow("Score Following", self.obs_image)
            cv2.waitKey(1)
        # else:
            # print(f"DEBUG: ScoreFollowingEnv.render(): Not displaying with cv2.imshow for mode {self.render_mode}") # DEBUG


        # + : Return the generated image data if initialized with rgb_array mode
        # + : For other modes (like 'human' or 'computer' where display is handled internally), return None.
        # print(f"DEBUG: ScoreFollowingEnv.render(): Returning image data for render_mode '{self.render_mode}' if rgb_array, else None.") # DEBUG
        if self.render_mode == 'rgb_array':
            return self.obs_image # + : Return the generated image data (NumPy array)
        else:
            return None # + : Explicitly return None for display modes


    def close(self):
        # + : Debug log for close call
        # print("DEBUG: ScoreFollowingEnv.close() called.")

        # + : Close OpenCV windows if any were opened by this environment
        # + : Check if render_mode was one that might open windows
        if self.render_mode in ['computer', 'human']:
             cv2.destroyAllWindows()
             # print("DEBUG: OpenCV windows destroyed.") # DEBUG


        if hasattr(self, 'viewer') and self.viewer:
            # print("DEBUG: Closing standard gym viewer.") # DEBUG
            self.viewer.close()
            self.viewer = None
        # + : Ensure AudioThread in this environment is stopped and cleaned up
        if hasattr(self, 'audioThread') and self.audioThread and self.audioThread.is_alive():
             # print("DEBUG: Stopping and joining AudioThread.") # DEBUG
             self.audioThread.end_stream()
             self.audioThread.join() # Wait for thread to finish
             # print("DEBUG: AudioThread stopped.") # DEBUG

        # + : Standard practice to set resource attributes to None after closing
        self.audioThread = None
        self.obs_image = None
        self.viewer = None # Redundant if checked above, but safe


    # - : _seed() 方法已不適用於 Gymnasium，種子設定改由 reset(seed=...) 處理
    # def _seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def _write_text(self, obs_image, pos_px, color):
        """Draw diagnostics text starting at a pixel y position."""

        # + : 恢復原始專案的文字內容和順序
        # + : 假設 self.rl_pool.sheet_speed 和 self.rl_pool.tracking_error() 是可用的

        # Original: pixel speed: (1st line in author's image)
        write_text('pixel speed: {:4.1f}'.format(self.rl_pool.sheet_speed), pos_px, obs_image, color=color)

        # Original: last reward: (2nd line in author's image)
        write_text('last reward: {:4.2f}'.format(self.last_reward if self.last_reward is not None else 0.0), pos_px + 30, obs_image, color=color)

        # Original: score: (3rd line in author's image)
        write_text("score: {:6.2f}".format(self.cum_reward if self.cum_reward is not None else 0.0), pos_px + 60, obs_image, color=color)

        # Original: action: (4th line, based on previous diffs, not explicitly in author's image example but good to have)
        action_text = "action: {:+6.1f}".format(self.last_action if self.last_action is not None else 0.0)
        write_text(action_text, pos_px + 90, obs_image, color=color)  # + : action 顯示在最後一行


    def prepare_score_for_render(self):
        # + : Ensure output is suitable for rendering (e.g., uint8, BGR if needed later)
        # + : prepare_sheet_for_render likely handles this, but adding check for safety
        score_img = prepare_sheet_for_render(self.score, resz_x=self.resz_imag, resz_y=self.resz_imag)
        return score_img # Assume prepare_sheet_for_render returns appropriate format

    def prepare_perf_for_render(self):
        # + : 根據建議，先處理 self.performance 的縮放
        perf = self.performance
        # + : 只有在 float 0-1 範圍時才放大到 0-255
        if perf.dtype != np.uint8:
            perf = np.uint8(np.clip(perf * 255, 0, 255)) # + : 在此處乘以 255
        perf_img = prepare_spec_for_render(perf, resz_spec=self.resz_spec)
        return perf_img # Assume prepare_spec_for_render returns appropriate format