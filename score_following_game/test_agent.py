import copy
import cv2
import os
import torch
# - : gym 已棄用，改用 gymnasium
# import gym
import gymnasium as gym
# + : 匯入 gymnasium 並使用別名 gym，以符合新版 API 標準
import matplotlib.cm as cm
import numpy as np
# + : DEBUG: 匯入 sys 模組以重定向 stdout
import sys

from matplotlib.colors import LinearSegmentedColormap, Normalize
from score_following_game.agents.human_agent import HumanAgent
from score_following_game.agents.optimal_agent import OptimalAgent
from score_following_game.agents.networks_utils import get_network
from score_following_game.data_processing.data_pools import get_single_song_pool
from score_following_game.data_processing.utils import load_game_config
from score_following_game.environment.score_following_env import ScoreFollowingEnv
from score_following_game.environment.render_utils import prepare_sheet_for_render, prepare_spec_for_render
from score_following_game.experiment_utils import initialize_trained_agent, get_make_env, make_env_tismir,\
    setup_evaluation_parser
from score_following_game.integrated_gradients import IntegratedGradients, prepare_grad_for_render
from score_following_game.reinforcement_learning.algorithms.models import Model
from score_following_game.utils import render_video, get_opencv_bar

# + : DEBUG: Add a global step counter for logging
global_step_counter = 0

# render mode for the environment ('human', 'computer', 'video')
# render_mode = 'computer' # Use this for live display
render_mode = 'video' 
mux_audio = True

if __name__ == "__main__":
    """ main """

    # + : DEBUG: 重定向標準輸出到 log 檔案
    original_stdout = sys.stdout # + : 保存原始的標準輸出
    log_file_path = "DebugLog.txt"
    # + : 打印一條消息到原始標準輸出，告知輸出將被重定向
    print(f"Redirecting debug output to {log_file_path}...", file=original_stdout)
    try:
        # + : 以寫入模式打開 log 檔案，使用 utf-8 編碼確保中文字符正確
        log_file = open(log_file_path, "w", encoding="utf-8")
        # + : 將標準輸出重定向到檔案
        sys.stdout = log_file

        print("--- test_agent.py script started ---") # + : DEBUG LOG (現在會輸出到檔案)

        parser = setup_evaluation_parser()
        parser.add_argument('--agent_type', help='which agent to test [rl|optimal|human].',
                            choices=['rl', 'optimal', 'human'], type=str, default="rl")
        # - : 原始 --plot_stats 的 default 為 False
        # parser.add_argument('--plot_stats', help='plot additional statistics.', action='store_true', default=False)
        parser.add_argument('--plot_stats', help='plot additional statistics.', action='store_true', default=True) # + : 恢復舊版行為，預設啟用統計繪圖
        parser.add_argument('--plot_ig', help='plot integrated gradients.', action='store_true', default=False)

        args = parser.parse_args()
        print(f"DEBUG: Parsed arguments: {args}") # + : DEBUG LOG
        print(f"DEBUG: Desired render_mode for script output: {render_mode}") # + : DEBUG LOG


        if args.agent_type == 'rl':
            print("DEBUG: Agent type is RL. Processing exp_name, net, game_config...") # + : DEBUG LOG
            # --- Determine exp_name (experiment name) ---
            # - : 原始 exp_name 定義。在 --params 僅為檔名時 (e.g., "model.pth")，
            #   os.path.split(args.params)[0] 會是空字串，導致 exp_name 也為空字串。
            # exp_name_old = os.path.basename(os.path.split(args.params)[0])

            # + : 使用更穩健的方式來決定 exp_name
            if args.params:
                # + : 嘗試從 --params 的目錄部分獲取 exp_name
                exp_dir = os.path.dirname(args.params)
                exp_name_from_dir = os.path.basename(exp_dir)
                if exp_name_from_dir: # + : 如果 --params 包含目錄結構 (e.g., "results/exp1/model.pth")
                    exp_name = exp_name_from_dir # + : 使用目錄名作為 exp_name (e.g., "exp1")
                else: # + : 如果 --params 僅為檔名 (e.g., "model.pth")，exp_dir 為空，exp_name_from_dir 也為空
                    exp_name = os.path.splitext(os.path.basename(args.params))[0] # + : 使用檔案名稱 (不含副檔名) 作為 exp_name (e.g., "model")
            else: # + : 如果未提供 --params
                exp_name = 'manual-test' # + : 設定預設的 exp_name
            # + : 此時 exp_name 是一個更可靠的字串 (e.g., "exp1", "model", or "manual-test")
            print(f"DEBUG: Determined exp_name: {exp_name}") # + : DEBUG LOG


            # --- Infer args.net if not provided ---
            if args.net is None:
                print(f"DEBUG: args.net is None. Inferring from exp_name '{exp_name}'...") # + : DEBUG LOG
                # - : 原始 args.net 推斷邏輯，依賴可能不穩定的 exp_name_old
                # args.net = exp_name_old.split('-')[1]
                # + : 使用新的、更穩健的 exp_name 推斷 args.net，並提供預設值
                exp_parts_for_net = exp_name.split('-')
                if len(exp_parts_for_net) > 1:
                    args.net = exp_parts_for_net[1]
                    # + : args.net 已從 exp_name 推斷 (e.g., if exp_name is "rl-network_type-...")
                else:
                    args.net = 'default' # + : exp_name 格式不符，args.net 設為預設值
                    print(f"DEBUG: Could not infer net from exp_name, using default: {args.net}") # + : DEBUG LOG
            print(f"DEBUG: Final args.net: {args.net}") # + : DEBUG LOG


            # --- Infer args.game_config if not provided ---
            if args.game_config is None:
                print(f"DEBUG: args.game_config is None. Inferring from exp_name '{exp_name}'...") # + : DEBUG LOG
                # - : 原始 game_config 推斷邏輯，依賴可能不穩定的 exp_name_old
                # args.game_config = 'game_configs/{}.yaml'.format(exp_name_old.split("-")[3].rsplit("_", 2)[0])
                # + : 使用新的、更穩健的 exp_name 推斷 game_config
                game_config_parts = exp_name.split("-")
                if len(game_config_parts) > 3: # + : 檢查 exp_name 是否符合 "something-net-features-configbase_..." 格式
                     args.game_config = 'game_configs/{}.yaml'.format(game_config_parts[3].rsplit("_", 2)[0])
                     # + : game_config 已成功推斷
                else:
                    # + : exp_name 格式不符，無法推斷 game_config。args.game_config 將維持 None。
                    # + : 如果 args.params 存在且 exp_name 不是 'manual-test'，則發出警告。
                    if args.params and exp_name not in ['manual-test', 'default']: # + : 避免對 'manual-test' 或未提供 --params 的情況發出警告
                         print(f"Warning: Could not infer game_config from experiment name '{exp_name}' "
                               f"(derived from --params='{args.params}'). Expected format for automatic inference "
                               f"is like '<anyprefix>-<network>-<features>-<configbasename>_...'. "
                               f"If --game_config was not specified and is required by the program, this may lead to an error.")
            print(f"DEBUG: Final args.game_config: {args.game_config}") # + : DEBUG LOG


        print(f"DEBUG: Loading game_config from: {args.game_config}") # + : DEBUG LOG
        config = load_game_config(args.game_config)
        print(f"DEBUG: Loaded game_config content (first few keys): {{'actions': {config.get('actions')}, 'spec_shape': {config.get('spec_shape')}, ...}}") # + : DEBUG LOG


        if args.agent_type == 'optimal':
            # the action space for the optimal agent needs to be continuous
            config['actions'] = []
            print("DEBUG: Agent type is optimal. Cleared actions in config.") # + : DEBUG LOG

        print(f"DEBUG: Getting single song pool for piece: {args.piece}, dataset: {args.data_set}") # + : DEBUG LOG
        pool = get_single_song_pool(
            dict(config=config, song_name=args.piece, directory=args.data_set, real_perf=args.real_perf))
        print("DEBUG: Song pool created.") # + : DEBUG LOG

        observation_images = []

        # initialize environment
        # + : 初始化潛在的 wrapped 環境 (例如 DifferenceWrapper 包裹 ScoreFollowingEnv)
        # + : 根據腳本期望的 render_mode，設定環境內部的渲染模式。
        # + : 如果期望輸出影片 ('video') 或電腦顯示 ('computer')，使用 'rgb_array' 讓 render() 返回像素陣列。
        # + : 如果期望人機互動 ('human')，則使用 'human' 讓 render() 顯示視窗。
        env_render_mode = 'rgb_array' if render_mode in ['video', 'computer'] else render_mode
        print(f"DEBUG: Initializing environment with internal render_mode: {env_render_mode}") # + : DEBUG LOG
        # + : env 現在是 wrapped 環境 (包含 Wrapper)
        env = make_env_tismir(pool, config, render_mode=env_render_mode)
        print(f"DEBUG: Environment created. Type of env: {type(env)}") # + : DEBUG LOG
        # + : 使用 env_render_mode 初始化環境，確保 render() 方法能回傳預期格式。


        # + : 在主迴圈開始前，將 env unwrapped 到 ScoreFollowingEnv 層，以便訪問其底層 render() 方法。
        # + : 這裡的 unwrapped_env 僅用於獲取底層環境的渲染畫面，不應用於 step/reset。
        unwrapped_env = env # + : 先保留一個對 wrapped env 的引用
        print(f"DEBUG: Starting env unwrapping. Initial unwrapped_env type: {type(unwrapped_env)}") # + : DEBUG LOG
        # + : 循環 unwrapping 直到找到 ScoreFollowingEnv 實例
        while not isinstance(unwrapped_env, ScoreFollowingEnv):
            # + : 如果當前 unwrapped_env 不是 ScoreFollowingEnv，則進入下一層 (wrapped 環境的 .env 屬性)
            print(f"DEBUG: Unwrapping... current type: {type(unwrapped_env)}, has .env: {hasattr(unwrapped_env, 'env')}") # + : DEBUG LOG
            if not hasattr(unwrapped_env, 'env'):
                print("ERROR: Environment does not have .env attribute for unwrapping. Stopping.") # + : DEBUG LOG
                # + : DEBUG: 在發生 unwrapping 錯誤時，明確記錄並結束
                print("--- test_agent.py script aborted due to unwrapping error ---")
                sys.exit(1) # + : 使用非零退出碼表示錯誤
            unwrapped_env = unwrapped_env.env # type: ignore
            # + : 持續更新 unwrapped_env 變數，直到指向最底層的 ScoreFollowingEnv
        # + : 現在 unwrapped_env 變數 নিশ্চিত地指向 ScoreFollowingEnv 的實例，可用於 render() 調用。
        print(f"DEBUG: Env unwrapped. Final unwrapped_env type: {type(unwrapped_env)}") # + : DEBUG LOG


        if args.agent_type == 'human' or args.agent_type == 'optimal':
            print(f"DEBUG: Agent type is '{args.agent_type}'. Calling agent.play_episode...") # + : DEBUG LOG
            agent = HumanAgent(pool) if args.agent_type == 'human' else OptimalAgent(pool)
            # Note: The play_episode function within the agent itself needs to be updated for Gymnasium API if it interacts directly with env.step/reset
            # + : 傳遞原始的 wrapped env 給 play_episode，讓它自己處理 unwrapping 或依賴 agent 內部邏輯
            alignment_errors, action_sequence, observation_images, episode_reward = agent.play_episode(env, render_mode)
            print(f"DEBUG: agent.play_episode finished. Episode reward: {episode_reward}, images collected: {len(observation_images)}") # + : DEBUG LOG

        else: # RL Agent
            print("DEBUG: RL Agent path. Initializing network, model, agent...") # + : DEBUG LOG
            # compile network architecture
            n_actions = len(config["actions"])
            print(f"DEBUG: Number of actions: {n_actions}") # + : DEBUG LOG
            net = get_network("networks_sheet_spec", args.net, n_actions=n_actions,
                              shapes=dict(perf_shape=config['spec_shape'], score_shape=config['sheet_shape']))
            print(f"DEBUG: Network '{args.net}' compiled.") # + : DEBUG LOG

            # load network parameters
            # - : 原始 torch.load 沒有指定 map_location，可能導致在非 CUDA 裝置上載入 CUDA 模型時報錯。
            # net.load_state_dict(torch.load(args.params))
            # + : 使用 map_location='cpu' 來確保模型張量載入到 CPU，兼容非 CUDA 環境 (如 macOS M3 的 MPS)。
            print(f"DEBUG: Loading network parameters from: {args.params}") # + : DEBUG LOG
            net.load_state_dict(torch.load(args.params, map_location=torch.device('cpu')))
            print("DEBUG: Network parameters loaded.") # + : DEBUG LOG


            # set model to evaluation mode
            net.eval()
            print("DEBUG: Network set to eval mode.") # + : DEBUG LOG

            # create agent
            # - : 原始程式碼檢查 use_cuda，但模型載入已強制到 CPU，此變數可能誤導。
            # use_cuda = torch.cuda.is_available() # Keep the check if needed elsewhere, but note model is on CPU.
            # + : 移除或註釋掉不必要的 use_cuda 檢查，因模型已強制載入到 CPU。
            # 如果後續的模型運算需要在 GPU (MPS) 上進行，則需要將模型和數據 tensor 手動移動到 MPS 裝置。
            # 在此範例中，模型已經載入到 CPU，且未發現明確將模型移到 MPS 的程式碼，因此保持在 CPU 上運行。

            model = Model(net, optimizer=None)
            print("DEBUG: Model wrapper created.") # + : DEBUG LOG

            # Note: initialize_trained_agent might still take a use_cuda argument, but it won't affect model device.
            # If the agent logic needs the model on a specific device (CPU/MPS), it should handle the device placement.
            agent = initialize_trained_agent(model, use_cuda=False, deterministic=False) # + : Explicitly set use_cuda=False as model is on CPU
            print("DEBUG: RL Agent initialized.") # + : DEBUG LOG


            observation_images = []

            # get observations
            episode_reward = 0
            # - : 將 reset 調用從原始的 wrapped env 移到 unwrapped_env (這是錯誤的)
            # observation, info = unwrapped_env.reset() # type: ignore
            # - : gym reset() API 返回單一 observation
            # observation = env.reset()
            # + : 在 wrapped env (env) 上調用 reset()，以獲取包含 Wrapper 處理結果的完整 observation
            print("DEBUG: Calling env.reset()...") # + : DEBUG LOG
            observation, info = env.reset() # type: ignore
            print(f"DEBUG: env.reset() done. Observation keys: {list(observation.keys()) if isinstance(observation, dict) else 'Not a dict'}") # + : DEBUG LOG
            if isinstance(observation, dict):
                 for k, v_obs in observation.items():
                     print(f"DEBUG:   obs['{k}'].shape: {v_obs.shape if hasattr(v_obs, 'shape') else type(v_obs)}, .dtype: {v_obs.dtype if hasattr(v_obs, 'dtype') else 'N/A'}") # + : DEBUG LOG

            # + : gymnasium reset() API 返回 observation 和 info dict. 在 wrapped env 上調用。

            reward = 0
            # - : gym step() API 返回的 done 變數已被棄用
            # done = False
            terminated = False # + : gymnasium 使用 terminated 標示因環境內部條件結束 (例如達到目標)
            truncated = False # + : gymnasium 使用 truncated 標示因外部條件結束 (例如時間限制)
            print("DEBUG: Initialized terminated=False, truncated=False.") # + : DEBUG LOG


            if args.plot_ig:
                print("DEBUG: Plotting Integrated Gradients is enabled.") # + : DEBUG LOG
                # Note: IG likely expects tensors on a specific device ('cuda' was used).
                # This needs adjustment for MPS or CPU. For now, keep 'cuda' but expect potential issues
                # if IG itself does not handle MPS or CPU tensors correctly or requires device mapping.
                # IG = IntegratedGradients(net, 'cuda', steps=20)
                # + : 調整 IntegratedGradients 的裝置設置為 CPU，因為模型已載入到 CPU
                IG = IntegratedGradients(net, 'cpu', steps=20) # + : 將 IG 裝置改為 'cpu'
                print("DEBUG: IntegratedGradients initialized for CPU.") # + : DEBUG LOG
                # Create a separate 'plain' environment for Integrated Gradients calculation
                # Note: The underlying make_env_tismir or get_make_env might need internal updates for Gymnasium
                # + : 這裡的 plain_env 邏輯保留，它創建一個 Wrapped 環境
                # + : 初始化 plain_env 時也設定正確的 render_mode
                plain_env_render_mode = 'rgb_array' if render_mode in ['video', 'computer'] else render_mode
                print(f"DEBUG: Initializing plain_env with internal render_mode: {plain_env_render_mode}") # + : DEBUG LOG
                plain_env = get_make_env(copy.deepcopy(pool), config, make_env_fnc=make_env_tismir, render_mode=plain_env_render_mode)() # type: ignore
                print(f"DEBUG: plain_env created. Type: {type(plain_env)}") # + : DEBUG LOG

                # + : Unwrap plain_env 到 ScoreFollowingEnv 層，用於獲取其底層 render() 畫面
                plain_unwrapped_env = plain_env # + : 保留 plain env 的引用
                print(f"DEBUG: Starting plain_env unwrapping. Initial type: {type(plain_unwrapped_env)}") # + : DEBUG LOG
                while not isinstance(plain_unwrapped_env, ScoreFollowingEnv):
                    if not hasattr(plain_unwrapped_env, 'env'):
                        print("ERROR: plain_env does not have .env attribute for unwrapping. Stopping.") # + : DEBUG LOG
                        # + : DEBUG: 在發生 unwrapping 錯誤時，明確記錄並結束
                        print("--- test_agent.py script aborted due to plain_env unwrapping error ---")
                        sys.exit(1) # + : 使用非零退出碼表示錯誤
                    plain_unwrapped_env = plain_unwrapped_env.env # type: ignore
                print(f"DEBUG: plain_env unwrapped. Final type: {type(plain_unwrapped_env)}") # + : DEBUG LOG


                # - : gym reset() API
                # plain_env.reset()
                # + : 在 plain wrapped env 上調用 reset()
                print("DEBUG: Calling plain_env.reset()...") # + : DEBUG LOG
                obs_plain, info_plain = plain_env.reset() # type: ignore
                print(f"DEBUG: plain_env.reset() done. obs_plain keys: {list(obs_plain.keys()) if isinstance(obs_plain, dict) else 'Not a dict'}") # + : DEBUG LOG
                if isinstance(obs_plain, dict):
                    for k, v_obs in obs_plain.items():
                        print(f"DEBUG:   obs_plain['{k}'].shape: {v_obs.shape if hasattr(v_obs, 'shape') else type(v_obs)}, .dtype: {v_obs.dtype if hasattr(v_obs, 'dtype') else 'N/A'}") # + : DEBUG LOG
                # + : gymnasium reset() API 返回 observation 和 info dict. 在 wrapped env 上調用。


            colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#000000', '#e0f3f8', '#abd9e9', '#74add1',
                      '#4575b4', '#313695']
            colors = list(reversed(colors))
            cmap = LinearSegmentedColormap.from_list('cmap', colors)
            norm = Normalize(vmin=-1, vmax=1)

            pos_grads = []
            neg_grads = []
            abs_grads = []
            values = []
            tempo_curve = []

            print("DEBUG: Starting main agent loop...") # + : DEBUG LOG
            # Loop until the episode ends (terminated or truncated)
            # - : 原始迴圈條件使用 done
            # while True: # Implicitly checked done inside
            while not (terminated or truncated): # + : 迴圈條件改為檢查 terminated 或 truncated
                global_step_counter += 1 # + : DEBUG
                print(f"\nDEBUG: --- Main Loop Step: {global_step_counter} ---") # + : DEBUG LOG
                print(f"DEBUG: Current observation keys (from env.step): {list(observation.keys()) if isinstance(observation, dict) else 'Not a dict'}") # + : DEBUG LOG
                if isinstance(observation, dict):
                    for k, v_obs in observation.items():
                        print(f"DEBUG:   obs['{k}'].shape: {v_obs.shape if hasattr(v_obs, 'shape') else type(v_obs)}, .dtype: {v_obs.dtype if hasattr(v_obs, 'dtype') else 'N/A'}") # + : DEBUG LOG

                # choose action
                # - : 原始程式碼錯誤地傳遞了已轉換的 observation_tensor
                # observation_tensor = agent.prepare_state(observation)
                # action = agent.select_action(observation_tensor)
                # + : 直接將 wrapped 環境返回的原始 observation (NumPy dict，應包含 perf_diff) 傳遞給 agent.select_action
                action = agent.select_action(observation) # + : 將 wrapped 環境返回的觀察值傳遞給 agent 的 select_action
                print(f"DEBUG: Step {global_step_counter}: Agent selected action: {action}") # + : DEBUG LOG


                # perform step and observe
                # - : 將 step 調用從原始的 wrapped env 移到 unwrapped_env (這是錯誤的)
                # observation, reward, terminated, truncated, info = unwrapped_env.step(action) # type: ignore
                # - : gym step() API 返回四元組 (obs, reward, done, info)
                # observation, reward, done, info = env.step(action)
                # + : 在 wrapped env (env) 上調用 step()，以獲取包含 Wrapper 處理結果的完整 observation
                print(f"DEBUG: Step {global_step_counter}: Calling env.step({action})...") # + : DEBUG LOG
                observation, reward, terminated, truncated, info = env.step(action) # type: ignore
                # + : gymnasium step() API 返回五元組 (obs, reward, terminated, truncated, info). 在 wrapped env 上調用。
                print(f"DEBUG: Step {global_step_counter}: env.step() returned. Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}") # + : DEBUG LOG
                print(f"DEBUG: Step {global_step_counter}: New observation keys (from env.step): {list(observation.keys()) if isinstance(observation, dict) else 'Not a dict'}") # + : DEBUG LOG
                if isinstance(observation, dict):
                    for k, v_obs in observation.items():
                        print(f"DEBUG:   New obs['{k}'].shape: {v_obs.shape if hasattr(v_obs, 'shape') else type(v_obs)}, .dtype: {v_obs.dtype if hasattr(v_obs, 'dtype') else 'N/A'}") # + : DEBUG LOG


                episode_reward += reward

                # + : 在 step() 後調用 unwrapped_env.render() 獲取底層環境的當前畫面
                # + : 注意：這裡仍從 unwrapped_env 獲取 render frame，因為 Wrapper 可能不暴露這個方法或返回不想要的格式。
                # + : 這是處理 render 邏輯與 agent 邏輯分離的一種方式。
                print(f"DEBUG: Step {global_step_counter}: Calling unwrapped_env.render()...") # + : DEBUG LOG
                render_frame = unwrapped_env.render()
                # + : render() 在 render_mode='rgb_array' 時返回 NumPy 陣列，在 'human' 或 'computer' 時可能返回 None
                if render_frame is None:
                    print(f"DEBUG: Step {global_step_counter}: unwrapped_env.render() returned None.") # + : DEBUG LOG
                else:
                    print(f"DEBUG: Step {global_step_counter}: unwrapped_env.render() returned frame with shape: {render_frame.shape}, dtype: {render_frame.dtype}") # + : DEBUG LOG


                # + : 檢查 render_frame 是否有效 (非 None)
                if render_frame is not None: # + : 檢查 render() 回傳值是否有效
                    print(f"DEBUG: Step {global_step_counter}: render_frame is not None. Processing frame...") # + : DEBUG LOG

                    # - : 原始程式碼從 unwrapped_env.obs_image 獲取 bar_img，現在改為使用 render_frame
                    # bar_img = unwrapped_env.obs_image
                    bar_img = copy.copy(render_frame) # + : 複製 render_frame 作為 bar_img，準備疊加資訊
                    print(f"DEBUG: Step {global_step_counter}: Copied render_frame to bar_img. bar_img.shape: {bar_img.shape}") # + : DEBUG LOG


                    if args.plot_ig or args.plot_stats:
                        print(f"DEBUG: Step {global_step_counter}: Plotting IG or Stats is enabled.") # + : DEBUG LOG
                        # + : 在需要使用張量模型的區塊準備 observation_tensor
                        # + : 注意：這裡傳入的 observation 是 wrapped env 返回的，應該包含 perf_diff
                        observation_tensor = agent.prepare_state(observation) # + : 準備 observation_tensor 給模型使用
                        print(f"DEBUG: Step {global_step_counter}: Prepared observation_tensor for model. Keys: {list(observation_tensor.keys())}") # + : DEBUG LOG
                        if 'perf' in observation_tensor:
                            print(f"DEBUG:   observation_tensor['perf'].shape: {observation_tensor['perf'].shape}, .dtype: {observation_tensor['perf'].dtype}") # + : DEBUG LOG
                        if 'score' in observation_tensor:
                            print(f"DEBUG:   observation_tensor['score'].shape: {observation_tensor['score'].shape}, .dtype: {observation_tensor['score'].dtype}") # + : DEBUG LOG


                        model_return = model(observation_tensor)
                        print(f"DEBUG: Step {global_step_counter}: Model forward pass done.") # + : DEBUG LOG


                        # - : 原始程式碼從 unwrapped_env.obs_image 獲取 shape，現在改為從 bar_img 獲取
                        # bar_height = unwrapped_env.obs_image.shape[0]
                        bar_height = bar_img.shape[0] # + : 從 bar_img 獲取高度

                        # - : NumPy 廢棄型別別名 np.uint8 -> np.uint8 (維持不變，非廢棄)
                        # + : 調整 spacer 的型別與 bar_img 匹配 (通常是 uint8)
                        spacer = 255 * np.ones((bar_height, 5, 3), bar_img.dtype) # + : spacer 與 bar_img 相同型別
                        # + : np.uint8 非 NumPy 廢棄型別，維持原樣

                        if args.plot_ig:
                            print(f"DEBUG: Step {global_step_counter}: Plotting IG...") # + : DEBUG LOG
                            # Step the plain environment to get original observations for IG
                            # - : gym step() API 返回四元組
                            # obs_org, r, d, _ = plain_env.step(action)
                            # + : 在 plain wrapped env 上調用 step()
                            obs_org, r, terminated_plain, truncated_plain, info_plain = plain_env.step(action) # type: ignore
                            # + : gymnasium step() 回傳五元組. 在 wrapped env 上調用。 obs_org 應包含 perf_diff。
                            print(f"DEBUG: Step {global_step_counter} (IG): plain_env.step() done. obs_org keys: {list(obs_org.keys())}") # + : DEBUG LOG
                            if isinstance(obs_org, dict):
                                for k, v_obs in obs_org.items():
                                    print(f"DEBUG:   obs_org['{k}'].shape: {v_obs.shape if hasattr(v_obs, 'shape') else type(v_obs)}, .dtype: {v_obs.dtype if hasattr(v_obs, 'dtype') else 'N/A'}") # + : DEBUG LOG


                            # + : 在 plain_env 也調用 plain_unwrapped_env.render() 獲取原始畫面用於疊加
                            plain_render_frame = plain_unwrapped_env.render() # + : 獲取 plain_env 的 render frame
                            if plain_render_frame is None:
                                print(f"ERROR: Step {global_step_counter} (IG): plain_unwrapped_env.render() returned None! Cannot plot IG overlay.") # + : DEBUG LOG
                            else:
                                print(f"DEBUG: Step {global_step_counter} (IG): plain_unwrapped_env.render() frame shape: {plain_render_frame.shape}, dtype: {plain_render_frame.dtype}") # + : DEBUG LOG


                                # invert and create grayscale score
                                # + : 確保取的是 score[0]
                                org_score = 1 - obs_org['score'][0]
                                # - : NumPy 廢棄型別別名 np.uint8 -> np.uint8 (維持不變，非廢棄)
                                # + : 轉換為正確的型別，與 cv2.addWeighted 兼容 (通常是 uint8)
                                org_score = np.uint8(cm.gray(org_score) * 255) # + : 維持 uint8

                                # create grayscale perf
                                # + : 確保取的是 perf[0]
                                org_perf = np.uint8(cm.gray(obs_org['perf'][0]) * 255) # + : 維持 uint8

                                # get gradients
                                # + : Pass observation_tensor (prepared from wrapped env obs) to IG
                                guided_score_grads, guided_perf_grads = IG.generate_gradients([observation_tensor['perf'],
                                                                                               observation_tensor['score']])
                                # prepare saliency map for score and delta-score
                                grads_score = guided_score_grads[0]
                                # + : prepare_grad_for_render 可能需要特定型別或範圍，確保兼容
                                grads_score = prepare_grad_for_render(grads_score, (config['score_shape'][2], config['score_shape'][1]), norm, cmap)
                                # + : 確保 grads_score 是 OpenCV 兼容的型別 (通常是 uint8 或 float32)
                                if grads_score.dtype != np.uint8:
                                     # 假設 grads_score 在 [0, 1] 範圍，轉換為 [0, 255]
                                     grads_score = np.uint8(np.clip(grads_score * 255, 0, 255)) # + : 添加 clip 確保值在範圍內，並轉換為 uint8

                                # prepare saliency map for performance and delta-performance
                                grads_perf = guided_perf_grads[0]
                                # + : prepare_grad_for_render 可能需要特定型別或範圍，確保兼容
                                grads_perf = prepare_grad_for_render(grads_perf, (config['perf_shape'][2], config['perf_shape'][1]),
                                                                     norm, cmap)
                                # + : 確保 grads_perf 是 OpenCV 兼容的型別 (通常是 uint8 或 float32)
                                if grads_perf.dtype != np.uint8:
                                     # 假設 grads_perf 在 [0, 1] 範圍，轉換為 [0, 255]
                                     grads_perf = np.uint8(np.clip(grads_perf * 255, 0, 255)) # + : 添加 clip 確保值在範圍內，並轉換為 uint8


                                # add score gradients to score
                                # + : 使用 plain_render_frame 作為疊加的原始圖片
                                # + : 檢查 org_score[:, :, :-1].shape 是否兼容，假設 org_score 是 (H, W, C) 且 C>=3
                                # + : 如果 org_score 是灰度 (H, W)，需要轉換為 (H, W, 3)
                                if org_score.ndim == 2:
                                    org_score = cv2.cvtColor(org_score, cv2.COLOR_GRAY2BGR) # + : 轉換灰度為 BGR
                                # + : 確保 added_image_score 和 org_img_part 的 shape 兼容
                                added_image_score = cv2.addWeighted(grads_score, 1.0, org_score, 0.4, 0) # + : 使用轉換後的 org_score

                                added_image_score = prepare_sheet_for_render(added_image_score, plain_unwrapped_env.resz_x, plain_unwrapped_env.resz_y, transform_to_bgr=False)

                                # add performance gradients to performance
                                # + : 使用 plain_render_frame 作為疊加的原始圖片
                                # + : 檢查 org_perf[:, :, :-1].shape 是否兼容，假設 org_perf 是 (H, W, C) 且 C>=3
                                # + : 如果 org_perf 是灰度 (H, W)，需要轉換為 (H, W, 3)
                                if org_perf.ndim == 2:
                                     org_perf = cv2.cvtColor(org_perf, cv2.COLOR_GRAY2BGR) # + : 轉換灰度為 BGR
                                # + : 確保 added_image_perf 和 org_img_part 的 shape 兼容
                                added_image_perf = cv2.addWeighted(grads_perf, 1.0, org_perf, 0.4, 0) # + : 使用轉換後的 org_perf

                                added_image_perf = prepare_spec_for_render(added_image_perf, plain_unwrapped_env.resz_spec, transform_to_bgr=False)

                                # - : 原始程式碼從 env.obs_image 獲取並修改它，現在改為使用 plain_render_frame 作為基礎來創建原圖部分
                                # org_img = copy.copy(unwrapped_env.obs_image)
                                # unwrapped_env.obs_image[0:added_image_score.shape[0], 0:added_image_score.shape[1], :] = added_image_score
                                # c0 = unwrapped_env.obs_image.shape[1] // 2 - added_image_perf.shape[1] // 2
                                # c1 = c0 + added_image_perf.shape[1]
                                # unwrapped_env.obs_image[added_image_score.shape[0]:, c0:c1, :] = added_image_perf
                                # + : 創建用於疊加的原圖部分，從 plain_render_frame 複製，並確保其形狀兼容
                                # + : 假設 plain_render_frame 是 RGB (H, W, 3)
                                org_img_part = plain_render_frame.copy() # + : 複製原圖部分
                                # + : 在原圖部分疊加 score gradients (確保疊加區域和疊加物的 shape 兼容)
                                h_score, w_score = added_image_score.shape[:2]
                                h_perf, w_perf = added_image_perf.shape[:2]
                                # + : 檢查疊加區域是否超出原圖範圍
                                if h_score <= org_img_part.shape[0] and w_score <= org_img_part.shape[1]:
                                     org_img_part[0:h_score, 0:w_score, :] = added_image_score
                                else:
                                     print(f"WARNING: Step {global_step_counter} (IG): added_image_score shape {added_image_score.shape} exceeds org_img_part shape {org_img_part.shape}. Skipping score overlay.")

                                # + : 在原圖部分疊加 performance gradients (確保疊加區域和疊加物的 shape 兼容)
                                c0 = org_img_part.shape[1] // 2 - w_perf // 2
                                c1 = c0 + w_perf
                                # + : 檢查疊加區域是否超出原圖範圍
                                if added_image_score.shape[0] + h_perf <= org_img_part.shape[0] and c1 <= org_img_part.shape[1]:
                                     org_img_part[added_image_score.shape[0]:added_image_score.shape[0] + h_perf, c0:c1, :] = added_image_perf
                                else:
                                    print(f"WARNING: Step {global_step_counter} (IG): added_image_perf shape {added_image_perf.shape} at position exceeds org_img_part shape {org_img_part.shape}. Skipping perf overlay.")


                                # + : 將疊加了 IG 的原圖部分拼接到 bar_img 後面
                                bar_img = np.concatenate((bar_img, spacer, org_img_part), axis=1)
                                print(f"DEBUG: Step {global_step_counter} (IG): IG overlay complete. bar_img.shape: {bar_img.shape}") # + : DEBUG LOG


                        if args.plot_stats:
                            print(f"DEBUG: Step {global_step_counter}: Plotting Stats...") # + : DEBUG LOG
                            # Note: Stats are added to bar_img regardless of whether plot_ig is true.
                            # If plot_ig is true, bar_img already contains original image + IG.
                            # If plot_ig is false, bar_img contains just the original rendered frame.
                            # The concatenation logic needs to account for this.

                            value = model_return['value'].detach().cpu().item()
                            values.append(value)
                            tempo_curve.append(pool.sheet_speed)
                            print(f"DEBUG: Step {global_step_counter} (Stats): Value: {value:.2f}, Speed: {pool.sheet_speed:.2f}, Error: {np.abs(pool.tracking_error()):.2f}, Episode Reward: {episode_reward:.2f}") # + : DEBUG LOG


                            # get value function bar plot
                            value_bgr = get_opencv_bar(value, bar_heigth=bar_height, max_value=25,
                                                       color=(255, 255, 0), title="value")

                            # get pixel speed bar plot
                            speed_bgr = get_opencv_bar(pool.sheet_speed, bar_heigth=bar_height, min_value=-15, max_value=15,
                                                       color=(255, 0, 255), title="speed")

                            # get tracking error bar
                            error_bgr = get_opencv_bar(np.abs(pool.tracking_error()),
                                                       bar_heigth=bar_height, max_value=config['score_shape'][-1] // 2,
                                                       color=(0, 0, 255), title="error")

                            # get score progress bar
                            score_bgr = get_opencv_bar(episode_reward, bar_heigth=bar_height, max_value=pool.get_current_song_timesteps(),
                                                       color=(0, 255, 255), title="reward")

                            # get potential score progress bar
                            pot_score_bgr = get_opencv_bar(len(tempo_curve), bar_heigth=bar_height,
                                                           max_value=pool.get_current_song_timesteps(),
                                                           color=(0, 255, 255), title="max")

                            # + : 將 bar_img (可能已包含 IG) 與 stats bars 拼接
                            bar_img = np.concatenate((bar_img, spacer, value_bgr, spacer, speed_bgr, spacer, error_bgr, spacer,
                                                      score_bgr, spacer, pot_score_bgr), axis=1)
                            print(f"DEBUG: Step {global_step_counter} (Stats): Stats overlay complete. bar_img.shape: {bar_img.shape}") # + : DEBUG LOG


                    if render_mode == 'video':
                        observation_images.append(bar_img)
                        print(f"DEBUG: Step {global_step_counter}: Appended bar_img to observation_images. Count: {len(observation_images)}") # + : DEBUG LOG
                    elif render_mode == 'computer': # + : 只有 render_mode 為 'computer' 才顯示視窗
                        print(f"DEBUG: Step {global_step_counter}: Displaying bar_img with cv2.imshow().") # + : DEBUG LOG
                        cv2.imshow("Stats Plot", bar_img)
                        cv2.waitKey(1)
                    # + : render_mode 'human' 不在此分支處理，由 agent.play_episode 或環境內部處理顯示。
                else: # render_frame is None
                    print(f"DEBUG: Step {global_step_counter}: render_frame was None, skipping frame processing and appending/displaying.") # + : DEBUG LOG


                if terminated or truncated:
                    print(f"DEBUG: Step {global_step_counter}: Episode ended. Terminated: {terminated}, Truncated: {truncated}. Breaking loop.") # + : DEBUG LOG
                    break # + : 從 while not (terminated or truncated) 迴圈中跳出

                # - : 原始 break 條件檢查 done
                # if done:
                #     break
                # + : 新版 break 條件已移至 while 迴圈條件檢查 terminated 或 truncated

            print(f"DEBUG: Main agent loop finished after {global_step_counter} steps.") # + : DEBUG LOG


        # write video
        # + : 檢查是否需要渲染影片，並且是否確實收集到了畫面
        if args.agent_type != 'human' and render_mode == 'video':
            print(f"DEBUG: Attempting to render video. Number of frames collected: {len(observation_images)}") # + : DEBUG LOG
            if observation_images: # + : 增加檢查 observation_images 非空
                render_video(observation_images, pool, fps=config['spectrogram_params']['fps'], mux_audio=mux_audio,
                             real_perf=args.real_perf)
                print("DEBUG: Video rendering process initiated.") # + : DEBUG LOG
            else:
                print("DEBUG: No frames collected, skipping video rendering.") # + : DEBUG LOG
        # + : 如果是 'computer' 模式，在結束時保持窗口顯示，直到用戶按下任意鍵
        elif args.agent_type != 'human' and render_mode == 'computer':
            # + : Crude check: See if any image data was collected/processed (more reliable than checking render_frame in isolation)
            if observation_images or (global_step_counter > 0 and render_mode == 'computer'): # + : 檢查是否收集到影片幀 或 在 computer 模式下迴圈運行過
                 print("Rendering complete. Press any key in the plot window to exit.")
                 cv2.waitKey(0)
                 cv2.destroyAllWindows()
                 print("DEBUG: OpenCV windows closed.") # + : DEBUG LOG
            else:
                 print("DEBUG: No frames were displayed in 'computer' mode, or loop didn't run. Exiting.") # + : DEBUG LOG

        # + : 否則 (如 'human' 模式或沒有畫面輸出)，直接退出
        else:
            print(f"DEBUG: Agent type is '{args.agent_type}' or render_mode is not 'video'/'computer'. No final rendering/display step here.") # + : DEBUG LOG

        print("--- test_agent.py script finished successfully ---") # + : DEBUG LOG

    except Exception as e:
        # + : DEBUG: 捕獲任何未處理的異常，記錄到 log 檔案
        print(f"\n--- test_agent.py script encountered an error ---") # + : DEBUG LOG
        print(f"ERROR: An unexpected error occurred: {e}") # + : DEBUG LOG
        import traceback
        traceback.print_exc(file=sys.stdout) # + : 打印詳細的 traceback 到 log 檔案
        print(f"--- test_agent.py script aborted due to error ---") # + : DEBUG LOG

    finally:
        # + : DEBUG: 恢復標準輸出並關閉 log 檔案
        sys.stdout = original_stdout # + : 恢復標準輸出到控制台
        log_file.close() # + : 關閉 log 檔案
        # + : 打印一條消息到原始標準輸出，告知 log 已保存
        print(f"Debug output redirection ended. Log saved to {log_file_path}.")