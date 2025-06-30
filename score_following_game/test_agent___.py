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


# render mode for the environment ('human', 'computer', 'video')
# render_mode = 'computer'
render_mode = 'video'
mux_audio = True

if __name__ == "__main__":
    """ main """

    parser = setup_evaluation_parser()
    parser.add_argument('--agent_type', help='which agent to test [rl|optimal|human].',
                        choices=['rl', 'optimal', 'human'], type=str, default="rl")
    parser.add_argument('--plot_stats', help='plot additional statistics.', action='store_true', default=False)
    parser.add_argument('--plot_ig', help='plot integrated gradients.', action='store_true', default=False)

    args = parser.parse_args()


    if args.agent_type == 'rl':
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

        # --- Infer args.net if not provided ---
        if args.net is None:
            # - : 原始 args.net 推斷邏輯，依賴可能不穩定的 exp_name_old
            # args.net = exp_name_old.split('-')[1]
            # + : 使用新的、更穩健的 exp_name 推斷 args.net，並提供預設值
            exp_parts_for_net = exp_name.split('-')
            if len(exp_parts_for_net) > 1:
                args.net = exp_parts_for_net[1]
                # + : args.net 已從 exp_name 推斷 (e.g., if exp_name is "rl-network_type-...")
            else:
                args.net = 'default' # + : exp_name 格式不符，args.net 設為預設值
                # + : 可選擇性加入警告，若 'default' 網路不存在或不適用。
                # print(f"Info: Could not infer network type from exp_name '{exp_name}'. Using default network 'default'.")

        # --- Infer args.game_config if not provided ---
        if args.game_config is None:
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
                if args.params and exp_name != 'manual-test': # + : 避免對 'manual-test' 或未提供 --params 的情況發出警告
                     print(f"Warning: Could not infer game_config from experiment name '{exp_name}' "
                           f"(derived from --params='{args.params}'). Expected format for automatic inference "
                           f"is like '<anyprefix>-<network>-<features>-<configbasename>_...'. "
                           f"If --game_config was not specified and is required by the program, this may lead to an error.")
                # + : args.game_config 維持為 None。後續 load_game_config(None) 可能會報錯，這是預期的。

    config = load_game_config(args.game_config)

    if args.agent_type == 'optimal':
        # the action space for the optimal agent needs to be continuous
        config['actions'] = []

    pool = get_single_song_pool(
        dict(config=config, song_name=args.piece, directory=args.data_set, real_perf=args.real_perf))

    observation_images = []

    # initialize environment
    env = make_env_tismir(pool, config, render_mode='human' if args.agent_type == 'human' else 'video')

    if args.agent_type == 'human' or args.agent_type == 'optimal':

        agent = HumanAgent(pool) if args.agent_type == 'human' else OptimalAgent(pool)
        # Note: The play_episode function within the agent itself needs to be updated for Gymnasium API if it interacts directly with env.step/reset
        alignment_errors, action_sequence, observation_images, episode_reward = agent.play_episode(env, render_mode)

    else:

        # compile network architecture
        n_actions = len(config["actions"])
        net = get_network("networks_sheet_spec", args.net, n_actions=n_actions,
                          shapes=dict(perf_shape=config['spec_shape'], score_shape=config['sheet_shape']))

        # load network parameters
        net.load_state_dict(torch.load(args.params))

        # set model to evaluation mode
        net.eval()

        # create agent
        use_cuda = torch.cuda.is_available()

        model = Model(net, optimizer=None)

        agent = initialize_trained_agent(model, use_cuda=use_cuda, deterministic=False)

        observation_images = []

        # get observations
        episode_reward = 0
        # - : gym reset() API 返回單一 observation
        # observation = env.reset()
        observation, info = env.reset() # type: ignore
        # + : gymnasium reset() API 返回 observation 和 info dict

        reward = 0
        # - : gym step() API 返回的 done 變數已被棄用
        # done = False
        terminated = False # + : gymnasium 使用 terminated 標示因環境內部條件結束 (例如達到目標)
        truncated = False # + : gymnasium 使用 truncated 標示因外部條件結束 (例如時間限制)

        if args.plot_ig:
            IG = IntegratedGradients(net, 'cuda', steps=20)
            # Create a separate 'plain' environment for Integrated Gradients calculation
            # Note: The underlying make_env_tismir or get_make_env might need internal updates for Gymnasium
            plain_env = get_make_env(copy.deepcopy(pool), config, make_env_fnc=make_env_tismir, render_mode=render_mode)() # type: ignore

            # Unwrap environment if necessary (e.g., if wrappers are applied)
            while not isinstance(plain_env, ScoreFollowingEnv):
                plain_env = plain_env.env # type: ignore

            # - : gym reset() API
            # plain_env.reset()
            obs_plain, info_plain = plain_env.reset() # type: ignore
            # + : gymnasium reset() API 返回 observation 和 info dict

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

        # Loop until the episode ends (terminated or truncated)
        # - : 原始迴圈條件使用 done
        # while True: # Implicitly checked done inside
        while not (terminated or truncated): # + : 迴圈條件改為檢查 terminated 或 truncated
            # choose action
            action = agent.select_action(observation)

            # perform step and observe
            # - : gym step() API 返回四元組 (obs, reward, done, info)
            # observation, reward, done, info = env.step(action)
            observation, reward, terminated, truncated, info = env.step(action) # type: ignore
            # + : gymnasium step() API 返回五元組 (obs, reward, terminated, truncated, info)

            episode_reward += reward

            if env.obs_image is not None:

                bar_img = env.obs_image

                if args.plot_ig or args.plot_stats:
                    observation_tensor = agent.prepare_state(observation)
                    model_return = model(observation_tensor)

                    bar_height = env.obs_image.shape[0]
                    # - : NumPy 廢棄型別別名 np.uint8 -> np.uint8 (維持不變，非廢棄)
                    spacer = 255 * np.ones((bar_height, 5, 3), np.uint8)
                    # + : np.uint8 非 NumPy 廢棄型別，維持原樣

                    if args.plot_ig:
                        # Step the plain environment to get original observations for IG
                        # - : gym step() API 返回四元組
                        # obs_org, r, d, _ = plain_env.step(action)
                        obs_org, r, terminated_plain, truncated_plain, info_plain = plain_env.step(action) # type: ignore
                        # + : gymnasium step() API 返回五元組，更新解構變數

                        # invert and create grayscale score
                        org_score = 1 - obs_org['score'][0]
                        # - : NumPy 廢棄型別別名 np.uint8 -> np.uint8 (維持不變，非廢棄)
                        org_score = np.uint8(cm.gray(org_score) * 255)
                        # + : np.uint8 非 NumPy 廢棄型別，維持原樣

                        # create grayscale perf
                        # - : NumPy 廢棄型別別名 np.uint8 -> np.uint8 (維持不變，非廢棄)
                        org_perf = np.uint8(cm.gray(obs_org['perf'][0]) * 255)
                        # + : np.uint8 非 NumPy 廢棄型別，維持原樣

                        # get gradients
                        guided_score_grads, guided_perf_grads = IG.generate_gradients([observation_tensor['perf'],
                                                                                       observation_tensor['score']])
                        # prepare saliency map for score and delta-score
                        grads_score = guided_score_grads[0]
                        grads_score = prepare_grad_for_render(grads_score, (config['score_shape'][2], config['score_shape'][1]), norm, cmap)

                        # prepare saliency map for performance and delta-performance
                        grads_perf = guided_perf_grads[0]
                        grads_perf = prepare_grad_for_render(grads_perf, (config['perf_shape'][2], config['perf_shape'][1]),
                                                             norm, cmap)

                        # add score gradients to score
                        added_image_score = cv2.addWeighted(grads_score, 1.0, org_score[:, :, :-1], 0.4, 0)
                        added_image_score = prepare_sheet_for_render(added_image_score, plain_env.resz_x, plain_env.resz_y, transform_to_bgr=False)

                        # add performance gradients to performance
                        added_image_perf = cv2.addWeighted(grads_perf, 1.0, org_perf[:, :, :-1], 0.4, 0)
                        added_image_perf = prepare_spec_for_render(added_image_perf, plain_env.resz_spec, transform_to_bgr=False)

                        org_img = copy.copy(env.obs_image)
                        env.obs_image[0:added_image_score.shape[0], 0:added_image_score.shape[1], :] = added_image_score

                        c0 = env.obs_image.shape[1] // 2 - added_image_perf.shape[1] // 2
                        c1 = c0 + added_image_perf.shape[1]
                        env.obs_image[added_image_score.shape[0]:, c0:c1, :] = added_image_perf

                        bar_img = np.concatenate((bar_img, spacer, org_img), axis=1)

                    if args.plot_stats:

                        value = model_return['value'].detach().cpu().item()
                        values.append(value)
                        tempo_curve.append(pool.sheet_speed)

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

                        bar_img = np.concatenate((bar_img, spacer, value_bgr, spacer, speed_bgr, spacer, error_bgr, spacer,
                                                  score_bgr, spacer, pot_score_bgr), axis=1)

                if render_mode == 'video':
                    observation_images.append(bar_img)
                else:
                    cv2.imshow("Stats Plot", bar_img)
                    cv2.waitKey(1)

            # - : 原始 break 條件檢查 done
            # if done:
            #     break
            # + : 新版 break 條件已移至 while 迴圈條件檢查 terminated 或 truncated

    # write video
    if args.agent_type != 'human' and render_mode == 'video':
        render_video(observation_images, pool, fps=config['spectrogram_params']['fps'], mux_audio=mux_audio,
                     real_perf=args.real_perf)
