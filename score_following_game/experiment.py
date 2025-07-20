import getpass
import os
import pickle
import torch
import numpy as np

from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv # 雖然不用，但保留導入以備未來切換
from score_following_game.agents.networks_utils import get_network
from score_following_game.agents.optim_utils import get_optimizer, cast_optim_params
# --- 修改點：恢復了所有必要的導入 ---
from score_following_game.data_processing.data_pools import get_data_pools, get_shared_cache_pools
from score_following_game.data_processing.data_production import create_song_producer, create_song_cache
# --- 修改點結束 ---
from score_following_game.data_processing.utils import load_game_config
from score_following_game.evaluation.evaluation import PerformanceEvaluator as Evaluator
from score_following_game.experiment_utils import setup_parser, setup_logger, setup_agent, make_env_tismir, get_make_env
from score_following_game.reinforcement_learning.torch_extentions.optim.lr_scheduler import RefinementLRScheduler
from score_following_game.reinforcement_learning.algorithms.models import Model
from time import gmtime, strftime

if __name__ == '__main__':
    """ main """

    parser = setup_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    # compile unique result folder
    time_stamp = strftime("%Y%m%d_%H%M%S", gmtime())
    tr_set = os.path.basename(args.train_set)
    config_name = os.path.basename(args.game_config).split(".yaml")[0]
    user = getpass.getuser()
    exp_dir = args.agent + "-" + args.net + "-" + tr_set + "-" + config_name + "_" + time_stamp + "-" + user

    args.experiment_directory = exp_dir

    # create model parameter directory
    args.dump_dir = os.path.join(args.param_root, exp_dir)
    if not os.path.exists(args.dump_dir):
        os.makedirs(args.dump_dir)

    args.log_dir = os.path.join(args.log_root, args.experiment_directory)

    # initialize tensorboard logger
    log_writer = None if args.no_log else setup_logger(args=args)

    args.log_writer = log_writer

    # cast optimizer parameters to float
    args.optim_params = cast_optim_params(args.optim_params)

    # load game config
    config = load_game_config(args.game_config)

    # --- 核心修改：從 "一次性全加載" 恢復到 "Producer-Cache" 模式 ---

    # --- 步驟 1: 刪除或註解掉導致 OOM 的程式碼 ---
    # print("Loading all training data into memory upfront (single-process)...")
    # rl_pools = get_data_pools(config, directory=args.train_set, real_perf=args.real_perf, n_worker=args.n_worker)
    # print(f"Successfully loaded {len(rl_pools)} songs.")
    
    # +++ 步驟 2: 恢復 producer-cache 機制，這是解決記憶體問題的關鍵 +++
    print("INFO: Initializing producer-cache data loading pipeline...")
    CACHE_SIZE = 50  # 記憶體中只保留 50 首歌的數據
    cache = create_song_cache(CACHE_SIZE)
    # 創建一個背景進程，負責讀取和預處理歌曲
    producer_process = create_song_producer(cache, config=config, directory=args.train_set, real_perf=args.real_perf)
    
    # 雖然是單行程訓練，但我們仍然從 cache 中獲取一個 pool 給這個行程用
    # 注意：nr_pools 參數設為 1，因為我們只創建一個環境
    rl_pools = get_shared_cache_pools(cache, config, nr_pools=1, directory=args.train_set)
    
    producer_process.start()  # 啟動背景進程
    print("INFO: Background data producer started. Memory usage will now be stable.")
    # --- 核心修改結束 ---


    env_fnc = make_env_tismir

    if args.agent == 'reinforce':
        # Reinforce Agent 的部分保持不變，它本來就是單環境
        env = get_make_env(rl_pools[0], config, env_fnc, render_mode=None)()
    else:
        # +++ 步驟 3: 維持使用單一、非平行的環境來驗證記憶體問題 +++
        # 我們不恢復 ShmemVecEnv，繼續使用這個穩定的單環境模式
        # env = ShmemVecEnv(...) # 保持註解
        
        print("INFO: Running in single-environment mode to ensure stability.")
        env = get_make_env(rl_pools[0], config, env_fnc, render_mode=None)()

    # compile network architecture
    net = get_network('networks_sheet_spec', args.net, env.action_space.n,
                      shapes=dict(perf_shape=config['spec_shape'], score_shape=config['sheet_shape']))

    # load initial parameters
    if args.ini_params:
        net.load_state_dict(torch.load(args.ini_params))

    # initialize optimizer
    optimizer = get_optimizer(args.optim, net.parameters(), **args.optim_params)

    # initialize model
    model = Model(net, optimizer, max_grad_norm=args.max_grad_norm, value_coef=args.value_coef,
                  entropy_coef=args.entropy_coef)

    # initialize refinement scheduler
    lr_scheduler = RefinementLRScheduler(optimizer=optimizer, model=model, n_refinement_steps=args.max_refinements,
                                         patience=args.patience, learn_rate_multiplier=args.lr_multiplier,
                                         high_is_better=not args.low_is_better)
    
    # initialize model evaluation
    # 注意：評估集通常較小，一次性加載是可以接受的
    evaluation_pools = get_data_pools(config, directory=args.eval_set, real_perf=args.real_perf)

    evaluator = Evaluator(env_fnc, evaluation_pools, config=config, trials=args.eval_trials, render_mode=None)

    args.model = model
    args.env = env
    args.lr_scheduler = lr_scheduler
    args.evaluator = evaluator
    args.n_actions = 1

    # +++ 步驟 4: 維持 n_worker=1 的設置 +++
    # 因為我們使用的是單一環境，這個覆寫仍然是必要的，以確保 Agent 內部緩衝區大小正確
    print(f"INFO: Overriding n_worker from {args.n_worker} to 1 for single-environment training.")
    args.n_worker = 1

    # 現在，傳遞給 setup_agent 的 args 中，n_worker 的值就是 1
    agent = setup_agent(args=args)

    max_updates = args.max_updates * args.t_max
    agent.train(env, max_updates)

    # +++ 步驟 5: 正確清理 producer 和 cache +++
    # store the song history to a file
    if not args.no_log:
        with open(os.path.join(args.log_dir, 'song_history.pkl'), 'wb') as f:
            # 修正之前發現的 bug，使用 cache 變數而不是 producer_process.cache
            print("INFO: Saving song history from cache...")
            pickle.dump(cache.get_history(), f)

    # stop the producer thread
    print("INFO: Terminating background data producer process.")
    producer_process.terminate()  # 結束背景進程

    if not args.no_log:
        log_writer.close()