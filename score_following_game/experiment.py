import getpass
import os
import pickle
import torch
# - : 移除不必要的 numpy 匯入，因未使用 np.int, np.float 等廢棄別名
# import numpy as np
# + : 導入 numpy 模組
import numpy as np

# - : ShmemVecEnv 是來自舊版 OpenAI Baselines，未來可能需要替換為 Gymnasium 或 Stable-Baselines3 的向量化環境以確保完全兼容性。
#   但根據指示，目前僅檢查，不更換 ShmemVecEnv 實作。
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from score_following_game.agents.networks_utils import get_network
from score_following_game.agents.optim_utils import get_optimizer, cast_optim_params
from score_following_game.data_processing.data_pools import get_data_pools, get_shared_cache_pools
from score_following_game.data_processing.data_production import create_song_producer, create_song_cache
from score_following_game.data_processing.utils import load_game_config
# - : Evaluator 內部可能需要進行 Gym -> Gymnasium 的遷移，此處僅為匯入
from score_following_game.evaluation.evaluation import PerformanceEvaluator as Evaluator
# - : setup_parser, setup_logger, setup_agent, make_env_tismir, get_make_env 內部可能需要進行 Gym -> Gymnasium 的遷移，此處僅為匯入
from score_following_game.experiment_utils import setup_parser, setup_logger, setup_agent, make_env_tismir, get_make_env
from score_following_game.reinforcement_learning.torch_extentions.optim.lr_scheduler import RefinementLRScheduler
# - : Model 內部可能需要進行 Gym -> Gymnasium 的遷移，此處僅為匯入
from score_following_game.reinforcement_learning.algorithms.models import Model
from time import gmtime, strftime

if __name__ == '__main__':
    """ main """

    parser = setup_parser()
    args = parser.parse_args()

    # - : np.random.seed 雖非廢棄，但新版建議使用 np.random.default_rng。此處保留原有用法以維持邏輯不變。
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

    # initialize song cache, producer and data pools
    CACHE_SIZE = 50
    cache = create_song_cache(CACHE_SIZE)
    producer_process = create_song_producer(cache, config=config, directory=args.train_set, real_perf=args.real_perf)
    rl_pools = get_shared_cache_pools(cache, config, nr_pools=args.n_worker, directory=args.train_set)

    producer_process.start()

    # + : make_env_tismir 和 get_make_env 預期在 experiment_utils.py 中已完成 Gym -> Gymnasium 的遷移
    env_fnc = make_env_tismir

    if args.agent == 'reinforce':
        # + : 假設 get_make_env 返回的環境創建函數已適配 Gymnasium API (reset/step)
        env = get_make_env(rl_pools[0], config, env_fnc, render_mode=None)()
    else:
        # + : ShmemVecEnv 需要能夠處理由 get_make_env 返回的 Gymnasium 環境。
        # + : 假設 get_make_env 返回的環境創建函數已適配 Gymnasium API (reset/step)
        # + : 注意：舊版 baselines 的 ShmemVecEnv 可能無法完全兼容 Gymnasium 的 reset/step 返回值 (obs, info / obs, reward, terminated, truncated, info)
        # + : 這部分需要實際運行測試確認，或考慮替換為 Gymnasium / SB3 的 VecEnv。根據指示，目前維持原樣。
        env = ShmemVecEnv([get_make_env(rl_pools[i], config, env_fnc, render_mode=None) for i in range(args.n_worker)])

    # compile network architecture
    # + : env.action_space.n 的用法在 Gym 和 Gymnasium 中兼容
    net = get_network('networks_sheet_spec', args.net, env.action_space.n,
                      shapes=dict(perf_shape=config['spec_shape'], score_shape=config['sheet_shape']))

    # load initial parameters
    if args.ini_params:
        net.load_state_dict(torch.load(args.ini_params))

    # initialize optimizer
    optimizer = get_optimizer(args.optim, net.parameters(), **args.optim_params)

    # initialize model
    # + : Model 類別預期在其內部 (reinforcement_learning/algorithms/models.py) 已適配 Gymnasium API
    model = Model(net, optimizer, max_grad_norm=args.max_grad_norm, value_coef=args.value_coef,
                  entropy_coef=args.entropy_coef)

    # initialize refinement scheduler
    lr_scheduler = RefinementLRScheduler(optimizer=optimizer, model=model, n_refinement_steps=args.max_refinements,
                                         patience=args.patience, learn_rate_multiplier=args.lr_multiplier,
                                         high_is_better=not args.low_is_better)

    # use cuda if available
    if args.use_cuda:
        model.cuda()

    # initialize model evaluation
    evaluation_pools = get_data_pools(config, directory=args.eval_set, real_perf=args.real_perf)

    # + : Evaluator 類別預期在其內部 (evaluation/evaluation.py) 已適配 Gymnasium API
    evaluator = Evaluator(env_fnc, evaluation_pools, config=config, trials=args.eval_trials, render_mode=None)

    args.model = model
    args.env = env
    args.lr_scheduler = lr_scheduler
    args.evaluator = evaluator
    args.n_actions = 1
    # + : setup_agent 預期在其內部 (experiment_utils.py 或相關 agent 檔案) 已適配 Gymnasium API
    agent = setup_agent(args=args)

    max_updates = args.max_updates * args.t_max
    # + : agent.train() 方法預期在其內部 (reinforcement_learning/algorithms/agent.py 或具體演算法檔案) 已適配 Gymnasium 的 env.step/reset
    agent.train(env, max_updates)

    # store the song history to a file
    if not args.no_log:
        with open(os.path.join(args.log_dir, 'song_history.pkl'), 'wb') as f:
            pickle.dump(producer_process.cache.get_history(), f)

    # stop the producer thread
    producer_process.terminate()

    if not args.no_log:
        log_writer.close()