
import numpy as np
import torch.optim as optim

from ast import literal_eval as make_tuple


def cast_optim_params(optim_params):
    """

    :param optim_params:
    :return:
    """
    for k in optim_params.keys():

        # check if argument is a parameter tuple
        if isinstance(optim_params[k], str) and '(' in optim_params[k]:
            optim_params[k] = make_tuple(optim_params[k])
        else:
            try:
                # - : np.float 已在 NumPy 1.20 中棄用，並在 1.24 中移除，應改用 Python 原生的 float。
                # optim_params[k] = np.float(optim_params[k])
                optim_params[k] = float(optim_params[k])
                # + : 改用 Python 原生的 float 進行型別轉換，以符合 NumPy 更新。
            except:
                pass

    return optim_params


def get_optimizer(optimizer_name, params, **kwargs):
    """
    Compile pytorch optimizer

    :param optimizer_name:
    :param params:
    :param kwargs:
    :return:
    """
    constructor = getattr(optim, optimizer_name)
    optimizer = constructor(params, **kwargs)
    return optimizer
