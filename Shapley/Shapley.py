# -*- coding: utf-8 -*-
# Copyright (c) Authors are Hided.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
from multiprocessing import Pool
import numpy as np
from tqdm import  trange


from .utils import (
    eval_utility,  split_permutation_num
)


def mc_shap(x_train, y_train, x_test, y_test, model,
            m, proc_num=1, flag_abs=False) -> np.ndarray:
    """
    Calculating the Shapley value of data points with
    Monte Carlo Method (multi-process supported)

    :param x_train:  features of train dataset
    :param y_train:  labels of train dataset
    :param x_test:   features of test dataset
    :param y_test:   labels of test dataset
    :param model:    the selected model
    :param m:        the permutation number
    :param proc_num: (optional) Assign the proc num with multi-processing
                     support. Defaults to ``1``.
    :param flag_abs: (optional) Whether use the absolution marginal
                     contribution. Defaults to ``False``.
    :return: Shapley value array `sv`
    :rtype: numpy.ndarray
    """

    if proc_num < 0:
        raise ValueError('Invalid proc num.')

    # assign the permutation of each process
    args = split_permutation_num(m, proc_num)
    pool = Pool()
    func = partial(_mc_shap_sub_task, x_train, y_train, x_test,
                   y_test, model, flag_abs)
    ret = pool.map(func, args)
    pool.close()
    pool.join()
    ret_arr = np.asarray(ret)
    return np.sum(ret_arr, axis=0) / m


def _mc_shap_sub_task(x_train, y_train, x_test, y_test, model,
                      flag_abs, local_m) -> np.ndarray:
    local_state = np.random.RandomState(None)

    n = len(y_train)
    sv = np.zeros(n)
    idxs = np.arange(n)
    for _ in trange(local_m):
        local_state.shuffle(idxs)
        old_u = 0
        for j in range(1, n + 1):
            temp_x, temp_y = x_train[idxs[:j]], y_train[idxs[:j]]
            temp_u = eval_utility(temp_x, temp_y, x_test, y_test, model)
            contribution = ((temp_u - old_u) if not flag_abs
                            else abs(temp_u - old_u))
            sv[idxs[j - 1]] += contribution
            old_u = temp_u
    return sv
