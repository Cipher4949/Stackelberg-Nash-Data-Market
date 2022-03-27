import math
import numpy as np
from ..Shapley import Shapley

def cal_tau_i_coef(omega_i, lambda_i, tau_common_coef):
    tau_i_coef = tau_common_coef / math.sqrt(omega_i, lambda_i)
    return tau_i_coef

def cal_tau_coef(omega, lambda_, N, m):
    tau_common_coef = 0
    for i in range(m):
        tau_common_coef += math.sqrt(omega[i] / lambda_[i])
    tau_common_coef /= 2 * N
    tau_coef = np.zeros(m)
    for i in range(m):
        tau_coef[i] = cal_tau_i_coef(omega[i], lambda_[i], tau_common_coef)
    return tau_coef

def cal_pM(theta1, rho1, lambda_, m, acc):
    lambda_reciprocal_sum = 0
    for i in range(m):
        lambda_reciprocal_sum += 1 / lambda_[i]
    cons_C = rho1 * acc / 4 * lambda_reciprocal_sum
    cons_D = acc * acc / 2 / theta1 * lambda_reciprocal_sum
    pM = (-cons_D + math.sqrt(cons_D * cons_D + 4 * cons_C *cons_D)) / (2 * cons_C * cons_D)
    return pM