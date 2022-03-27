import math
import numpy as np
from ..Shapley import Shapley
from ..DP import DP

#Backward part
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


#Forward part
def cal_tau_from_eps(eps):
    tau = 2 * math.acos(1 / (eps + 1) ) / math.pi
    return tau

def cal_eps_from_tau(tau):
    eps = math.cos(1 / (tau * math.pi / 2) )
    return eps

def cal_v(rho):
    return math.log(1 + rho, math.e)

def cal_phi(chi, tau, acc, theta1, theta2, m):
    sum = 0
    for i in range(m):
        sum += chi[i] * tau[i]
    return theta1 * cal_v(sum) + theta2 * cal_v(acc)

def cal_Phi(chi, tau, acc, theta1, theta2, m, pM, qM):
    return cal_phi(chi, tau, acc, theta1, theta2, m) - pM * qM

def cal_traincost(chi, acc, sigma1, sigma2):
    sum = np.sum(chi)
    return sigma1 * sum**2 + sigma2 * acc**2

def cal_Omega(pM, qM, pD, qD, chi, acc, sigma1, sigma2):
    return pM * qM - pD * qD - cal_traincost(chi, acc, sigma1, sigma2)

def cal_seller_loss(tau, chi, lambda_):
    return lambda_ * (tau * chi)**2

def cal_Psi_i(pD, tau_i, chi_i, lambda_i):
    return pD * chi_i * tau_i - cal_seller_loss(tau_i, chi_i, lambda_i)

#whole workflow
def Stackelberg_Nash_DataMarket(omega):
    return
