from hashlib import new
import sys
sys.path.append("..")
import math
import numpy as np
from sklearn import metrics
from dynashap.dynamic import mc_shap
from DP.DP import LapNoise, OneD_DP


#Backward part
def cal_tau_i_coef(omega_i, lambda_i, tau_common_coef):
    tau_i_coef = tau_common_coef / math.sqrt(omega_i * lambda_i)
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

def cal_chi(omega, tau, m, N):
    numerator = np.zeros(m)
    denominator = 0
    for i in range(m):
        numerator[i] = omega[i] * tau[i]
        denominator += numerator[i]
    chi = np.zeros(m)
    denominator /= N
    for i in range(m):
        chi[i] = numerator[i] / denominator
    return chi

def cal_qD(chi, tau, m):
    qD = 0
    for i in range(m):
        qD += chi[i] * tau[i]
    return qD

def cal_qM(chi, tau, m, acc):
    qM = 0
    for i in range(m):
        qM += chi[i] * tau[i]
    return qM * acc

def cal_v(rho, attr):
    return math.log(1 + rho * attr)

def cal_phi(chi, tau, acc, theta1, theta2, m, rho1, rho2):
    sum = cal_qD(chi, tau, m)
    return theta1 * cal_v(rho1, sum) + theta2 * cal_v(rho2, acc)

def cal_Phi(chi, tau, acc, theta1, theta2, m, pM, qM, rho1, rho2):
    return cal_phi(chi, tau, acc, theta1, theta2, m, rho1, rho2) - pM * qM

def cal_traincost(chi, acc, sigma1, sigma2):
    sum = np.sum(chi)
    return sigma1 * sum**2 + sigma2 * acc**2

def cal_Omega(pM, qM, pD, qD, chi, acc, sigma1, sigma2):
    return pM * qM - pD * qD - cal_traincost(chi, acc, sigma1, sigma2)

def cal_seller_loss(tau, chi, lambda_):
    return lambda_ * (tau * chi)**2

def cal_Psi_i(pD, tau_i, chi_i, lambda_i):
    return pD * chi_i * tau_i - cal_seller_loss(tau_i, chi_i, lambda_i)

#main workflow
def Stackelberg_Nash_DataMarket(x_test, y_test,#test_data
                                theta1, theta2, rho1, rho2, acc,#buyer
                                sigma1, sigma2,#broker
                                lambda_, omega, m, N, x_in, y_in,#seller
                                model):
                                #x_train is a 3D-list which contains m matrices
                                #each matrices represents each seller's data
    tau_coef = cal_tau_coef(omega, lambda_, N, m)
    pM = cal_pM(theta1, rho1, lambda_, m, acc)
    pD = pM * acc / 2
    tau = tau_coef * pD
    epss = np.zeros(m)
    for i in range(m):
        epss[i] = cal_eps_from_tau(tau[i])
    chi = cal_chi(omega, tau, m, N)

    #generate train data based on chi and epsilon
    x_train = np.zeros((N, len(x_in[0][0])))
    y_train = np.zeros(N)
    idx = 0
    for i in range(m):
        for j in range(int(chi[i])):
            x_train[idx] = OneD_DP(x_in[i][j], epss[i])
            y_train[idx] = y_in[i][j] + LapNoise(epss[i])
            idx += 1
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    true_acc = metrics.explained_variance_score(y_test, y_pred)
    qM = cal_qM(chi, tau, m, true_acc)
    qD = cal_qD(chi, tau, m)
    Phi = cal_Phi(chi, tau, true_acc, theta1, theta2, m, pM, qM, rho1, rho2)#buyer
    Omega = cal_Omega(pM, qM, pD, qD, chi, true_acc, sigma1, sigma2)#broker
    Psi = np.zeros(m)
    for i in range(m):
        Psi[i] = cal_Psi_i(pD, tau[i], chi[i], lambda_[i])
    data_shapley = mc_shap(x_train, y_train, x_test, y_test, model, 100)
    min_data_shapley = np.min(data_shapley) - 0.00001
    new_omega = np.zeros(m)
    idx = 0
    for i in range(m):
        for j in range(int(chi[i])):
            new_omega[i] += data_shapley[idx] - min_data_shapley
            idx += 1
    max_seller_shapley = np.max(new_omega)
    for i in range(m):
        new_omega[i] /= max_seller_shapley
    print(new_omega)
    return Phi, Omega, Psi, new_omega
    #return profits and refresh omega(weight)
