import numpy as np
def LapNoise(eps):
    beta = 1 / eps
    rand1 = np.random.randint(1, 1000)
    rand2 = np.random.random()
    if (rand1 % 2) == 0:
        noise = beta * np.log(rand2)
    else:
        noise = -beta * np.log(1.0 - rand2)
    return noise

def OneD_DP(data, eps):
    for i in data:
        i += LapNoise(eps)
    return data

def TwoDs_DP(data, eps):
    for i in data:
        for j in i:
            j += LapNoise(eps)
    return data