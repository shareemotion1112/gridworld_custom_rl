import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from myMath import factorize


def create_random_df(n_row = 1000, n_col = 100):
    arr = np.zeros((n_row, n_col))

    for i in range(arr.shape[0]):
        # arr[i, :] = [np.random.random() for i in range(n_col)]
        arr[i, :] = np.random.uniform(-1, 1, n_col)

    result = pd.DataFrame(arr)
    result.columns = [str(k) for k in np.arange(result.shape[1])]
    return result



from Selftraining import Selftraining


D_l = create_random_df(n_row = 1000, n_col = 101)
D_u = create_random_df(n_row = 10000, n_col = 100)
y_name = '100'


x_factor, offset, avg, sd, level = factorize(D_l[y_name])

D_l[y_name] = x_factor

gnb = Selftraining(D_l=D_l, D_u=D_u, y_name=y_name, levels=list(level))

# gnb.cal_mu_and_sd_matrix()

D_l_torch = torch.tensor(D_l.values, dtype = torch.float32, device = torch.device("cuda:0"))

res1, res2, res3 = gnb.get_prob_gnb(D_l_torch)
print(res1)
print(res2)
print(res3)