import os
from platform import platform
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
print(f"gpu acceleration : {torch.has_mps}")
import numpy as np
import pandas as pd
import platform

from SelfTraining.myMath import factorize
from sklearn.model_selection import train_test_split

def create_random_df(n_row = 1000, n_col = 100):
    arr = np.zeros((n_row, n_col))

    for i in range(arr.shape[0]):
        # arr[i, :] = [np.random.random() for i in range(n_col)]
        arr[i, :] = np.random.uniform(-1, 1, n_col)

    result = pd.DataFrame(arr)
    result.columns = [str(k) for k in np.arange(result.shape[1])]
    return result



from SelfTraining.Selftraining import Selftranining


D_l = create_random_df(n_row = 1000, n_col = 101)
D_u = create_random_df(n_row = 10000, n_col = 100)
y_name = '100'


x_factor, offset, avg, sd, level = factorize(D_l[y_name])

D_l[y_name] = x_factor

trainX, testX = train_test_split(D_l, test_size=0.2)
test_indx = testX.index

xgboost_hyperparameter = {'learning_rate': 0.1, 
                        'n_estimators': 1000, 
                        'max_depth': 2, 
                        'min_child_weight': 1}

gnb = Selftranining(D_l=D_l, D_u=D_u, y_name=y_name, levels=list(level), test_indx=test_indx, xgboost_hyperparameter=xgboost_hyperparameter)

# gnb.cal_mu_and_sd_matrix()


if platform.platform()[:5] == 'macOS':
    device = torch.device('mps')
else:
    device = torch.device('cuda:0')

D_l_torch = torch.tensor(D_l.values, dtype = torch.float32, device = device)

res1, res2, res3 = gnb.get_probability_GNB(D_l_torch)
print(res1)
print(res2)
print(res3)