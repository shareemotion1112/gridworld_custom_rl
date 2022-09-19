import torch
import numpy as np
import math
import pandas as pd



def sharpening(prob_arr, temperature=0.01):
    result = np.zeros_like(prob_arr.shape[0])
    
    for i in range(prob_arr.shape[0]):
        tmp = prob_arr[i, :]
        denominator = np.nansum(tmp ** (1/temperature))
        numerator = tmp ** (1/temperature)
        
        if denominator == 0:
            denominator = 1e-10

        result[i, :] = numerator / denominator
    return result

def cal_probability_density_by_gaussian(x, mu, sd):
    return torch.exp( -((x - mu) / sd)**2 / 2 ) / torch.sqrt( torch.tensor(2*math.pi) ) / sd

def remove_column_of_torch_tensor(df_t : torch.tensor, ind : np.int32):
    return df_t[:, np.arange(df_t.shape[1]) != ind]


# def get_prior_probability(D_l, y_name, level):
#     prior_prob = np.zeros((len(level), 1))
#     for ind, lv in enumerate(level):
#         D_l_lv = D_l.loc[D_l[y_name] == lv]
#         prior_prob[ind, 0] = D_l_lv.shape[0] / D_l.shape[0]
#     return prior_prob


def factorize(x : np.array, sd_interval = 0.5, mu_input = None, sd_input = None):
    avg = 0; sd = 0;

    if mu_input is None:
        avg = np.nanmean(x)
    else:
        avg = mu_input

    if sd_input is None:
        sd = np.nanstd(x)
    else:
        sd = sd_input

    x_factor = np.zeros_like(x)

    for i in range(-20, 20, 1):
        x_factor[np.where( np.logical_and( x < (avg + sd_interval * sd * (i + 1)), x >= (avg + sd_interval * sd * i) ) )] = i

    offset = np.nanmin(x_factor)
    x_factor = x_factor - offset
    level = np.unique(x_factor)
    return x_factor, offset, avg, sd, level


def convert_to_real(y_factor, avg, sd, sd_interval, offset):
    y_factor_offset = y_factor + offset
    y = sd * sd_interval * y_factor_offset + avg

    if str(type(y_factor)) == "<class 'numpy.ndarray'>":
        y_update = y + sd * sd_interval * (y_factor - [math.floor(el) for el in list(y_factor)])
    else:
        y_update = y + sd * sd_interval * (y_factor - y_factor.apply(math.floor))

    return np.array(y_update)
    

def MAE(y1, y2):
    return np.mean( np.abs(y1 - y2) )

def MAPE(y1, y2):
    return np.mean(np.abs((y1 - y2) / y1) * 100)