# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 08:09:57 2022

@author: coolc
"""
import os
root_dir = os.path.dirname(os.path.realpath('__file__'))
import sys
sys.path.append(root_dir)

from copyreg import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
from myMath import *
import torch
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import scipy



class Selftranining:

    def __init__(self, 
                D_l, D_u, y_name, test_indx, xgboost_hyperparameter, levels : list, \
                dnorm_fix = 1e2, \
                adjust_limit_of_n_sample = 10, \
                sd_threshold = 1e-5, \
                is_adjust_dnorm_fix = True, \
                is_save_file = True, \
                is_flexmatch = False, \
                use_interaction_features = False):

        self.dnorm_fix = dnorm_fix
        self.adjust_limit_of_n_sample = adjust_limit_of_n_sample
        self.sd_threshold = sd_threshold
        self.is_adjust_dnorm_fix = is_adjust_dnorm_fix
        
        self.cuda_device = torch.device('cuda:0')

        # convert to torch.tensor of D_l and D_u
        self.D_l = D_l; self.D_u = D_u
        self.D_l_torch = torch.tensor(D_l.values, dtype=torch.float64, device = self.cuda_device)
        self.D_u_torch = torch.tensor(D_u.values, dtype=torch.float64, device = self.cuda_device)


        # y_name, and levels
        self.y_name = y_name
        self.y_ind = np.where(self.D_l.columns == self.y_name)[0][0]
        self.levels = levels

        # mu and sd
        self.mu_matrix = None
        self.sd_matrix = None
        self.mu_matrix_torch_D_l = torch.zeros((len(self.levels), self.D_l.shape[1] - 1), dtype=torch.float64, device=self.cuda_device)
        self.sd_matrix_torch_D_l = torch.zeros((len(self.levels), self.D_l.shape[1] - 1), dtype=torch.float64, device=self.cuda_device)
        # D_u는 Y값이 없기 때문에 mu, sd matrix를 만들 수 없음


        self.test_indx = test_indx
        self.xgboost_hyperparameter = xgboost_hyperparameter
        
        self.is_flexmatch = is_flexmatch

        self.root_dir = None
        self.file_name = ""
        self.model_filename = ""
        self.is_save_file = is_save_file
        self.use_interfaction_features = use_interaction_features

    
    def predict_by_GNB(self, data_torch : torch.tensor, prior_prob_matrix_torch: torch.tensor):
        """
        prior_prob_matrix can be calculated by get_prior_prob in myMath
        """

        y_logs = torch.log(prior_prob_matrix_torch).reshape(1, -1).repeat(data_torch.shape[0], 1)

        decision_matrix_torch, prob_row_norm_torch = self.get_decision_and_probability_per_class(data_torch, y_logs)

        # print(f"y logs : {y_logs}, {y_logs.shape}")

        return decision_matrix_torch.cpu().numpy(), prob_row_norm_torch.cpu().numpy()


    def calculate_prior_probability(self):
        """
        calculate prior probaility.
        also, calculate mu and sd matrix of labeled data
        """
        y_prob_torch = torch.zeros( (len(self.levels), 1), dtype=torch.float64, device = self.cuda_device)
        
        for ind, lv in enumerate(self.levels):
            data_lv_torch = self.D_l_torch[self.D_l_torch[:, self.y_ind] == lv, :]           

            y_prob_torch[ind, 0] = ( data_lv_torch.shape[0] / self.D_l_torch.shape[0] )           

            data_lv_torch_wo_Y = remove_column_of_torch_tensor(data_lv_torch, self.y_ind)

            self.mu_matrix_torch_D_l[ind, :] = torch.mean(data_lv_torch_wo_Y, 0, False)
            self.sd_matrix_torch_D_l[ind, :] = torch.std(data_lv_torch_wo_Y, 0, False)

        return y_prob_torch


    def get_decision_and_probability_per_class(self, data_unlabeled : torch.tensor, y_logs_torch : torch.tensor):
        prob_rows_torch = torch.zeros((data_unlabeled.shape[0], len(self.levels)), dtype=torch.float64, device=self.cuda_device)
        decision_matrix_torch = torch.zeros( data_unlabeled.shape[0], dtype=torch.float64, device=self.cuda_device)

        # calculate the probability belong to certain class
        for i in range(len(self.levels)):
            prob_tmps_torch = cal_probability_density_by_gaussian(data_unlabeled, self.mu_matrix_torch_D_l[i, :], self.sd_matrix_torch_D_l[i, :])
            prob_tmps_torch = torch.where(prob_tmps_torch != 0, prob_tmps_torch, 1e-4)
            prob_rows_torch[:, i] = torch.nansum(torch.log(prob_tmps_torch), dim = 1)

        prob_row_tmp_torch = torch.exp( (prob_rows_torch + y_logs_torch) / self.dnorm_fix )

        # if dnorm_fix is too small/high, adjust dnorm fix parameter will help you.
        if self.is_adjust_dnorm_fix == True:
            iter = 0; limit = 5
            while torch.max(prob_row_tmp_torch)  > 0.2 and torch.max(prob_row_tmp_torch) < 0.99:
                if torch.max(prob_row_tmp_torch) < 0.2:
                    self.dnorm_fix = self.dnorm_fix * 5
                elif torch.max(prob_row_tmp_torch) > 0.99:
                    self.dnorm_fix = self.dnorm_fix / 5
                prob_row_tmp_torch = torch.exp( (prob_rows_torch + y_logs_torch) / self.dnorm_fix )
                if iter > limit:
                    break;
                iter += 1

        sum_arr = torch.nansum(prob_row_tmp_torch, dim = 1).reshape(-1, 1)        
        sum_arr_row = sum_arr.repeat(1, len(self.levels))

        prob_row_norm_torch = prob_row_tmp_torch / sum_arr_row

        decision_matrix_torch = torch.argmax(prob_row_norm_torch, 1)

        return decision_matrix_torch, prob_row_norm_torch


    def get_probability_GNB(self, data_torch : torch.tensor):

        self.sd_matrix_torch_D_l = torch.where(self.sd_matrix_torch_D_l == 0, self.sd_threshold, self.sd_matrix_torch_D_l)

        # cal log(prior_prob)
        y_prob_torch = self.calculate_prior_probability()
        y_prob_torch = torch.where(y_prob_torch == torch.tensor([0], dtype=torch.float64, device=self.cuda_device), \
                                    torch.tensor([1], dtype=torch.float64, device=self.cuda_device), y_prob_torch)        
        y_logs_torch = torch.log(y_prob_torch.reshape(1, -1)).repeat(data_torch.shape[0], 1)

        # cal decision matrix
        data_torch_wo_Y = remove_column_of_torch_tensor(data_torch, self.y_ind)
        decision_matrix_torch, prob_row_norm_torch = self.get_decision_and_probability_per_class(data_torch_wo_Y, y_logs_torch)

        # make result
        result = {'prob' : prob_row_norm_torch.cpu().numpy(), 'class' : decision_matrix_torch.cpu().numpy()}

        return result, self.mu_matrix_torch_D_l, self.sd_matrix_torch_D_l


    def get_good_index_array(prob_array: pd.DataFrame, cut:np.float_):
        max_array = prob_array.max(axis = 1)
        return max_array.index[max_array > cut]

    def get_new_D_l_batch(self, D_l_batch_new, D_u_batch, good_indx, y_class):

        D_u_good = D_u_batch.loc[good_indx, :]
        y_pseudo = y_class.loc[good_indx, :].astype('float')
        y_pseudo_series = pd.Series(y_pseudo.squeeze(), name=self.y_name)

        D_u_pseudo = pd.concat([D_u_good, y_pseudo_series], axis = 1)
        D_l_batch_new = pd.concat([D_l_batch_new, D_u_pseudo], axis = 0)

        D_u_batch_wo_good_indx = D_u_batch.drop(good_indx, axis=0)

        return D_l_batch_new, D_u_batch_wo_good_indx


    def get_correlation_by_multivariate_analysis(self, result_pred_GNB, result_pred_train_GNB, \
                    D_l_batch, y_factor_test, mu_y_train, sd_y_train, sd_interval, offset, is_save = False):
                    
        y_train_pred = np.array(result_pred_train_GNB)
        y_real = np.array(D_l_batch.loc[:, self.y_name]).reshape(-1, 1)
        y_test_pred = result_pred_GNB.iloc[:, 1:]

        mlr = Ridge(alpha = 0.1)
        mlr.fit(y_train_pred, y_real)
        y_pred_GNB = mlr.predict(y_test_pred)

        y_pred_GNB_weight = np.zeros_like(y_pred_GNB)
        for i, y_el in enumerate(np.array(y_pred_GNB)):
            y_pred_GNB_weight[i] = y_el[0].astype(float)

        result_pred_GNB['prediction'] = convert_to_real(pd.Series(y_pred_GNB_weight), mu_y_train, sd_y_train, sd_interval, offset)

        correlation_factor = scipy.stats.pearsonr(y_factor_test, y_pred_GNB_weight)[0]
        correlation_real = scipy.stats.pearsonr( result_pred_GNB['real'], result_pred_GNB['prediction'] )[0]
        mae = MAE( np.array(result_pred_GNB['real']), np.array(result_pred_GNB['prediction']) )
        mape = MAPE( np.array(result_pred_GNB['real']), np.array(result_pred_GNB['prediction']) )

        if is_save == True and np.abs(correlation_real) > 0.5:
            pickle.dump(mlr, open('./models/' + self.model_filename + "_mlr", 'wb'))

        return correlation_factor, correlation_real, y_pred_GNB_weight, mae, mape
