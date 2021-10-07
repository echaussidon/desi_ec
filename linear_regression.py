# coding: utf-8
# Author : Edmond Chaussidon (CEA)
#
# Permet de faire une regression lineaire avec minuit rapidement

## Pour avoir une implementation en 1D avec une matrice de covariance regarder fnl/fnl_angulaire

import numpy as np

from iminuit import Minuit, describe
from iminuit.util import make_func_code

class LeastSquares:
    def __init__(self, model, regulator, x, y, cov_inv):
        self.model = model
        self.regulator = regulator
        self.x = np.array(x)
        self.y = np.array(y)
        self.cov_inv = np.array(cov_inv)
        self.func_code = make_func_code(describe(self.model)[1:])

    def __call__(self, *par):
        ym = self.model(self.x, *par)
        chi2 = (self.y - ym).T.dot(self.cov_inv).dot(self.y - ym) + self.regulator*(np.nanmean(ym) - 1)**2
        return chi2


def regression_least_square(model, regulator, data_x, data_y, data_y_cov_inv, nbr_params, use_minos=False, print_covariance=False, print_param=True, return_errors=False, **dict_ini):
    chisq = LeastSquares(model, regulator, data_x, data_y, data_y_cov_inv)
    m = Minuit(chisq, forced_parameters=[f"a{i}" for i in range(0, nbr_params)], **dict_ini)
    # make the regression:
    m.migrad()
    if print_param:
        print(m.params)
    if use_minos:
        print(m.minos())
    if print_covariance:
        print(repr(m.covariance)) 
    if return_errors:
        return [m.values[f"a{i}"] for i in range(0, nbr_params)], [m.errors[f"a{i}"] for i in range(0, nbr_params)]
    else:
        return [m.values[f"a{i}"] for i in range(0, nbr_params)]

## EXEMPLE de fonction pour appeler la regression lineaire :
#        * il faut modifier model
#        * il faut modifier dict_ini
#        * il faut adapter les erreurs
#        * ici le code normalise les données uniquement sur la zone d'entrainement
def make_linear_regressor(X, Y, keep_to_train, regulator=0.0, print_level=1, print_param=True):

    nbr_features = X.shape[1]
    print(f"[INFO] Number of features used : {nbr_features}")
    nbr_params = nbr_features + 1

    X_train, Y_train = X[keep_to_train == 1], Y[keep_to_train == 1]

    print("[WARNING :] We normalize and center features on the training footprint (don't forget to normalized also the non training footprint...)")
    X = (X - X_train.mean())/X_train.std()
    X_train = (X_train - X_train.mean())/X_train.std()
    print(f"[INFO] Mean of Mean and Std training features : {X_train.mean().mean()} -- {X_train.std().mean()}")
    print(f"[INFO] Mean and median of norm_target_density : {Y_train.mean()} -- {np.median(Y_train)}\n")

    def model(x, *par): # Estimateur == modele utilise
        return par[0]*np.ones(x.shape[0]) + np.array(par[1:]).dot(x.T)

    dict_ini = {f'a{i}': 1 if i==0 else 0 for i in range(0, nbr_params)}
    dict_ini.update({f'error_a{i}' : 0.001 for i in range(0, nbr_params)})
    dict_ini.update({f'limit_a{i}': (-1, 2) if i==0 else (-1,1) for i in range(0, nbr_params)})
    dict_ini.update({'errordef':1}) #for leastsquare
    dict_ini.update({'print_level':print_level}) # to remove message from iminuit set 0

    Y_cov_inv = 1/np.sqrt(Y_train) * np.eye(Y_train.size)

    param = regression_least_square(model, regulator, X_train, Y_train, Y_cov_inv, nbr_features, print_param=print_param, use_minos=False, return_errors=False, **dict_ini)

    return model(X, *param)


## MINIMAL EXAMPLE without normalization ect ...
# Fit only one parameter
def make_linear_regressor(model, X, Y, Y_cov_inv, regulator=0.0, print_level=1, print_param=True):

    nbr_features = 1 
    
    dict_ini = {'a0': -0.0}
    dict_ini.update({f'error_a0' : 0.1})
    dict_ini.update({f'limit_a0': (-20, 50)})
    dict_ini.update({'errordef':1}) # for leastsquare
    dict_ini.update({'print_level':print_level}) # to remove message from iminuit set 0

    param, err_param = regression_least_square(model, regulator, X, Y, Y_cov_inv, nbr_features, print_param=print_param, use_minos=False, return_errors=True, **dict_ini)

    return model(X, *param), param, err_param