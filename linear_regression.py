# coding: utf-8
# Author : Edmond Chaussidon (CEA)
#
# Permet de faire une regression lineaire avec minuit rapidement

import numpy as np

from iminuit import Minuit, describe
from iminuit.util import make_func_code

class LeastSquares:
    def __init__(self, model, regulator, x, y, y_err):
        self.model = model
        self.regulator = regulator
        self.x = np.array(x)
        self.y = np.array(y)
        self.y_err = np.array(y_err)
        self.func_code = make_func_code(describe(self.model)[1:])

    def __call__(self, *par):
        ym = self.model(self.x, *par)
        sel = (self.y_err != 0)
        chi2 = np.nansum((self.y[sel] - ym[sel])**2/(self.y_err[sel])**2) + self.regulator*(np.nanmean(ym[sel]) - 1)**2
        return chi2

def regression_least_square(model, regulator, data_x, data_y, data_y_err, nbr_params, **dict_ini):
    chisq = LeastSquares(model, regulator, data_x, data_y, data_y_err)
    m = Minuit(chisq, print_level=1, forced_parameters=[f"a{i}" for i in range(0, nbr_params)], **dict_ini)
    m.migrad()
    print(m.get_param_states())
    return [m.values[f"a{i}"] for i in range(0, nbr_params)]


## EXEMPLE de fonction pour appeler la regression lineaire :
#        * il faut modifier model
#        * il faut modifier dict_ini
#        * il faut adapter les erreurs
#        * ici le code normalise les données uniquement sur la zone d'entrainement
def make_linear_regressor(X, Y, keep_to_train, regulator=0.0):

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

    Y_err_train = np.sqrt(Y_train)

    param = regression_least_square(model, regulator, X_train, Y_train, Y_err_train, nbr_features, **dict_ini)

    return model(X, *param)
