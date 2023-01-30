# coding: utf-8
# Author : Edmond Chaussidon (CEA)
#
# Permet de faire une regression lineaire (une minimzation de Xi2 en realité) avec minuit rapidement

# Pour avoir une implementation en 1D avec une matrice de covariance regarder fnl/fnl_angulaire

# Attention a la nouvelle version de iminuit !! --> mise a jour le 20/04/2022 pour que cela marche avec iminuit.__version__ > 2.8 --> Des adaptations sont à prévoir dans des vieux codes..

import numpy as np

from iminuit import Minuit, describe
from iminuit.util import make_func_code


class LeastSquares:
    def __init__(self, model, x, y, cov_inv, regulator=0, regulator_val=1):
        self.model = model
        self.regulator = regulator
        self.regulator_val = regulator_val
        self.x = np.array(x)
        self.y = np.array(y)
        self.cov_inv = np.array(cov_inv)
        self.func_code = make_func_code(describe(self.model)[1:])

    def __call__(self, *par):
        ym = self.model(self.x, *par)
        chi2 = (self.y - ym).T.dot(self.cov_inv).dot(self.y - ym) + self.regulator * (np.nanmean(ym) - self.regulator_val)**2
        return chi2


def regression_least_square(model, data_x, data_y, data_y_cov_inv, nbr_params,
                            use_hesse=False, use_minos=False,
                            regulator=0, regulator_val=1,
                            print_covariance=False, print_param=True,
                            return_errors=True, **dict_ini):
    # create Xi2 model:
    chisq = LeastSquares(model, data_x, data_y, data_y_cov_inv, regulator, regulator_val)

    name_params = [f"x{i}" for i in range(0, nbr_params)]
    # initialize the Minuit object:
    m = Minuit(chisq, name=name_params, **{name: dict_ini[name] for name in name_params})
    m.errors[:] = [dict_ini['error_' + name] for name in name_params]
    m.limits[:] = [dict_ini['limits_' + name] for name in name_params]
    m.fixed[:] = [dict_ini['fixed_' + name] for name in name_params]
    m.errordef = dict_ini['errordef']
    m.print_level = dict_ini['print_level']

    # make the regression:
    m.migrad()

    if use_hesse:  # compute parameter uncertainties with hesse
        print(m.hesse())
    if use_minos:  # compute parameter uncertainties with minos
        print(m.minos())
    if print_param:
        print(m.params)
    if print_covariance:
        print(repr(m.covariance))

    if return_errors:
        return [m.values[name] for name in name_params], [m.errors[name] for name in name_params]
    else:
        return [m.values[name] for name in name_params]


# EXEMPLE de fonction pour appeler la regression lineaire :
#        * il faut modifier model
#        * il faut modifier dict_ini
#        * il faut adapter les erreurs
#        * ici le code normalise les données uniquement sur la zone d'entrainement
def make_linear_regressor_1(X, Y, keep_to_train, regulator=0.0, regulator_val=1, print_level=1, print_param=True):

    nbr_features = X.shape[1]
    print(f"[INFO] Number of features used : {nbr_features}")
    nbr_params = nbr_features + 1

    X_train, Y_train = X[keep_to_train == 1], Y[keep_to_train == 1]

    print("[WARNING :] We normalize and center features on the training footprint (don't forget to normalized also the non training footprint...)")
    X = (X - X_train.mean()) / X_train.std()
    X_train = (X_train - X_train.mean()) / X_train.std()
    print(f"[INFO] Mean of Mean and Std training features : {X_train.mean().mean()} -- {X_train.std().mean()}")
    print(f"[INFO] Mean and median of norm_target_density : {Y_train.mean()} -- {np.median(Y_train)}\n")

    def model(x, *par):  # Estimateur == modele utilise
        return par[0] * np.ones(x.shape[0]) + np.array(par[1:]).dot(x.T)

    dict_ini = {f'x{i}': 1 if i == 0 else 0 for i in range(0, nbr_params)}
    dict_ini.update({f'error_x{i}': 0.001 for i in range(0, nbr_params)})
    dict_ini.update({f'limit_x{i}': (-1, 2) if i == 0 else (-1, 1) for i in range(0, nbr_params)})
    dict_ini['errordef'] = 1  # for leastsquare
    dict_ini['print_level'] = print_level  # to remove message from iminuit set 0

    # can also use np.diag
    Y_cov_inv = 1 / np.sqrt(Y_train) * np.eye(Y_train.size)

    param = regression_least_square(model, X_train, Y_train, Y_cov_inv, nbr_features, regulator=regulator, regulator_val=regulator_val, print_param=print_param, use_minos=False, return_errors=False, **dict_ini)

    return model(X, *param)


# MINIMAL EXAMPLE without normalization ect ...
# Fit two parameters
def make_linear_regressor_2(model, X, Y, Y_cov_inv, regulator=0.0, print_level=1, print_param=True):

    # number of parameters to fit
    nbr_params = 2

    # create dictionary for initial condition
    dict_ini = dict()

    # parameter 1:
    dict_ini['x0'] = 1.5
    dict_ini['error_x0'] = 0.1
    dict_ini['limits_x0'] = (0.5, 2.0)
    dict_ini['fixed_x0'] = False

    # parameter 2:
    dict_ini['x1'] = 2.0
    dict_ini['error_x1'] = 0.3
    dict_ini['limits_x1'] = (1.5, 2.3)
    dict_ini['fixed_x1'] = False

    # parameter for Minuit:
    dict_ini['errordef'] = 1  # for leastsquare
    dict_ini['print_level'] = print_level  # to remove message from iminuit set 0

    # use hesse or minos increase the time but the errors are better:
    param, err_param = regression_least_square(model, X, Y, Y_cov_inv, nbr_params, print_param=print_param, use_hesse=False, use_minos=False, return_errors=True, **dict_ini)

    return model(X, *param), param, err_param
