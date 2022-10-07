"""
coding: utf-8

Author : Edmond Chaussidon (CEA)

Code to fit fnl local with scale-dependent bias. 

Only basic model for the moment. Fit is performed with linear regression (leat square from iminuit)

"""

from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator

from linear_regression import regression_least_square

import numpy as np
import matplotlib.pyplot as plt

from cosmo_ps import PowerSpectrum
from cosmo_tracer import QSO_tracer

import tqdm
import pickle


def create_model_2d(Tracer, 
                    k=np.linspace(1e-3, 1.0, int(1e6)), bias=np.linspace(1.5, 3.5, 10), fnl=np.linspace(-5, 5, 40),
                    load_model=False, save_model=False, path_model=None,
                    use_new_Plin=False, new_k=None, new_Plin=None,
                    n_jobs=4, verbose=10):
    
    # Compte the model at discretizied points
    if not load_model:
        def compute_model(Tracer, k, bias, fnl, i, j, use_new_Plin=use_new_Plin, new_k=new_k, new_Plin=new_Plin):
            ps = PowerSpectrum(Tracer.copy(bias=bias[i]), fnl[j])
            if use_new_Plin:
                if (new_k is None) or (new_Plin is None):
                    raise ValueError('new_k or new_Plin is not set')
                ps.set_Plin_from_array(new_k, new_Plin)
            return i, j, ps.monopole(k)

        model = np.NaN * np.zeros((k.size, bias.size, fnl.size))
        
        results = Parallel(n_jobs=n_jobs, batch_size=(bias.size * fnl.size) // n_jobs, verbose=verbose)(delayed(compute_model)(Tracer, k, bias, fnl, i, j) for i in range(bias.size) for j in range(fnl.size))

        # normalement le i, j sont bien renvoyés dans le bon sens mais pour etre sur on les retourne pour ne pasa voir de surprise
        i, j, model_val = zip(*results)
        for l in range(len(i)):
            model[:, i[l], j[l]] = model_val[l]
        
        if save_model:
            if path_model is None:
                print('WARNING: Model is not saved since path_model is None')
            else:
                to_save = {'k': k, 'bias': bias, 'fnl': fnl, 'model': model}
                with open(path_model, 'wb') as f:
                    pickle.dump(to_save, f)
    
    else:
        print(f'WARNING: Load precomputed model saved in {path_model}')
        with open(path_model, 'rb') as f:
            loaded = pickle.load(f)
        k, bias, fnl, model = loaded['k'], loaded['bias'], loaded['fnl'], loaded['model']
        print(f'The model loaded is for:')
        print(f'    * k_min, k_max, n°k: {k[0]}, {k[-1]}, {k.size}')
        print(f'    * bias_min, bias_max, n_bias: {bias[0]}, {bias[-1]}, {bias.size}')  
        print(f'    * fnl_min, fnl_max, n_fnl: {fnl[0]}, {fnl[-1]}, {fnl.size}')  

    # Interpolate the model
    # use RegularGridInterpolator instead of LinearNDInterpolator
    model = RegularGridInterpolator((k, bias, fnl), model)

    return model


# Plot model to validate it:
def plot_model_2d(model_2d, z0, k, bias, fnl, savename='model_bias_fnl.pdf'):
    
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    colors = plt.cm.jet(np.linspace(0, 1, fnl.size))
    for i, p in enumerate(fnl):
        b = 2.6
        axs[0].loglog(k, model_2d(k, b, p), color=colors[i])
    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=fnl[0], vmax=fnl[-1], clip=False), cmap='jet'), ax=axs[0], label='fnl', format=FuncFormatter(lambda x, pos: '{:1.2f}'.format(x)))
    axs[0].set_xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$")
    axs[1].set_ylabel(r"$P_{hh}(k)$ $[h^{-3} \mathrm{Mpc}^{3}]$")
    axs[0].set_title(f'bias={b} -- pop=1 -- z={z0}')

    colors = plt.cm.jet(np.linspace(0, 1, bias.size))
    for i, b in enumerate(bias):
        axs[1].loglog(k, model_2d(k, b, 0.0), color=colors[i])
    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=bias[0], vmax=bias[-1], clip=False), cmap='jet'), label='bias', ax=axs[1], format=FuncFormatter(lambda x, pos: '{:1.2f}'.format(x)))
    axs[1].set_xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$")
    axs[1].set_ylabel(r"$P_{hh}(k)$ $[h^{-3} \mathrm{Mpc}^{3}]$")
    axs[1].set_title(f'pop=1 -- fnl={0} -- z={z0}')

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()
    

# Define regression with iminuit:
def regression(model, X, Y, Y_cov_inv,
               bias_ini=2.0, bias_limits=(1.5, 4.0), fix_bias=False,
               fnl_ini=0.0, fnl_limits=(-20, 20), fix_fnl=False,
               noise_ini=0.0, noise_limits=(-1e5, 1e5), fix_noise=False,
               print_level=0, print_param=False):
    # number of parameters to fit
    nbr_params = 3

    # create dictionary for initial condition
    dict_ini = dict()

    # parameter bias:
    dict_ini['x0'] = bias_ini
    dict_ini['error_x0'] = 0.01
    dict_ini['limits_x0'] = bias_limits
    dict_ini['fixed_x0'] = fix_bias

    # parameter fnl:
    dict_ini['x1'] = fnl_ini
    dict_ini['error_x1'] = 0.01
    dict_ini['limits_x1'] = fnl_limits
    dict_ini['fixed_x1'] = fix_fnl

    # parameter for residual shot noise
    dict_ini['x2'] = noise_ini
    dict_ini['error_x2'] = 0.01
    dict_ini['limits_x2'] = noise_limits
    dict_ini['fixed_x2'] = fix_noise
    
    # parameter for Minuit:
    dict_ini['errordef'] = 1  # for leastsquare
    dict_ini['print_level'] = print_level  # to remove message from iminuit set 0

    param, err_param = regression_least_square(model, X, Y, Y_cov_inv, nbr_params, print_param=print_param, use_minos=False, return_errors=True, **dict_ini)

    return model(X, *param), param, err_param