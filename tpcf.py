# coding: utf-8
# Author : Edmond Chaussidon (CEA)
# function to read catalog, write input / read output for CUTE and to compute 2pcf.

import sys
import os
import logging
logger = logging.getLogger("TPCF")

import numpy as np
import matplotlib.pyplot as plt
import fitsio
import pandas as pd
import healpy as hp
from astropy.io import ascii

from wrapper import time_measurement
from scipy import interpolate
import scipy.stats as stats

from configparser import ConfigParser

#------------------------------------------------------------------------------#
@time_measurement
def generate_sample_mcmc(Nsample, x_posterior, y_posterior, t_max=100, show_result=False):
    np.random.seed(seed=2207) # to fix the randomness

    lb, hb = x_posterior[0], x_posterior[-1] # lower and upper bound
    f = interpolate.interp1d(x_posterior, y_posterior) # f is the posterior distribution

    def estimation_MCMC_unif(init, t_max):
        # Metropolis Hastings sampling from the posterior distribution
        # Use uniform law to generate new sample (--> can be changed to gaussian)
        X, ones_vect = init, np.ones(Nsample)
        y, p = stats.uniform.rvs(loc=lb, scale=hb-lb, size=(init.size, t_max)), np.random.random(size=(init.size, t_max))
        f_y = f(y)

        for time in range(1,t_max):
            rho = np.minimum(ones_vect, f_y[:, time]/f(X))
            value_to_update = p[:, time] < rho
            X[value_to_update] = y[value_to_update, time]
        return X

    init = x_posterior.mean()*np.ones(Nsample) # on commence au milieu de X (dans le cas d'une distribution centrée c'est ok sinon faire aleatoire)
    samples = estimation_MCMC_unif(init, t_max)

    if show_result:
        plt.figure(figsize=(4.5,4.5))
        plt.plot(x_posterior, y_posterior, linestyle=':', marker='*', color='red', label='Post')
        plt.hist(samples, density=1, bins=50, color='blue', range=(0, 4), label='Sample')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return samples

#------------------------------------------------------------------------------#
#Class for imaging systematic weight
log = logging.getLogger('SysWeight')

class SysWeight(object):

    def __init__(self, tracer="LRG", survey='SV3', engine='EC', Nside=None, use_stream=False):
        """
        Survey is eihter SV3 or MAIN.
        Engine is either EC (moi dans mon dossier), MR (Medhi) or RZ (Rongpu)
        Tracer is either LRG, LRG_LOWDENS, ELG, ELG_HIP, QSO for SV3 and QSO for MAIN
        """

        self.tracer=tracer
        self.survey=survey
        self.engine=engine
        if Nside is not None:
            self.nside = Nside
        elif tracer in ["QSO", "ELG_VLO"]:
            self.nside=256
        else:
            self.nside=512

        # for engine=EC
        suffixe=''
        if use_stream:
            suffixe='_with_stream'

        if engine=='EC':
            dir_weight="/global/cfs/cdirs/desi/users/edmondc/Imaging_weight"
            weight_file = os.path.join(dir_weight, f"{survey}/{tracer}_imaging_weight_{self.nside}{suffixe}.npy")
            logging.info(f"Read imaging weight: {weight_file}")
            self.map = np.load(weight_file)

        elif engine=='MR' or engine == 'RZ':
            dir_weight = f'/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/SystematicMaps/w_sys_{engine}'
            weight_file = os.path.join(dir_weight, f"{survey}/{tracer}_imaging_weight_{self.nside}.npy")
            logging.info(f"Read imaging weight: {weight_file}")
            self.map = np.load(weight_file)

    def __call__(self, ra, dec):
        """
        Return weight for Ra, Dec list
        """
        pix = hp.ang2pix(self.nside, ra, dec, nest=True, lonlat=True)
        return self.map[pix]

    def plot_map(self):
        from plot import plot_moll, to_tex
        plot_moll(self.map - 1, min=-0.2, max=0.2, label=to_tex(self.tracer))

#------------------------------------------------------------------------------#
def read_fits(filename):
    logger.info(f'Read fits file from : {filename}')
    return fitsio.FITS(filename)[1]

def read_fits_to_pandas(filename, ext_name=1, columns=None):
    # ext_name can be int or string
    logger.info(f'Read ext: {ext_name} from {filename}')
    if columns is None:
        dataFrame = pd.DataFrame(fitsio.FITS(filename)[ext_name].read().byteswap().newbyteorder())
    else:
        dataFrame = pd.DataFrame(fitsio.FITS(filename)[ext_name][columns].read().byteswap().newbyteorder())
    return dataFrame


def make_selection(darray, criterions):
    # darray = catalog in darray type, like data[1][:] when fits file is open with fitsio.FITS
    # criterions = list of criterion : [feature_name, operation, value] only value is not in str

    def apply_criterion_for_selection(darray, criterion):
        feature_name, operation, value = criterion[0], criterion[1], criterion[2]
        logger.info(f"    * {feature_name} {operation} {value}")
        if operation == '==':
            return darray[feature_name] == value
        elif operation == '!=':
            return darray[feature_name] != value
        elif operation == '<':
            return darray[feature_name] < value
        elif operation == '<=':
            return darray[feature_name] <= value
        elif operation == '>':
            return darray[feature_name] > value
        elif operation == '>=':
            return darray[feature_name] >= value
        elif operation == '&':
            return (darray[feature_name]&2**value) != 0
        elif operation == 'in':
            return np.isin(darray[feature_name], value)

    sel = np.ones(darray.shape[0], dtype=bool)
    logger.info("We apply the selection:")
    for criterion in criterions:
        sel &= apply_criterion_for_selection(darray, criterion)
    return sel

#------------------------------------------------------------------------------#
# FOR PYCORR !
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# FOR CUTE !
#------------------------------------------------------------------------------#
def save_catalog_txt(catalog, selection, use_redshift='from_cat', add_redshift=None, use_weight='from_cat', add_weight=None, filename='oups.fits'):
    """Create catalog [RA, DEC, Z, WEIGHT] in .txt format.
    Parameters
    ----------
    catalog      : catalog containing RA, DEC (Z, WEIGHT) columns
    selection    : boolean array to select which objects is saved
    use_redshift : 'from_cat' -- 'from_add'
    add_redshift : redshift array for 'from_add'
    use_weight   : 'from_cat' -- 'from_add' -- 'from_one'
    add_weight   : weight array  for 'from_add'
    filename     : filename for the .txt file
    """

    ra = catalog['RA'][:][selection]
    dec = catalog['DEC'][:][selection]

    if use_redshift == 'from_cat':
        z = catalog['Z'][:][selection]
    else:
        z = add_redshift[selection]

    if use_weight == 'from_cat':
        weight = catalog['WEIGHT'][:][selection]
    elif use_weight == 'from_add':
        weight = add_weight[selection]
    else:
        weight = np.ones(selection.sum())

    ascii.write([ra, dec, z, weight], filename , names=['ra', 'dec', 'z', 'w'],
                format='no_header', overwrite=True)
    logger.info(f"Write catalog in {filename} with {ra.size} points")


def write_CUTE_ini(param):
    if 'ini_filename' in param.keys():
        logger.info(f"Write {param['ini_filename']}")
        file = open(param['ini_filename'], "w") # overwrite
        file.write("# input-output files and parameters\n")
    else:
        sys.exit("MISSING PARAM FOR CUTE INI (ini_filename)")
    if 'data_filename' in param.keys():
        file.write(f"data_filename= {param['data_filename']}\n")
    else:
        sys.exit("MISSING PARAM FOR CUTE INI (data_filename)")
    if 'randoms_filename' in param.keys():
        file.write(f"random_filename= {param['randoms_filename']}\n")
    else:
        sys.exit("MISSING PARAM FOR CUTE INI (randoms_filename)")
    if 'output_with_RR' in param.keys():
        logger.warning(f"We use a previous calculation of RR which is in : {param['output_with_RR']} --> CHECK IF IT IS THE SAME RANDOMS AND THE CUTE PARAMETERS\n")
        file.write(f"RR_filename= {param['output_with_RR']}\n")
    file.write("input_format= 2\n")
    if 'output_filename' in param.keys():
        file.write(f"output_filename= {param['output_filename']}\n")
    else:
        sys.exit("MISSING PARAM FOR CUTE INI (output filename)")
    file.write("\n")
    file.write("# estimation parameters\n")
    if 'corr_type' in param.keys():
        file.write(f"corr_type= {param['corr_type']}\n")   # angular monopole 3D_ps 3D_rm full
    else:
        sys.exit("MISSING PARAM FOR CUTE INI (corr_type)")
    file.write("\n")
    file.write("# cosmological parameters\n") # planck cosmology --> ok
    file.write("omega_M= 0.315\n")
    file.write("omega_L= 0.685\n")
    file.write("w= -1\n")
    file.write("\n")
    file.write("# binning\n")
    if 'log_bin' in param.keys():
        file.write(f"log_bin= {param['log_bin']}\n")
    else:
        file.write("log_bin= 0\n")
    if 'dim1_min_logbin' in param.keys():
        file.write(f"dim1_min_logbin= {param['dim1_min_logbin']}\n")
    else:
        file.write("dim1_min_logbin= 0\n")
    if 'dim1_max' in param.keys():
        file.write(f"dim1_max= {param['dim1_max']}\n")
    else:
        file.write("dim1_max= 100\n")
    if 'dim1_nbin' in param.keys():
        file.write(f"dim1_nbin= {param['dim1_nbin']}\n")
    else:
        file.write("dim1_nbin= 60\n")
    if 'dim2_max' in param.keys():
        file.write(f"dim2_max= {param['dim2_max']}\n")
    else:
        file.write("dim2_max= 1.\n")
    if 'dim2_nbin' in param.keys():
        file.write(f"dim2_nbin= {param['dim2_nbin']}\n")
    else:
        file.write("dim2_nbin= 5.\n")

    file.write("dim3_min= 0.5\n")
    file.write("dim3_max= 2\n")
    file.write("dim3_nbin= 1\n")
    file.write("\n")
    file.write("# pixels for radial correlation\n")
    file.write("radial_aperture= 10\n")
    file.write("\n")
    file.write("# pm parameters\n")
    if 'use_pm' in param.keys():
        file.write(f"use_pm= {param['use_pm']}\n")
    else:
        file.write("use_pm= 0\n")
    if 'n_pix_sph' in param.keys():
        file.write(f"n_pix_sph= {param['n_pix_sph']}\n")
    else:
        file.write("n_pix_sph= 0\n")

    file.close()


def get_CUTE_ini(filename):
    """
    Create a configparser from CUTE init file. The output works like a dictionary.
    """
    config = ConfigParser()
    with open(filename) as stream:
        # Need to add a section to read it with configparser
        logger.info(f"Read CUTE initial parameters from {filename}")
        config.read_string("[top]\n" + stream.read())
    return config['top']


@time_measurement
def CUTE(param, nbr_nodes=4, nbr_threads=16, keep_trace_txt='output_cute.txt'):
    write_CUTE_ini(param)
    cute_ini = param['ini_filename']
    logger.info(f"RUN CUTE ({param['corr_type']}) for {cute_ini} with {nbr_nodes} nodes and {nbr_threads} threads.")
    logger.info(f"Terminal ouput is saved in {keep_trace_txt}")
    CUTE_CALL = f'mpiexec -np {nbr_nodes} /global/homes/e/edmondc/Software/CUTE/CUTE/CUTE  {cute_ini}'
    os.system(f"module load openmpi && module load gsl && export OMP_NUM_THREADS={nbr_threads} && {CUTE_CALL} |& tee {keep_trace_txt}")


def extract_cute_result_1D(filename, return_dd=False):
    logger.info(f"Read CUTE result from {filename}")
    data_xi = ascii.read(filename, format='no_header', names=['R','Xi','DD','DR','RD','RR'])

    dd=np.array(data_xi['DD'])
    dr=np.array(data_xi['DR'])
    rd=np.array(data_xi['RD'])
    rr=np.array(data_xi['RR'])

    r=np.array(data_xi['R'])
    xi=np.array(data_xi['Xi'])

    err_xi = (1+xi)/np.sqrt(dd) ## au premier ordre c'est bien ca (en negliant les termes en alphabeta, beta^2, gamma^2, gammabeta)
    err_r = np.zeros(len(r))/rr

    if return_dd:
        return r, xi, err_r, err_xi, dd, dr, rd, rr
    else:
        return r, xi, err_r, err_xi


def extract_cute_result_2D(filename, filename_ini, return_dd=False):
    param_ini = get_CUTE_ini(filename_ini)
    nbin_x1, nbin_x2 = int(param_ini['dim1_nbin']), int(param_ini['dim2_nbin'])
    x1_max, x2_max = float(param_ini['dim1_max']), float(param_ini['dim2_max'])

    logger.info(f"Read CUTE result from {filename}")
    data_xi = ascii.read(filename, format='no_header', names=['X1', 'X2', 'Xi','DD','DR','RD','RR'])

    x1_grid = data_xi['X1'].reshape(nbin_x1, nbin_x2)
    x1 = x1_grid[0, :]
    x2_grid = data_xi['X2'].reshape(nbin_x1, nbin_x2)
    x2 = x2_grid[:, 0]

    xi = data_xi['Xi'].reshape(nbin_x1, nbin_x2)
    dd = data_xi['DD'].reshape(nbin_x1, nbin_x2)
    dr = data_xi['DR'].reshape(nbin_x1, nbin_x2)
    rd = data_xi['RD'].reshape(nbin_x1, nbin_x2)
    rr = data_xi['RR'].reshape(nbin_x1, nbin_x2)

    err_xi = (1+xi)/np.sqrt(dd) ## au premier ordre c'est bien ca (en negliant les termes en alphabeta, beta^2, gamma^2, gammabeta)
    err_xi[np.isinf(err_xi)] = np.NaN

    # attention xi est donnée en xi(x1, x2) --> si on veut afficher xi(x2, x1) --> il faut donc transposer la matrice !
    #extent = [x2[0], x2[-1], x1[0], x1[-1]]

    if return_dd:
        return x1_grid, x2_grid, xi, err_xi, dd, dr, rd, rr
    else:
        return x1_grid, x2_grid, xi, err_xi


def extract_cute_result(filename, filename_ini=None, return_dd=False):
    if filename[-12:] == 'monopole.txt' or filename[-11:] == 'angular.txt':
        return extract_cute_result_1D(filename, return_dd)
    elif filename[-9:] == '3D_ps.txt' or filename[-9:] == '3D_rm.txt':
        return extract_cute_result_2D(filename, filename_ini, return_dd)
    else:
        sys.exit("USE correct filename format (*_corr_type.txt)")
