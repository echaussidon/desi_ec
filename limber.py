# coding: utf-8
# Author : Edmond Chaussidon (CEA)
#
# Permet de faire le fit de limber

import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
import scipy.special  as special

import nbodykit.cosmology as cosmology

from astropy.io import ascii

from linear_regression import regression_least_square
from corr import compute_result
import corr #pour ouvrir le fichier RF_g

###Load Planck cosmology:
c_fid = cosmology.Planck15

### Build Quasar luminosity function:
# dN/dz
data_rf_g = ascii.read(corr.__file__[:-7] + 'Data/RF_g.txt', format='no_header', names=['DR8_RF','z'])
dr8_rf_g = np.array(data_rf_g['DR8_RF']) #QFL * completeness_RF_D8 par bin de z
z_g = np.array(data_rf_g['z'])           #les bins de z

# smooth function
z_smooth = np.linspace(z_g.min(), z_g.max(), 100)
QLF_inter = UnivariateSpline(z_g, dr8_rf_g)
QLF_inter.set_smoothing_factor(2.0)
QLF_smooth =  QLF_inter(z_smooth)
QLF_integral = quad(QLF_inter, 0., 5)[0]

### Convertir degree avec comoving_transverse_distance cf: https://en.wikipedia.org/wiki/Distance_measures_(cosmology)
## rp en Mpc/h !! (cosmoprimo and nbodykit give the comoving distance in Mpc/h which is not the case in astropy)
def rp2deg(rp, z=-1):
    if z==-1:
        z = 1.7 # redshift moyen de l'echantillon QSO avec QLF x completeness attendu par DESI
    degree = np.rad2deg(rp/c_fid.comoving_transverse_distance(z))
    return degree

def deg2rp(degree, z=-1):
    if z==-1:
        z = 1.7 # redshift moyen de l'echantillon QSO avec QLF x completeness attendu par DESI
    rp = np.deg2rad(degree)*c_fid.comoving_transverse_distance(z)
    return rp

### Define Limber function and fit:

def limber(theta, *param):
    r0 = param[0]
    gamma = param[1]
    def dn_dz(z):
        return QLF_inter(z)/QLF_integral

    def dchi_dz(z):
        return c_fid.C/c_fid.H0/c_fid.efunc(z)

    def chi(z):
        return c_fid.comoving_transverse_distance(z)

    def integrand(z, gamma):
        return ((dn_dz(z))**2)*(chi(z)**(1.0-gamma))/(dchi_dz(z))

    z_max = 5.0
    integral = quad(integrand, 0., z_max, args=(gamma))[0]

    return (r0**gamma) * np.sqrt(np.pi) * integral * (np.deg2rad(theta))**(1-gamma) * special.gamma((gamma-1.0)/2.) / special.gamma(gamma/2.)

def plot_limber(ax, r0, gamma, color=None, linestyle='--', linewidth=1, alpha=1, label='', label_param=True, min_theta=0.01, max_theta=1):
    if label_param:
        label = f'{label} ({r0:.2f}, {gamma:.2f})'
    x = np.logspace(np.log10(min_theta),np.log10(max_theta), 1000)
    ax.plot(x, limber(x, r0, gamma), color=color, linestyle=linestyle, linewidth=linewidth, label=label, alpha=alpha)

def Fit_Limber(r, xi, err_xi, r_min=0.01, r_max=0.6, use_minos=False, print_covariance=True):
    #attention ici: r_min et r_max sont en degree !
    nbr_params = 2 # r_0 et gamma

    r_fit = r[(r>r_min) & (r<r_max)]
    xi_fit = xi[(r>r_min) & (r<r_max)]
    err_xi_fit = err_xi[(r>r_min) & (r<r_max)]

    dict_ini = {'a0': 5, 'a1': 1}
    dict_ini.update({'error_a0' : 0.001, 'error_a1' : 0.001})
    dict_ini.update({'limit_a0': (0, 20), 'limit_a1': (0, 10)})
    dict_ini.update({'errordef':1}) #for leastsquare

    param = regression_least_square(limber, 0.0, r_fit, xi_fit, err_xi_fit, nbr_params, use_minos=use_minos, print_covariance=print_covariance, **dict_ini)

    return param

def limber_from_cute(filename, ax=None, color=None, linestyle='--', linewidth=1, marker=None, markerfacecolor=None, label='', alpha=1, min_theta=0.001, max_theta=1.0, label_param=True, use_minos=False, print_covariance=True):
    r, xi, err_r, err_xi = compute_result(filename=filename)
    par = Fit_Limber(r, xi, err_xi, min_theta, max_theta, use_minos, print_covariance)

    if ax != None:
        plot_limber(ax, par[0], par[1], color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label, label_param=True, min_theta=min_theta, max_theta=max_theta)
