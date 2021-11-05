## coding: utf-8
## Author : Edmond Chaussidon (CEA)
## Code for Tracer modelisation: class tracer

import os
import tqdm
import numpy as np
from scipy.interpolate import interp1d

from cosmoprimo.fiducial import DESI

# ---------------------------------------------------------------------------------------------------- #
# Reference cosmology --> LOAD DESI cosmology (fiducial cosmology for DESI measurment)
c_fid = DESI(engine='class')
# Compute background with engine as engine for computation (already set)
c_fid.get_background()
# Initialize the Fourier class to compute the power spectrum from a background
c_fid.get_fourier()

# ---------------------------------------------------------------------------------------------------- #
class DN_DZ(object):
    """
    Class to compute (interpolating) the dn_dz from different input.
    """

    def __init__(self, filename):
        """
        Initialize :class:`DN_DZ`.

        Ce n'est qu'un debut pour l'instant, on ne lit qu'a partir d'un fichier txt que j'ai construit

        Parameters
        ----------

        """

        self.filename = filename
        self.input_from_txt(self.filename)


    def input_from_txt(self, filename):
        """
        LOAD n(z) normalized from a .txt file ! see: f_NL/Compute_n_z.ipynb

        Parameters
        ----------
        filename : path of the .txt file

        """
        d = np.loadtxt(filename)
        self.interp = interp1d(d[0], d[1], kind='quadratic', bounds_error=False, fill_value=(0,0))


    def __call__(self, z):
        """Return (interpolated) dn_dz at redshift ``z`` (scalar or array)."""
        return self.interp(z)

# ---------------------------------------------------------------------------------------------------- #
class Tracer(object):
    """
    Implementation of the Tracer which probes the matter in the Universe
    """


    def __init__(self, name, cosmo, bias, pop, z0, dn_dz, area, density_deg2, z_width, shot_noise_limited):
        """
        Initialize :class:`Tracer`

        Parameters
        ----------
        name : name of the tracer, useful for legend
        cosmo : cosmo class from cosmoprimo
        bias : float
               bias of the consider tracer
        pop : float
              parameter to describe if the tracor is due to recent merger or not. Should be 1 (old merger) < pop < 1.6 (recent merger)
        z0 : float
            mean redshift of the sample of the consider tracer
        dn_dz : callable function
                function which describes the n(z) of the tracer
        area : float
               Surface in deg2 of the observation of the tracer
        density_deg2 : float
                       density of the tracer in deg2
        z_width : float
                  effective Delta_z of the dn_dz
        shot_noise_limited : bool
                             Are you in the shot noise regime ?
        """

        self.name = name

        # tracor parameters
        self.bias = bias
        self.pop = pop
        self.z0 = z0
        self.dn_dz = dn_dz

        # fiducial cosmology
        self.cosmo = cosmo

        # Survey information
        self.area = area
        self.density_deg2 = density_deg2
        self.z_width = z_width

        self.V_survey = self.Volume()
        self.n_survey = self.Density()

        self.shot_noise_limited = shot_noise_limited


    def __str__(self):
        string = "\nBuild Tracer with the following parameters:"
        string += "\n    * name: " + str(self.name)
        string += "\n    * bias: " + str(self.bias)
        string += "\n    * pop: " + str(self.pop)
        string += "\n    * z0: " + str(self.z0)
        string += "\n    * z_width: " + str(self.z_width)
        string += "\n    * area: " + str(self.area)
        string += "\n    * density_deg2: " + str(self.density_deg2)
        string += f"\n    * Survey Volume (Gpc/h)^3 = {self.V_survey/1.0e9:2.2f}"
        string += f"\n    * Survey density =  {self.n_survey:.2e}"
        string += f"\n    * Is in shoot noise limited region ? {self.shot_noise_limited}\n"
        return string


    def Volume(self):
        f_sky = self.area/(4*180.*180./np.pi)
        V = 4.0/3.0*np.pi*f_sky
        V *= self.cosmo.get_background().comoving_radial_distance(self.z0+self.z_width/2.0)**3 - self.cosmo.get_background().comoving_radial_distance(self.z0-self.z_width/2.0)**3
        return V


    def Density(self):
        n = self.area*self.density_deg2/self.V_survey
        return n


    def __copy__(self):
        """
        Proper way to copy the class
        """
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new


    def copy(self, **kwargs): #super malin ca !
        new = self.__copy__()
        new.__dict__.update(kwargs)

        #On oublie pas de mettre à jours le volume et la densité !
        new.V_survey = new.Volume()
        new.n_survey = new.Density()

        return new


# ---------------------------------------------------------------------------------------------------- #
def LRG_tracer():
    """
    Define Standard DESI LRG tracer
    """
    ## Param for LRG as tracer:
    bias = 2.3  #https://arxiv.org/pdf/1607.05383.pdf
    pop = 1.0
    z0 = 0.7
    z_width = 1.0

    ## Info sur le survey:
    density_deg2 = 500.0
    Area = 14000 # DESI geometry
    ## shot noise limited regime ?
    shot_noise_limited = True

    ## Load n(z)
    dn_dz = DN_DZ(os.path.join(os.path.dirname(__file__), 'Data/dn_dz_lrg.txt'))

    return Tracer("LRG", c_fid, bias, pop, z0, dn_dz, Area, density_deg2, z_width, shot_noise_limited)


def QSO_tracer():
    """
    Define Standard DESI QSO tracer
    """
    ## Param for QSO as tracer:
    bias = 2.5
    pop = 1.0
    z0 = 1.7
    z_width = 1.4

    ## Info sur le survey:
    density_deg2 = 200.0
    Area = 14000 # DESI geometry
    ## shot noise limited regime ?
    shot_noise_limited = True

    ## Load n(z)
    dn_dz = DN_DZ(os.path.join(os.path.dirname(__file__), 'Data/dn_dz_qso.txt'))

    return Tracer("QSO", c_fid, bias, pop, z0, dn_dz, Area, density_deg2, z_width, shot_noise_limited)


def LBG_tracer():
    """
    Define LBG tracer
    """
    ## Param for LBG tracer:
    bias = 3.2
    pop = 1.0
    z0 = 2.76
    z_width = 1.4

    ## Info sur le survey:
    density_deg2 = 500.0
    Area = 10000
    ## shot noise limited regime ?
    shot_noise_limited = True

    ## Load n(z)
    dn_dz = DN_DZ(os.path.join(os.path.dirname(__file__), 'Data/dn_dz_bgs.txt'))

    return Tracer("LBG", c_fid, bias, pop, z0, dn_dz_bgs, Area, density_deg2, z_width, shot_noise_limited)
