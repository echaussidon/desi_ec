## coding: utf-8
## Author : Edmond Chaussidon (CEA)
## Code for Tracer modelisation: class tracer

import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import quad

import tqdm

from cosmoprimo import *
from cosmoprimo import constants

# ---------------------------------------------------------------------------------------------------- #   
## We define the fiducial cosmo (Planck2018 Universe) with cosmoprimo.

# Reference cosmology
c_fid = Planck2018FullFlatLCDM(engine='class')

# Compute background with engine as engine for computation (already set)
c_fid.get_background()

# Initialize the Fourier class to compute the power spectrum from a background
c_fid.get_fourier()


# ---------------------------------------------------------------------------------------------------- #    
## TO DO (a finir mais attendre de voir ce que l'on aura vraiment avec SV3 ect...)
class DN_DZ(object):
    """
    Class to compute (interpolating) the dn_dz from different input.
    """

    def __init__(self, kind='quadratic', bounds_error=False, fill_value=(0, 0)):
        """
        Initialize :class:`DN_DZ`.

        Parameters
        ----------

        """
        
        self.kind = kind
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        
    def input_from_txt(self, filename, normalized=True):
        """
        TO DO

        Parameters
        ----------

        """
        from astropy.io import ascii
        from scipy.integrate import quad
        print('aaaa')
        
        
    def compute_with_hist_from_fits(self, filename, bins=25, z_range=(0, 3.0), colname='Z', hdu_num=1):
        """
        TO DO

        Parameters
        ----------

        """
        d = fitsio.FITS(filename)
        dens, bins = np.histogram(d[hdu_num][colname][:], bins=bins, range=z_range, density=1)
        self.interp = interp1d((bins[:-1] + bins[1:])/2, dens, kind=self.kind, bounds_error=self.bounds_error, fill_value=self.fill_value)

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

    ## Build dN/dz from QLF and RF completeness (une fois que Evrest est sorti prendre le dn/dz d'Everest !!)
    import corr
    from astropy.io import ascii
    # DR8_RF == QFL * completeness_RF_D8 par bin de z 
    data_rf_g = ascii.read(corr.__file__[:-7] + '/RF_g.txt', format='no_header', names=['DR8_RF','z'])
    # Set 0 value for z=0
    dr8_rf_g, z_g = np.r_[np.array([0]), np.array(data_rf_g['DR8_RF'])], np.r_[np.array([0]), np.array(data_rf_g['z']) ]
    dd = interp1d(z_g, dr8_rf_g,  kind='quadratic', bounds_error=False, fill_value=(0,0))
    normalization = quad(dd, 0., 5, limit=100)[0]
    def dn_dz_qso(z): 
        return dd(z)/normalization

    return Tracer("QSO", c_fid, bias, pop, z0, dn_dz_qso, Area, density_deg2, z_width, shot_noise_limited)


def LRG_tracer():
    """
    Define Standard DESI LRG tracer
    """
    ## Param for ELG as tracer:
    bias = 2.3  #https://arxiv.org/pdf/1607.05383.pdf
    pop = 1.0
    z0 = 0.7
    z_width = 1.0

    ## Info sur le survey:
    density_deg2 = 500.0
    Area = 14000 # DESI geometry

    ## shot noise limited regime ?
    shot_noise_limited = True  

    ## Build dN/dz
    # use linear to avoid negative value in the interpolation
    import fitsio 
    LRG_path = '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/everest/LSScats/test/LRG_main_14_clustering.ran.fits'
    #LRG_path = '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/everest/LSScats/test/LRG_main_clustering.dat.fits'
    dens, bins = np.histogram(fitsio.FITS(LRG_path)[1]['Z'][:], bins=25, range=(0, 2.0), density=1)
    dn_dz_lrg = interp1d(bins[:-1], dens, kind='linear', bounds_error=False, fill_value=(0,0))

    return Tracer("LRG", c_fid, bias, pop, z0, dn_dz_lrg, Area, density_deg2, z_width, shot_noise_limited)


def LBG_tracer():
    """
    Define LBG tracer
    """
    ## Param for LBG tracer:
    bias = 3.2
    pop = 1.6
    z0 = 2.76
    z_width = 1.4
    
    ## Info sur le survey:
    density_deg2 = 500.0
    Area = 10000
    
    ## shot noise limited regime ?
    shot_noise_limited = True
    
    ## Build dN/dz
    z = np.array([2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3.05, 3.15, 3.25, 3.35, 3.45, 3.55, 3.65, 3.75, 3.85, 3.95])
    n_z = np.array([343., 484., 516., 866., 812., 1010., 948., 1012., 1219., 1228., 1069., 899., 694., 355., 124., 21., 8., 4., 5., 2.])
    
    n_z_interp = interp1d(z, n_z,  kind='quadratic', bounds_error=False, fill_value=(0,0))
    #smooth 
    from scipy.ndimage import gaussian_filter
    z_s = np.linspace(0, 5, 1000)
    n_z_interp = interp1d(z_s, gaussian_filter(n_z_interp(z_s), sigma=15),  kind='quadratic', bounds_error=False, fill_value=(0,0))
    
    normalization = quad(n_z_interp, 1.5, 4.5, limit=100)[0]
    def dn_dz_bgs(z):
        return n_z_interp(z)/normalization
    
    return Tracer("LRG", c_fid, bias, pop, z0, dn_dz_bgs, Area, density_deg2, z_width, shot_noise_limited)
    

