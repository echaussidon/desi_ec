## coding: utf-8
## Author : Edmond Chaussidon (CEA)
## Code for fnl modelisation: class tracer, power spectrum, correlation function

import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d, interp2d

import tqdm

from cosmoprimo import *
from cosmoprimo import constants

# ---------------------------------------------------------------------------------------------------- #    
# efficient implemantion to get chi_to_z 
# chi_to_z = DistanceToRedshift(c_fid.get_background().comoving_radial_distance)

class DistanceToRedshift(object):
    """
    Class that holds a conversion distance -> redshift.
    From : https://github.com/cosmodesi/pyrecon/blob/main/pyrecon/utils.py/
    """

    def __init__(self, distance, zmax=100., nz=2048, interp_order=3):
        """
        Initialize :class:`DistanceToRedshift`.
        Creates an array of redshift -> distance in log(redshift) and instantiates
        a spline interpolator distance -> redshift.
        Parameters
        ----------
        distance : callable
            Callable that provides distance as a function of redshift (array).
        zmax : float, default=100.
            Maximum redshift for redshift <-> distance mapping.
        nz : int, default=2048
            Number of points for redshift <-> distance mapping.
        interp_order : int, default=3
            Interpolation order, e.g. ``1`` for linear interpolation, ``3`` for cubic splines.
        """
        
        self.distance = distance
        self.zmax = zmax
        self.nz = nz
        zgrid = np.logspace(-8, np.log10(self.zmax), self.nz)
        self.zgrid = np.concatenate([[0.], zgrid])
        self.rgrid = self.distance(self.zgrid)
        from scipy import interpolate
        self.interp = interpolate.UnivariateSpline(self.rgrid, self.zgrid, k=interp_order, s=0)

    def __call__(self, distance):
        """Return (interpolated) redshift at distance ``distance`` (scalar or array)."""
        return self.interp(distance)

# ---------------------------------------------------------------------------------------------------- #    
# Usefull fonction for cosmological computation

def chi_to_z(s, cosmo):
    return DistanceToRedshift(cosmo.get_background().comoving_radial_distance)(s)

def dchi_dz(z, cosmo):
    # attention aux unités
    H0 = 100*1000 # en m.s^-1.h.Mpc^-1
    return constants.c/(H0*cosmo.get_background().efunc(z))

def chi(z, cosmo):
    ## attention ici ca marche car on travail dans un univers plat !
    return cosmo.get_background().comoving_radial_distance(z)

# ---------------------------------------------------------------------------------------------------- #  
## Class : PowerSpectrum / CorrelationFunction / AngularCorrelationFunction


## TODO : faire un cas pour eviter l'erreur qaund k > 100 pour le spectre de puissance --> mettre 0 ?
class PowerSpectrum(object):
    """
    Implementation of the Power spectrum with f_nl following Slosar 2008
    """
    
    def __init__(self, tracer, fnl=0.0):
        """
        Initialize :class:`PowerSpectrum`
        
        Parameters
        ----------
        tracer : Tracer Class
                 tracer class containing the description of the tracer
        fnl : float
              parameter to depict the local non-gaussianity type (following the CMB convention)

        """
        
        # tracer parameters
        self.tracer = tracer.copy()
        
        self.fnl = fnl
        
        # cosmo / background / fourier
        self.bg = self.tracer.cosmo.get_background()
        self.fo = self.tracer.cosmo.get_fourier()
        
        # Compute the linear power spectrum at z0 with fo 
        self.Plin = self.fo.pk_interpolator().to_1d(z=self.tracer.z0)
        
        # Compute the amplitude for Delta_b
        delta_c = 1.686 #the spherical collapse linear over-density
        DH_inv = self.bg.H0/(constants.c/1000) # we want c in km/s  H0 s/km
        Omega0 = self.bg.Omega_m(0.0)
        self.amp = 3*self.fnl*(self.tracer.bias - self.tracer.pop)*delta_c*Omega0*DH_inv**2
    
    
    def set_Plin_from_array(self, k, new_Plin, keep_bias=True):
        """
        Modify the value of Plin from k, new_Plin array. Help us to propagate the errors since 
        we will suppose that they are independant and everything is linear !
        
        Parameters
        ----------
        k : array_like
            wavelenght on which new_Plin is evaluated
        new_Plin : array_like
            the new value at k of Plin
        with_bias : bool
            if False set bias to one and amp to zeros
            if True keep the same attributes.
        """
        
        self.Plin = PowerSpectrumInterpolator1D(k, new_Plin)
        
        if ~keep_bias:
            # self(k) will be exactly self.Plin(k) 
            self.tracer.bias = 1.0
            self.fnl = 0.0
            self.amp = 0.0
        
        
    def T_from_class(self, k):
        """
            Compute the CLASS linear transfer function at z0 (copy and adapt from nbodykit)
            using:
            .. math::

                T(k) = \left [ P_L(k) / k^n_s  \right]^{1/2}.

            We normalize the transfer function :math:`T(k)` to unity as
            :math:`k \rightarrow 0` at :math:`z=0`.

            Parameters
            ---------
            k : float, array_like
                the wavenumbers in units of :math:`h \mathrm{Mpc}^{-1}`

            Returns
            -------
            Tk : float, array_like
                the transfer function evaluated at ``k``, normalized to unity on
                large scales
            """

        # find the low-k amplitude to normalize to 1 as k->0 at z = 0
        #KMIN = 1e-8

        k = np.asarray(k)
        nonzero = k>0

        linearP = self.Plin(k[nonzero]) / self.tracer.cosmo['h']**3 # in Mpc^3
        primordialP = (k[nonzero]*self.tracer.cosmo['h'])**self.tracer.cosmo['n_s'] # put k into Mpc^{-1}

        # return shape
        Tk = np.ones(nonzero.shape)

        # at k=0, this is 1.0 * D(z), where T(k) = 1.0 at z=0
        Tk[~nonzero] = self.bg.growth_factor(self.tracer.z0)

        # fill in all k>0
        Tk[nonzero] = (linearP / primordialP)**0.5

        # Normalize the transfert function at low-k amplitude
        Tk /= Tk[0]

        return Tk

    
    def __call__(self, k):
        """
        Compute the power spectrum for k
        
        Parameters
        ----------
        k : float, array_like
            Array of wavelenght on which the power spectrum will be evaluated
        Returns
        -------
        P : float, array_like
            The computed power spectrum. Size of k
            
        """
        
        #attention : lors d'un super, call ne peut etre appele --> faire eval appele dans call
        
        Pk_total =  self.Plin(k) * (self.tracer.bias + self.amp/(self.T_from_class(k)*k**2))**2
        return Pk_total
    
    
    def pk_to_xi(self, k):
        """
        Compute the correlation function throught PowerToCorrelation function of cosmoprimo which used FFT log algorithm.
        
        Parameters
        ----------
        k : float, array_like
            Array of wavelenght in which the power spectrum will be evaluated
            
        Returns
        -------
        Xi : CorrelationFunction class
        """
        r, Xi = PowerToCorrelation(k)(self(k))
        return CorrelationFunction(self.tracer, r, Xi)

    
    def pk_to_cl(self, ell):            
        """
        Compute the angular power spectra from the power spectrum for the considered tracor.
        .. math::

            C_{\ell} = \dfrac{2}{\pi} \int_0^{\infty} f_{\ell}(k)^2  P(k) k^2 dk

        Parameters
        ---------
        l : int, array_like
            order of the spherical bessel function

        Returns
        -------
        Cl : float, array_like
        """
        
        def compute_f_l_k(ell, dn_dz, cosmo):
            """
            Compute the integration of the spherical bessel function 
            weighted by the selection function using the fftlog implementation
            of cosmoprimo in CorrelationToPower
            .. math::

                f_{\ell}(k) = \int_0^{\infty} j_{\ell}(kr(z)) \dfrac{dn}{dz} dz 

            Parameters
            ---------
            ell : int, array_like
                order of the spherical bessel function
            dn_dz : callable function
                    return the n(z) at a consider redshift
            cosmo : cosmoprimo cosmo instance
                    Necessary to compute dchi_dz

            Returns
            -------
            f_l_k : float, array_like of shape (k.size, l.size)
                    Spherical bessel function evaluated in l, k  
            """
            
            def kernel_to_integrate(s, dn_dz, cosmo):
                z_at_s = chi_to_z(s, cosmo)
                return 1/s**2 * dn_dz(z_at_s) / dchi_dz(z_at_s, cosmo)  
            # array to perform the fftlog
            s = np.logspace(-2, 3.5, 1000)
            # fftlog
            k, f_k = CorrelationToPower(s, ell=ell)(kernel_to_integrate(s, dn_dz, cosmo))
            # attention à la normalisation de CorrelationToPower
            return k, np.real(f_k/(4*np.pi*((-1j)**ell[:, None])))

        k, f_l_k = compute_f_l_k(ell, self.tracer.dn_dz, self.tracer.cosmo)
        delta_k = k[:, 1:] - k[:, :-1]
        # attention a la taille de delta_k
        return 2/np.pi * np.sum(f_l_k[:,:-1]**2*k[:,:-1]**2*delta_k*self(k[:,:-1]), axis=1)
    
    
    def sigma_P(self, k):
        """
        Give PowerSpectrum theoritical errors following Feldman94  
        
        **REMARK:** For now, Cov_P is always diagonal
        
        Parameters
        ----------
        k : array_like
            Array of wavelenght on which the power spectrum will be evaluated
        
        Returns
        -------
        sigma_P : float, array_like
            The predicted theoritical errors for the power spectrum.
            
        """
        
        sigma_P = np.sqrt(2/self.tracer.V_survey)
        
        if(self.tracer.shot_noise_limited):
            sigma_P *= (self(k) + 1/self.tracer.n_survey)
        
        return sigma_P
    
    
    def generate_data(self, k, N_sim=1, bin_errors=True):
        """
        Generate data from Power spectrum with errors based on Feldman1994 (see Seo2003 or Greib2015)
        
        Warning: We decrease the size of k of 1 to compute the delta_k ! 
        This is necessary because the errors depend on the volume of the bin ect ...
        
        Parameters
        ----------
        k : array_like
            Array of wavelenght on which the power spectrum will be evaluated
        N_sim : int
            Number of sample generated
        bin_errors : bool
            if True, use bin mean covariance value for the error
            
        Returns
        -------
        k_gen : array_like
            Wavelength on which the data are generated
        Pk_gen : array_like
            Generated power spectrum with gaussian distribution around the expected errors --> size N_sim x k_gen.size 
            --> You can do : plt.plot(k_gen, Pk_gen.T) to see each sample
        sigma_Pk : array_like
            The expected errors
        """
        # bin width
        dk= k[1:] - k[:-1]
        # define k in the middle of the bin
        k_gen = k[1:] - dk/2

        sigma_Pk = self.sigma_P(k_gen)
        # Error for binned power spectrum estimator --> mean over the k-bin
        if bin_errors:
            sigma_Pk *= np.sqrt((2*np.pi)**3 / (4*np.pi*dk*k_gen**2)) 

        # generate data with errors:
        delta = np.random.normal(size=(N_sim, k_gen.size))
        
        Pk_gen = np.broadcast_to(self(k_gen), (N_sim, k_gen.size)) + np.broadcast_to(sigma_Pk, (N_sim, k_gen.size))*delta
        
        # to avoid to call Pk_gen[0] to have access to the only sample
        if N_sim == 1:
            Pk_gen = Pk_gen[0]

        return k_gen, Pk_gen, sigma_Pk

    
class CorrelationFunction(object):
    """
    Implementation of the Correlation function. For now, initialisation with PowerSpectrum.pk_to_xi()
    """
    
    def __init__(self, tracer, s, xi):
        """
        Initialise :class:`CorrelationFunction`
        Parameters
        ----------
        tracer : Tracer Class
                 tracer class containing the description of the tracer
        s : array_like
            Sorted array contaning the comoving distance from PowerToCorrelation(k)
            The CorrelationFunction will be only defined on this interval. Assume to be ordered
        Xi : array_like 
            Array contaning the correlation function from PowerToCorrelation(k)
        """
        
        self.tracer = tracer
        self.s_ref = s
        self.s_ref_min = np.min(self.s_ref)
        self.s_ref_max = np.max(self.s_ref)
        self.Xi = CorrelationFunctionInterpolator1D(s, xi)
    
    
    def __call__(self, s):
        """
        Compute the correlation function for s
        
        Parameters
        ----------
        s : float, array_like
            Array of comoving distance on which the correlation function will be evaluated
        Returns
        -------
        Xi_eval : float, array_like
            The computed correlation function. Size of s. 
            Return Nan values when s is not included in self.s_ref (the interpolation cannot work)
        """
        
        Xi_eval = np.NaN * np.zeros(s.shape)
        mask = (s < self.s_ref_min) | (s > self.s_ref_max)
        Xi_eval[~mask] = self.Xi(s[~mask])
        
        return Xi_eval

    
    def xi_to_w(self, theta, z_min=0., z_max=5.0, nbins=700):
        """ 
        Compute the angular correlation from the 3d correlation function.
        # speed up the process with numpy matrix computation 
        # test with broadcasting or [:, :, None] to avoid the k loop --> does not decrease time computation
        args: 
            * theta (numpy array): angles given in degrees
            * z_min, z_max, nbins (float, float, int): they define the range of the integration 
              and the bin size for the numerical estimation of the integral.
            * display_tqdm (bool): display or not the tqdm advance bar
        return:
            * w (numpy array): the angular correlation with same size than theta.

        """
        z1, z2 = np.linspace(z_min, z_max, nbins), np.linspace(z_min, z_max, nbins) 
        dz1, dz2 = (z1[1] - z1[0]), (z2[1] - z2[0])
        Z1, Z2 = np.meshgrid(z1, z2)

        # Compute these values only once ! (same for every angle)
        f1xf2 = self.tracer.dn_dz(Z1)*dz1 * self.tracer.dn_dz(Z2)*dz2

        chi1, chi2 = chi(Z1, self.tracer.cosmo), chi(Z2, self.tracer.cosmo)
        chi1xx2_plus_chi2xx2 = chi1**2 + chi2**2
        chi1xchi2 = chi1 * chi2

        cos_theta = np.cos(theta * np.pi/180)

        # to store the result
        w = np.zeros(theta.size)

        for k in range(theta.size):
            # integration methode des rectangles point gauche
            r_12 = np.sqrt(chi1xx2_plus_chi2xx2 - 2*chi1xchi2*cos_theta[k])
            # compute only where Xi is well defined with the interpolation
            mask = (r_12 < self.s_ref_min) | (r_12 > self.s_ref_max)
            w[k] = np.nansum(self(r_12[~mask]) * f1xf2[~mask])
            
        return AngularCorrelationFunction(self.tracer, theta, w)

    
    def set_cov_with_theory(self, Pk, k):
        """
        Compute analytically the theoric covariance of the Correlation Function 
        Parameters
        ----------
        Pk : PowerSpectrum()
        k : float, array_like
            k vector with which the integration will be performed. Need to be in logspace
        Returns
        -------
        self.cov : callable 2d function with NaN value when the interpolation cannot be performed.
        """

        def integrand(Pk, k, r_prime):
            return np.sinc(k*r_prime / np.pi) * Pk.sigma_P(k)**2

        int_j0_calculator = PowerToCorrelation(k, ell=0)

        # collect the TF output space
        r_prime, _ = int_j0_calculator(Pk(k))

        xi_cov = np.zeros((r_prime.size, r_prime.size))
        # la ft donne tous les points d'un coup donc on ne va pas modifier le code pour avoir uniquement qu'une triangulaire ... -> ok !
        for i in tqdm.tqdm(range(r_prime.size)):
            _, tmp = int_j0_calculator(integrand(Pk, k, r_prime[i]))
            xi_cov[:, i] = tmp

        # On a un problem avec la triangulaire supérieure ..
        # On ne garde donc que le calcul de la triangulaire inférieure
        xi_cov =  np.tril(xi_cov) + np.tril(xi_cov, k=-1).T

        self.cov = interp2d(r_prime, r_prime, xi_cov, bounds_error=False, fill_value=np.NaN)

        
    def set_cov_with_sim(self, Pk, k, N_sim):
        """
        Compute the covariance of the Correlation Function generating data with errors in k-space
        Parameters
        ----------
        Pk : PowerSpectrum()
        k : float, array_like
            k vector with which the data will be generated
        N_sim : int
            Number of data used to compute the covariance matrix
        Returns
        -------
        self.cov : callable 2d function with NaN value when the interpolation cannot be performed.
        """
        k_sim, Pk_sim, _ = Pk.generate_data(k, N_sim, bin_errors=False)
        s_sim, Xi_sim = PowerToCorrelation(k_sim)(Pk_sim)

        self.cov =  interp2d(s_sim, s_sim, np.cov(Xi_sim.T), bounds_error=False, fill_value=np.NaN) 
    
    
    def compute_covariance_matrix_for_bin(self, s_edge):
        """
        Compute the average bin covariance
        
        Parameters
        ----------
        s_edge : float, array_like
            comoving bin edge on which the covariance have to be computed
            
        Returns
        -------
        cov_matrix : float array_like
            average bin covariance used for generating sample of size s_edge-1 x s_edge-1
        """
        
        delta_s = s_edge[1:] - s_edge[:-1]
        s_sim = (s_edge[1:] + s_edge[:-1])/2
        
        cov_matrix = np.zeros((s_sim.size, s_sim.size))
        
        for i in tqdm.tqdm(range(s_sim.size)):
            for j in range(s_sim.size):
                r, r_prime = np.linspace(s_sim[i] - delta_s[i]/2, s_sim[i] + delta_s[i]/2, 50), np.linspace(s_sim[j] - delta_s[j]/2, s_sim[j] + delta_s[j]/2, 50)
                dr, dr_prime = r[1] - r[0], r_prime[1] - r_prime[0]
                cov_matrix[i, j] = np.sum(self.cov(r, r_prime) * (r**2*dr)[None, :] * (r_prime**2*dr_prime)[:, None]) / np.sum((r**2*dr)[None, :] * (r_prime**2*dr_prime)[:, None])
        return cov_matrix
        
        
    def generate_data(self, s_edge, N_sim=1, bin_errors=True):
        """
        Generate N_sim sample using a multivariate normal law following the covariance of the correlation function.
        Warning: if bin_errors = False, the errors are not for the 'binned correlation function' --> need to take the mean of the covariance matrix.
        
        Parameters
        ----------
        s_sim : float, array_like
            comoving bin edge on which the data will be generated 
        N_sim : float, array_like
            Number of simulation required
        bin_errors : bool
            if True, use bin mean covariance value for the error
            
        Returns
        -------
        s_sim : array_like
            vctor on which the data are generated

        Xi_sim : float array_like
            Xi generated thanks to a multivariate normal law
            
        cov_matrix : float array_like of size s_sim x s_sim
        """
        
        if not ('cov' in self.__dict__.keys()):
            raise ValueError("Set Covariance before generating data...")
        else:
            s_sim = (s_edge[1:] + s_edge[:-1]) / 2
            if bin_errors:
                cov_matrix = self.compute_covariance_matrix_for_bin(s_edge)
            else:
                cov_matrix = self.cov(s_sim, s_sim)
                
        return s_sim, np.random.multivariate_normal(self(s_sim), cov_matrix, N_sim), cov_matrix
    

class AngularCorrelationFunction(object):
    """
    Implementation of the Angular Correlation function. For now, initialisation from CorrelationFunction.xi_to_w()
    """
    def __init__(self, tracer, theta, w):
        """
        Initialise :class:`CorrelationFunction`
        Parameters
        ----------
        tracer : Tracer Class
                 tracer class containing the description of the tracer
        theta : array_like
                Sorted array contaning the angular distance (degree).
                The AngularCorrelationFunction will be only defined on this interval. Assume to be ordered
        Xi : array_like 
            Array contaning the correlation function from PowerToCorrelation(k)
        """
        
        self.tracer = tracer
        self.theta_ref = theta
        self.w = CorrelationFunctionInterpolator1D(theta, w)
    
    
    def __call__(self, theta):
        """
        Compute the correlation function for theta
        
        Parameters
        ----------
        theta : float, array_like
            Array of angular distance (degree) on which the angular correlation function will be evaluated
        Returns
        -------
        w_eval : float, array_like
            The computed angular correlation function. Size of theta
            Return Nan values when theta is not included in self.theta_ref (the interpolation cannot work)
        """
        
        w_eval = np.NaN * np.zeros(theta.shape)
        mask = (theta < self.theta_ref[0]) | (theta > self.theta_ref[-1])
        w_eval[~mask] = self.w(theta[~mask])
        
        return w_eval

    
    def set_cov_with_theory(self, Xi, theta, z_min=0., z_max=5.0, nbins=200):
        """
        Compute analytically the covariance of the Angular Correlation Function using the correlation function covariance
        Parameters
        ----------
        Xi : CorrelationFunction() with covariance set
        theta : float, array_like
            theta vector with which the covariance matrix will be generated
        N_sim : int
            Number of data used to compute the covariance matrix
        Returns
        -------
        self.cov : callable 2d function with NaN value when the interpolation cannot be performed.
        """
        if not hasattr(Xi, 'cov'):
            raise ValueError("Set Covariance before generating data...")
        else:
            # symetric Z1, Z2 / Z3, Z4

            z_tmp  = np.linspace(z_min, z_max, nbins) 
            dz = (z_tmp[1] - z_tmp[0])
            Z1, Z2 = np.meshgrid(z_tmp, z_tmp)

            # Compute these values only once ! (same for every angle)
            f1xf2 = Xi.tracer.dn_dz(Z1)*dz * Xi.tracer.dn_dz(Z2)*dz

            chi1, chi2 = chi(Z1, self.tracer.cosmo), chi(Z2, self.tracer.cosmo)
            chi1xx2_plus_chi2xx2 = chi1**2 + chi2**2
            chi1xchi2 = chi1 * chi2

            cos_theta= np.cos(theta * np.pi/180)

            w_cov = np.zeros((theta.size, theta.size))

            # on utilise un mask pour ne pas faire des calculs inutile
            for k in tqdm.tqdm(range(theta.size)):
                r_12 = np.sqrt(chi1xx2_plus_chi2xx2 - 2*chi1xchi2*cos_theta[k])
                mask = (r_12 < Xi.s_ref_min) | (r_12 > Xi.s_ref_max)
                # la matrice est symmétrique (pas de calcul inutile !)
                for l in range(k+1):
                    r_12_prime = np.sqrt(chi1xx2_plus_chi2xx2 - 2*chi1xchi2*cos_theta[l])
                    mask_tot = mask | (r_12_prime < Xi.s_ref_min) | (r_12_prime > Xi.s_ref_max)

                    res = Xi.cov(r_12[~mask_tot], r_12_prime[~mask_tot])
                    res *= f1xf2[~mask_tot] * f1xf2[~mask_tot]
                    w_cov[k, l] = np.nansum(res)

            w_cov += np.tril(w_cov, k=-1).T
    
            self.cov = interp2d(theta, theta, w_cov, bounds_error=False, fill_value=np.NaN)
    
    
    def set_cov_with_sim(self, Pk, k_gen, theta_sim, N_sim):
        """
        Compute the covariance of the Angular Correlation Function generating data with errors in k-space
        Parameters
        ----------
        Pk : PowerSpectrum()
        k : float, array_like
            k vector with which the data will be generated
        N_sim : int
            Number of data used to compute the covariance matrix
        Returns
        -------
        :attr:`cov` : callable 2d function with NaN value when the interpolation cannot be performed.
        """
        k_sim, Pk_sim, _ = Pk.generate_data(k_gen, N_sim, bin_errors=False)
        s_sim, Xi_sim = PowerToCorrelation(k_sim)(Pk_sim)
    
        w_sim = []
        for i in tqdm.tqdm(range(N_sim)):
            w_sim += [CorrelationFunction(Pk.tracer, s_sim, Xi_sim[i,:]).xi_to_w(theta_sim, nbins=500)(theta_sim)]
        w_sim = np.array(w_sim)    
        
        self.cov = interp2d(theta_sim, theta_sim, np.cov(w_sim.T), bounds_error=False, fill_value=np.NaN)

    
    def compute_covariance_matrix_for_bin(self, theta_edge):
        """
        Compute the average bin covariance
        
        Parameters
        ----------
        theta_edge : float, array_like
            comoving bin edge on which the covariance have to be computed
            
        Returns
        -------
        cov_matrix : float array_like
            average bin covariance used for generating sample of size theta_edge-1 x theta_edge-1
        """
        
        delta_theta = theta_edge[1:] - theta_edge[:-1]
        theta_sim = (theta_edge[1:] + theta_edge[:-1])/2
        
        cov_matrix = np.zeros((theta_sim.size, theta_sim.size))
        
        for i in tqdm.tqdm(range(theta_sim.size)):
            for j in range(theta_sim.size):
                print("NOOOON IL FAT ETRE EN RADIAN POUR LE SIN et mettre les dtheta")
                theta, theta_prime = np.linspace(theta_sim[i] - delta_theta[i]/2, theta_sim[i] + delta_theta[i]/2, 50), np.linspace(theta_sim[j] - delta_theta[j]/2, theta_sim[j] + delta_theta[j]/2, 50)
                cov_matrix[i, j] = np.sum(self.cov(theta, theta_prime) * np.sin(np.radians(theta))[None, :] * np.sin(np.radians(theta_prime))[None, :]) / np.sum(np.sin(np.radians(theta))[None, :] * np.sin(np.radians(theta_prime))[None, :])
        return cov_matrix
        
        
    def generate_data(self, theta_edge, N_sim=1, bin_errors=True):
        """
        Generate N_sim sample using a multivariate normal law following the covariance of the correlation function.
        Warning: if bin_errors = False, the errors are not for the 'binned correlation function' --> need to take the mean of the covariance matrix.
        
        Parameters
        ----------
        theta_edge : float, array_like
            comoving bin edge on which the data will be generated 
        N_sim : float, array_like
            Number of simulation required
        bin_errors : bool
            if True, use bin mean covariance value for the error
            
        Returns
        -------
        theta_sim : array_like
            vctor on which the data are generated

        w_sim : float array_like
            Xi generated thanks to a multivariate normal law
            
        cov_matrix : float array_like of size theta_sim x theta_sim
        """
        
        if not ('cov' in self.__dict__.keys()):
            raise ValueError("Set Covariance before generating data...")
        else:
            theta_sim = (theta_edge[1:] + theta_edge[:-1]) / 2
            if bin_errors:
                cov_matrix = self.compute_covariance_matrix_for_bin(theta_edge)
            else:
                cov_matrix = self.cov(theta_sim, theta_sim)
                
        return theta_sim, np.random.multivariate_normal(self(theta_sim), cov_matrix, N_sim), cov_matrix