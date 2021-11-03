Usefull functions for DESI
==========================

This package contains usefull functions/routines for my work in [DESI](https://www.desi.lbl.gov/) (targets selection and clustering analysis with quasars)

**Author:** Edmond Chaussidon (Phd student at IRFU/CEA-Saclay)
**Contact:** edmond.chaussidon@cea.fr

License :
---------

Feel free to use/modify these scripts while you link and quote them to you work. Please contact the author to obtain the corresponding reference.

Python code :
-------------

## Tool:
  * *wrapper*: contains usefull wrappers like *time_measurement*
  * *logger*: Parametrize logger
  * *linear_regression*: Quick and usefull implementation of least-square regression with iminuit
  * *plot*: cartesian and mollweide visualization of a HEALPix map
  * *plot_spectre**: plot spectrum from coadd file in NERSC

## Matplotlib style:
  * *ec_style.mplstyle*: basic style for matplotlib (**Warning:** latex is activated by default)
  * *ec_style_article.mplsyle*: basic style for matplotlib with smaller size corresponding to a 2 column paper.

## TS and systematics:
  * *systematics*: function to calculate systematic plot and parametrization for each features
  * *corr*: routines to read/write/compute data (input/output) to CUTE (OUTDATED) (see comments)
  * *limber*: Limber function implementation and fits (used Nbodykit for cosmo.Planck15)
  * *ts_utils*: utils to load RF probability, apply cut, draw systematics from a pixmap, ect...
  * *desi_footprint*: load information about DESI footprint and the different photometric footprint in function of the different tracer.

## Clustering:
  * *cosmo_tracer*: contain class tracer to load easily the information for the different DESI tracers. (the corresponding dn_dz is in Data)
  * *cosmo_ps*: Implemenation of the Powerspectrum / CorrelationFunction class with fnl
  * *tpcf*: utils to load fits file, apply selection and compute Correlation function from CUTE and NOW pycorr


*?? :
----
