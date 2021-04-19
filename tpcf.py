# coding: utf-8
# Author : Edmond Chaussidon (CEA)
# function to read catalog, write input for CUTE to calculate 2pcf.

import sys
import os

import numpy as np
import fitsio
from astropy.io import ascii

#------------------------------------------------------------------------------#
def read_fits(filename):
    print('[INFO] Read fits file from : ', filename)
    return fitsio.FITS(filename)[1]


def make_selection(darray, criterions):
    # darray = catalog in darray type, like data[1][:] when fits file is open with fitsio.FITS
    # criterions = list of criterion : [feature_name, operation, value] only value is not in str

    def apply_criterion_for_selection(darray, criterion):
        feature_name, operation, value = criterion[0], criterion[1], criterion[2]
        print("    * ", feature_name, operation, value)
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

    sel = np.ones(darray.size, dtype=bool)
    print("[INFO] We apply the selection:")
    for criterion in criterions:
        sel &= apply_criterion_for_selection(darray, criterion)
    return sel


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
    print(f"[INFO] Write catalog in {filename} with {ra.size} points")


def CUTE_ini_file(param):
    if 'ini_filename' in param.keys():
        print(f"\n[INFO] Write {param['ini_filename']}")
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
        print(f"[WARNING] We use a previous calculation of RR which is in : {param['output_with_RR']} --> CHECK IF IT IS THE SAME RANDOMS AND THE CUTE PARAMETERS\n")
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
    file.write("use_pm= 0\n")
    file.write("n_pix_sph= 0\n")
    file.close()


def CUTE(param, nbr_nodes=4, nbr_threads=16, keep_trace_txt='output_cute.txt'):
    CUTE_ini_file(param)
    cute_ini = param['ini_filename']
    print(f"[INFO] RUN CUTE ({param['corr_type']}) for {cute_ini} with {nbr_nodes} nodes and {nbr_threads} threads. Terminal ouput is saved in {keep_trace_txt}\n")
    CUTE_CALL = f'mpiexec -np {nbr_nodes} /global/homes/e/edmondc/Software/CUTE/CUTE/CUTE  {cute_ini}'
    os.system(f"module load openmpi && module load gsl && export OMP_NUM_THREADS={nbr_threads} && {CUTE_CALL} |& tee {keep_trace_txt}")


def extract_cute_result(filename, return_dd=False) : #ok angular and the monopole are similar here : --> faire une fonction plus large et reorganiser patch ect .. (mettre dnas limber plutot)
    print("[INFO] Read cute result from ", filename)
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
        return r, xi, err_r, err_xi, dd, rr, dr, rd, rr
    else:
        return r, xi, err_r, err_xi
