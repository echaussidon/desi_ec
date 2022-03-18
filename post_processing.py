import os
import logging
import argparse

import fitsio
import numpy as np

from mpi_utils import gather_array

from pypower import CatalogFFTPower, setup_logging


logger = logging.getLogger('Post processing')

# To avoid error from NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


def logger_info(logger, msg, rank, mpiroot=0):
    """Print something with the logger only for the rank == mpiroot to avoid duplication of message."""
    if rank == mpiroot:
        logger.info(msg)


def load_bigfile(path, dataset='1/', comm=None):
    """
    Load BigFile with BigFileCatalog removing FuturWarning. Just pass the commutator to spread the file in all the processes.

    Parameters
    ----------
    path : str
        Path where the BigFile is.
    dataset: str
        Which sub-dataset do you want to read from the BigFile ?
    comm : MPI commutator
        Pass the current commutator if you want to use MPI.

    Return
    ------
    cat : BigFileCatalog
        BigFileCatalog object from nbodykit.

    """
    from nbodykit.source.catalog.file import BigFileCatalog

    # to remove the following warning:
    # FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)

        logger_info(logger, f'Read {path}', comm.Get_rank())
        # read simulation output
        cat = BigFileCatalog(path, dataset=dataset, comm=comm)
        # add BoxSize attributs (mandatory for fof)
        cat.attrs['BoxSize'] = np.array([cat.attrs['boxsize'][0], cat.attrs['boxsize'][0], cat.attrs['boxsize'][0]])

    return cat


def load_fiducial_cosmo():
    """ Load fiducial DESI cosmology."""
    from cosmoprimo.fiducial import DESI
    # Load fiducial cosmology
    cosmo = DESI(engine='class')
    #precompute the bakcground
    bg = cosmo.get_background()
    return cosmo


def build_halos_catalog(particles, linking_length=0.2, nmin=5, particle_mass=1e12):
    """
    Determine the halos of Dark Matter with the FOF algorithm of nbody kit. The computation is parallelized with MPI if the file is open with a commutator.

    Parameters
    ----------
    particles : nbodykit BigFileCatalog
        Catalog containing the position of all the particle of the simulation. (Not necessary a BigFileCatalog)
    linking_lenght : float
        The linking length, either in absolute units, or relative to the mean particle separation.
    nmin = int
        Minimal number of particles to determine a halos.
    particle_mass : float
        Mass of each DM particle in Solar Mass unit.

    Return
    ------
    halos :  numpy dtype array

    """
    from nbodykit.algorithms.fof import FOF
    from nbodykit.algorithms.fof import fof_catalog

    # Run the fof algorithm
    fof = FOF(particles, linking_length, nmin)

    # build halos catalog:
    halos = fof_catalog(fof._source, fof.labels, fof.comm, peakcolumn=None, periodic=fof.attrs['periodic'])
    # remove halos with lenght == 0
    halos = halos[halos['Length'] > 0]

    # meta-data
    attrs = particles.attrs.copy()
    attrs['particle_mass'] = particle_mass

    return halos, attrs


def collect_argparser():
    pid = os.getpid()
    parser = argparse.ArgumentParser(description="Post processing of fastpm-python simulation. It run FOF halo finder, save the halo catalog into a .fits format. \
                                                  It computes also the power spectrum for the particle and the power spectrum for the halos with a given mass selection.")

    parser.add_argument("--path_to_sim", type=str, required=False, default='/global/u2/e/edmondc/Scratch/Mocks/',
                        help="Path to the Scratch where the simulations are saved")
    parser.add_argument("--sim", type=str, required=False, default='test',
                        help="Simulation name (e.g) fastpm-fnl-0")
    parser.add_argument("--aout", type=str, required=False, default='1.0000',
                        help="scale factor at which the particles are saved (e.g) '0.3300' or '1.0000'")

    parser.add_argument("--nmesh", type=int, required=False, default=1024,
                        help="nmesh used for the power spectrum computation")
    # parser.add_argument("--kedges", type=str, required=False, default='test',
    #                     help="Simulation name (e.g) fastpm-fnl-0")
    # parser.add_argument("--aout", type=str, required=False, default='1.0000',
    #                     help="scale factor at which the particles are saved (e.g) '0.3300' or '1.0000'")

    parser.add_argument("--min_mass_halos", type=float, required=False, default=1e13,
                        help="minimal mass of the halos to be kept")

    # Finir de mettre les parameteres libres en argpars

    return parser.parse_args()

if __name__ == '__main__':

    setup_logging()

    # to remove the following warning:
    # VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or
    # shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
    # self.edges = numpy.asarray(edges)
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    start_ini = MPI.Wtime()

    args = collect_argparser()

    sim = os.path.join(args.path_to_sim, args.sim)
    aout = args.aout

    start = MPI.Wtime()
    particles = load_bigfile(os.path.join(sim, f'fpm-{aout}'), comm=comm)
    logger_info(logger, f"Number of DM particles: {particles.csize} read in {MPI.Wtime() - start:2.2f} s.", rank)

    start = MPI.Wtime()
    # need to use wrap = True since some particles are outside the box
    # no neeed to select rank == 0 it is automatic in .save method
    CatalogFFTPower(data_positions1=particles['Position'].compute(), wrap=True, edges=np.geomspace(5e-3, 3e0, 80), ells=(0), nmesh=args.nmesh,
                    boxsize=particles.attrs['boxsize'][0], boxcenter=particles.attrs['boxsize'][0]//2, resampler='tsc', interlacing=2, los='x', position_type='pos',
                    mpicomm=comm).poles.save(os.path.join(sim, f'particle-power-{aout}.npy'))
    logger_info(logger, f'CatalogFFTPower with particles done in {MPI.Wtime() - start:2.2f} s.', rank)

    # take care if -N != 1 --> particles will be spread in the different nodes --> csize instead .size to get the full lenght
    start = MPI.Wtime()
    cosmo = load_fiducial_cosmo()
    particle_mass = (cosmo.get_background().rho_cdm(1/float(aout) - 1) + cosmo.get_background().rho_b(1/float(aout) - 1)) /cosmo.h *1e10 * particles.attrs['boxsize'][0]**3 / particles.csize # units: Solar Mass
    halos, attrs = build_halos_catalog(particles, nmin=int(args.min_mass_halos//particle_mass), particle_mass=particle_mass)
    logger_info(logger, f"Find halos (with nmin = {int(args.min_mass_halos//particle_mass)}) done in {MPI.Wtime() - start:.2f} s.", rank)

    start = MPI.Wtime()
    # collect all the halo in a single processor before to save it in fits format.
    halos = gather_array(halos, root=0, mpicomm=comm)
    if rank == 0:
        f = fitsio.FITS(os.path.join(sim, f'halos-{aout}.fits'), 'rw', clobber=True)
        f.write(halos, extname='HALOS')
        f['HALOS'].insert_column('Position', halos['CMPosition'])
        f['HALOS'].insert_column('Velocity', halos['CMVelocity'])
        f['HALOS'].insert_column('Mass', attrs['particle_mass'] * halos['Length'])
        f.close()
        logger.info(f"Save {halos.shape[0]} halos done in {MPI.Wtime() - start:2.2f} s.")

    start = MPI.Wtime()
    if rank == 0:
        position = halos['CMPosition'][(halos['Length'] * attrs['particle_mass']) >= args.min_mass_halos]
    else:
        position = None #halos does not exist anymore in rank != 0 after gather_array
    CatalogFFTPower(data_positions1=position, edges=np.geomspace(5e-3, 3e0, 80), ells=(0), nmesh=args.nmesh,
                    boxsize=attrs['boxsize'][0], boxcenter=attrs['boxsize'][0]//2, resampler='tsc', interlacing=2, los='x', position_type='pos',
                    mpicomm=comm, mpiroot=0).poles.save(os.path.join(sim, f'halos-power-{aout}.npy'))
    logger_info(logger, f'CatalogFFTPower with halos done in {MPI.Wtime() - start:2.2f} s.', rank)

    logger_info(logger, f"Post processing took {MPI.Wtime() - start_ini:2.2f} s.", rank)
