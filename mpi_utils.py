"""
Copy from mockfactory: https://github.com/adematti/mockfactory/blob/736ef267159fefc59e01709c09fd47b1eeb7a3d0/mockfactory/mpi.py#L89
which use extensively nbodykit:
MPI routines, many taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/__init__.py
and https://github.com/bccp/nbodykit/blob/master/nbodykit/batch.py.


Contain method to gather numpy array across all the process without the limitation of 2GB from comm.gather()

see appendix H: http://www.pik-potsdam.de/~bloh/pdffile/pe_mpi_prog_guide_v3r10.pdf

"""

import logging
import functools
from contextlib import contextmanager

import numpy as np
from mpi4py import MPI


class CurrentMPIComm(object):
    """Class to faciliate getting and setting the current MPI communicator."""
    logger = logging.getLogger('CurrentMPIComm')

    _stack = [MPI.COMM_WORLD]

    @staticmethod
    def enable(func):
        """
        Decorator to attach the current MPI communicator to the input
        keyword arguments of ``func``, via the ``mpicomm`` keyword.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            kwargs.setdefault('mpicomm', None)
            if kwargs['mpicomm'] is None:
                kwargs['mpicomm'] = CurrentMPIComm.get()
            return func(*args, **kwargs)

        return wrapper

    @classmethod
    @contextmanager
    def enter(cls, mpicomm):
        """
        Enter a context where the current default MPI communicator is modified to the
        argument `comm`. After leaving the context manager the communicator is restored.
        Example:
        .. code:: python
            with CurrentMPIComm.enter(comm):
                cat = UniformCatalog(...)
        is identical to
        .. code:: python
            cat = UniformCatalog(..., comm=comm)
        """
        cls.push(mpicomm)

        yield

        cls.pop()

    @classmethod
    def push(cls, mpicomm):
        """Switch to a new current default MPI communicator."""
        cls._stack.append(mpicomm)
        if mpicomm.rank == 0:
            cls.logger.info('Entering a current communicator of size {:d}'.format(mpicomm.size))
        cls._stack[-1].barrier()

    @classmethod
    def pop(cls):
        """Restore to the previous current default MPI communicator."""
        mpicomm = cls._stack[-1]
        if mpicomm.rank == 0:
            cls.logger.info('Leaving current communicator of size {:d}'.format(mpicomm.size))
        cls._stack[-1].barrier()
        cls._stack.pop()
        mpicomm = cls._stack[-1]
        if mpicomm.rank == 0:
            cls.logger.info('Restored current communicator to size {:d}'.format(mpicomm.size))

    @classmethod
    def get(cls):
        """Get the default current MPI communicator. The initial value is ``MPI.COMM_WORLD``."""
        return cls._stack[-1]


@CurrentMPIComm.enable
def gather_array(data, root=0, mpicomm=None):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    Gather the input data array from all ranks to the specified ``root``.
    This uses `Gatherv`, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype
    Parameters
    ----------
    data : array_like
        The data on each rank to gather.
    root : int, Ellipsis, default=0
        The rank number to gather the data to. If root is Ellipsis or None,
        broadcast the result to all ranks.
    mpicomm : MPI communicator, default=None
        The MPI communicator.
    Returns
    -------
    recvbuffer : array_like, None
        the gathered data on root, and `None` otherwise
    """
    if root is None: root = Ellipsis

    if np.isscalar(data):
        if root is Ellipsis:
            return np.array(mpicomm.allgather(data))
        gathered = mpicomm.gather(data, root=root)
        if mpicomm.rank == root:
            return np.array(gathered)
        return None

    if not isinstance(data, np.ndarray):
        raise ValueError('`data` must be numpy array in gather_array')

    # need C-contiguous order
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    local_length = data.shape[0]

    # check dtypes and shapes
    shapes = mpicomm.allgather(data.shape)
    dtypes = mpicomm.allgather(data.dtype)

    # check for structured data
    if dtypes[0].char == 'V':

        # check for structured data mismatch
        names = set(dtypes[0].names)
        if any(set(dt.names) != names for dt in dtypes[1:]):
            raise ValueError('mismatch between data type fields in structured data')

        # check for 'O' data types
        if any(dtypes[0][name] == 'O' for name in dtypes[0].names):
            raise ValueError('object data types ("O") not allowed in structured data in gather_array')

        # compute the new shape for each rank
        newlength = mpicomm.allreduce(local_length)
        newshape = list(data.shape)
        newshape[0] = newlength

        # the return array
        if root is Ellipsis or mpicomm.rank == root:
            recvbuffer = np.empty(newshape, dtype=dtypes[0], order='C')
        else:
            recvbuffer = None

        for name in dtypes[0].names:
            d = gather_array(data[name], root=root, mpicomm=mpicomm)
            if root is Ellipsis or mpicomm.rank == root:
                recvbuffer[name] = d

        return recvbuffer

    # check for 'O' data types
    if dtypes[0] == 'O':
        raise ValueError('object data types ("O") not allowed in structured data in gather_array')

    # check for bad dtypes and bad shapes
    if root is Ellipsis or mpicomm.rank == root:
        bad_shape = any(s[1:] != shapes[0][1:] for s in shapes[1:])
        bad_dtype = any(dt != dtypes[0] for dt in dtypes[1:])
    else:
        bad_shape, bad_dtype = None, None

    if root is not Ellipsis:
        bad_shape, bad_dtype = mpicomm.bcast((bad_shape, bad_dtype), root=root)

    if bad_shape:
        raise ValueError('mismatch between shape[1:] across ranks in gather_array')
    if bad_dtype:
        raise ValueError('mismatch between dtypes across ranks in gather_array')

    shape = data.shape
    dtype = data.dtype

    # setup the custom dtype
    duplicity = np.product(np.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newlength = mpicomm.allreduce(local_length)
    newshape = list(shape)
    newshape[0] = newlength

    # the return array
    if root is Ellipsis or mpicomm.rank == root:
        recvbuffer = np.empty(newshape, dtype=dtype, order='C')
    else:
        recvbuffer = None

    # the recv counts
    counts = mpicomm.allgather(local_length)
    counts = np.array(counts, order='C')

    # the recv offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # gather to root
    if root is Ellipsis:
        mpicomm.Allgatherv([data, dt], [recvbuffer, (counts, offsets), dt])
    else:
        mpicomm.Gatherv([data, dt], [recvbuffer, (counts, offsets), dt], root=root)

    dt.Free()

    return recvbuffer
