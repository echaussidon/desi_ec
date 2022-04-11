"""
Copy from mockfactory: https://github.com/adematti/mockfactory/blob/736ef267159fefc59e01709c09fd47b1eeb7a3d0/mockfactory/mpi.py#L89
which use extensively nbodykit:
MPI routines, many taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/__init__.py
and https://github.com/bccp/nbodykit/blob/master/nbodykit/batch.py.

MPI utils tool box !! --> cf mackfactory/mpi.py for more functions ect ...

Contain method to gather numpy array across all the process without the limitation of 2GB from comm.gather()

see appendix H: http://www.pik-potsdam.de/~bloh/pdffile/pe_mpi_prog_guide_v3r10.pdf

"""

import logging
import functools
from contextlib import contextmanager

import numpy as np
from mpi4py import MPI


# ca permet de passer com=None et avec @enable, il va chercher le comm tout seul !
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


@CurrentMPIComm.enable
def broadcast_array(data, root=0, mpicomm=None):
    """
    Broadcast the input data array across all ranks, assuming `data` is
    initially only on `root` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype.
    Parameters
    ----------
    data : array_like or None
        On `root`, this gives the data to broadcast.
    root : int, default=0
        The rank number that initially has the data.
    mpicomm : MPI communicator, default=None
        The MPI communicator.
    Returns
    -------
    recvbuffer : array_like
        The chunk of `data` that each rank gets.
    """

    # check for bad input
    if mpicomm.rank == root:
        isscalar = np.isscalar(data)
    else:
        isscalar = None
    isscalar = mpicomm.bcast(isscalar, root=root)

    if isscalar:
        return mpicomm.bcast(data, root=root)

    if mpicomm.rank == root:
        bad_input = not isinstance(data, np.ndarray)
    else:
        bad_input = None
    bad_input = mpicomm.bcast(bad_input, root=root)
    if bad_input:
        raise ValueError('`data` must by numpy array on root in broadcast_array')

    if mpicomm.rank == root:
        # need C-contiguous order
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = mpicomm.bcast(shape_and_dtype, root=root)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in broadcast_array; please specify specific data type')

    # initialize empty data on non-root ranks
    if mpicomm.rank != root:
        np_dtype = np.dtype((dtype, shape))
        data = np.empty(0, dtype=np_dtype)

    # setup the custom dtype
    duplicity = np.product(np.array(shape, 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # the return array
    recvbuffer = np.empty(shape, dtype=dtype, order='C')

    # the send offsets
    counts = np.ones(mpicomm.size, dtype='i', order='C')
    offsets = np.zeros_like(counts, order='C')

    # do the scatter
    mpicomm.Barrier()
    mpicomm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt], root=root)
    dt.Free()
    return recvbuffer


@CurrentMPIComm.enable
def scatter_array(data, counts=None, root=0, mpicomm=None):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    Scatter the input data array across all ranks, assuming `data` is
    initially only on `root` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype
    Parameters
    ----------
    data : array_like or None
        On `root`, this gives the data to split and scatter.
    counts : list of int
        List of the lengths of data to send to each rank.
    root : int, default=0
        The rank number that initially has the data.
    mpicomm : MPI communicator, default=None
        The MPI communicator.
    Returns
    -------
    recvbuffer : array_like
        The chunk of `data` that each rank gets.
    """
    if counts is not None:
        counts = np.asarray(counts, order='C')
        if len(counts) != mpicomm.size:
            raise ValueError('counts array has wrong length!')

    # check for bad input
    if mpicomm.rank == root:
        bad_input = not isinstance(data, np.ndarray)
    else:
        bad_input = None
    bad_input = mpicomm.bcast(bad_input, root=root)
    if bad_input:
        raise ValueError('`data` must by numpy array on root in scatter_array')

    if mpicomm.rank == root:
        # need C-contiguous order
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = mpicomm.bcast(shape_and_dtype, root=root)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in scatter_array; please specify specific data type')

    # initialize empty data on non-root ranks
    if mpicomm.rank != root:
        np_dtype = np.dtype((dtype, shape[1:]))
        data = np.empty(0, dtype=np_dtype)

    # setup the custom dtype
    duplicity = np.product(np.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newshape = list(shape)

    if counts is None:
        newshape[0] = newlength = local_size(shape[0], mpicomm=mpicomm)
    else:
        if counts.sum() != shape[0]:
            raise ValueError('the sum of the `counts` array needs to be equal to data length')
        newshape[0] = counts[mpicomm.rank]

    # the return array
    recvbuffer = np.empty(newshape, dtype=dtype, order='C')

    # the send counts, if not provided
    if counts is None:
        counts = mpicomm.allgather(newlength)
        counts = np.array(counts, order='C')

    # the send offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # do the scatter
    mpicomm.Barrier()
    mpicomm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt], root=root)
    dt.Free()
    return recvbuffer


@CurrentMPIComm.enable
def local_size(size, mpicomm=None):
    """
    Divide global ``size`` into local (process) size.
    Parameters
    ----------
    size : int
        Global size.
    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.
    Returns
    -------
    localsize : int
        Local size. Sum of local sizes over all processes equals global size.
    """
    start = mpicomm.rank * size // mpicomm.size
    stop = (mpicomm.rank + 1) * size // mpicomm.size
    localsize = stop - start
    return localsize


@CurrentMPIComm.enable
def send_array(data, dest, tag=0, blocking=True, mpicomm=None):
    """
    Send input array ``data`` to process ``dest``.

    Parameters
    ----------
    data : array
        Array to send.

    dest : int
        Rank of process to send array to.

    tag : int, default=0
        Message identifier.

    blocking : bool, default=False
        Blocking?

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.
    """
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    shape, dtype = data.shape, data.dtype

    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in send_array; please specify specific data type')

    duplicity = np.product(np.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    if blocking: send, Send = mpicomm.send, mpicomm.Send
    else: send, Send = mpicomm.isend, mpicomm.Isend
    send((shape, dtype), dest=dest, tag=tag)
    Send([data, dt], dest=dest, tag=tag)


@CurrentMPIComm.enable
def recv_array(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, mpicomm=None):
    """
    Receive array from process ``source``.

    Parameters
    ----------
    source : int, default=MPI.ANY_SOURCE
        Rank of process to receive array from.

    tag : int, default=0
        Message identifier.

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.

    Returns
    -------
    data : array
    """
    shape, dtype = mpicomm.recv(source=source, tag=tag)
    data = np.zeros(shape, dtype=dtype)

    duplicity = np.product(np.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    mpicomm.Recv([data, dt], source=source, tag=tag)
    return data
