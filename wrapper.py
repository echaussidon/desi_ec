import functools
import time

#------------------------------------------------------------------------------#
# TIME

def time_measurement(func):
    """Timestamp decorator for dedicated functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        execTime = time.time() - startTime
        mlsec = repr(execTime).split('.')[1][:3]
        readable = time.strftime("%H:%M:%S.{}".format(mlsec), time.gmtime(execTime))
        print('----> Function "{}" took : {} sec \n'.format(func.__name__, readable))
        return result
    return wrapper
