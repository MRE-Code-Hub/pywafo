'''
Misc
'''


import fractions
import numbers
import sys
import warnings
from time import strftime, gmtime
import numpy as np
from numpy import (sqrt, arctan2, sin, cos, exp, log, log1p,
                   inf, pi, zeros, ones, meshgrid, ndarray)
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import gammaln, betaln  # pylint: disable=no-name-in-module
from scipy.integrate import trapezoid, simpson


from wafo.plotbackend import plotbackend as plt
from wafo._misc_numba import findrfc, findcross, findexrema
from wafo._nieslony_numba import findrfc_astm


FLOATINFO = np.finfo(float)
_tiny_name = 'tiny' if np.__version__ < '1.22' else 'smallest_normal'
_TINY = getattr(FLOATINFO, _tiny_name)
_EPS = FLOATINFO.eps

__all__ = ['now', 'spaceline', 'narg_smallest', 'args_flat', 'is_numlike',
           'JITImport', 'printf',
            'detrendma', 'ecross', 'findcross', 'findextrema',
           'findpeaks', 'findrfc', 'findtp', 'findtc', findrfc_astm,
           'findoutliers', 'common_shape', 'argsreduce', 'stirlerr',
           'getshipchar',
           'betaloge', 'gravity', 'nextpow2', 'discretize',
           'polar2cart', 'cart2polar', 'pol2cart', 'cart2pol',
           'meshgrid', 'ndgrid', 'trangood', 'tranproc',
           'plot_histgrm', 'num2pistr',
           'lazywhere', 'lazyselect',
           'lfind',
           'moving_average', 'moving_max', 'moving_median',
           'piecewise',
           'check_random_state', 'to_scalar']


def to_scalar(x):
    """Convert 1 element numpy arrays to scalar
    
    Notes
    -----
    Raises error for multiellement arrays

    Examples
    --------
    >>> to_scalar(np.array([1]))
    1
    >>> to_scalar(np.array([[2]]))
    2
    """
    if type(x) is ndarray:
        if x.size == 1:
            return x.ravel().item()
        raise ValueError(f"Expected array with 1 element, got {x.size}")
    return x


def around(val, decimals=0):
    """Evenly round to the given number of decimals

    Parameters
    ----------
    val: real scalar or array-like

    Returns
    -------
    rval: real scalar or list of real scalars

    Notes
    -----
    This function is useful if you want rvals to have exactly n decimals.

    Examples
    --------
    >>> around(1.23456789)
    1
    >>> around(1.23456789, decimals=2)
    1.23
    >>> around(1.23456789, decimals=3)
    1.235
    >>> around(1.23456789, decimals=4)
    1.2346
    >>> around(1.23456789, decimals=5)
    1.23457
    >>> around([1.23456789, 2.23456789])
    [1, 2]
    >>> around([1.23456789, 2.23456789], decimals=2)
    [1.23, 2.23]
    >>> around([1.23456789, 2.23456789], decimals=3)
    [1.235, 2.235]
    >>> around([1.23456789, 2.23456789], decimals=4)
    [1.2346, 2.2346]
    >>> around([1.23456789, 2.23456789], decimals=5)
    [1.23457, 2.23457]

    """
    if decimals == 0:
        if np.ndim(val) == 0:
            return int(np.around(val))
        return list(int(np.around(v)) for v in val)
    scale = 10**decimals
    if np.ndim(val) == 0:
        return int(np.around(val * scale)) / scale
    return list(int(np.around(v * scale)) / scale for v in val)


def xor(a, b):
    """ Returns True only when inputs differ."""
    return a ^ b


def lfind(haystack, needle, maxiter=None):
    """Return indices to the maxiter first needles in the haystack as an iterable generator.

    Parameters
    ----------
    haystack: list or tuple of items
    needle: item to find
    maxiter: scalar integer maximum number of occurences

    Returns
    -------
    indices_gen: generator object

    Examples
    --------
    >>> haystack = (1, 3, 4, 3, 10, 3, 1, 2, 5, 7, 10)
    >>> list(lfind(haystack, 3))
    [1, 3, 5]
    >>> [i for i in lfind(haystack, 3, 2)]
    [1, 3]
    >>> [i for i in lfind(haystack, 0, 2)]
    []
    """
    maxiter = inf if maxiter is None else maxiter
    ix = -1
    i = 0
    while i < maxiter:
        i += 1
        try:
            ix = haystack.index(needle, ix + 1)
            yield ix
        except (ValueError, KeyError):
            break


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None (or np.random), return the RandomState singleton used
    by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.

    Examples
    --------
    >>> rs0 = check_random_state(seed=None)
    >>> rs1 = check_random_state(seed=1)
    >>> rs2 = check_random_state(seed=np.random.RandomState(1))

    check_random_state(seed=2.5)
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand  # pylint: disable=protected-access
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    msg = '{} cannot be used to seed a numpy.random.RandomState instance'
    raise ValueError(msg.format(seed))


def piecewise(condlist, funclist, xi=None, fillvalue=0.0, args=(), **kw):
    """
    Evaluate a piecewise-defined function.

    Given a set of conditions and corresponding functions (or constants),
    evaluate each function on the input data wherever its condition is true.


    Parameters
    ----------
    condlist : list of bool arrays
        Each boolean array corresponds to a function in `funclist`. Wherever
        `condlist[i]` is True, `funclist[i]` is used to compute the output.
        All conditions must be broadcastable to the same shape.

        If ``len(funclist) == len(condlist) + 1``, the last function in
        `funclist` is treated as the "otherwise" (default) case and is applied
        where none of the conditions are True.

    funclist : list of callables or scalars
        Each element corresponds to a condition in `condlist`. If an element
        is callable, it is evaluated as:
            ``f(x0, x1, ..., xn, *args, **kw)``
        using only the values where the corresponding condition is True.

        If an element is not callable, it is treated as a constant and assigned
        directly to the output where the condition is True.
    xi : tuple of array_like, optional
        Input arrays passed to the functions in `funclist`, i.e., (x0, x1,...., xn).
        These arrays must be broadcastable to a common shape. If `xi` is a single array, it may
        be passed directly (it will be converted to a tuple internally).

        If `xi` is None, functions should not expect array inputs.

    fillvalue : scalar
        Value used to fill elements where no condition is True. Default is 0.0.

    args : tuple, optional
        Additional positional arguments passed to each callable function.
    **kw : dict, optional
        Additional keyword arguments passed to each callable function.

    Returns
    -------
    out : ndarray
        An array with the same shape as the broadcasted inputs. Each element
        is computed according to the first condition that evaluates to True.
        Elements not covered by any condition are set to `fillvalue`.

    See Also
    --------
    lazyselect, lazywhere,

    Notes
    -----
    - Only the necessary elements of `xi` are passed to each function,
      improving efficiency.
    - Functions in `funclist` must return values compatible with the number
      of True elements in their corresponding condition.
    - This function is similar to `numpy.piecewise` but evaluates functions
      only on selected elements (lazy evaluation).


    The result is::

          |--
          |funclist[0](x0[condlist[0]], x1[condlist[0]],..., xn[condlist[0]])
    out = |funclist[1](x0[condlist[1]], x1[condlist[1]],..., xn[condlist[1]])
          |...
          |funclist[n2](x0[condlist[n2]], x1[condlist[n2]],..,xn[condlist[n2]])
          |--

    Examples
    --------
    Define the sigma function, which is -1 for ``x < 0`` and +1 for ``x >= 0``.

    >>> x = np.linspace(-2.5, 2.5, 6)
    >>> np.allclose(piecewise([x < 0, x >= 0], [-1, 1]),
    ...    [-1, -1, -1,  1,  1,  1])
    True

    Define the absolute value, which is ``-x`` for ``x <0`` and ``x`` for
    ``x >= 0``.

    >>> np.allclose(piecewise([x < 0, x >= 0], [lambda x: -x, lambda x: x], xi=(x,)),
    ...             [ 2.5,  1.5,  0.5,  0.5,  1.5,  2.5])
    True

    Define the absolute value, which is ``-x*y`` for ``x*y <0`` and ``x*y`` for
    ``x*y >= 0``
    >>> X, Y = np.meshgrid(x, x)
    >>> np.allclose(piecewise([X * Y < 0, ], [lambda x, y: -x * y,
    ...                                       lambda x, y: x * y], xi=(X, Y)),
    ...        [[ 6.25,  3.75,  1.25,  1.25,  3.75,  6.25],
    ...         [ 3.75,  2.25,  0.75,  0.75,  2.25,  3.75],
    ...         [ 1.25,  0.75,  0.25,  0.25,  0.75,  1.25],
    ...         [ 1.25,  0.75,  0.25,  0.25,  0.75,  1.25],
    ...         [ 3.75,  2.25,  0.75,  0.75,  2.25,  3.75],
    ...         [ 6.25,  3.75,  1.25,  1.25,  3.75,  6.25]])
    True

    >>> x = np.arange(5)
    >>> np.allclose(piecewise([x > 1, x > 3], [1, 3], xi=(x,)), [0, 0, 1, 1, 3])
    True
    """
    
    def otherwise_condition(condlist):
        return ~np.logical_or.reduce(condlist, axis=0)

    # Validate lengths
    num_cond, num_fun = len(condlist), len(funclist)
    if num_cond not in (num_fun, num_fun - 1):
        raise ValueError("function list and condition list must match")

    # Broadcast conditions
    condlist = list(np.broadcast_arrays(*condlist))

    # Add default condition if needed
    if len(condlist) == len(funclist) - 1:
        condlist.append(otherwise_condition(condlist))

    # Prepare inputs
    if xi is None:
        arrays = ()
        shape = condlist[0].shape
        dtype = np.result_type(fillvalue)
    else:
        if not isinstance(xi, tuple):
            xi = (xi,)
        arrays = np.broadcast_arrays(*xi)
        shape = arrays[0].shape
        dtype = np.result_type(*arrays, fillvalue)

    out = np.full(shape, fillvalue, dtype)

    for cond, func in zip(condlist, funclist):

        if not cond.any():
            continue

        if callable(func):
            vals = tuple(arr[cond] for arr in arrays) + args
            np.place(out, cond, func(*vals, **kw))
        else:
            np.putmask(out, cond, func)
    return out


def lazywhere(condition, arrays, f, fillvalue=None, f2=None):
    """
    Lazily evaluate a conditional expression.

    Parameters
    ----------
    condition : array-like of bool
        Boolean mask. Where True, `f(*arrays)` is used; elsewhere either
        `fillvalue` or `f2(*arrays)` is used.

    arrays : tuple of array_like
        Input arrays passed to `f` and `f2`. Must be broadcastable to a
        common shape.

    f : callable
        Function applied where `condition` is True. Must accept the arrays
        as positional arguments and return values compatible with the mask.

    fillvalue : scalar, optional
        Value used where `condition` is False. If not provided, `f2` must be given.

    f2 : callable, optional
        Function applied where `condition` is False.

    Returns
    -------
    out : ndarray
        Result array with the same shape as the broadcasted inputs.

    Notes
    -----
    Equivalent to:
        np.where(condition, f(*arrays), fillvalue)
    or:
        np.where(condition, f(*arrays), f2(*arrays))

    but evaluates `f` and `f2` only where needed.  The functions must 
    return values compatible with the number of True/False
    elements in `condition`.

    Examples
    --------
    >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
    >>> def f(a, b):
    ...     return a*b
    >>> np.allclose(lazywhere(a > 2, (a, b), f, fillvalue=np.nan),
    ...            [np.nan,  np.nan,  21.,  32.], equal_nan=True)
    True
    >>> def f2(a, b):
    ...     return (a*b)**2
    >>> np.allclose(lazywhere(a > 2, (a, b), f, f2=f2), [  25.,  144.,   21.,   32.])
    True

    """
    
    
    if len(arrays) == 0:
        raise ValueError("arrays must contain at least one array")

    if fillvalue is None and f2 is None:
        raise ValueError("One of (fillvalue, f2) must be given.")
    if fillvalue is not None and f2 is not None:
        raise ValueError("Only one of (fillvalue, f2) can be given.")

    arrays = np.broadcast_arrays(*arrays)
    condition = np.asarray(condition, dtype=bool)
    condition = np.broadcast_to(condition, arrays[0].shape)

    shape = arrays[0].shape

    # dtype determination
    if f2 is None:
        dtype = np.result_type(*arrays, fillvalue)
        out = np.full(shape, fillvalue, dtype=dtype)
    else:
        dtype = np.result_type(*arrays)
        out = np.empty(shape, dtype=dtype)

        inv = ~condition
        if inv.any():
            vals = tuple(arr[inv] for arr in arrays)
            np.place(out, inv, f2(*vals))

    if condition.any():
        vals = tuple(arr[condition] for arr in arrays)
        np.place(out, condition, f(*vals))
    return out


def lazyselect(condlist, funclist, arrays, default=0):
    """
    Lazily select from functions based on conditions.

    Similar to `np.select`, but functions are evaluated only on the elements
    where their corresponding condition is True.

    Parameters
    ----------
    condlist : sequence of array_like of bool
        Conditions determining which function to apply. Later conditions
        override earlier ones if multiple are True.

    funclist : sequence of callables
        Functions corresponding to each condition. Each must accept
        `arrays` as arguments and return values compatible with the mask.

    arrays : tuple of array_like
        Input arrays, broadcastable to a common shape.

    default : scalar, optional
        Value used where no conditions are True.

    Returns
    -------
    out : ndarray
        Output array with the same shape as broadcasted inputs.

    Notes
    -----
    - Functions are evaluated only on the subset of elements where their
      condition is True (lazy evaluation).
    - If multiple conditions are True, the last one takes precedence.


    See also
    --------
    lazywhere, piecewise

    Notes
    -----
    Lazily evaluates the choicelist = [fun(*arrays) for fun in funclist],
    depending on conditions and returns an array drawn from elements in choicelist,

    Notice it assumes that all `arrays` are of the same shape, or can be
    broadcasted together.

    All functions in `funclist` must accept array arguments in the order
    given in `arrays` and must return an array of the same shape as broadcasted
    `arrays`.

    Examples
    --------
    >>> x = np.arange(6)
    >>> np.allclose(np.select([x <3, x > 3], [x**2, x**3], default=0),
    ...             [  0,   1,   4,   0,  64, 125])
    True
    >>> np.allclose(lazyselect([x < 3, x > 3], [lambda x: x**2, lambda x: x**3], (x,)),
    ...             [   0.,    1.,    4.,    0.,   64.,  125.])
    True

    >>> a = -np.ones_like(x)
    >>> np.allclose(lazyselect([x < 3, x > 3],
    ...                        [lambda x, a: x**2, lambda x, a: a * x**3],
    ...                        (x, a)),
    ...             [   0.,    1.,    4.,    0.,  -64., -125.])
    True

    """
    
    if len(condlist) != len(funclist):
        raise ValueError("condlist and funclist must have the same length")
    if len(condlist) == 0:
        raise ValueError("condlist cannot be empty")
    if len(arrays) == 0:
        raise ValueError("arrays must contain at least one array")

    # Broadcast arrays
    arrays = np.broadcast_arrays(*arrays)

    # Normalize and broadcast conditions
    condlist = [np.asarray(cond, dtype=bool) for cond in condlist]
    condlist = [np.broadcast_to(cond, arrays[0].shape) for cond in condlist]

    # Output dtype
    dtype = np.result_type(*arrays, default)
    out = np.full(arrays[0].shape, default, dtype=dtype)

    # Apply functions
    for cond, func in zip(condlist, funclist):
        if not cond.any():
            continue

        vals = tuple(arr[cond] for arr in arrays)
        np.place(out, cond, func(*vals))

    return out


def rotation_matrix(heading, pitch, roll):
    """
    Compute a 3×3 rotation matrix from heading, pitch, and roll (degrees).

    Uses the ZYX (yaw–pitch–roll) convention:
        R = Rz(heading) @ Ry(pitch) @ Rx(roll)

    Parameters
    ----------
    heading, pitch, roll : float
        Rotation angles in degrees:
        - heading (yaw) about z-axis
        - pitch about y-axis
        - roll about x-axis
    
    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix

    Examples
    --------
    >>> import numpy as np
    >>> np.allclose(rotation_matrix(heading=0, pitch=0, roll=0),
    ...       [[ 1.,  0.,  0.],
    ...        [ 0.,  1.,  0.],
    ...        [ 0.,  0.,  1.]])
    True

    >>> np.allclose(rotation_matrix(heading=180, pitch=0, roll=0),
    ...      [[ -1.,   0.,   0.],
    ...       [  0.,  -1.,   0.],
    ...       [  0.,   0.,   1.]])
    True
    >>> np.allclose(rotation_matrix(heading=0, pitch=180, roll=0),
    ...      [[ -1.,  0.,   0.],
    ...       [  0.,  1.,   0.],
    ...       [  0.,  0.,  -1.]])
    True
    >>> np.allclose(rotation_matrix(heading=0, pitch=0, roll=180),
    ...      [[  1.,  0.,   0.],
    ...       [ 0.,  -1.,   0.],
    ...       [ 0.,   0.,  -1.]])
    True
    """
    if heading == 0 and pitch == 0 and roll == 0:        
        return np.eye(3)  # No transform if H=P=R=0

    # Convert to radians
    rheading, rpitch, rroll = np.deg2rad([heading, pitch, roll])    
    # Trig values
    cH, sH = cos(rheading), sin(rheading)
    cP, sP = cos(rpitch), sin(rpitch)
    cR, sR = cos(rroll), sin(rroll)

    # Rotation matrix (ZYX)
    return np.array([
            [cH*cP,  cH*sP*sR - sH*cR,  cH*sP*cR + sH*sR],
            [sH*cP,  sH*sP*sR + cH*cR,  sH*sP*cR - cH*sR],
            [-sP,    cP*sR,             cP*cR]
        ], dtype=float)


def rotate(x, y, z, heading=0, pitch=0, roll=0):
    """
    Rotate 3D coordinates using heading, pitch, and roll angles.

    Uses the same ZYX (yaw–pitch–roll) convention as `rotation_matrix`:
        R = Rz(heading) @ Ry(pitch) @ Rx(roll)

    Parameters
    ----------
    x, y, z : array_like or scalar
        Coordinates of the points. Must be broadcastable to the same shape.

    heading, pitch, roll : float
        Rotation angles in degrees.

    Returns
    -------
    x_out, y_out, z_out : ndarray or scalar
        Rotated coordinates with the same shape as inputs.

    Examples
    --------
    >>> import numpy as np
    >>> x, y, z = 1, 1, 1
    >>> np.allclose(rotate(x, y, z, heading=0, pitch=0, roll=0),
    ...    (1.0, 1.0, 1.0))
    True
    >>> np.allclose(rotate(x, y, z, heading=90, pitch=0, roll=0),
    ...            (-1.0, 1.0, 1.0))
    True
    >>> np.allclose(rotate(x, y, z, heading=0, pitch=90, roll=0),
    ...            (1.0, 1.0, -1.0))
    True
    >>> np.allclose(rotate(x, y, z, heading=0, pitch=0, roll=90),
    ...            (1.0, -1.0, 1.0))
    True
    """
    
    R = rotation_matrix(heading, pitch, roll)

    # Stack inputs (supports scalars + arrays)
    v = np.stack((x, y, z), axis=0)

    # Apply rotation
    result = R @ v

    return result[0], result[1], result[2]

    rot_param = rotation_matrix(heading, pitch, roll).ravel()
    x_out = x * rot_param[0] + y * rot_param[1] + z * rot_param[2]
    y_out = x * rot_param[3] + y * rot_param[4] + z * rot_param[5]
    z_out = x * rot_param[6] + y * rot_param[7] + z * rot_param[8]
    return x_out, y_out, z_out


def rotate_2d(x, y, angle_deg):
    """
    Rotate points in the xy plane counter clockwise.

    Parameters
    ----------
    x, y : array_like or scalar
        Coordinates of the points. Must be broadcastable.

    angle_deg : float
        Rotation angle in degrees (counterclockwise).

    Returns
    -------
    x_out, y_out : ndarray or scalar
        Rotated coordinates.

    Examples
    --------
    >>> np.allclose(rotate_2d(x=1, y=0, angle_deg=0), (1.0, 0.0))
    True
    >>> np.allclose(rotate_2d(x=1, y=0, angle_deg=90), (0, 1.0))
    True
    >>> np.allclose(rotate_2d(x=1, y=0, angle_deg=180), (-1.0, 0))
    True
    >>> np.allclose(rotate_2d(x=1, y=0, angle_deg=360), (1.0, 0))
    True
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = cos(angle_rad)
    sin_a = sin(angle_rad)
    return cos_a * x - sin_a * y, sin_a * x + cos_a * y


def now(show_seconds=True):
    '''
    Return current date and time as a string
    '''
    if show_seconds:
        return strftime("%a, %d %b %Y %H:%M:%S", gmtime())
    return strftime("%a, %d %b %Y %H:%M", gmtime())


def _assert(cond, txt=''):
    if not cond:
        raise ValueError(txt)


def spaceline(start_point, stop_point, num=10):
    '''Return `num` evenly spaced points between start and stop.

    Parameters
    ----------
    start_point : vector, size=3
        The starting point of the sequence.
    stop_point : vector, size=3
        The end point of the sequence.
    num : int, optional
        Number of samples to generate. Default is 10.

    Returns
    -------
    space_points : ndarray of shape n x 3
        There are `num` equally spaced points in the closed interval
        ``[start, stop]``.

    See Also
    --------
    linspace : similar to spaceline, but in 1D.
    arange : Similiar to `linspace`, but uses a step size (instead of the
             number of samples).
    logspace : Samples uniformly distributed in log space.

    Examples
    --------
    >>> import wafo.misc as wm
    >>> np.allclose(wm.spaceline((2,0,0), (3,0,0), num=5),
    ...      [[ 2.  ,  0.  ,  0.  ],
    ...       [ 2.25,  0.  ,  0.  ],
    ...       [ 2.5 ,  0.  ,  0.  ],
    ...       [ 2.75,  0.  ,  0.  ],
    ...       [ 3.  ,  0.  ,  0.  ]])
    True
    >>> np.allclose(wm.spaceline((2,0,0), (0,0,3), num=5),
    ...      [[ 2.  ,  0.  ,  0.  ],
    ...       [ 1.5 ,  0.  ,  0.75],
    ...       [ 1.  ,  0.  ,  1.5 ],
    ...       [ 0.5 ,  0.  ,  2.25],
    ...       [ 0.  ,  0.  ,  3.  ]])
    True
    >>> np.allclose(wm.spaceline((2,0,0), (0,0,3), num=1),
    ...      [[ 1.  ,  0.  ,  1.5 ]])
    True
    '''

    num = int(num)
    if num <= 0:
        raise ValueError("num must be positive")
    
    start = np.asarray(start_point)
    stop  = np.asarray(stop_point)
    dtype = np.result_type(start, stop, float)

    start = start.astype(dtype, copy=False)
    stop  = stop.astype(dtype, copy=False)


    if start.shape != stop.shape:
        raise ValueError("start_point and stop_point must have the same shape")
    
    # Special case: midpoint
    if num == 1:
        return ((start + stop) / 2)[None, :]

    # Vectorized interpolation
    t = np.linspace(0, 1, num)[:, None]
    return start + t * (stop - start)


def narg_smallest(arr, n=1):
    ''' Return the n smallest indices to the arr

    Parameters
    ----------
    arr : array_like
        Input array.
    n : int, optional
        Number of smallest elements to return (default 1).

    Returns
    -------
    indices : ndarray of int
        Indices of the n smallest elements (sorted by value).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([37, 11, 4, 23, 4, 6, 3, 2, 7, 4, 0])
    >>> ix = narg_smallest(t, 3)
    >>> np.allclose(ix,
    ...             [10,  7,  6])
    True
    >>> np.allclose(t[ix], [0, 2, 3])
    True
    '''

    arr = np.asarray(arr).ravel()
    n = int(n)

    if n <= 0:
        return np.array([], dtype=int)

    n = min(n, arr.size)

    ix = np.argpartition(arr, n - 1)[:n]
    return ix[np.argsort(arr[ix])]


def args_flat(*args):
    '''
    Return x,y,z positions as a N x 3 ndarray

    Parameters
    ----------
    pos : array_like, shape (N, 3)
        Positions as rows [x, y, z].
    or
    x, y, z : array_like
        Coordinate arrays (broadcastable to same shape).

    Returns
    ------
    pos : ndarray, shape N x 3
        Flattened positions.
    common_shape : None or tuple
        Shape of broadcasted inputs (if x, y, z given),
        or None if pos was given.

    Examples
    --------
    >>> x = [1,2,3]
    >>> pos, c_shape = args_flat(x,2,3)
    >>> np.allclose(pos, [[1, 2, 3],
    ...                   [2, 2, 3],
    ...                   [3, 2, 3]])
    True
    >>> c_shape == (3,)
    True
    >>> pos1, c_shape1 = args_flat([1,2,3])
    >>> np.allclose(pos1, [[1, 2, 3]])
    True
    >>> c_shape1 is None
    True
    >>> pos1, c_shape1 = args_flat(1,2,3)
    >>> np.allclose(pos1, [[1, 2, 3]])
    True
    >>> c_shape1
    ()
    >>> pos1, c_shape1 = args_flat([1],2,3)
    >>> np.allclose(pos1, [[1, 2, 3]])
    True
    >>> c_shape1 == (1,)
    True

    '''
    nargin = len(args)

    if nargin not in (1, 3):
        raise ValueError("Number of arguments must be 1 or 3")

    # Case 1: single (N,3) input
    if nargin == 1:  
        
        pos = np.asarray(args[0])

        if pos.ndim == 1:
            if pos.size != 3:
                raise ValueError("Single point must have length 3")
            pos = pos[None, :]
        elif pos.ndim == 2:
            if pos.shape[1] != 3:
                raise ValueError("POS array must be shape (N, 3)")
        else:
            raise ValueError("Invalid shape for POS")

        return pos, None

    # Case 2: x, y, z inputs
    x, y, z = np.broadcast_arrays(*args[:3])
    c_shape = x.shape
    return np.column_stack((x.ravel(), y.ravel(), z.ravel())), c_shape


def index2sub(shape, index, order='C'):
    """
    
    Convert linear indices to subscripts.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the array.

    index : int or array_like of int
        Linear index or indices into the array.

    order : {'C', 'F'}, optional
        Indexing order:
        - 'C' : row-major (C-style)
        - 'F' : column-major (Fortran-style)

    Returns
    -------
    subs : tuple of ndarray or int
        A tuple of index arrays (or ints for scalar input), one per dimension.


    Examples
    --------
    >>> shape = (3,3,4)
    >>> a = np.arange(np.prod(shape)).reshape(shape)
    >>> order = 'C'
    >>> np.allclose(a[1, 2, 3], 23)
    True
    >>> i = sub2index(shape, 1, 2, 3, order=order)
    >>> np.allclose(a.ravel(order)[i], 23)
    True
    >>> np.allclose(index2sub(shape, i, order=order), (1, 2, 3))
    True

    See also
    --------
    sub2index
    """
    
    shape = tuple(shape)

    if order not in ('C', 'F'):
        raise ValueError("order must be 'C' or 'F'")

    return np.unravel_index(index, shape, order=order)


def sub2index(shape, *subscripts, order='C'):
    """
    Convert subscripts to linear indices.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the array.

    *subscripts : int or array_like
        Subscripts for each dimension. The number of subscripts must match
        the number of dimensions in `shape`. Subscripts are broadcast together.

    order : {'C', 'F'}, optional
        Indexing order:
        - 'C' : row-major (default)
        - 'F' : column-major

    Returns
    -------
    index : ndarray or int
        Linear index or indices corresponding to the given subscripts.


    Examples
    --------
    >>> shape = (3, 3, 4)
    >>> a = np.arange(np.prod(shape)).reshape(shape)
    >>> order = 'C'
    >>> i = sub2index(shape, 1, 2, 3, order=order)
    >>> np.allclose(a[1, 2, 3], 23)
    True
    >>> np.allclose(a.ravel(order)[i], 23)
    True
    >>> np.allclose(index2sub(shape, i, order=order), (1, 2, 3))
    True

    See also
    --------
    index2sub
    """
    
    shape = tuple(shape)

    if len(subscripts) != len(shape):
        raise ValueError("Number of subscripts must match number of dimensions")

    if order not in ('C', 'F'):
        raise ValueError("order must be 'C' or 'F'")

    return np.ravel_multi_index(subscripts, shape, order=order)


def is_numlike(x):
    """
    Return True if *x* behaves like a numeric scalar.

    The function considers an object "numlike" if it is:
    - a Python or NumPy scalar number (int, float, complex), or
    - a NumPy ndarray of size 1 with a numeric dtype.

    Parameters
    ----------
    x : any
        Object to test.

    Returns
    -------
    bool
        True if `x` represents a numeric scalar value or a single-element
        numeric array, otherwise False.

    Examples
    --------
    >>> is_numlike(1)
    True
    >>> is_numlike(1j)
    True
    >>> is_numlike(1.5)
    True
    >>> is_numlike(np.float64(1))
    True

    >>> is_numlike(np.array(1))
    True
    >>> is_numlike(np.array([1]))
    True

    >>> is_numlike(np.array([1, 2]))
    False
    >>> is_numlike(np.array('1'))
    False
    >>> is_numlike('1')
    False
    >>> is_numlike([1])
    False

    Notes
    -----
    - NumPy arrays are only considered numlike if they contain exactly one
      element and have a numeric dtype.
    - Multi-element arrays and non-numeric types are not considered numlike.
    """
    # Case 1: Python / NumPy scalar
    if isinstance(x, numbers.Number):
        return True

    # Case 2: NumPy array
    if isinstance(x, np.ndarray):
        return x.size == 1 and np.issubdtype(x.dtype, np.number)

    return False


class JITImport(object):

    '''
    Just In Time Import of module

    Examples
    --------
    >>> np = JITImport('numpy')
    >>> np.exp(0)==1.0
    True
    '''

    def __init__(self, module_name):
        self._module_name = module_name
        self._module = None

    def __getattr__(self, attr):
        try:
            return getattr(self._module, attr)
        except AttributeError as exc:
            if self._module is None:
                self._module = __import__(self._module_name, None, None, ['*'])
                # assert(isinstance(self._module, types.ModuleType), 'module')
                return getattr(self._module, attr)
            raise exc


def printf(format_, *args):  # @ReservedAssignment
    sys.stdout.write(format_ % args)


def moving_average(x, L, axis=0):
    """
    Returns moving average from data using a window of size 2*L+1.

    Parameters
    ----------
    x : vector or matrix of column vectors
        of data
    L : scalar, integer
        defines the size of the moving average window
    axis: scalar integer
        axis along which the moving average is computed. Default axis=0.

    Returns
    -------
    y : ndarray
        moving average

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> exp = np.exp; cos = np.cos; randn = np.random.randn

    >>> x = np.linspace(0,1,200)
    >>> y = exp(x) + cos(5*2*pi*x) + 1e-1*randn(x.size)
    >>> trend = moving_average(y, 20)
    >>> np.allclose(trend[:4], [ 1.1189971,  1.1189971,  1.1189971,  1.1189971], atol=1e-1)
    True
    >>> periodic = y - trend

    >>> x2 = np.linspace(1, 5, 5)
    >>> np.allclose(moving_average(x2, L=1), [2.,  2.,  3.,  4.,  4.])
    True
    >>> np.allclose(moving_average(x2, L=10), [3.,  3.,  3.,  3.,  3.])
    True
    >>> x3 = np.vstack((x2, x2+5))
    >>> np.allclose(moving_average(x3, L=1, axis=1), [[2.,  2.,  3.,  4.,  4.],
    ...                                               [7.,  7.,  8.,  9.,  9.]])
    True
    >>> np.allclose(moving_average(x3, L=10, axis=1), [[3.,  3.,  3.,  3.,  3.],
    ...                                                [8.,  8.,  8.,  8.,  8.]])
    True

    >>> import matplotlib.pyplot as plt
    >>> h0 = plt.plot(x, y, label='data')
    >>> h1 = plt.plot(x, trend, 'r', label='trend')
    >>> h2 = plt.plot(x, exp(x), 'r--', label='true trend')
    >>> h3 = plt.plot(x, periodic, 'm', label='periodic')
    >>> h4 = plt.plot(x, cos(5*2*pi*x), 'm--', label='true periodic')
    >>> h5 = plt.legend()

    >>> plt.close('all')

    See also
    --------
    Reconstruct
    """
    #return _moving_fun(np.mean, x, L, axis)  # alternative is slower and more memory consuming
    
    x = np.asarray(x)
    L = int(L)

    if L <= 0:
        raise ValueError("L must be positive")

    axis = np.lib.array_utils.normalize_axis_index(axis, x.ndim)

    # Move axis to front
    x = np.swapaxes(x, axis, 0)

    n = x.shape[0]
    window = 2 * L + 1

    y = np.empty(x.shape, dtype=np.result_type(x.dtype, np.float64))

    if n <= window:
        y[:] = x.mean(axis=0)
        return np.swapaxes(y, 0, axis)

    mn = x[0:window].mean(axis=0)

    # First L+1 values correspond to initial window
    y[0:L + 1] = mn

    ix = np.r_[L + 1:(n - L)]

    y[ix] = ((x[ix + L] - x[ix - L - 1]) / window).cumsum(axis=0) + mn

    # Right edge
    y[n - L:] = y[n - L - 1]

    return np.swapaxes(y, 0, axis)

    
def _moving_fun(fun, x, L, axis):
    """Core implementation for moving window functions."""
    
    x1 = np.asarray(x)
    L = int(L)

    if L <= 0:
        raise ValueError("L must be positive")
    
    axis = np.lib.array_utils.normalize_axis_index(axis, x.ndim)

    # Move target axis to front
    x1 = np.swapaxes(x1, axis, 0)

    n = x1.shape[0]
    y = np.empty_like(x1, dtype=np.result_type(x1.dtype, np.float64))

    if n <= 2 * L + 1:
        y[:] = fun(x1, axis=0)
        return np.swapaxes(y, 0, axis)

    rolling = sliding_window_view(x1, 2 * L + 1, axis=0)

    # Reduce over last axis
    result = fun(rolling, axis=-1)

    y[L:n - L, ...] = result

    # Edge fill
    y[:L, ...] = y[L, ...]
    y[n - L:, ...] = y[n - L - 1, ...]

    return np.swapaxes(y, 0, axis)


def moving_max(x, L, axis=0):
    """
    Returns the moving maximum from data using a window of size 2*L+1
    
    Parameters
    ----------
    x : vector or matrix of column vectors
        of data
    L : scalar, integer
        defines the size of the moving average window
    axis: scalar integer
        axis along which the moving average is computed. Default axis=0.

    Returns
    -------
    y : ndarray
        moving maximum

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.linspace(1, 5, 5)
    >>> np.allclose(moving_max(x1, L=1),[3.0, 3.0, 4.0, 5.0, 5.0])
    True
    >>> np.allclose(moving_max(x1, L=2), [5.0, 5.0, 5.0, 5.0, 5.0])
    True

    >>> x2 = np.vstack((x1, x1+5))
    >>> np.allclose(moving_max(x2, L=1, axis=1),  [[3.0, 3.0, 4.0, 5.0, 5.0], [8.0, 8.0, 9.0, 10.0, 10.0]])
    True
    >>> np.allclose(moving_max(x2, L=2, axis=1),  [[5.0, 5.0, 5.0, 5.0, 5.0], [10.0, 10.0, 10.0, 10.0, 10.0]])
    True
    
    """
    return _moving_fun(np.max, x, L, axis)


def moving_median(x, L, axis=0):
    """
    Returns the moving mmedia from data using a window of size 2*L+1
    
    Parameters
    ----------
    x : vector or matrix of column vectors
        of data
    L : scalar, integer
        defines the size of the moving average window
    axis: scalar integer
        axis along which the moving average is computed. Default axis=0.

    Returns
    -------
    y : ndarray
        moving meduab

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.linspace(1, 5, 5)
    >>> np.allclose(moving_median(x1, L=1),[2.0, 2.0, 3.0, 4.0, 4.0])
    True
    >>> np.allclose(moving_median(x1, L=2),  [3.0, 3.0, 3.0, 3.0, 3.0])
    True

    >>> x2 = np.vstack((x1, x1+5))
    >>> np.allclose(moving_median(x2, L=1, axis=1), [[2.0, 2.0, 3.0, 4.0, 4.0], 
    ...                                               [7.0, 7.0, 8.0, 9.0, 9.0]])
    True
    >>> np.allclose(moving_median(x2, L=2, axis=1),  [[3.0, 3.0, 3.0, 3.0, 3.0], 
    ...                                               [8.0, 8.0, 8.0, 8.0, 8.0]])
    True
    
    """
    return _moving_fun(np.median, x, L, axis)


def detrendma(x, L, axis=0):
    """
    Removes a trend from data using a moving average
           of size 2*L+1.  If 2*L+1 > len(x) then the mean is removed

    Parameters
    ----------
    x : vector or matrix of column vectors
        of data
    L : scalar, integer
        defines the size of the moving average window
    axis: scalar integer
        axis along which the moving average is computed. Default axis=0.

    Returns
    -------
    y : ndarray
        detrended data

    Examples
    --------
    >>> import numpy as np
    >>> import wafo.misc as wm
    >>> exp = np.exp; cos = np.cos; randn = np.random.randn

    >>> x = np.linspace(0,1,200)
    >>> noise = 0.1*randn(x.size)
    >>> noise = 0.1*np.sin(100*x)
    >>> y = exp(x) + cos(5*2*pi*x) + noise
    >>> periodic = wm.detrendma(y, 20)
    >>> trend = y - periodic
    >>> np.allclose(trend[:5],
    ...    [ 1.14134814,  1.14134814,  1.14134814,  1.14134814,  1.14134814])
    True
    >>> y1 = wm.detrendma(y, 200)
    >>> np.allclose((y-y1), 1.7239972279640454)
    True
    >>> x2 = np.linspace(1, 5, 5)
    >>> np.allclose(wm.detrendma(x2, L=1), [-1, 0, 0, 0, 1])
    True

    >>> import matplotlib.pyplot as plt
    >>> h0 = plt.plot(x, y, label='data')
    >>> h1 = plt.plot(x, trend, 'r', label='trend')
    >>> h2 = plt.plot(x, exp(x), 'r--', label='true trend')
    >>> h3 = plt.plot(x, periodic, 'm', label='periodic')
    >>> h4 = plt.plot(x, cos(5*2*pi*x), 'm--', label='true periodic')
    >>> h5 = plt.legend()

    >>> plt.close('all')

    See also
    --------
    Reconstruct, moving_average
    """
    x1 = np.atleast_1d(x)
    trend = moving_average(x1, L, axis)
    return x1 - trend


def ecross(t, f, ind, v=0):
    '''
    Extracts exact level v crossings

    ECROSS interpolates t and f linearly to find the exact level v
    crossings, i.e., the points where f(t0) = v

    Parameters
    ----------
    t,f : vectors
        of arguments and functions values, respectively.
    ind : ndarray of integers
        indices to level v crossings as found by findcross.
    v : scalar or vector (of size(ind))
        defining the level(s) to cross.

    Returns
    -------
    t0 : vector
        of  exact level v crossings.

    Examples
    --------
    >>> import wafo.misc as wm
    >>> ones = np.ones
    >>> t = np.linspace(0,7*np.pi,250)
    >>> x = np.sin(t)
    >>> ind = wm.findcross(x,0.75)
    >>> np.allclose(ind, [  9,  25,  80,  97, 151, 168, 223, 239])
    True
    >>> t0 = wm.ecross(t, x, ind, 0.75)
    >>> np.allclose(t0, [0.84910514, 2.2933879 , 7.13205663, 8.57630119,
    ...        13.41484739, 14.85909194, 19.69776067, 21.14204343])
    True

    >>> from matplotlib import pyplot as plt
    >>> h0 = plt.plot(t, x, '.', label='data')
    >>> h1 = plt.plot(t[ind], x[ind], 'r.', label='approximate crossings')
    >>> h2 = plt.plot(t, ones(t.shape)*0.75, label='0.75 level')
    >>> h3 = plt.plot(t0, ones(t0.shape)*0.75, 'g.', label='exact crossings')
    >>> h4 = plt.legend()

    >>> plt.close('all')

    See also
    --------
    findcross
    '''
    # Tested on: Python 2.5
    # revised pab Feb2004
    # By pab 18.06.2001
    return (t[ind] + (v - f[ind]) * (t[ind + 1] - t[ind]) /
            (f[ind + 1] - f[ind]))


def findpeaks(data, n=2, min_h=None, min_p=0.0):
    '''
    Find peaks of vector or matrix possibly rainflow filtered

    Parameters
    ----------
    data : matrix or vector
    n : scalar integer
         The n highest peaks are found (if exist). (default 2)
    min_h : real scalar
        The threshold in the rainflowfilter (default 0.05*range(data.ravel())).
        A zero value will return all the peaks of data.
    min_p : 0..1
        Only the peaks that are higher than min_p*max(max(data))
        min_p*(the largest peak in data) are returned (default  0).

    Returns
    -------
    ix : array-like
        linear index to peaks of data

    Examples
    --------

    Find highest 8 peaks that are not
    less that 0.3*"global max" and have
    rainflow amplitude larger than 5.
    >>> import numpy as np
    >>> import wafo.misc as wm
    >>> x = np.arange(0,10,0.01)
    >>> data = x**2 + 10*np.sin(3*x) + 0.5*np.sin(50*x)
    >>> ind = wm.findpeaks(data, n=8, min_h=5, min_p=0.3)
    >>> np.allclose(ind, [908, 694, 481])
    True

    >>> import matplotlib.pyplot as plt
    >>> h0 = plt.plot(x, data, label='data')
    >>> h1 = plt.plot(x[ind], data[ind], 'r.', label='peaks')
    >>> h2 = plt.legend()

    >>> plt.close('all')

    See also
    --------
    findtp
    '''

    data1 = np.asarray(data, dtype=np.float64)
    dmax = data1.max()

    if min_h is None:
        dmin = data1.min()
        min_h = 0.05 * (dmax - dmin)

    ndim = data1.ndim
    data2 = np.atleast_2d(data1)
    nrows, mcols = data2.shape

    ind_p = []

    for iy in range(nrows):

        i_tp = findtp(data2[iy], min_h)

        if len(i_tp):
            peaks_row = i_tp[1::2]
        else:
            peaks_row = np.array([data2[iy].argmax()], dtype=np.int64)

        if ndim == 1:
            ind = peaks_row
        else:
            if iy == 0:
                mask = data2[iy, peaks_row] > data2[iy + 1, peaks_row]
            elif iy == nrows - 1:
                mask = data2[iy, peaks_row] > data2[iy - 1, peaks_row]
            else:
                mask = ((data2[iy, peaks_row] > data2[iy - 1, peaks_row]) &
                        (data2[iy, peaks_row] > data2[iy + 1, peaks_row]))

            ind_p.append(peaks_row[mask] + iy * mcols)

    if ndim > 1:
        ind = np.hstack(ind_p) if len(ind_p) else np.empty(0, dtype=np.int64)
    else:
        ind = np.asarray(ind, dtype=np.int64)

    if ind.size == 0:
        return ind

    peaks = data2.ravel()[ind]

    # filter first
    if min_p > 0:
        keep = peaks > min_p * dmax
        ind = ind[keep]
        peaks = peaks[keep]

    # rank after filtering
    idx = peaks.argsort()[::-1]
    nmax = min(n, len(ind))

    return ind[idx[:nmax]]

def findtp(x, h=0.0, kind=None,  mode='inclusive'):
    '''
    Return indices to turning points (tp) of data, optionally rainflow filtered.

    Parameters
    ----------
    x : vector
        signal
    h : real, scalar
        rainflow threshold
         if  h<0, then ind = range(len(x))
         if  h=0, then  tp  is a sequence of turning points (default)
         if  h>0, then all rainflow cycles with height smaller than
                  h  are removed.
    kind : string
        defines the type of wave or indicate the ASTM rainflow counting method.
        Possible options are 'astm' 'mw' 'Mw' or 'none'.
        If None all rainflow filtered min and max
        will be returned, otherwise only the rainflow filtered
        min and max, which define a wave according to the
        wave definition, will be returned.
    mode : {'inclusive', 'strict'}
        Threshold rule:
        - 'inclusive' : keep cycles with range >= h
        - 'strict'    : keep cycles with range > h

    Returns
    -------
    ind : array-like
        indices to the turning points in the original sequence.

    Examples
    --------
    >>> import wafo.misc as wm
    >>> t = np.linspace(0, 30, 500).reshape((-1, 1))
    >>> x = np.hstack((t, np.cos(t) + 0.3 * np.sin(5*t)))

    >>> itp = wm.findtp(x[:,1], 0, 'mw')
    >>> itph = wm.findtp(x[:,1], 0.3, 'mw')
    >>> tp = x[itp, :]
    >>> tph = x[itph, :]
    >>> np.allclose(itp[:10], [0, 5, 18, 24, 38, 46, 57, 70, 76, 91])
    True
    >>> np.allclose(itph, [57, 109, 161, 214, 266, 318, 370, 423, 475])
    True

    >>> import matplotlib.pyplot as plt
    >>> h0 = plt.plot(x[:, 0], x[:, 1], label='data')
    >>> h1 = plt.plot(tp[:, 0], tp[:, 1], 'ro', label='turning points')
    >>> h2 = plt.plot(tph[:, 0], tph[:, 1], 'k.', label='filtered turning points')
    >>> h3 = plt.legend()

    >>> plt.close('all')

    See also
    ---------
    findtc
    findcross
    findextrema
    findrfc
    '''
    n = len(x)
    if h < 0.0:
        return np.arange(n)

    ind = findextrema(x)

    if ind.size < 2:
        return None

    # In order to get the exact up-crossing intensity from rfc by
    # mm2lc(tp2mm(rfc))  we have to add the indices
    # to the last value (and also the first if the
    # sequence of turning points does not start with a minimum).

    if kind == 'astm':
        # the Nieslony approach always put the first loading point as the first
        # turning point.
        # add the first turning point is the first of the signal
        if ind[0] != 0:
            ind = np.r_[0, ind, n - 1]
        else:  # only add the last point of the signal
            ind = np.r_[ind, n - 1]
    else:
        if x[ind[0]] > x[ind[1]]:  # adds indices to  first and last value
            ind = np.r_[0, ind, n - 1]
        else:  # adds index to the last value
            ind = np.r_[ind, n - 1]

    if h > 0.0:
        ind1 = findrfc(x[ind], h, mode=mode, assume_tp=True)
        ind = ind[ind1]

    if kind in ('mw', 'Mw'):
        # make sure that the first is a Max if wdef == 'Mw'
        # or make sure that the first is a min if wdef == 'mw'
        first_is_max = (x[ind[0]] > x[ind[1]])

        remove_first = xor(first_is_max, kind.startswith('Mw'))
        if remove_first:
            ind = ind[1::]

        # make sure the number of minima and Maxima are according to the
        # wavedef. i.e., make sure Nm=length(ind) is odd
        if np.mod(ind.size, 2) != 1:
            ind = ind[:-1]
    return ind


def findtc(x_in, v=None, kind=None):
    """
    Return indices to troughs and crests of data.

    Parameters
    ----------
    x : vector
        surface elevation.
    v : real scalar
        reference level (default  v = mean of x).

    kind : string
        defines the type of wave. Possible options are
        'dw', 'uw', 'tw', 'cw' or None.
        If None indices to all troughs and crests will be returned,
        otherwise only the paired ones will be returned
        according to the wavedefinition.

    Returns
    --------
    tc_ind : vector of ints
        indices to the trough and crest turningpoints of sequence x.
    v_ind : vector of ints
        indices to the level v crossings of the original
        sequence x. (d,u)

    Examples
    --------
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,30,500).reshape((-1,1))
    >>> x = np.hstack((t, np.cos(t)))

    >>> idtc, idv = wm.findtc(x[:, 1], 0, 'dw')
    >>> tc = x[idtc,:]
    >>> udc = x[idv, :]
    >>> np.allclose(idtc, [52, 105, 157, 209, 261, 314, 366, 418])
    True

    >>> np.allclose(idv, [26,  78, 130, 182, 235, 287, 339, 391, 444])
    True

    >>> iutc, iuv = wm.findtc(x[:, 1], 0, 'uw')
    >>> np.allclose(iutc, [105, 157, 209, 261, 314, 366, 418, 470])
    True
    >>> np.allclose(iuv, [78, 130, 182, 235, 287, 339, 391, 444, 496])
    True

    >>> import matplotlib.pyplot as plt
    >>> h0 = plt.plot(tc[:,0], tc[:,1],'ro', label='trough or crest')
    >>> h1 = plt.plot(udc[:,0], udc[:,1],'go', label='up or down crossing')
    >>> h2 = plt.plot(x[:,0], x[:,1], 'k.', label='data')
    >>> h3 = plt.legend()

    >>> plt.close('all')

    See also
    --------
    findtp
    findcross,
    wavedef
    """

    x = np.atleast_1d(x_in)
    if v is None:
        v = x.mean()

    v_ind = findcross(x, v, kind)
    n_c = v_ind.size
    if n_c <= 2:
        warnings.warn('There are no waves!')
        return zeros(0, dtype=int), zeros(0, dtype=int)

    # determine the number of trough2crest (or crest2trough) cycles
    is_even = np.mod(n_c + 1, 2)
    n_tc = int((n_c - 1 - is_even) / 2)

    # allocate variables before the loop increases the speed
    ind = zeros(n_c - 1, dtype=int)

    first_is_down_crossing = (x[v_ind[0]] > x[v_ind[0] + 1])
    if first_is_down_crossing:
        f1, f2 = np.argmin, np.argmax
    else:
        f1, f2 = np.argmax, np.argmin

    for i in range(n_tc):
        # trough or crest
        j = 2 * i
        ind[j] = f1(x[v_ind[j] + 1:v_ind[j + 1] + 1])
        # crest or trough
        ind[j + 1] = f2(x[v_ind[j + 1] + 1:v_ind[j + 2] + 1])

    if (2 * n_tc + 1 < n_c) and (kind in (None, 'tw', 'cw')):
        # trough or crest
        ind[n_c - 2] = f1(x[v_ind[n_c - 2] + 1:v_ind[n_c - 1] + 1])

    return v_ind[:n_c - 1] + ind + 1, v_ind


def findoutliers(x, zcrit=0.0, dcrit=None, ddcrit=None, verbose=False):
    """
    Return indices to spurious points of data

    Parameters
    ----------
    x : vector
        of data values.
    zcrit : real scalar
        critical distance between consecutive points.
    dcrit : real scalar
        critical distance of Dx used for determination of spurious
        points.  (Default 1.5 standard deviation of x)
    ddcrit : real scalar
        critical distance of DDx used for determination of spurious
        points.  (Default 1.5 standard deviation of x)

    Returns
    -------
    inds : ndarray of integers
        indices to spurious points.
    indg : ndarray of integers
        indices to the rest of the points.

    Notes
    -----
    Consecutive points less than zcrit apart  are considered as spurious.
    The point immediately after and before are also removed. Jumps greater than
    dcrit in Dxn and greater than ddcrit in D^2xn are also considered as
    spurious.
    (All distances to be interpreted in the vertical direction.)
    Another good choice for dcrit and ddcrit are:

        dcrit = 5*dT  and ddcrit = 9.81/2*dT**2

    where dT is the timestep between points.

    Examples
    --------
    >>> import numpy as np
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,30,500).reshape((-1,1))
    >>> xx = np.hstack((t, np.cos(t)))
    >>> dt = np.diff(xx[:2,0])
    >>> dcrit = 5*dt
    >>> ddcrit = 9.81/2*dt*dt
    >>> zcrit = 0
    >>> inds, indg = wm.findoutliers(xx[:,1], verbose=True)
    Found 0 missing points
    dcrit is set to 1.05693
    ddcrit is set to 1.05693
    Found 0 spurious positive jumps of Dx
    Found 0 spurious negative jumps of Dx
    Found 0 spurious positive jumps of D^2x
    Found 0 spurious negative jumps of D^2x
    Found 0 consecutive equal values
    Found the total of 0 spurious points


    #waveplot(xx,'-',xx(inds,:),1,1,1)

    See also
    --------
    waveplot, reconstruct
    """

    def _find_nans(xn):
        i_missing = np.flatnonzero(np.isnan(xn))
        if verbose:
            print('Found %d missing points' % i_missing.size)
        return i_missing

    def _find_spurious_jumps(dxn, dcrit, name='Dx'):
        i_p = np.flatnonzero(dxn > dcrit)
        if i_p.size > 0:
            i_p += 1  # the point after the jump
        if verbose:
            print('Found {0:d} spurious positive jumps of {1}'.format(i_p.size,
                                                                      name))

        i_n = np.flatnonzero(dxn < -dcrit)  # the point before the jump
        if verbose:
            print('Found {0:d} spurious negative jumps of {1}'.format(i_n.size,
                                                                      name))
        if i_n.size > 0:
            return np.hstack((i_p, i_n))
        return i_p

    def _find_consecutive_equal_values(dxn, zcrit):

        mask_small = (np.abs(dxn) <= zcrit)
        i_small = np.flatnonzero(mask_small)
        if verbose:
            if zcrit == 0.:
                print('Found %d consecutive equal values' % i_small.size)
            else:
                print('Found %d consecutive values less than %g apart.' %
                      (i_small.size, zcrit))
        if i_small.size > 0:
            i_small += 1
            # finding the beginning and end of consecutive equal values
            i_step = np.flatnonzero((np.diff(mask_small))) + 1
            # indices to consecutive equal points
            # removing the point before + all equal points + the point after

            return np.hstack((i_step - 1, i_small, i_step, i_step + 1))
        return i_small

    xn = np.asarray(x).flatten()

    _assert(2 < xn.size, 'The vector must have more than 2 elements!')

    i_missing = _find_nans(xn)
    if np.any(i_missing):
        xn[i_missing] = 0.  # set NaN's to zero
    if dcrit is None:
        dcrit = 1.5 * xn.std()
        if verbose:
            print('dcrit is set to %g' % dcrit)

    if ddcrit is None:
        ddcrit = 1.5 * xn.std()
        if verbose:
            print('ddcrit is set to %g' % ddcrit)

    dxn = np.diff(xn)
    ddxn = np.diff(dxn)

    ind = np.hstack((_find_spurious_jumps(dxn, dcrit, name='Dx'),
                     _find_spurious_jumps(ddxn, ddcrit, name='D^2x'),
                     _find_consecutive_equal_values(dxn, zcrit)))

    indg = ones(xn.size, dtype=bool)

    if ind.size > 1:
        ind = np.unique(ind)
        indg[ind] = 0
    indg, = np.nonzero(indg)

    if verbose:
        print('Found the total of %d spurious points' % np.size(ind))

    return ind, indg


def common_shape(*args, ** kwds):
    """Return the common shape of a sequence of arrays.

    Parameters
    -----------
    *args : arraylike
        sequence of arrays
    **kwds :
        shape

    Returns
    -------
    shape : tuple
        common shape of the elements of args.

    Raises
    ------
    An error is raised if some of the arrays do not conform
    to the common shape according to the broadcasting rules in numpy.

    Examples
    --------
    >>> import numpy as np
    >>> import wafo.misc as wm
    >>> A = np.ones((4,1))
    >>> B = 2
    >>> C = np.ones((1,5))*5
    >>> wm.common_shape(A,B,C) == (4, 5)
    True
    >>> wm.common_shape(A,B,C, shape=(3,4,1))  == (3, 4, 5)
    True

    See also
    --------
    broadcast, broadcast_arrays

    """

    shape = kwds.get('shape')
    x0 = 1 if shape is None else np.ones(shape)
    return tuple(np.broadcast(x0, *args).shape)


def argsreduce(condition, *args):
    """ Return the elements of each input array that satisfy some condition.

    Parameters
    ----------
    condition : array_like
        An array whose nonzero or True entries indicate the elements of each
        input array to extract. The shape of 'condition' must match the common
        shape of the input arrays according to the broadcasting rules in numpy.
    arg1, arg2, arg3, ... : array_like
        one or more input arrays.

    Returns
    -------
    narg1, narg2, narg3, ... : ndarray
        sequence of extracted copies of the input arrays converted to the same
        size as the nonzero values of condition.

    Examples
    --------
    >>> import wafo.misc as wm
    >>> import numpy as np
    >>> rand = np.random.random_sample
    >>> A = rand((4,5))
    >>> B = 2
    >>> C = rand((1,5))
    >>> cond = np.ones(A.shape)
    >>> [A1,B1,C1] = wm.argsreduce(cond,A,B,C)
    >>> B1.shape == (20,)
    True
    >>> cond[2,:] = 0
    >>> [A2,B2,C2] = wm.argsreduce(cond,A,B,C)
    >>> B2.shape == (15,)
    True

    See also
    --------
    numpy.extract
    """

s    condition = np.asarray(condition, dtype=bool)
    condition, *arrays = np.broadcast_arrays(condition, *args)

    return tuple(arr[condition] for arr in arrays)


    # Ensure boolean condition
    condition = np.asarray(condition, dtype=bool)

    # Broadcast condition and all args to common shape
    condition, *arrays = np.broadcast_arrays(condition, *args)

    # Extract values
    return tuple(np.extract(condition, arr) for arr in arrays)


    newargs = np.atleast_1d(*args)
    if not isinstance(newargs, tuple):
        newargs = [newargs, ]
    expand_arr = (condition == condition)
    return [np.extract(condition, arr1 * expand_arr) for arr1 in newargs]


def stirlerr(n):
    '''
    Returns error of Stirling approximation,
        i.e., log(n!) - log( sqrt(2*pi*n)*(n/exp(1))**n )

    Examples
    --------
    >>> import wafo.misc as wm
    >>> np.allclose(wm.stirlerr(2),  0.0413407)
    True
    >>> np.allclose(wm.stirlerr(5), 0.01664469)
    True
    >>> np.allclose(wm.stirlerr(8), 0.01041127)
    True
    >>> np.allclose(wm.stirlerr(12), 0.00694284)
    True
    >>> np.allclose(wm.stirlerr(25), 0.00333316)
    True
    >>> np.allclose(wm.stirlerr(70), 0.00119047)
    True
    >>> np.allclose(wm.stirlerr(100), 0.00083333)
    True


    See also
    --------
    binom


    References
    ----------
    Catherine Loader (2000).
    Fast and Accurate Computation of Binomial Probabilities
    <http://lists.gnu.org/archive/html/octave-maintainers/2011-09/pdfK0uKOST642.pdf>
    '''

    S0 = 0.083333333333333333333  # /* 1/12 */
    S1 = 0.00277777777777777777778  # /* 1/360 */
    S2 = 0.00079365079365079365079365  # /* 1/1260 */
    S3 = 0.000595238095238095238095238  # /* 1/1680 */
    S4 = 0.0008417508417508417508417508  # /* 1/1188 */

    n1 = np.atleast_1d(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = gammaln(n1 + 1) - log(sqrt(2 * pi * n1) * (n1 / exp(1)) ** n1)

    nn = n1 * n1

    n500 = 500 < n1
    y[n500] = (S0 - S1 / nn[n500]) / n1[n500]
    n80 = np.logical_and(80 < n1, n1 <= 500)
    if np.any(n80):
        y[n80] = (S0 - (S1 - S2 / nn[n80]) / nn[n80]) / n1[n80]
    n35 = np.logical_and(35 < n1, n1 <= 80)
    if np.any(n35):
        nn35 = nn[n35]
        y[n35] = (S0 - (S1 - (S2 - S3 / nn35) / nn35) / nn35) / n1[n35]

    n15 = np.logical_and(15 < n1, n1 <= 35)
    if np.any(n15):
        nn15 = nn[n15]
        y[n15] = (
            S0 - (S1 - (S2 - (S3 - S4 / nn15) / nn15) / nn15) / nn15) / n1[n15]

    return y


def _get_max_deadweight(**ship_property):
    names = list(ship_property)
    _assert(len(ship_property) == 1, 'Only one ship property allowed!')
    name = names[0]
    value = np.array(ship_property[name])
    valid_props = dict(le='length', be='beam', dr='draught',
                       ma='max_deadweigth',
                       se='service_speed', pr='propeller_diameter')
    prop = valid_props[name[:2]]
    prop2max_dw = dict(length=lambda x: (x / 3.45) ** (2.5),
                       beam=lambda x: ((x / 1.78) ** (1 / 0.27)),
                       draught=lambda x: ((x / 0.8) ** (1 / 0.24)),
                       service_speed=lambda x: ((x / 1.14) ** (1 / 0.21)),
                       propeller_diameter=lambda x: (((x / 0.12) ** (4 / 3) /
                                                      3.45) ** (2.5)),
                       max_deadweight=lambda x: x
                       )
    max_deadweight = prop2max_dw.get(prop, lambda x: x)(value)
    return max_deadweight, prop


def getshipchar(**ship_property):
    '''
    Return ship characteristics from value of one ship-property

    Parameters
    ----------
    **ship_property : scalar
        the ship property used in the estimation. Options are:
           'max_deadweight','length','beam','draft','service_speed',
           'propeller_diameter'.
           The length was found from statistics of 40 vessels of size 85 to
           100000 tonn. An exponential curve through 0 was selected, and the
           factor and exponent that minimized the standard deviation of the
           relative error was selected. (The error returned is the same for
           any ship.) The servicespeed was found for ships above 1000 tonns
           only. The propeller diameter formula is from [1]_.

    Returns
    -------
    sc : dict
        containing estimated mean values and standard-deviations of ship
        characteristics:
            max_deadweight    [kkg], (weight of cargo, fuel etc.)
            length            [m]
            beam              [m]
            draught           [m]
            service_speed      [m/s]
            propeller_diameter [m]

    Examples
    --------
    >>> import wafo.misc as wm
    >>> true_sc = {'beam': 29.0, 'beamSTD': 2.9000000000000004,
    ...    'draught': 9.6,  'draughtSTD': 2.112,
    ...    'length': 216.0, 'lengthSTD': 2.011309883194276,
    ...    'max_deadweight': 30969.0, 'max_deadweightSTD': 3096.9,
    ...    'propeller_diameter': 6.761165385916601, 'propeller_diameterSTD': 0.20267047566705432,
    ...    'service_speed': 10.0, 'service_speedSTD': 0}
    >>> wm.getshipchar(service_speed=10) == true_sc
    True
    >>> sc = wm.getshipchar(service_speed=10)
    >>> sc == true_sc
    True

    Other units: 1 ft = 0.3048 m and 1 knot = 0.5144 m/s


    References
    ----------
    .. [1] Gray and Greeley, (1978),
    "Source level model for propeller blade rate radiation for the world's
    merchant fleet", Bolt Beranek and Newman Technical Memorandum No. 458.
    '''

    max_deadweight, prop = _get_max_deadweight(**ship_property)
    property_std = prop + 'STD'

    length = np.round(3.45 * max_deadweight ** 0.40)
    length_err = length ** 0.13

    beam = np.round(1.78 * max_deadweight ** 0.27 * 10) / 10
    beam_err = beam * 0.10

    draught = np.round(0.80 * max_deadweight ** 0.24 * 10) / 10
    draught_err = draught * 0.22

    # S    = round(2/3*(L)**0.525)
    speed = np.round(1.14 * max_deadweight ** 0.21 * 10) / 10
    speed_err = speed * 0.10

    p_diam = 0.12 * length ** (3.0 / 4.0)
    p_diam_err = 0.12 * length_err ** (3.0 / 4.0)

    max_deadweight = np.round(max_deadweight)
    max_deadweight_std = 0.1 * max_deadweight

    shipchar = dict(beam=beam, beamSTD=beam_err,
                    draught=draught, draughtSTD=draught_err,
                    length=length, lengthSTD=length_err,
                    max_deadweight=max_deadweight, max_deadweightSTD=max_deadweight_std,
                    propeller_diameter=p_diam, propeller_diameterSTD=p_diam_err,
                    service_speed=speed, service_speedSTD=speed_err)

    shipchar[property_std] = 0
    return shipchar


def binomln(z, w):
    '''
    Natural Logarithm of binomial coefficient.

    CALL binomln(z,w)

    BINOMLN computes the natural logarithm of the binomial
    function for corresponding elements of Z and W.   The arrays Z and
    W must be real and nonnegative. Both arrays must be the same size,
    or either can be scalar.  BINOMLN is defined as:

    y = LOG(binom(Z,W)) = gammaln(Z)-gammaln(W)-gammaln(Z-W)

    and is obtained without computing BINOM(Z,W). Since the binom
    function can range over very large or very small values, its
    logarithm is sometimes more useful.
    This implementation is more accurate than the log(BINOM(Z,W) implementation
    for large arguments

    Examples
    --------

    >>> np.allclose(binomln(3,2), 1.09861229)
    True

    See also
    --------
    binom
    '''

    # return gammaln(z+1.) - gammaln(w+1.) - gammaln(z-w+1.)
    zmw = z-w
    k = np.where(zmw < 2, zmw, w)
    out = -log(z + 1.) - betaloge(z - k + 1., k + 1.)
    if np.isscalar(zmw):
        if k < 2:
            if k < 0:
                return -np.inf
            if k == 0:
                return 0.0
            if k == 1:
                return log(z)
    else:
        n = np.broadcast_to(z, np.shape(k))
        out[k < 0] = -np.inf
        out[k == 0] = 0.0
        out[k == 1] = log(n[k == 1])
    return out


def _betaln3(a, b):
    """ Returns betaln valid for huge arguments: 10 <= a  and b < huge

    More accurate than special.betaln
    """
    apb = a + b
    corr = stirlerr(a) + stirlerr(b) - stirlerr(apb)
    return (- 0.5 * log(b) + 0.5 * log(2 * pi) + corr
            + (a - 0.5) * log(a/apb) + b * log1p(-a/apb))


def betaloge(z, w):
    '''
    Natural Logarithm of beta function.

    CALL betaloge(z,w)

    BETALOGE computes the natural logarithm of the beta
    function for corresponding elements of Z and W.   The arrays Z and
    W must be real and nonnegative. Both arrays must be the same size,
    or either can be scalar.  BETALOGE is defined as:

    y = LOG(BETA(Z,W)) = gammaln(Z)+gammaln(W)-gammaln(Z+W)

    and is obtained without computing BETA(Z,W). Since the beta
    function can range over very large or very small values, its
    logarithm is sometimes more useful.
    This implementation is more accurate than the BETALN implementation
    for large arguments

    Examples
    --------
    >>> import wafo.misc as wm
    >>> np.allclose(wm.betaloge(3,2), -2.48490665)
    True

    See also
    --------
    betaln, beta
    '''
    huge = FLOATINFO.max
    a = np.minimum(z, w)
    b = np.maximum(z, w)
    out = np.full(b.shape, np.nan, dtype=float)
    r00 = (a == 0)
    out[r00] = np.inf
    r01 = (0 < a) & (huge <= b)
    out[r01] = -np.inf
    r_3 = (10 <= a) & (b < huge)
    if np.any(r_3):
        out[r_3] = _betaln3(a[r_3], b[r_3])
    r12 = (0 < a) & (a < 10) & (b < huge)
    if np.any(r12):
        out[r12] = betaln(a[r12], b[r12])
    return out


def gravity(phi=45):
    ''' Returns the constant acceleration of gravity

    GRAVITY calculates the acceleration of gravity
    using the international gravitational formulae [1]_:

      g = 9.78049*(1+0.0052884*sin(phir)**2-0.0000059*sin(2*phir)**2)
    where
      phir = phi*pi/180

    Parameters
    ----------
    phi : {float, int}
         latitude in degrees

    Returns
    --------
    g : ndarray
        acceleration of gravity [m/s**2]

    Examples
    --------
    >>> import wafo.misc as wm
    >>> import numpy as np
    >>> phi = np.linspace(0,45,5)
    >>> np.allclose(wm.gravity(phi),
    ...            [ 9.78049   ,  9.78245014,  9.78803583, 9.79640552,  9.80629387])
    True

    See also
    --------
    wdensity

    References
    ----------
    .. [1] Irgens, Fridtjov (1987)
            "Formelsamling i mekanikk:
            statikk, fasthetsl?re, dynamikk fluidmekanikk"
            tapir forlag, University of Trondheim,
            ISBN 82-519-0786-1, pp 19

    '''

    phir = phi * pi / 180.  # change from degrees to radians
    return 9.78049 * (1. + 0.0052884 * sin(phir) ** 2.
                      - 0.0000059 * sin(2 * phir) ** 2.)


def nextpow2(x):
    '''
    Return next higher power of 2

    Examples
    --------
    >>> import wafo.misc as wm
    >>> np.allclose(wm.nextpow2(10), 4)
    True
    >>> np.allclose(wm.nextpow2(np.arange(5)), 3)
    True
    '''
    t = np.isscalar(x) or len(x)
    if t > 1:
        f, n = np.frexp(t)
    else:
        f, n = np.frexp(np.abs(x))

    if f == 0.5:
        n = n - 1
    return n


def discretize(fun, a, b, tol=0.005, n=5, method='linear'):
    '''
    Automatic discretization of function

    Parameters
    ----------
    fun : callable
        function to discretize
    a, b : real scalars
        evaluation limits
    tol : real, scalar
        absoute error tolerance
    n : scalar integer
        number of values to start the discretization with.
    method : string
        defining method of gridding, options are 'linear' and 'adaptive'

    Returns
    -------
    x : discretized values
    y : fun(x)

    Examples
    --------
    >>> import wafo.misc as wm
    >>> import numpy as np
    >>> fun = lambda x: 1./x
    >>> a, b = 0.1, 1.0
    >>> x, y = wm.discretize(fun, a, b)
    >>> np.allclose(x[-5:], [0.94375, 0.9578125, 0.971875, 0.9859375, 1.0])
    True
    >>> len(x)
    65

    >>> xa, ya = wm.discretize(fun, a, b, method='adaptive')
    >>> np.allclose(xa[-5:], [0.6625, 0.71875, 0.775, 0.8875, 1.0])
    True
    >>> len(xa)
    25

    >>> import matplotlib.pyplot as plt
    >>> h0 = plt.plot(x, y, '.', label='linear')
    >>> h1 = plt.plot(xa, ya, 'ro', label='adaptive', fillstyle='none')
    >>> h2 = plt.legend()

    >>> plt.close('all')

    '''
    if method.startswith('a'):
        return _discretize_adaptive(fun, a, b, tol, n)
    return _discretize_linear(fun, a, b, tol, n)


def _discretize_linear(fun, a, b, tol=0.005, n=5):
    '''
    Automatic discretization of function, linear gridding
    '''
    x = np.linspace(a, b, n)
    y = fun(x)

    err0 = inf
    err = 10000
    nmax = 2 ** 20
    num_tries = 0
    while num_tries < 5 and err > tol and n < nmax:
        err0 = err
        x0 = x
        y0 = y
        n = 2 * (n - 1) + 1
        x = np.linspace(a, b, n)
        y = fun(x)
        y00 = np.interp(x, x0, y0)
        err = 0.5 * np.amax(np.abs(y00 - y) / (np.abs(y00) + np.abs(y) + _TINY + tol))
        num_tries += int(abs(err - err0) <= tol / 2)
    return x, y


def _discretize_adaptive(fun, a, b, tol=0.005, n=5):
    '''
    Automatic discretization of function, adaptive gridding.
    '''
    n += (np.mod(n, 2) == 0)  # make sure n is odd
    x = np.linspace(a, b, n)
    fx = fun(x)

    n2 = (n - 1) // 2
    erri = np.hstack((zeros((n2, 1)), ones((n2, 1)))).ravel()
    err = erri.max()
    err0 = inf
    num_tries = 0
    # reltol = abstol = tol
    for j in range(50):
        if num_tries < 5 and err > tol:
            err0 = err
            # find top errors

            ix, = np.where(erri > tol)
            # double the sample rate in intervals with the most error
            y = (np.vstack(((x[ix] + x[ix - 1]) / 2,
                            (x[ix + 1] + x[ix]) / 2)).T).ravel()
            fy = fun(y)
            fy0 = np.interp(y, x, fx)

            abserr = np.abs(fy0 - fy)
            erri = 0.5 * (abserr / (np.abs(fy0) + np.abs(fy) + _TINY + tol))
            # converged = abserr <= np.maximum(abseps, releps * abs(fy))
            # converged = abserr <= np.maximum(tol, tol * abs(fy))
            err = erri.max()

            x = np.hstack((x, y))

            ix = x.argsort()
            x = x[ix]
            erri = np.hstack((zeros(len(fx)), erri))[ix]
            fx = np.hstack((fx, fy))[ix]
            num_tries += int(abs(err - err0) <= tol / 2)
        else:
            break
    else:
        warnings.warn('Recursion level limit reached j=%d' % j)

    return x, fx


def polar2cart(theta, rho, z=None):
    '''
    Transform polar coordinates into 2D cartesian coordinates.

    Returns
    -------
    x, y : array-like
        Cartesian coordinates, x = rho*cos(theta), y = rho*sin(theta)

    Examples
    --------
    >>> np.allclose(polar2cart(0, 1, 1), (1, 0, 1))
    True
    >>> np.allclose(polar2cart(0, 1), (1, 0))
    True

    See also
    --------
    cart2polar
    '''
    x, y = rho * cos(theta), rho * sin(theta)
    if z is None:
        return x, y
    return x, y, z


pol2cart = polar2cart


def cart2polar(x, y, z=None):
    ''' Transform 2D cartesian coordinates into polar coordinates.

    Returns
    -------
    theta : array-like
        radial angle, arctan2(y,x)
    rho : array-like
        radial distance, sqrt(x**2+y**2)

    Examples
    --------
    >>> np.allclose(cart2polar(1, 0, 1), (0, 1, 1))
    True
    >>> np.allclose(cart2polar(1, 0), (0, 1))
    True

    See also
    --------
    polar2cart
    '''
    t, r = arctan2(y, x), np.hypot(x, y)
    if z is None:
        return t, r
    return t, r, z


cart2pol = cart2polar


def ndgrid(*args, **kwargs):
    """
    Same as calling meshgrid with indexing='ij' (see meshgrid for
    documentation).

    Examples
    --------
    >>> x, y = ndgrid([1,2,3],[4,5,6])
    >>> np.allclose(x, [[1, 1, 1],
    ...                 [2, 2, 2],
    ...                 [3, 3, 3]])
    True
    >>> np.allclose(y, [[4, 5, 6],
    ...                 [4, 5, 6],
    ...                 [4, 5, 6]])
    True
    """
    kwargs['indexing'] = 'ij'
    return meshgrid(*args, **kwargs)


def trangood(x, f, min_n=None, min_x=None, max_x=None, max_n=inf):
    """
    Make sure transformation is efficient.

    Parameters
    ------------
    x, f : array_like
        input transform function, (x,f(x)).
    min_n : scalar, int
        minimum number of points in the good transform.
               (Default  x.shape[0])
    min_x : scalar, real
        minimum x value to transform. (Default  min(x))
    max_x : scalar, real
        maximum x value to transform. (Default  max(x))
    max_n : scalar, int
        maximum number of points in the good transform
              (default inf)
    Returns
    -------
    x, f : array_like
        the good transform function.

    Notes
    -----
    TRANGOOD interpolates f linearly  and optionally
    extrapolate it linearly outside the range of x
    with X uniformly spaced.

    See also
    ---------
    tranproc,
    numpy.interp
    """
    xo, fo = np.atleast_1d(x, f)

    _assert(xo.ndim == 1, 'x must be a vector.')
    _assert(fo.ndim == 1, 'f  must be a vector.')

    i = xo.argsort()
    xo, fo = xo[i], fo[i]
    del i
    dx = np.diff(xo)
    _assert(all(dx > 0), 'Duplicate x-values not allowed.')

    nf = fo.shape[0]

    max_x = xo[-1] if max_x is None else max_x
    min_x = xo[0] if min_x is None else min_x
    min_n = nf if min_n is None else min_n
    min_n = max(min_n, 2)
    max_n = max(max_n, 2)

    ddx = np.diff(dx)
    xn = xo[-1]
    x0 = xo[0]
    L = float(xn - x0)
    if nf < min_n or max_n < nf or np.any(np.abs(ddx) > 10 * _EPS * L):
        # pab 07.01.2001: Always choose the stepsize df so that
        # it is an exactly representable number.
        # This is important when calculating numerical derivatives and is
        # accomplished by the following.
        dx = L / (min(min_n, max_n) - 1)
        dx = (dx + 2.) - 2.
        xi = np.arange(x0, xn + dx / 2., dx)
        # New call pab 11.11.2000: This is much quicker
        fo = np.interp(xi, xo, fo)
        xo = xi

    # x is now uniformly spaced
    dx = xo[1] - xo[0]

    # Extrapolate linearly outside the range of ff
    if min_x < xo[0]:
        x1 = dx * np.arange(np.floor((min_x - xo[0]) / dx), -2)
        f2 = fo[0] + x1 * (fo[1] - fo[0]) / (xo[1] - xo[0])
        fo = np.hstack((f2, fo))
        xo = np.hstack((x1 + xo[0], xo))

    if max_x > xo[-1]:
        x1 = dx * np.arange(1, np.ceil((max_x - xo[-1]) / dx) + 1)
        f2 = f[-1] + x1 * (f[-1] - f[-2]) / (xo[-1] - xo[-2])
        fo = np.hstack((fo, f2))
        xo = np.hstack((xo, x1 + xo[-1]))

    return xo, fo


def tranproc(x, f, x0, *xi):
    """
    Transforms process X and up to four derivatives using the transformation f.

    Parameters
    ----------
    x,f : array-like
        [x,f(x)], transform function, y = f(x).
    x0, x1,...,xn : vectors
        where xi is the i'th time derivative of x0. 0<=n<=4.

    Returns
    -------
    y0, y1,...,yn : vectors
        where yi is the i'th time derivative of y0 = f(x0).

    By the basic rules of derivation:
    Y1 = f'(X0)*X1
    Y2 = f''(X0)*X1^2 + f'(X0)*X2
    Y3 = f'''(X0)*X1^3 + f'(X0)*X3 + 3*f''(X0)*X1*X2
    Y4 = f''''(X0)*X1^4 + f'(X0)*X4 + 6*f'''(X0)*X1^2*X2
      + f''(X0)*(3*X2^2 + 4*X1*X3)

    The derivation of f is performed numerically with a central difference
    method with linear extrapolation towards the beginning and end of f,
    respectively.

    Examples
    --------
    Derivative of g and the transformed Gaussian model.
    >>> import wafo.misc as wm
    >>> import wafo.transform.models as wtm
    >>> tr = wtm.TrHermite()
    >>> x = np.linspace(-5, 5, 501)
    >>> g = tr(x)
    >>> gder = wm.tranproc(x, g, x, ones(g.shape[0]))
    >>> np.allclose(gder[1][:5],
    ... [ 1.09938766,  1.39779849,  1.39538745,  1.39298656,  1.39059575])
    True

    >>> import matplotlib.pyplot as plt
    >>> import wafo.stats as ws

    >>> h0 = plt.plot(x, g, label='Hermite transform')
    >>> h1 = plt.plot(x, x, label='Linear transform')
    >>> h2 = plt.legend()
    >>> plt.close('all')

    >>> h3 = plt.plot(x, ws.norm.pdf(g) * gder[1], label='Transformed model')
    >>> h4 = plt.plot(x, ws.norm.pdf(x), label='Gaussian model')
    >>> h5 = plt.legend()
    >>> plt.close('all')

    See also
    --------
    trangood.
    """
    def _default_step(xo, num_derivatives):
        hn = xo[1] - xo[0]
        if hn ** num_derivatives < sqrt(_EPS):
            msg = ('Numerical problems may occur for the derivatives in ' +
                   'tranproc.\n' +
                   'The sampling of the transformation may be too small.')
            warnings.warn(msg)
        return hn

    def _diff(xo, fo, x0, num_derivatives):
        hn = _default_step(xo, num_derivatives)
        # Transform X with the derivatives of  f.
        fder = np.vstack((xo, fo))
        fxder = np.zeros((num_derivatives, x0.size))
        for k in range(num_derivatives):  # Derivation of f(x) using a difference method.
            n = fder.shape[-1]
            fder = np.vstack([(fder[0, 0:n - 1] + fder[0, 1:n]) / 2,
                              np.diff(fder[1, :]) / hn])
            fxder[k] = tranproc(fder[0], fder[1], x0)
        return fxder

    def _der_1(fxder, xi):
        """First time derivative of y: y1 = f'(x)*x1"""
        return fxder[0] * xi[0]

    def _der_2(fxder, xi):
        """Second time derivative of y: y2 = f''(x)*x1.^2+f'(x)*x2"""
        return fxder[1] * xi[0] ** 2. + fxder[0] * xi[1]

    def _der_3(fxder, xi):
        """Third time derivative of y:
        y3 = f'''(x)*x1.^3+f'(x)*x3 +3*f''(x)*x1*x2
        """
        return (fxder[2] * xi[0] ** 3 + fxder[0] * xi[2] +
                3 * fxder[1] * xi[0] * xi[1])

    def _der_4(fxder, xi):
        """Fourth time derivative of y:
            y4 = f''''(x)*x1.^4+f'(x)*x4 +
                 6*f'''(x)*x1^2*x2+f''(x)*(3*x2^2+4x1*x3)
        """
        return (fxder[3] * xi[0] ** 4. + fxder[0] * xi[3] +
                6. * fxder[2] * xi[0] ** 2. * xi[1] +
                fxder[1] * (3. * xi[1] ** 2. + 4. * xi[0] * xi[1]))

    xo, fo, x0 = np.atleast_1d(x, f, x0)
    xi = np.atleast_1d(*xi)
    if not isinstance(xi, list):
        xi = [xi, ]
    num_derivatives = len(xi)  # num_derivatives = number of derivatives
    nmax = np.ceil((xo.ptp()) * 10 ** (7. / max(num_derivatives, 1)))
    xo, fo = trangood(xo, fo, min_x=min(x0), max_x=max(x0), max_n=nmax)

    n = f.shape[0]

    xu = (n - 1) * (x0 - xo[0]) / (xo[-1] - xo[0])

    fi = np.asarray(np.floor(xu), dtype=int)
    fi = np.where(fi == n - 1, fi - 1, fi)

    xu = xu - fi
    y0 = fo[fi] + (fo[fi + 1] - fo[fi]) * xu

    y = y0
    if num_derivatives > 4:
        warnings.warn('Transformation of derivatives of order>4 is ' +
                      'not supported.')
        num_derivatives = 4
    if num_derivatives > 0:
        y = [y0]
        fxder = _diff(xo, fo, x0, num_derivatives)

        # Calculate the transforms of the derivatives of X.
        dfuns = [_der_1, _der_2, _der_3, _der_4]
        for dfun in dfuns[:num_derivatives]:
            y.append(dfun(fxder, xi))

    return y


# pylint: disable=redefined-builtin
def good_bins(data=None, range=None, num_bins=None, odd=False, loose=True):  # @ReservedAssignment
    ''' Return good bins for histogram

    Parameters
    ----------
    data : array-like
        the data
    range : (float, float)
        minimum and maximum range of bins (default data.min(), data.max())
    num_bins : scalar integer
        approximate number of bins wanted
        (default depending on num_data=len(data))
    odd : bool
        placement of bins (0 or 1) (default 0)
    loose : bool
        if True add extra space to min and max
        if False the bins are made tight to the min and max

    Examples
    --------
    >>> import wafo.misc as wm
    >>> np.allclose(wm.good_bins(range=(0,5), num_bins=6),
    ...             [-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.])
    True
    >>> np.allclose(wm.good_bins(range=(0,5), num_bins=6, loose=False),
    ...             [ 0.,  1.,  2.,  3.,  4.,  5.])
    True
    >>> np.allclose(wm.good_bins(range=(0,5), num_bins=6, odd=True),
    ...            [-1.5, -0.5,  0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5])
    True
    >>> np.allclose(wm.good_bins(range=(0,5), num_bins=6, odd=True, loose=False),
    ...             [-0.5,  0.5,  1.5,  2.5,  3.5,  4.5,  5.5])
    True
    '''
    def _default_range(range_, x):
        return range_ if range_ else (x.min(), x.max())

    def _default_bins(num_bins, x):
        if num_bins is None:
            num_bins = int(np.ceil(4 * np.sqrt(np.sqrt(len(x)))))
        return num_bins

    def _default_step(mn, mx, num_bins):
        d = float(mx - mn) / num_bins * 2
        e = np.floor(np.log(d) / np.log(10))
        m = np.clip(np.floor(d / 10 ** e), a_min=0, a_max=5)
        if 2 < m < 5:
            m = 2
        return m * 10 ** e

    if data is not None:
        data = np.atleast_1d(data)

    mn, mx = _default_range(range, data)
    num_bins = _default_bins(num_bins, data)
    d = _default_step(mn, mx, num_bins)
    mn = (np.floor(mn / d) - loose) * d - odd * d / 2
    mx = (np.ceil(mx / d) + loose) * d + odd * d / 2
    limits = np.arange(mn, mx + d / 2, d)
    return limits


def _make_bars(limits, bin_):
    limits.shape = (-1, 1)
    xx = limits.repeat(3, axis=1)
    xx.shape = (-1,)
    xx = xx[1:-1]
    bin_.shape = (-1, 1)
    yy = bin_.repeat(3, axis=1)
    # yy[0,0] = 0.0 # pdf
    yy[:, 0] = 0.0  # histogram
    yy.shape = (-1,)
    yy = np.hstack((yy, 0.0))
    return xx, yy


# pylint: disable=redefined-builtin
def _histogram(data, bins=None, range=None, density=False, weights=None):  # @ReservedAssignment

    """
    Examples
    --------
    >>> import numpy as np
    >>> data = np.linspace(0, 10)
    >>> xx, yy, limits = _histogram(data)
    >>> len(limits)
    12
    >>> xx, yy, limits = _histogram(data, bins=[0, 5, 11])
    >>> np.allclose(xx, [ 0,  0,  5,  5,  5, 11, 11])
    True
    >>> np.allclose(yy, [  0.,  25.,  25.,   0.,  25.,  25.,   0.])
    True
    >>> np.allclose(limits, [[ 0], [ 5], [11]])
    True

    """
    x = np.atleast_1d(data)
    if bins is None:
        bins = int(np.ceil(4 * np.sqrt(np.sqrt(len(x)))))
    bin_, limits = np.histogram(data,
                                bins=bins,
                                range=range,
                                weights=weights,
                                density=density)
    xx, yy = _make_bars(limits, bin_)
    return xx, yy, limits


def plot_histgrm(data, bins=None, range=None,  # @ReservedAssignment
                 density=False, weights=None, lintype='b-'):
    '''
    Plot histogram

    Parameters
    -----------
    data : array-like
        the data
    bins : int or sequence of scalars, optional
        If an int, it defines the number of equal-width
        bins in the given range (4 * sqrt(sqrt(len(data)), by default).
        If a sequence, it defines the bin edges, including the
        rightmost edge, allowing for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(data.min(), data.max())``.  Values outside the range are
        ignored.
    density : bool, optional
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability *density* function
        at the bin, normalized such that the *integral* over the range is 1.
    weights : array_like, optional
        An array of weights, of the same shape as `data`.  Each value in `data`
        only contributes its associated weight towards the bin count
        (instead of 1).  If `normed` is True, the weights are normalized,
        so that the integral of the density over the range remains 1
    lintype : specify color and lintype, see PLOT for possibilities.

    Returns
    -------
    h : list of plot-objects


    Examples
    --------
    >>> import wafo.misc as wm
    >>> import wafo.stats as ws
    >>> R = ws.weibull_min.rvs(2,loc=0,scale=2, size=100)
    >>> bins = wm.good_bins(R)

    >>> x = np.linspace(-3,16,200)
    >>> pdf = ws.weibull_min.pdf(x, 2, 0, 2)

    >>> import matplotlib.pyplot as plt
    >>> h0 = wm.plot_histgrm(R, bins, density=True)
    >>> h1 = plt.plot(x, pdf,'r')

    >>> plt.close('all')

    See also
    --------
    wafo.misc.good_bins
    numpy.histogram
    '''

    xx, yy, limits = _histogram(data, bins, range, density, weights)
    return plt.plot(xx, yy, lintype, limits, limits * 0)


def num2pistr(x, n=3, numerator_max=10, denominator_max=10):
    r'''
    Convert a scalar to a text string in fractions of pi.

    Examples
    --------
    >>> import utilities.numpy_utils as wm
    >>> wm.num2pistr(np.pi*3/4)==r'3\pi/4'
    True
    >>> wm.num2pistr(-np.pi/4)==r'-\pi/4'
    True
    >>> wm.num2pistr(-np.pi)==r'-\pi'
    True
    >>> wm.num2pistr(-1/4)=='-0.25'
    True
    '''
    
    def _denominator_text(den):
        return '' if abs(den) == 1 else f'/{den}'

    def _numerator_text(num):
        if abs(num) == 1:
            return '-' if num == -1 else ''
        return f'{num:d}'

    frac = fractions.Fraction(x / pi).limit_denominator()
    num, den = frac.numerator, frac.denominator

    if (abs(den) <= denominator_max and 
        abs(num) <= numerator_max and 
        num != 0):
        return rf'{_numerator_text(num)}\pi{_denominator_text(den)}'

    return f'{x:.{n}g}'



def fourier(data, t=None, period=None, m=None, method='trapezoid'):
    '''
    Returns Fourier coefficients.

    Parameters
    ----------
    data : array-like
        vector or matrix of row vectors with data points shape p x n.
    t : array-like
        vector with n values indexed from 1 to n.
    period : real scalar, (default t[-1]-t[0])
        primitive period of signal, i.e., smallest period.
    m : scalar integer
        defines number of harmonics desired (default m = n)
    method : string
        integration method used ("trapezoid" or "simpson")

    Returns
    -------
    a,b  = Fourier coefficients size m x p

    FOURIER finds the coefficients for a Fourier series representation
    of the signal x(t) (given in digital form).  It is assumed the signal
    is periodic over T.  N is the number of data points, and M-1 is the
    number of coefficients.

    The signal can be estimated by using M-1 harmonics by:
                        M-1
     x[i] = 0.5*a[0] + sum (a[n]*c[n,i] + b[n]*s[n,i])
                       n=1
    where
       c[n,i] = cos(2*pi*(n-1)*t[i]/T)
       s[n,i] = sin(2*pi*(n-1)*t[i]/T)

    Note that a[0] is the "dc value".
    Remaining values are a[1], a[2], ... , a[M-1].

    Examples
    --------
    >>> import wafo.misc as wm
    >>> import numpy as np
    >>> T = 2*np.pi
    >>> t = np.linspace(0, 4*T)
    >>> x = np.sin(t)
    >>> a, b = wm.fourier(x, t, period=T, m=5)
    >>> np.allclose(a, [ 0.,  0.,  0.,  0.,  0.])
    True
    >>> np.allclose(b.ravel(), [ 0.,  4.,  0.,  0.,  0.])
    True

    See also
    --------
    fft
    '''
    x = np.atleast_2d(data)
    p, n = x.shape
    t = np.arange(n) if t is None else np.atleast_1d(t)

    n = len(t) if n is None else n
    m = n if m is None else m
    period = t[-1] - t[0] if period is None else period
    intfun = trapezoid if method.startswith('trap') else simpson

    # Define the vectors for computing the Fourier coefficients
    t.shape = (1, -1)
    a = zeros((m, p))
    b = zeros((m, p))
    a[0] = intfun(x, t, axis=-1)

    # Compute M-1 more coefficients
    tmp = 2 * pi * t / period
    for i in range(1, m):
        a[i] = intfun(x * cos(i * tmp), t, axis=-1)
        b[i] = intfun(x * sin(i * tmp), t, axis=-1)

    a = a / pi
    b = b / pi

    # Alternative:  faster for large M, but gives different results than above.
#    nper = np.diff(t([1 end]))/period; # Number of periods given
#    if nper == round(nper):
#        N1 = n/nper
#    else:
#        N1 = n
#
#
#
# Fourier coefficients by fft
#    Fcof1 = 2*ifft(x(1:N1,:),[],1);
#    Pcor = [1; exp(sqrt(-1)*(1:M-1).'*t(1))] # correction term to get
#                                             # the correct integration limits
#    Fcof = Fcof1(1:M,:).*Pcor(:,ones(1,P));
#    a = real(Fcof(1:M,:));
#    b = imag(Fcof(1:M,:));

    return a, b


if __name__ == "__main__":
    from wafo.testing import test_docstrings
    test_docstrings(__file__)
