"""
Created on 6. okt. 2016

@author: pab
"""
import warnings
import numpy as np
from numba import njit

_WAVE_KINDS = frozenset(('dw', 'uw', 'tw', 'cw'))


jit(int64(int64[:], int8[:]), nopython=True)def _findcross(ind, y):
    """
    Find zero-level crossings in a 1D signal.

    Parameters
    ----------
    ind : ndarray[int64]
        Output array (preallocated)
    y : ndarray[float64]
        Input signal

    Returns
    -------
    int
        Number of crossings stored in `ind`

    Notes
    -----
    Alternates between up- and down-crossings.
    
    Same implementation as findcross function found in c_functions.c.
    """

    n = len(y)
    
    if n < 2:
        return 0

    ix = 0

    zero = 0.0
    start = 0

    # ------------------------------------------------------------
    # Determine initial crossing direction
    # ------------------------------------------------------------
    if y[0] < zero:
        # First crossing must be upward
        direction = -1
    elif y[0] > zero:
        # First crossing must be downward
        direction = +1
    else:
        # Handle exact zero at start
        direction = 0

        for i in range(1, n):
            if y[i] < zero:
                ind[ix] = i - 1
                ix += 1
                direction = -1
                start = i
                break

            elif y[i] > zero:
                ind[ix] = i - 1
                ix += 1
                direction = +1
                start = i
                break

    # ------------------------------------------------------------
    # Main loop: detect alternating crossings
    # ------------------------------------------------------------
    for i in range(start, n - 1):

        # Up-crossing
        if direction == -1:
            if y[i] <= zero and zero < y[i + 1]:
                ind[ix] = i
                ix += 1
                direction = +1

        # Down-crossing
        elif direction == +1:
            if zero <= y[i] and y[i + 1] < zero:
                ind[ix] = i
                ix += 1
                direction = -1
        else:
            continue

    return ix


def findcross(x, v=0.0, kind=None):
    """
    Return indices of level-v crossings in a one-dimensional signal.

    Parameters
    ----------
    x : array_like
        One-dimensional signal.
    v : float, optional
        Crossing level (default 0.0)
    kind : {"u", "d", "dw", "uw", "cw", "tw", None, "all", "du"}, optional
        Selects the type of crossings or waves returned:
        "dw" : downcrossing wave
        "uw" : upcrossing wave
        "cw" : crest wave
        "tw" : trough wave
        "d"  : downcrossings only
        "u"  : upcrossings only
        None / "all" / "du" : all crossings

    Returns
    -------
    ind : ndarray
        Indices of the crossings in the original sequence x.
        Each index i identifies a crossing between x[i] and x[i+1].
        May be empty if no crossings are found.

    Notes
    -----
    A crossing is detected when the signal changes sign:

    - Up-crossing:   x[i] <= v < x[i+1]
    - Down-crossing: x[i] >= v > x[i+1]
    
    Crossings are returned in alternating order (up, down, up, ... or
    down, up, down, ...), matching the original WAFO/C implementation.
    
    For wave types ("dw", "uw", "cw", "tw"), a subset of the
    crossing sequence is returned defining the corresponding waves.
    
    If no crossings are found (for example, when the signal is constant and
    equal to the crossing level), an empty array is returned.


    Examples
    --------
    >>> import wafo.misc as wm
    >>> np.allclose(wm.findcross([0, 1, -1, 1], 0), [0, 1, 2])
    True
    >>> v = 0.75
    >>> t = np.linspace(0, 7*np.pi, 250)
    >>> x = np.sin(t)
    >>> ind = wm.findcross(x, v) # all crossings
    >>> np.allclose(ind, [9,  25,  80,  97, 151, 168, 223, 239])
    True

    >>> ind2 = wm.findcross(x, v, 'u')
    >>> np.allclose(ind2, [9,  80, 151, 223])
    True
    >>> ind3 = wm.findcross(x, v, 'd')
    >>> np.allclose(ind3, [25,  97, 168, 239])
    True
    >>> ind4 = wm.findcross(x, v, 'dw')
    >>> np.allclose(ind4, [25,  80,  97, 151, 168, 223, 239])
    True

    >>> from matplotlib import pyplot as plt
    >>> h0 = plt.plot(t, x, '.', label='data')
    >>> h1 = plt.plot(t[ind], x[ind], 'r.', label='all crossings')
    >>> h2 = plt.plot(t, np.full_like(t, v), label=f'{v} level')
    >>> h3 = plt.plot(t[ind2], x[ind2], 'o', label='upcrossings', fillstyle='none')
    >>> h4 = plt.legend()

    >>> plt.close('all')

    See also
    --------
    crossdef
    wavedef
    """
    
    kind = kind.lower() if kind is not None else None
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("x must be 1D")
    y = x - v
    ind = np.empty(y.size, dtype=np.int64)
    m = _findcross(ind, y)
    ind = ind[:m]
    if ind.size == 0 or kind in {'du', 'all', None}:
        return ind
    
    # Determine crossing direction
    is_up = y[ind] <= 0
    
    if kind == 'u':
        return ind[is_up]

    is_down = ~is_up
    if kind == 'd':
        return ind[is_down]

    if kind in _WAVE_KINDS:
        # Ensure correct first crossing
        first_is_down = is_down[0]
        want_down_first = kind in {'dw', 'tw'}

        if first_is_down != want_down_first:
            ind = ind[1:]
            if ind.size == 0:
                return ind

        # Ensure correct parity
        # dw/uw require an odd number of crossings
        # tw/cw require an even number of crossings
        if kind in {'dw', 'uw'}:  # odd length
            if ind.size % 2 == 0:
                ind = ind[:-1]
        else:  # tw or cw even length
            if ind.size % 2 == 1:
                ind = ind[:-1]
        return ind

    raise ValueError(f"Unknown kind: {kind}")


@njit
def _extract_turning_points(y, ind):
    """
    Extract indices of turning points (minima/maxima) from a signal.

    - Single-pass
    - Plateau-safe
    - Ensures strict alternation
    - Includes first and last points (RFC compatible)

    Parameters
    ----------
    y : ndarray[float64]
        Input signal
    ind : ndarray[int64]
        Output buffer (preallocated)

    Returns
    -------
    int
        Number of turning points written to ind
    """

    n = len(y)
    if n < 3:
        if n > 0:
            ind[0] = 0
            return 1
        return 0

    k = 0

    # ------------------------------------------------------------
    # include first point
    # ------------------------------------------------------------
    ind[k] = 0
    k += 1

    # ------------------------------------------------------------
    # find first non-zero slope
    # ------------------------------------------------------------
    i = 1
    while i < n and y[i] == y[i - 1]:
        i += 1

    if i == n:
        return 1  # constant signal

    prev_diff = y[i] - y[i - 1]

    # track plateau end
    plateau_idx = i

    # ------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------
    for j in range(i, n - 1):

        diff = y[j + 1] - y[j]

        if diff == 0.0:
            plateau_idx = j + 1
            continue

        # detect sign change using last plateau index
        if prev_diff > 0.0 and diff < 0.0:
            ind[k] = plateau_idx
            k += 1

        elif prev_diff < 0.0 and diff > 0.0:
            ind[k] = plateau_idx
            k += 1

        prev_diff = diff
        plateau_idx = j + 1

    # ------------------------------------------------------------
    # include last point
    # ------------------------------------------------------------
    ind[k] = n - 1
    k += 1

    return k


def findextrema(x, include_endpoints=False):
    '''
    Return indices to minima and maxima of a vector

    Parameters
    ----------
    x : array_like
        Input signal.
    include_endpoints : bool, optional
        If True, include first and last points.


    Returns
    -------
    ind : ndarray[int64]
        Indices of extrema.
    
    Notes
    -----
    - Flat regions (plateaus) are handled safely.
    - Output alternates strictly between minima and maxima.

    Examples
    --------
    >>> import numpy as np
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,7*np.pi,250)
    >>> x = np.sin(t)
    >>> ind = wm.findextrema(x)
    >>> np.allclose(ind, [18, 53, 89, 125, 160, 196, 231])
    True

    >>> import matplotlib.pyplot as plt
    >>> h0 = plt.plot(t, x, '.', label='data')
    >>> h1 = plt.plot(t[ind], x[ind], 'r.', label='extrema')
    >>> h2 = plt.legend()

    >>> plt.close('all')

    See also
    --------
    findcross
    crossdef
    '''
    
    x = np.asarray(x, dtype=np.float64, order='C')
    n = len(x)

    if n < 3:
        return np.empty(0, dtype=np.int64)

    tp_ind = np.empty(n, dtype=np.int64)
    m = _extract_turning_points(x, tp_ind)
    
    if m <= 2:
        return tp_ind[:m] if include_endpoints else np.empty(0, dtype=np.int64)

    if include_endpoints:
        return tp_ind[:m]
    return tp_ind[1:m-1]



# ============================================================
# RFC filter kernel (method 0 & 1 unified)
# ============================================================


@njit
def _keep_cycle(x_center, x_ref, h, inclusive):
    diff = x_center - x_ref
    if inclusive:
        return diff >= h
    else:
        return diff > h



@njit
def _findrfc_core(ind, y, h, inclusive=True):
    """
    RFC turning point filter (WAFO-style, legacy equivalent).

    Parameters
    ----------
    ind : ndarray[int64]
        Output indices (preallocated)
    y : ndarray[float64]
        Turning points
    h : float
        Threshold
    inclusive : bool
        If True: keep cycles with range >= h
        If False: keep cycles with range > h

    Returns
    -------
    int
        Number of indices written to ind

    Notes
    -----
    Refactored version of original C/WAFO algorithm.
    """

    n = len(y)
    n_cycles = n // 2
    ix = 0

    for i in range(n_cycles):

        center = 2 * i + 1
        x_center = y[center]

        # =========================================================
        # LEFT SEARCH (find minimum left of center)
        # =========================================================
        idx_min = 2 * i
        x_min = y[idx_min]

        j = i - 1
        while j >= 0 and y[2 * j + 1] <= x_center:
            val = y[2 * j]
            if val < x_min:
                x_min = val
                idx_min = 2 * j
            j -= 1

        # =========================================================
        # RIGHT SEARCH (find minimum right of center)
        # =========================================================
        idx_right = 2 * i + 2
        x_right = y[idx_right]

        # ---------------------------------------------------------
        # Case 1: Left dominates > no need for right scan
        # ---------------------------------------------------------
        if x_min >= x_right:
            if _keep_cycle(x_center, x_min, h, inclusive):
                ind[ix] = idx_min
                ix += 1
                ind[ix] = center
                ix += 1
            continue

        # ---------------------------------------------------------
        # Case 2: scan forward (right side)
        # ---------------------------------------------------------
        j = i + 1
        while j < n_cycles:
            if y[2 * j + 1] >= x_center:
                break

            val = y[2 * j + 2]
            if val <= x_right:
                x_right = val
                idx_right = 2 * j + 2

            j += 1
        else:
            # reached end without break
            if _keep_cycle(x_center, x_min, h, inclusive):
                ind[ix] = idx_min
                ix += 1
                ind[ix] = center
                ix += 1
            continue

        # =========================================================
        # FINAL DECISION
        # =========================================================
        if x_right <= x_min:
            if _keep_cycle(x_center, x_min, h, inclusive):
                ind[ix] = idx_min
                ix += 1
                ind[ix] = center
                ix += 1

        elif _keep_cycle(x_center, x_right, h, inclusive):
            ind[ix] = center
            ix += 1
            ind[ix] = idx_right
            ix += 1

    
    return ix


@njit
def _insertion_sort_dedup(a, n):
    """
    In-place insertion sort optimized for nearly sorted data,
    followed by in-place deduplication.

    Parameters
    ----------
    a : ndarray[int64]
        Input/output array
    n : int
        Number of valid elements in a

    Returns
    -------
    int
        Number of unique sorted elements
    """

    # ------------------------------------------------------------
    # Insertion sort (optimized for small disorder)
    # ------------------------------------------------------------
    for i in range(1, n):
        key = a[i]

        # fast path: already in order
        if key >= a[i - 1]:
            continue

        j = i - 1

        # shift larger elements to the right
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1

        a[j + 1] = key

    # ------------------------------------------------------------
    # Deduplicate in place
    # ------------------------------------------------------------
    if n == 0:
        return 0

    k = 1
    prev = a[0]

    for i in range(1, n):
        if a[i] != prev:
            a[k] = a[i]
            prev = a[i]
            k += 1

    return k

# ============================================================
# Public API
# ============================================================
def findrfc(
    y,
    h,
    mode="inclusive",
    assume_tp=True
):
    """
    Return indices of RFC-filtered turning points.

    Parameters
    ----------
    y : array_like            
        Turning points if assume_tp=True (default),
        otherwise the input signal.
    h : float
        Minimum cycle range retained by the RFC filter.
    mode : {'inclusive', 'strict'}
        Threshold rule:
        - 'inclusive' : keep cycles with range >= h
        - 'strict'    : keep cycles with range > h
    assume_tp : bool
        If True, input is already turning points.

    Returns
    -------
    indices : ndarray[int64]
        If assume_tp=True, indices into the supplied turning-point sequence.
        If assume_tp=False, indices into the original signal.
        Returned indices are sorted in ascending order.

    
    Examples
    --------
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,7*np.pi,250)
    >>> x = np.sin(t) + 0.1*np.sin(50*t)
    >>> ind = wm.findextrema(x)
    >>> ti, tp = t[ind], x[ind]

    >>> ind1 = wm.findrfc(tp, 0.3, mode="inclusive")
    >>> np.allclose(ind1, [0, 9, 32, 53, 74, 95, 116, 137])
    True

    >>> ind2 = wm.findrfc(tp, 0.3, mode="strict")
    >>> np.allclose(ind2, [  0,   9,  32,  53,  74,  95, 116, 137]) 
    True
    
    >>> ind3 = wm.findrfc(x, 0.3, assume_tp=False)
    >>> np.allclose(ind3, [  1,  16,  55,  90, 125, 161, 196, 231])
    True
    >>> np.allclose(x[ind3], tp[ind1])
    True

    >>> import matplotlib.pyplot as plt
    >>> h0 = plt.plot(t, x,'-', label='data')
    >>> h1 = plt.plot(ti, tp,'r.', label='turning points')
    >>> h2 = plt.plot(ti[ind1], tp[ind1], 'ko', label='filtered turning points')
    >>> h3 = plt.legend()

    >>> plt.close('all')

    See also
    --------
    rfcfilter,
    findtp.
    """
    y = np.asarray(y, dtype=np.float64)
    
    if y.ndim != 1:
        raise ValueError("y must be 1D")

    n = len(y)

    if n < 3:
        return np.empty(0, dtype=np.int64)

    # ------------------------------------------------------------
    # Select comparison rule
    # ------------------------------------------------------------
    if mode == "inclusive":
        inclusive = True
    elif mode == "strict":
        inclusive = False
    else:
        raise ValueError("mode must be 'inclusive' or 'strict'")

    # ------------------------------------------------------------
    # Extract turning points if needed
    # ------------------------------------------------------------
    if assume_tp:
        # Basic TP validity check
        if (y[0] < y[1] < y[2]) or (y[0] > y[1] > y[2]):
            warnings.warn(
                "Input does not appear to be a sequence of turning points; returning empty result."
            )
            return np.empty(0, dtype=np.int64)
        #  drop first max so that the cycles start with a minimum
        offset = 1 if y[0] > y[1] else 0

        tp_ind = np.arange(offset, n, dtype=np.int64)
        tp = y[offset:]
    else:
        tp_ind = np.empty(n, dtype=np.int64)
        ntp = _extract_turning_points(y, tp_ind)
        if ntp < 3:
            return np.empty(0, dtype=np.int64)

        #  drop first max so that the cycles start with a minimum
        offset = 1 if y[tp_ind[0]] > y[tp_ind[1]] else 0
        tp_ind = tp_ind[offset:ntp]
        tp = y[tp_ind]
    
    ntp = len(tp_ind)
    if ntp < 3:
        return np.empty(0, dtype=np.int64)

    # Buffer for indices in TP space
    ind_tp = np.empty(len(tp_ind), dtype=np.int64)
    
    m = _findrfc_core(ind_tp, tp, h, inclusive)

    # ------------------------------------------------------------
    # Map back to original indices
    # ------------------------------------------------------------
    tmp = tp_ind[ind_tp[:m]]
    k = _insertion_sort_dedup(tmp, m)
    return tmp[:k]
    # tmp.sort()
    # return tmp


@njit(inline="always")
def _amplitude_products(rA, iA, ixi, jyi):
    rrA = rA[ixi] * rA[jyi]
    iiA = iA[ixi] * iA[jyi]
    riA = rA[ixi] * iA[jyi]
    irA = iA[ixi] * rA[jyi]
    return rrA, iiA, riA, irA


@njit
def _finite_water_disufq(
    rvec,
    ivec,
    rA,
    iA,
    w,
    kw,
    h,
    g,
    nmin,
    nmax,
    m,
):
    """
    Finite-water-depth DISUFQ kernel.

    Uses kfact=2 to exploit spectral symmetry. The mirrored-frequency
    contributions are therefore omitted.
    """
    kfact = 2.0

    for ix in range(nmin - 1, nmax):
        kw1 = kw[ix]
        w1 = w[ix]

        tanh_kh = np.tanh(kw1 * h)

        # Group velocity
        cg = (
            0.5 * g
            * (tanh_kh + kw1 * h * (1.0 - tanh_kh * tanh_kh))
            / w1
        )

        a1 = 0.5 * g * (kw1 / w1) ** 2
        a2 = 0.5 * w1 * w1 / g
        a3 = g * kw1 / (w1 * cg)

        if kw1 * h < 300.0:
            a4 = kw1 / np.sinh(2.0 * kw1 * h)
        else:
            a4 = 0.0

        # Diagonal difference-frequency coefficient
        edij = (
            (a1 - a2 + a3)
            / (1.0 - g * h / (cg * cg))
            - a4
        )

        # Diagonal sum-frequency coefficient
        epij = (
            3.0 * (a1 - a2)
            / (1.0 - a1 / kw1 * np.tanh(2.0 * kw1 * h))
            + 3.0 * a2
            - a1
        )

        # --------------------------------------------------------
        # Diagonal contribution (ix == jy)
        # --------------------------------------------------------
        ixi = ix * m
        iz1 = 2 * ixi

        for i in range(m):
            rrA = rA[ixi] * rA[ixi]
            iiA = iA[ixi] * iA[ixi]
            riA = rA[ixi] * iA[ixi]

            # Sum-frequency contribution
            rvec[iz1] += kfact * (rrA - iiA) * epij
            ivec[iz1] += kfact * 2.0 * riA * epij

            # Difference-frequency contribution
            # contributes only to the mean.
            rvec[i] += 2.0 * (rrA + iiA) * edij

            ixi += 1
            iz1 += 1

        # --------------------------------------------------------
        # Off-diagonal contribution (ix < jy)
        # --------------------------------------------------------
        for jy in range(ix + 1, nmax):
            kw2 = kw[jy]
            w2 = w[jy]

            interaction = g * (kw1 / w1) * (kw2 / w2)

            s2 = (
                0.5 / g
                * (w1 * w1 + w2 * w2 + w1 * w2)
            )

            s3 = (
                0.5 * g
                * (w1 * kw2 * kw2 + w2 * kw1 * kw1)
                / (w1 * w2 * (w1 + w2))
            )

            sden = (
                1.0
                - g * (kw1 + kw2)
                / ((w1 + w2) * (w1 + w2))
                * np.tanh((kw1 + kw2) * h)
            )

            epij = (
                (interaction - s2 + s3) / sden
                + s2
                - 0.5 * interaction
            )

            d2 = (
                0.5 / g
                * (w1 * w1 + w2 * w2 - w1 * w2)
            )

            d3 = (
                -0.5 * g
                * (w1 * kw2 * kw2 - w2 * kw1 * kw1)
                / (w1 * w2 * (w1 - w2))
            )

            dden = (
                1.0
                - g * (kw1 - kw2)
                / ((w1 - w2) * (w1 - w2))
                * np.tanh((kw1 - kw2) * h)
            )

            edij = (
                (interaction - d2 + d3) / dden
                + d2
                - 0.5 * interaction
            )

            ixi = ix * m
            jyi = jy * m

            iz1 = ixi + jyi
            iv1 = jyi - ixi

            for _ in range(m):
                rrA, iiA, riA, irA = _amplitude_products(
                    rA, iA, ixi, jyi
                )

                # Sum-frequency contribution
                rvec[iz1] += kfact * 2.0 * (rrA - iiA) * epij
                ivec[iz1] += kfact * 2.0 * (riA + irA) * epij

                # Difference-frequency contribution
                rvec[iv1] += kfact * 2.0 * (rrA + iiA) * edij
                ivec[iv1] += kfact * 2.0 * (riA - irA) * edij

                ixi += 1
                jyi += 1
                iz1 += 1
                iv1 += 1

@njit
def _deep_water_disufq(
    rvec,
    ivec,
    rA,
    iA,
    kw,
    nmin,
    nmax,
    m,
):
    """
    Deep-water approximation of DISUFQ.

    Uses kfact=2 to exploit spectral symmetry. The mirrored-frequency
    contributions are therefore omitted.
    """
    kfact = 2.0

    for ix in range(nmin - 1, nmax):
        kw1 = kw[ix]

        # --------------------------------------------------------
        # Diagonal contribution (ix == jy)
        # --------------------------------------------------------
        ixi = ix * m
        iz1 = 2 * ixi

        for _ in range(m):
            rrA = rA[ixi] * rA[ixi]
            iiA = iA[ixi] * iA[ixi]
            riA = rA[ixi] * iA[ixi]

            rvec[iz1] += kfact * (rrA - iiA) * kw1
            ivec[iz1] += kfact * 2.0 * riA * kw1

            ixi += 1
            iz1 += 1

        # --------------------------------------------------------
        # Off-diagonal contribution (ix < jy)
        # --------------------------------------------------------
        for jy in range(ix + 1, nmax):
            kw2 = kw[jy]

            epij = 0.5 * (kw1 + kw2)
            edij = -0.5 * (kw2 - kw1)

            ixi = ix * m
            jyi = jy * m

            iz1 = ixi + jyi
            iv1 = jyi - ixi

            for _ in range(m):
                rrA, iiA, riA, irA = _amplitude_products(
                    rA, iA, ixi, jyi
                )

                # Sum-frequency contribution
                rvec[iz1] += kfact * 2.0 * (rrA - iiA) * epij
                ivec[iz1] += kfact * 2.0 * (riA + irA) * epij

                # Difference-frequency contribution
                rvec[iv1] += kfact * 2.0 * (rrA + iiA) * edij
                ivec[iv1] += kfact * 2.0 * (riA - irA) * edij

                ixi += 1
                jyi += 1
                iz1 += 1
                iv1 += 1


def disufq(rA, iA, w, kw, h, g, nmin, nmax):
    """
    Compute second-order sum- and difference-frequency contributions.

    Parameters
    ----------
    rA, iA : ndarray, shape (m, n)
        Real and imaginary Fourier amplitudes.
    w : ndarray, shape (n,)
        Angular frequencies [rad/s].
    kw : ndarray, shape (n,)
        Wavenumbers.
    h : float
        Water depth [m].
    g : float
        Gravitational acceleration.
    nmin, nmax : int
        Frequency-index range used in the summation.

    Returns
    -------
    rvec, ivec : ndarray, shape (m*n,)
        Real and imaginary second-order contributions.

    Notes
    -----
    The result may be transformed to the time domain as

        real(np.fft.fft(rvec + 1j * ivec))

    to obtain the second-order Stokes-wave contribution.

@jit(int64(float64[:], float64[:], float64[:, :]), nopython=True)
def _findrfc3_astm(array_ext, a, array_out):
    """
    rA = np.asarray(rA, dtype=np.float64)
    iA = np.asarray(iA, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    kw = np.asarray(kw, dtype=np.float64)

    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------
    if rA.ndim != 2:
        raise ValueError("rA and iA must be 2D arrays")

    if rA.shape != iA.shape:
        raise ValueError("rA and iA must have the same shape")

    if w.ndim != 1 or kw.ndim != 1:
        raise ValueError("w and kw must be 1D arrays")

    m, n = rA.shape

@jit(int64(float64[:], float64[:], float64[:], float64[:], float64[:, :]),
     nopython=True)
def _findrfc5_astm(array_ext, array_t, a, t, array_out):
    """
    Rain flow with time analysis

    if kw.size != n:
        raise ValueError(f"kw must have length {n}")

    if not (1 <= nmin <= nmax <= n):
        raise ValueError(
            f"Require 1 <= nmin <= nmax <= n, "
            f"got nmin={nmin}, nmax={nmax}, n={n}"
        )

    # ------------------------------------------------------------
    # Allocate output
    # ------------------------------------------------------------
    rvec = np.zeros(m * n, dtype=np.float64)
    ivec = np.zeros(m * n, dtype=np.float64)

    # ------------------------------------------------------------
    # Compute second-order interactions
    # ------------------------------------------------------------
    # WAFO convention:
    # h > 10000 m is treated as effectively infinite depth.
    if h > 10000.0:
        _deep_water_disufq(
            rvec, ivec,
            rA, iA,
            kw,
            nmin, nmax,
            m,
        )
    else:
        _finite_water_disufq(
            rvec, ivec,
            rA, iA,
            w, kw,
            h, g,
            nmin, nmax,
            m,
        )

    return rvec, ivec


if __name__ == '__main__':
    pass
