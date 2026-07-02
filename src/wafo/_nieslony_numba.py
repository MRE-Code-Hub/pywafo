"""
Copyright (c) 2003, Adam Niesłony
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


This file contains a Numba/Python translation of Adam Niesłony's
rainflow counting algorithm. The original copyright and license
have been retained.
"""
import numpy as np
from numba import njit

@njit
def _rfc_cycles_no_time(y, work, out):
    """
    ASTM rainflow counting (no time).

    Parameters
    ----------
    y : ndarray
        Turning points
    work : ndarray
        Working buffer
    out : ndarray
        Output (n, 3)

    Returns
    -------
    int
        Number of cycles written to out.
    """
    n = len(y)
    j = -1
    po = 0

    for i in range(n):
        j += 1
        work[j] = y[i]

        while j >= 2 and abs(work[j - 1] - work[j - 2]) <= abs(work[j] - work[j - 1]):

            ampl = abs(work[j - 1] - work[j - 2]) / 2.0
            mean = (work[j - 1] + work[j - 2]) / 2.0

            if j == 2:
                work[0] = work[1]
                work[1] = work[2]
                j = 1

                if ampl > 0:
                    out[po, 0] = ampl
                    out[po, 1] = mean
                    out[po, 2] = 0.5
                    po += 1
            else:
                work[j - 2] = work[j]
                j -= 2

                if ampl > 0:
                    out[po, 0] = ampl
                    out[po, 1] = mean
                    out[po, 2] = 1.0
                    po += 1

    # tail cycles
    for i in range(j):
        ampl = abs(work[i] - work[i + 1]) / 2.0
        mean = (work[i] + work[i + 1]) / 2.0

        if ampl > 0:
            out[po, 0] = ampl
            out[po, 1] = mean
            out[po, 2] = 0.5
            po += 1

    return po


@njit
def _rfc_cycles_with_time(y, t, work_y, work_t, out):
    """
    ASTM rainflow counting with time.

    Returns
    -------
    int
        Number of cycles written to out.

    """
    n = len(y)
    j = -1
    po = 0

    for i in range(n):
        j += 1
        work_y[j] = y[i]
        work_t[j] = t[i]

        while j >= 2 and abs(work_y[j - 1] - work_y[j - 2]) <= abs(work_y[j] - work_y[j - 1]):

            ampl = abs(work_y[j - 1] - work_y[j - 2]) / 2.0
            mean = (work_y[j - 1] + work_y[j - 2]) / 2.0
            period = 2.0 * (work_t[j - 1] - work_t[j - 2])
            t0 = work_t[j - 2]

            if j == 2:
                work_y[0] = work_y[1]
                work_y[1] = work_y[2]
                work_t[0] = work_t[1]
                work_t[1] = work_t[2]
                j = 1

                if ampl > 0:
                    out[po, 0] = ampl
                    out[po, 1] = mean
                    out[po, 2] = 0.5
                    out[po, 3] = t0
                    out[po, 4] = period
                    po += 1
            else:
                work_y[j - 2] = work_y[j]
                work_t[j - 2] = work_t[j]
                j -= 2

                if ampl > 0:
                    out[po, 0] = ampl
                    out[po, 1] = mean
                    out[po, 2] = 1.0
                    out[po, 3] = t0
                    out[po, 4] = period
                    po += 1

    # tail cycles
    for i in range(j):
        ampl = abs(work_y[i] - work_y[i + 1]) / 2.0
        mean = (work_y[i] + work_y[i + 1]) / 2.0
        period = 2.0 * (work_t[i + 1] - work_t[i])
        t0 = work_t[i]

        if ampl > 0:
            out[po, 0] = ampl
            out[po, 1] = mean
            out[po, 2] = 0.5
            out[po, 3] = t0
            out[po, 4] = period
            po += 1

    return po


def findrfc_astm(tp, t=None):
    """
    Rainflow cycle counting (ASTM standard).

    Parameters
    ----------
    tp : array_like
        Sequence of turning points, alternating between minima and maxima.
    t : array_like or scalar, optional
        Time vector with the same length as tp, or a scalar sampling interval.
        If a scalar is provided, a uniform time vector is generated.

    Returns
    -------
    cycles : ndarray
        One row per counted rainflow cycle.
        If t is None: shape (n, 3)
            cycles[:, 0] = cycle amplitude (= range / 2)
            cycles[:, 1] = mean
            cycles[:, 2] = cycle weight (0.5 half cycle, 1.0 full cycle)

        If t is not None: shape (n, 5)
            cycles[:, 0] = cycle amplitude (= range / 2)
            cycles[:, 1] = mean
            cycles[:, 2] = cycle weight (0.5 half cycle, 1.0 full cycle)
            cycles[:, 3] = start time
            cycles[:, 4] = cycle period (per ASTM/Nieslony definition)


    Notes
    -----
    This is a translation of Adam Nieslony's MATLAB/C MEX implementation 
    of the ASTM rainflow counting algorithm.

    Cycles are returned in the order generated by the ASTM algorithm.

    Improvements over the original:
        * No 16384-point limit
        * Numba JIT instead of MEX dependency
        * Cleaner output format (n,3) / (n,5)
        * Safer memory management
        * Easier testing and maintenance

    Examples
    --------
    >>> import numpy as np
    >>> tp = np.array([0, 5, -1, 4, -2, 3])
    >>> cycles = findrfc_astm(tp)
    >>> np.allclose(cycles, [[2.5, 2.5, 0.5],
    ...                      [2.5, 1.5, 1. ],
    ...                      [3.5, 1.5, 0.5],
    ...                      [2.5, 0.5, 0.5]])
    True

    The returned cycles represent:

        * one half cycle with amplitude 2.5 and mean 2.5
        * one full cycle with amplitude 2.5 and mean 1.5
        * one half cycle with amplitude 3.5 and mean 1.5
        * one half cycle with amplitude 2.5 and mean 0.5

    References
    ----------    
    Nieslony, A., "Rainflow Counting Algorithm",
    MATLAB Central File Exchange #3026.
    https://se.mathworks.com/matlabcentral/fileexchange/3026-rainflow-counting-algorithm
    (Retrieved July 2, 2026)

    """
    tp = np.asarray(tp, dtype=np.float64).ravel()
    n = tp.size
    
    if n < 3:
        ncol = 3 if t is None else 5
        return np.empty((0, ncol), dtype=np.float64)

    if t is None:
        out = np.empty((n, 3), dtype=np.float64)
        work = np.empty(n, dtype=np.float64)

        m = _rfc_cycles_no_time(tp, work, out)
        return out[:m]

    # With time
    t = np.asarray(t, dtype=np.float64)
    
    if t.size == 1:
        t = np.arange(n, dtype=np.float64) * float(t.flat[0])
    else:
        t = t.ravel()
        if t.size != n:
            raise ValueError("t must have same length as tp")

    out = np.empty((n, 5), dtype=np.float64)
    work_y = np.empty(n, dtype=np.float64)
    work_t = np.empty(n, dtype=np.float64)

    m = _rfc_cycles_with_time(tp, t, work_y, work_t, out)
    return out[:m]


if __name__ == '__main__':
    from utilities.testing import test_docstrings
    test_docstrings(__file__)
