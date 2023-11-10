"""
Control envelopes in time domain and frequency domain

For a pair of functions g(t) <-> h(f) we use the following
convention for the Fourier Transform:

         / +inf
        |
h(f) =  | g(t) * exp(-2j*pi*f*t) dt
        |
       / -inf

         / +inf
        |
g(t) =  | h(f) * exp(2j*pi*f*t) df
        |
       / -inf

Note that we are working with frequency in GHz, rather than
angular frequency.  Also note that the sign convention is opposite
to what is normally taken in physics.  But this is the convention
used here and in the DAC deconvolution code, so you should use it.
"""

import numpy as np
import copy
from scipy.integrate import quad
from scipy.special import erf
from scipy.signal import windows


class Envelope(object):
    """
    basic Envelope class (ref: labrad envelope.py)
    time unit: ns
    freq unit: GHz
    """

    def __init__(self, timeFunc=None, freqFunc=None, start=None, end=None):
        if timeFunc is None:
            if freqFunc is None:
                raise Exception(
                    'You have to input at least one valid timeFunc/freqFunc.')
            else:
                self.freqFunc = freqFunc
                self.reload_timeFunc()
        else:
            if freqFunc is None:
                self.timeFunc = timeFunc
                self.reload_freqFunc()
            else:
                self.timeFunc = timeFunc
                self.freqFunc = freqFunc
        self.start = start
        self.end = end

    def reload_timeFunc(self):
        """
        reload timeFunc using freqFunc, defined by the integral at the beginning
        """
        freqFunc = copy.deepcopy(self.freqFunc)

        def timeFunc(t):
            try:
                return quad(
                    func=lambda f, t: freqFunc(f) * np.exp(2j * np.pi * f * t),
                    a=-np.inf,
                    b=np.inf,
                    args=(t, ))[0]
            except:
                return np.array([
                    quad(func=lambda f, t: freqFunc(f) * np.exp(2j * np.pi * f
                                                                * t),
                         a=-np.inf,
                         b=np.inf,
                         args=(_t, ))[0] for _t in t
                ])

        self.timeFunc = timeFunc

    def reload_freqFunc(self):
        """
        reload freqFunc using timeFunc, defined by the integral at the beginning
        """
        timeFunc = copy.deepcopy(self.timeFunc)

        def freqFunc(f):
            try:
                return quad(func=lambda t, f: timeFunc(t) * np.exp(-2j * np.pi
                                                                   * f * t),
                            a=-np.inf,
                            b=np.inf,
                            args=(f, ))[0]
            except:
                return np.array([
                    quad(func=lambda t, f: timeFunc(t) * np.exp(-2j * np.pi * f
                                                                * t),
                         a=-np.inf,
                         b=np.inf,
                         args=(_f, ))[0] for _f in f
                ])

        self.freqFunc = freqFunc

    def __call__(self, x, fourier=False):
        """
        self(x)
        """
        if fourier:
            return self.freqFunc(x)
        else:
            return self.timeFunc(x)

    def __add__(self, other):
        """
        self + other
        """
        if isinstance(other, Envelope):
            start, end = timeRange((self, other))

            def timeFunc(t):
                return self.timeFunc(t) + other.timeFunc(t)

            def freqFunc(f):
                return self.freqFunc(f) + other.freqFunc(f)

            return Envelope(timeFunc, freqFunc, start=start, end=end)
        else:
            # if we try to add envelopes with the built in sum() function,
            # the first envelope is added to 0 before adding the rest.  To support
            # this, we add a special case here since adding 0 in time or fourier
            # is equivalent
            if other == 0:
                return self
            raise Exception(
                "Cannot add a constant to hybrid time/fourier envelopes")

    __radd__ = __add__  # other + self

    def __sub__(self, other):
        """
        self - other
        """
        if isinstance(other, Envelope):
            start, end = timeRange((self, other))

            def timeFunc(t):
                return self.timeFunc(t) - other.timeFunc(t)

            def freqFunc(f):
                return self.freqFunc(f) - other.freqFunc(f)

            return Envelope(timeFunc, freqFunc, start=start, end=end)
        else:
            raise Exception(
                "Cannot subtract a constant from hybrid time/fourier envelopes"
            )

    def __rsub__(self, other):
        """
        other - self
        """
        if isinstance(other, Envelope):
            start, end = timeRange((self, other))

            def timeFunc(t):
                return other.timeFunc(t) - self.timeFunc(t)

            def freqFunc(f):
                return other.freqFunc(f) - self.freqFunc(f)

            return Envelope(timeFunc, freqFunc, start=start, end=end)
        else:
            raise Exception(
                "Cannot subtract a constant from hybrid time/fourier envelopes"
            )

    def __mul__(self, other):
        """
        self * other
        """
        if isinstance(other, Envelope):
            raise Exception(
                "Hybrid time/fourier envelopes can only be multiplied by constants"
            )
        else:

            def timeFunc(t):
                return self.timeFunc(t) * other

            def freqFunc(f):
                return self.freqFunc(f) * other

            return Envelope(timeFunc, freqFunc, start=self.start, end=self.end)

    __rmul__ = __mul__  # other * self

    def __div__(self, other):
        """
        self / other
        """
        if isinstance(other, Envelope):
            raise Exception(
                "Hybrid time/fourier envelopes can only be divided by constants"
            )
        else:

            def timeFunc(t):
                return self.timeFunc(t) / other

            def freqFunc(f):
                return self.freqFunc(f) / other

            return Envelope(timeFunc, freqFunc, start=self.start, end=self.end)

    __truediv__ = __div__  # self / other

    def __rdiv__(self, other):
        """
        other / self
        """
        if isinstance(other, Envelope):
            raise Exception(
                "Hybrid time/fourier envelopes can only be divided by constants"
            )
        else:

            def timeFunc(t):
                return other / self.timeFunc(t)

            def freqFunc(f):
                return other / self.freqFunc(f)

            return Envelope(timeFunc, freqFunc, start=self.start, end=self.end)

    __rtruediv__ = __rdiv__  # other / self

    def __neg__(self):
        """
        -self
        """
        return -1 * self

    def __pos__(self):
        """
        +self
        """
        return self

    def copy(self):
        return copy.deepcopy(self)


_zero = lambda x: np.zeros_like(x, dtype=float)

# empty envelope
NOTHING = Envelope(_zero, _zero, start=None, end=None)


def gaussian(t0, w, amp=1.0, phase=0.0, df=0.0):
    """A gaussian pulse with specified center and full-width at half max."""
    sigma = w / np.sqrt(8 * np.log(2))  # convert fwhm to std. deviation

    def timeFunc(t):
        return amp * np.exp(-(t - t0)**2 / (2 * sigma**2) - 2j * np.pi * df *
                            (t - t0) + 1j * phase)

    sigmaf = 1 / (2 * np.pi * sigma)  # width in frequency space
    ampf = amp * np.sqrt(2 * np.pi * sigma**2)  # amp in frequency space

    def freqFunc(f):
        return ampf * np.exp(-(f + df)**2 / (2 * sigmaf**2) -
                             2j * np.pi * f * t0 + 1j * phase)
        # return ampf * np.exp(-(f+df)**2/(2*sigmaf**2) - 2j*np.pi*f*t0 + 1j*phase)

    return Envelope(timeFunc, freqFunc, start=t0 - w, end=t0 + w)


def rect(t0, len, amp, overshoot=0.0, overshoot_w=1.0):
    """A rectangular pulse with sharp turn on and turn off.

    Note that the overshoot_w parameter, which defines the FWHM of the gaussian overshoot peaks
    is only used when evaluating the envelope in the time domain.  In the fourier domain, as is
    used in the dataking code which uploads sequences to the boards, the overshoots are delta
    functions.
    """
    tmin = min(t0, t0 + len)
    tmax = max(t0, t0 + len)
    tmid = (tmin + tmax) / 2.0
    overshoot *= np.sign(amp)  # overshoot will be zero if amp is zero

    # to add overshoots in time, we create an envelope with two gaussians
    if overshoot:
        o_w = overshoot_w
        o_amp = 2 * np.sqrt(np.log(2) / np.pi) / o_w  # total area == 1
        o_env = gaussian(tmin, o_w, o_amp) + gaussian(tmax, o_w, o_amp)
    else:
        o_env = NOTHING

    def timeFunc(t):
        return (amp * (t >= tmin) * (t < tmax) + overshoot * o_env(t))

    # to add overshoots in frequency, use delta funcs (smoothed by filters)
    def freqFunc(f):
        return (
            amp * abs(len) * np.sinc(len * f) * np.exp(-2j * np.pi * f * tmid)
            + overshoot *
            (np.exp(-2j * np.pi * f * tmin) + np.exp(-2j * np.pi * f * tmax)))

    return Envelope(timeFunc, freqFunc, start=tmin, end=tmax)


def flattop(t0, len, amp, w=5, phase=0., overshoot=0.0, overshoot_w=1.0):
    """A rectangular pulse convolved with a gaussian to have smooth rise and fall."""
    tmin = min(t0, t0 + len)
    tmax = max(t0, t0 + len)

    overshoot *= np.sign(amp)  # overshoot will be zero if amp is zero

    # to add overshoots in time, we create an envelope with two gaussians
    a = 2 * np.sqrt(np.log(2)) / w
    if overshoot:
        o_w = overshoot_w
        o_amp = 2 * np.sqrt(np.log(2) / np.pi) / o_w  # total area == 1
        o_env = gaussian(tmin, o_w, o_amp) + gaussian(tmax, o_w, o_amp)
    else:
        o_env = NOTHING

    amp *= np.exp(1j * phase)

    def timeFunc(t):
        return (amp * (erf(a * (tmax - t)) - erf(a * (tmin - t))) / 2.0 +
                overshoot * o_env(t))

    # to add overshoots in frequency, use delta funcs (smoothed by filters)
    rect_env = rect(t0, len, 1.0)
    kernel = gaussian(0, w, 2 * np.sqrt(np.log(2) / np.pi) / w)  # area = 1

    def freqFunc(f):
        return (
            amp * rect_env(f, fourier=True) * kernel(f, fourier=True)
            +  # convolve with gaussian kernel
            overshoot *
            (np.exp(-2j * np.pi * f * tmin) + np.exp(-2j * np.pi * f * tmax)))

    return Envelope(timeFunc, freqFunc, start=tmin, end=tmax)


class Transformer(object):
    """
    Linear Transformer class.
    Takes as inputs a series of envelopes, and outputs a series of envelopes by matrix product
    """

    def __init__(self, transfer_matrix=None):
        """
        transfer_matrix = 
        [[H11(f), H12(f), ..., H1n(f)],
         [H21(f), H22(f), ..., H2n(f)],
         ...,
         [Hn1(f), Hn2(f), ..., Hnn(f)]]
        """
        self.transfer_matrix = transfer_matrix

    def __call__(self, envelopes):
        """
        out[i] = sum_j(Hij * in[j])
        """
        transfer_matrix = self.transfer_matrix
        if transfer_matrix is None:
            return envelopes
        else:
            assert len(transfer_matrix) == len(
                transfer_matrix[0]) == len(envelopes)
            env_num = len(envelopes)
            freqFuncs = [copy.deepcopy(env.freqFunc) for env in envelopes]

            def get_transformed_freqFunc(i):

                def fun(f):
                    output = 0
                    for j in range(env_num):
                        output += transfer_matrix[i][j](f) * freqFuncs[j](f)
                    return output

                return fun

            transformed_freqFuncs = [
                get_transformed_freqFunc(i) for i in range(env_num)
            ]
            transformed_envelopes = [
                Envelope(freqFunc=transformed_freqFuncs[i],
                         start=envelopes[i].start,
                         end=envelopes[i].end) for i in range(env_num)
            ]
            return transformed_envelopes


def ucsb_transformer(transfer_matrix_params=None):
    """
    ucsb transfer function ansatz
    each matrix element should be either a number or a dict
    with a form as {'coeff': coeff, 'amps': [...], 'freqs': [...]}
    """
    if transfer_matrix_params is None:
        return Transformer(transfer_matrix=None)
    else:
        assert len(transfer_matrix_params) == len(transfer_matrix_params[0])
        env_num = len(transfer_matrix_params)

        def get_transfunc_from_dict(coeff=1.0, amps=[], freqs=[]):

            def transfunc(f):
                output = 1
                for i in range(len(amps)):
                    output += (1j * amps[i] * f) / (1j * f + freqs[i])
                return coeff * output

            return transfunc

        def get_transfunc_from_value(value=0.0):

            def transfunc(f):
                return value * np.ones_like(f)

            return transfunc

        transfer_matrix = []
        for i in range(env_num):
            transfer_matrix.append([])
            for j in range(env_num):
                try:
                    transfer_matrix[i].append(
                        get_transfunc_from_dict(
                            **transfer_matrix_params[i][j]))
                except:
                    transfer_matrix[i].append(
                        get_transfunc_from_value(
                            **{'value': transfer_matrix_params[i][j]}))
        return Transformer(transfer_matrix=transfer_matrix)


def timeRange(envelopes):
    """Calculate the earliest start and latest end of a list of envelopes.

    Returns a tuple (start, end) giving the time range.  Note that one or
    both of start and end may be None if the envelopes do not specify limits.
    """
    starts = [env.start for env in envelopes if env.start is not None]
    start = min(starts) if len(starts) else None
    ends = [env.end for env in envelopes if env.end is not None]
    end = max(ends) if len(ends) else None
    return start, end
