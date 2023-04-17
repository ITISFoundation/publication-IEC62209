import os
import sys
import numpy as np
import pandas as pd
import skgstat as skg
import scipy.stats as stat
import scipy.spatial.distance as dist
from sampler import Sampler
from sample import Sample
from model import Model, make_variogram, variogram_param


# ============================================================================
# segmentation functions

# filter out all points that don't have at least probability prob to be above thresh
def _filter_space(x, z, e, prob, mpe):
    mpe = abs(mpe)
    lt = -mpe
    ut = mpe
    phi1_p = stat.norm.ppf(prob)
    pts = np.c_[x, z, e]
    l = pts[phi1_p <= (lt - pts[:,-2]) / pts[:,-1]]
    u = pts[phi1_p <= -(ut - pts[:,-2]) / pts[:,-1]]
    return l, u

# returns the probabilities to pas lt, respectively ut
def _pass_prob(z, e, mpe):
    z = np.asarray(z)
    e = np.asarray(e)
    mpe = abs(mpe)
    lt = np.asarray(-mpe)
    ut = np.asarray(mpe)
    passl = stat.norm.cdf((lt - z) / e)
    passu = 1. - stat.norm.cdf((ut - z) / e)
    return passl, passu

# dipoles:
# mpe10g = 1.6 for f < 750 MHz
# mpe10g = 1.5 for 750 <= f < 3700 MHz
# mpe10g = 1.6 for 3700 <= f < 6000 MHz
# mpe1g = 1.6 for f < 750 MHz
# mpe1g = 1.5 for 750 <= f < 3700 MHz
# mpe1g = 1.6 for 3700 <= f < 5600 MHz
# mpe1g = 1.7 for 5600 <= f < 6000 MHz
# VPIFAs:
# mpe10g = 1.6 for f = 750, 835, 3700 MHz
# mpe10g = 1.5 for f = 1950 MHz
# mpe1g = 1.8 for f = 750, 835 MHz
# mpe1g = 1.6 for f = 1950, 3700 MHz
def _mpe_map(row, ant_name, freq_name, mass='10g'):
    freq = row[freq_name]
    dipole = row[ant_name].startswith('D')
    if dipole: 
        if mass == '10g': 
            if 750 <= freq & freq < 3700:
                return 1.5
        else:
            if 750 <= freq & freq < 3700:
                return 1.5
            if 5600 <= freq & freq < 6000:
                return 1.7
    else:
        if mass == '10g': 
            if freq == 1950:
                return 1.5
        else:
            if freq in [750, 835]:
                return 1.8
    return 1.6


# ============================================================================
# with NoPrint(): is used when calling skgstat.OrdinaryKriging.transform to 
# prevent attrocious warnings 

class NoPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# ============================================================================
# Kriging class: holds variogram as a skg.Variogram and performs kriging.

class Kriging:
    def __init__(self, model, kriging_kwargs={}):
        xanchor = model.idata()
        zanchor = model.zdata()
        self.kriging = skg.OrdinaryKriging(model.variogram, coordinates=xanchor, values=zanchor, **kriging_kwargs)
        self.model = model

    def __call__(self, x, error=True, iso=False):
        """Applies the kriging to x and returns the predicted values with the predicted errors."""
        if isinstance(x, Sample):
            x = x.xdata()
        if not iso:
            x = self.model.iota(x)
        y = None
        with NoPrint():
            y = np.asarray(self.kriging.transform(x)) + self.model.calibration[0]
        if error:
            return y, self.error()       
        else:
            return y

    def error(self):
        """The error values of the last kriging call."""
        err = np.sqrt(np.abs(self.kriging.sigma)) * (1 + np.abs(self.model.variogram.nrmse))
        return err * self.model.calibration[1] 

    def recalibrate(self, loc=0, scale=1):
        """Recalibrate the model internal to self."""
        self.model.calibration = (loc, abs(scale))
        
    def online(self, start, end, num=50):
        """Applies kriging on segment linspace(start, end, num)."""
        x = np.linspace(start, end, num=num)
        y, e = self(x)
        return x, y, e

    def onplane(self, start, end1, end2, num=(25, 25)):
        """Applies kriging on the 2d-plane defined by basis (end1-start, end2-start)."""
        n = start.size
        n1 = num[0]
        n2 = num[1]
        u1, u2 = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n2))
        q1 = np.expand_dims(u1, axis=2)
        q2 = np.expand_dims(u2, axis=2)
        x = q1 * (end1-start) + q2 * (end2-start) + start
        ps = x.reshape(n1*n2, n)
        y, e = self(ps)
        y = y.reshape(n1, n2)
        e = e.reshape(n1, n2)
        return x, y, e

    def plot_line(self, start, end, num=50, err=1.96, func=None, ax=None):
        """Plot the kriging values on linspace(start, end, num)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 9))
        x, y, e = self.online(start, end, num=num)
        r = np.linspace(0, 1, num)
        ax.set_xticks(np.arange(2), [str(start), str(end)])
        ax.plot(r, y, 'b-')
        ax.fill(np.concatenate([r, r[::-1]]),
                np.concatenate([(y - err * e), (y + err * e)[::-1]]),
                alpha=.4, fc='b', ec='None')
        if func is not None:
            ax.plot(r, func(x), 'r:')
        return ax.figure

    def plot_plane(self, start, end1, end2, num=(25, 25), func=None, ax=None):
        """Plot the kriging values on the 2d-plane defined by basis (end1-start, end2-start)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 9))
        fig = ax.figure
        ax.remove()
        ax = fig.add_subplot(111, projection='3d')
        n1 = num[0]
        n2 = num[1]
        u1, u2 = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n2))
        ax.set_xticks(np.arange(2), [str(start), str(end1)])
        ax.set_yticks(np.arange(2), [str(start), str(end2)])
        x, y, e = self.onplane(start, end1, end2, num=num)
        ax.plot_surface(u1, u2, y)
        return ax.figure

    def residuals(self, sample):
        """Returns the normalized residuals, the residuals and errors induced by 
        kriging on the sample argument."""
        if not sample.isrealvalued():
            raise ValueError('sample is not real valued')
        xdata = sample.xdata()
        zdata = sample.zdata().ravel()
        zpred, zerr = self(xdata)
        resid = np.asarray(zpred - zdata)
        nresid = resid / zerr
        return nresid, resid, zerr

    def _search(self, sample, mpe, sens, niter):
        if not sample.isrealvalued():
            raise ValueError('sample is not real valued')
        # thresholds
        lt = None
        ut = None
        if isinstance(mpe, (list, tuple)):
            lt = min(mpe)
            ut = max(mpe)
        else:
            mpe = abs(mpe)
            lt = -mpe
            ut = mpe
        mt = (ut + lt) / 2
        x = sample.xdata().copy()
        f = lambda l: self(l, error=False, iso=True)
        r = self.model.range()
        s = self.model.sill()
        p = sens
        delta = lambda l: self.model.delta_gaussian(l)
        # limit search to sample range 
        mn = np.amin(x, axis=0)
        mx = np.amax(x, axis=0)
        # sensitivity
        sens = 0.5 * min(max(sens, 0), 1)
        # dimension
        n = x.shape[1]
        zos = np.zeros(n)
        # move points
        z = f(x)
        for j in range(0, niter):
            osf = 1. / (j+1)
            isf = 0.5 * osf
            for i, (xi, zi) in enumerate(zip(x, z)):
                # push upward
                if zi > mt:
                    d = isf * delta(zi-ut) if zi > ut else osf * delta(ut-zi)
                    xis = np.tile(xi, (2*n+1, 1)) + np.concatenate((np.zeros((1,n)), np.diagflat(zos+d), np.diagflat(zos-d)))
                    xis = xis[np.all((xis <= mx) & (xis >= mn), axis=1)]
                    zis = f(xis)
                    dis = np.amin(dist.cdist(np.delete(x, i, axis=0), xis, 'euclidean'), axis=0)
                    k = np.argmax((zis-mt) * (dis**sens))
                    x[i] = xis[k]
                    z[i] = zis[k]
                # push downward
                else:
                    d = isf * delta(lt-zi) if zi < lt else osf * delta(zi-lt)
                    xis = np.tile(xi, (2*n+1, 1)) + np.concatenate((np.zeros((1,n)), np.diagflat(zos+d), np.diagflat(zos-d)))
                    xis = xis[np.all((xis <= mx) & (xis >= mn), axis=1)]
                    zis = f(xis)
                    dis = np.amin(dist.cdist(np.delete(x, i, axis=0), xis, 'euclidean'), axis=0)
                    k = np.argmax((mt-zis) * (dis**sens))
                    x[i] = xis[k]
                    z[i] = zis[k]
        # return x, z
        df = pd.DataFrame(x, columns=sample.xvar)
        df[sample.zvar[0]] = z
        return Sample(df, xvar=sample.xvar, zvar=sample.zvar)

    def search(self, sample, mpe=1.5, sens=0.1, niter=8, snap=True):
        """Explore the data space and returns a sample of critical regions with 
        associated probabilities to pass one of the mpe thresholds."""
        iota = self.model.iota
        iotai = iota.inv()
        isample = iota(sample)
        osample = iotai(self._search(isample, mpe=mpe, sens=sens, niter=niter))
        if snap:
            osample = Sampler().snap(osample)
        return osample

    def pass_prob(self, sample, mpe=1.5):
        """Returns the probabilities of each element of sample to pass mpe thresholds."""
        if np.isscalar(mpe):
            mpe = [mpe] * sample.zshape()[0]
        mpe = np.asarray(mpe)
        # last kriging round
        z, e = self(sample)
        # probabilities to pass thresholds
        passl, passu = _pass_prob(z, e, mpe)
        passa = np.asarray(passl) + np.asarray(passu)
        return pd.DataFrame({'zval':z, 'err':e, 'mpe':mpe, 'passl':passl, 'passu':passu, 'pass':passa})

    def add_prob(self, sample, mpe=1.5, inplace=False):
        """Adds to sample the probabilities to pass the mpe thresholds."""
        if not sample.isrealvalued():
            raise ValueError('sample must be real valued')
        df = self.pass_prob(sample, mpe=mpe)
        zvar = sample.zvar[0]
        df.rename(columns={'zval':zvar}, inplace=True)
        sample.data.drop(zvar, axis=1, inplace=True)
        retsample = None
        if inplace:
            sample.add_df(df)
            retsample = sample
        else:
            sample2 = sample.copy()
            sample2.add_df(df)
            retsample = sample2
        # sort rows
        retsample.data.sort_values(by='pass', ascending=False, inplace=True)
        retsample = retsample.data.reset_index(drop=True)
        return retsample





