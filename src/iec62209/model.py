import json as js

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from skgstat import Variogram

from .iota import Iota
from .sample import Sample
from .sampler import latin_cube

# ============================================================================
# functions

def directional_metric(u, v, d, angle=30):
    """The euclidean metric along direction d."""
    l = np.abs(np.cos(np.deg2rad(angle))) * np.linalg.norm(d)
    w = v - u
    nw = np.linalg.norm(w)
    if (l * nw > np.abs(np.dot(d, w))):
        return -np.inf
    return nw

def make_variogram(x, z, vkwargs={}):
    """Builds an isotropic Variogram from x and z."""
    vg = None
    vkwargs['maxlag'] = 0.75
    if ('range' in vkwargs) and ('sill' in vkwargs):
        r = vkwargs['range']
        s = vkwargs['sill']
        n = vkwargs['nugget']
        vkwargs.pop('fit_method', None)
        vkwargs.pop('range', None)
        vkwargs.pop('sill', None)
        vkwargs.pop('nugget', None)
        vg = Variogram(x, z, **vkwargs)
        vg.fit(method='manual', range=r, sill=s, nugget=n)
    else:
        vg = Variogram(x, z, **vkwargs)
    return vg

def make_directional_variograms(x, z, direc, angle=45, vkwargs={}):
    """Builds a list of directional Variograms from x and z that are along direction direc."""
    direc = np.asarray(direc)
    vkwargs['maxlag'] = 0.75
    if direc.ndim <= 1:
        return Variogram(x, z, dist_func=lambda u, v: directional_metric(u, v, direc, angle), **vkwargs)
    else:
        vgs = []
        for d in direc:
            vg = Variogram(x, z, dist_func=lambda u, v: directional_metric(u, v, d, angle), **vkwargs)
            vgs.append(vg)
        return vgs

def variogram_param(variogram):
    """Returns the parameters of the variogram."""
    params = variogram.describe(flat=True)
    tokeep = ('model','estimator','dist_func','bin_func','fit_method','n_lags','maxlag','effective_range', 'sill', 'nugget')
    params = {k:params[k] for k in tokeep if k in params}
    params = {'range' if k == 'effective_range' else k:v for k,v in params.items()}
    return params

def ranges(variograms):
    """Returns the range of the variograms."""
    vgs = np.atleast_1d(variograms)
    ranges = []
    for vg in vgs:
        params = vg.parameters
        ranges.append(params[0])
    return ranges

def sills(variograms):
    """Returns the sills of the variograms."""
    vgs = np.atleast_1d(variograms)
    sills = []
    for vg in vgs:
        params = vg.parameters
        sills.append(params[1])
    return sills

# delta measure on l, with sensitivity p, based on gaussian semivariogram with range r and sill s
def delta_gaussian(l, p, r, s, n=0.):
    """The delta function associated to the gaussian variogram model."""
    l = np.abs(l)
    phi1_p = np.abs(stats.norm.ppf(p))
    if (l >= np.sqrt(2*s) * phi1_p):
        return r
    if (l <= np.sqrt(2*n) * phi1_p):
        return 0.
    l1 = (l / phi1_p)**2 / 2
    return np.sqrt(np.abs(np.log(1-l1/s))) * r / 2


# ============================================================================

class Model:
    """
    A class that holds the model data.

    Holds sample data as a Sample object, variogram as a Variogram, iota as a
    function applicable on sample. Note: a model.sample has a unique zvar.

    """
    def __init__(self, sample, metadata=None, iota=None, variogram=None, iqrfactor=5, calibration=(0., 1.), rescale=True, dirvg_kwargs={}, isovg_kwargs={}):
        if not sample.isrealvalued():
            raise ValueError('sample is not real valued')
        # metadata must be a json object
        self.metadata = metadata
        self.sample = sample
        self.iota = iota
        self.variogram = variogram
        self.dirvariograms = None
        self.calibration = calibration
        if iota is None:
            # build prescaled iota
            self.prescale()
        if variogram is None:
            # build directional variograms
            dirvgs = self.directional_variograms(iqrfactor=iqrfactor, vkwargs=dirvg_kwargs)
            self.dirvariograms = dirvgs
            # rescale iota
            if rescale:
                rgs = ranges(dirvgs)
                rgs = [r if r >= 0.1 else 1 for r in rgs]
                xscale = np.asarray(rgs)
                self.iota = self.iota.compose(Iota(np.diag(xscale)).inv())
            # set isotropic variogram with rescaled iota
            self.variogram = self.isotropic_variogram(iqrfactor=iqrfactor, vkwargs=isovg_kwargs)

    def param(self):
        """Returns the variogram parameters."""
        return variogram_param(self.variogram)

    def range(self):
        """Returns the variogram range."""
        return self.variogram.parameters[0]

    def sill(self):
        """Returns the variogram sill."""
        return self.variogram.parameters[1]

    def rmse(self):
        """Returns the variogram root mean square error."""
        return self.variogram.rmse

    def nrmse(self):
        """Returns the variogram normalized root mean square error."""
        return self.variogram.nrmse

    def iota(self, sample=None):
        """Applies self.iota to either self.sample or sample."""
        if sample is None:
            sample = self.sample
        return self.iota(sample.xdata())

    def xdata(self):
        """Returns all x-variables data as a numpy array."""
        return self.sample.xdata()

    def idata(self):
        """Returns iota applied to x-variables data as a numpy array."""
        return self.iota(self.sample.xdata())

    def zdata(self):
        """Returns all z-variables data as a list."""
        # there is only one zvar per model
        return self.sample.zdata().ravel()

    def recalibrate(self, loc=0, scale=1):
        """Recalibrates self."""
        self.model.calibration = (loc, abs(scale))

    def prescale(self):
        """
        Prescales self based on mean and std of the initial sample values.

        This step increases the accuracy of the directional metrics used to compute
        directional variograms.
        """
        df = self.sample.data
        xvar = self.sample.xvar
        xloc = []
        xscale = []
        for var in xvar:
            grp = df[var].to_numpy()
            xscale.append(np.std(grp))
            xloc.append(np.mean(grp))
        sigma = np.diag(np.asarray(xscale))
        mu = np.asarray(xloc)
        self.iota = Iota(sigma, mu).inv()

    def directional_variograms(self, iqrfactor=None, vkwargs={}):
        """
        Returns a list of the directional variogram along each x-dimension.

        A not None iqrfactor will filter out outliers before building the variograms.
        """
        x = None
        z = None
        if iqrfactor is not None and iqrfactor > 0:
            insample = self.sample.inliers_on(self.sample.zvar[0], iqrfactor=iqrfactor)
            x = self.iota(insample.xdata())
            z = insample.zdata().ravel()
        else:
            x = self.idata()
            z = self.zdata()
        n = x.shape[1]
        dirs = list(map(tuple, np.eye(n)))
        return make_directional_variograms(x, z, dirs, vkwargs=vkwargs)

    # assumes self.iota is valid and correct
    def isotropic_variogram(self, iqrfactor=None, vkwargs={}):
        """
        Returns the isotropic variogram after applying iota to x-data.

        A not None iqrfactor will filter out outliers before building the variograms.
        """
        x = None
        z = None
        if iqrfactor is not None and iqrfactor > 0:
            insample = self.sample.inliers_on(self.sample.zvar[0], iqrfactor=iqrfactor)
            x = self.iota(insample.xdata())
            z = insample.zdata().ravel()
        else:
            x = self.idata()
            z = self.zdata()
        return make_variogram(x, z, vkwargs=vkwargs)

    def variogram_errors(self):
        """Returns the bins and errors between the empirical and theoretical variograms."""
        vg = self.variogram
        bins = vg.bins
        exp = vg.experimental
        theo = np.array([vg.fitted_model(h) for h in bins])
        vgerr = exp - theo
        return bins, vgerr

    def plot_variogram(self, ax=None, show=False):
        """Plots the isotropic variogram induced by self."""
        vg = self.variogram
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(15, 12))
            plt.subplots_adjust(left=0.07, right=0.95, bottom=0.05, top=0.9, wspace=0.2, hspace=0.2)
            fig.suptitle('isotropic semivariogram', fontsize=16)
        vg_range, vg_sill, vg_nugget  = vg.parameters
        hist = False
        ax1 = None
        if isinstance(ax, (list, tuple, np.ndarray)) and (len(ax) > 1):
            hist = True
        vg.plot(ax, show=False, hist=hist)
        if hist:
            ax1 = ax[1]
            ax = ax[0]
            pat = ax1.patches
            for p in pat:
                p.set_color('blue')
        ax.set_xlabel('lag (range = ' + f'{vg_range:.6f}' + ')')
        ax.set_ylabel('semivar (sill = ' + f'{vg_sill:.6f}' + ')')
        ax.set_ylim(ymin=0.)
        ax.get_lines()[0].set_color('blue')
        ax.get_lines()[1].set_color('red')
        if hist:
            ax.get_shared_x_axes().join(ax, ax1)
            ax1.tick_params(labelbottom=True)
            ax1.set_xlabel('lag')
            ax1.set_ylabel('count')
        if show:
            plt.show()
        return ax.figure

    def plot_directional_variograms(self, show=False):
        """Plots the directional variograms along each dimension induced by self."""
        vgs = self.dirvariograms
        if vgs is None:
            return None
        fig, axs = plt.subplots(4, 2, figsize=(15, 12))
        plt.subplots_adjust(left=0.07, right=0.95, bottom=0.05, top=0.9, wspace=0.2, hspace=0.4)
        fig.suptitle('directional semivariograms', fontsize=16)
        for i, v in enumerate(vgs):
            var = self.sample.xvar[i]
            i1 = i // 2
            i2 = i % 2
            ax = axs[i1, i2]
            v.plot(ax, show=False, hist=False)
            ax.set_title(f'along {var}')
            ax.set_xlabel('lag')
            ax.set_ylabel('semivar')
        if show:
            plt.show()
        return axs[0, 0].figure

    def delta_gaussian(self, l, p=0.1):
        """The delta function induced by self and the gaussian model."""
        return delta_gaussian(l, p, self.range(), self.sill())

    def initial_search_sample(self, mpe=1.5, minsize=10, maxsize=1000):
        """Computes and returns the initial sample used to start the exploration algorithm."""
        mpe = np.abs(mpe)
        x = self.idata()
        z = self.zdata()
        l = np.minimum(np.abs(z-mpe), np.abs(z+mpe))
        mins = np.amin(x, axis=0)
        maxs = np.amax(x, axis=0)
        ptp = np.ptp(x, axis=0)
        keys = self.sample.xvar
        vals = np.column_stack((mins, maxs)).tolist()
        dom = dict(zip(keys, vals))
        lbar = max(np.mean(l), mpe / 100.)
        nu = int(np.prod(1 + np.ceil(ptp / self.delta_gaussian(lbar))))
        nu = max(min(nu, maxsize), minsize)
        isample = Sample(latin_cube(dom, nu), xvar=self.sample.xvar)
        isample.set_zvar(self.sample.zvar[0])
        iotai = self.iota.inv()
        return iotai(isample)

    def to_json(self):
        """Returns a json object (a dict) representation of self."""
        return { 'metadata':copy.deepcopy(self.metadata()),
            'sample':self.sample.to_json(),
            'iota':self.iota.to_json(),
            'vgx': self.variogram.coordinates.tolist(),
            'vgz': self.variogram.values.tolist(),
            'vgparam': self.param(),
            'calibration': list(self.calibration) }

    @classmethod
    def from_json(cls, json):
        """Reconstitutes a Model for the given json object."""
        if isinstance(json, (str, bytes, bytearray)):
            json = js.loads(json)
        metadata = copy.deepcopy(json['metadata'])
        sample = Sample.from_json(json['sample'])
        iota = Iota.from_json(json['iota'])
        variogram = make_variogram(json['vgx'], json['vgz'], vkwargs=json['vgparam'])
        calibration = tuple(json['calibration'])
        return Model(sample, metadata=metadata, iota=iota, variogram=variogram, calibration=calibration)

    def save(self, filename):
        """Serializes self to file."""
        with open(filename, 'w', encoding='utf8') as outfile:
            js.dump(self.to_json(), outfile, ensure_ascii=False, separators=(',', ':'))

    @classmethod
    def load(cls, filename):
        """Deserializes self from file."""
        with open(filename) as infile:
            return cls.from_json(js.load(infile))


