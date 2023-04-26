import json as js
import numbers
import os
import os.path as path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import nan
from scipy.spatial import distance
from skgstat import Variogram

from . import statis as ut
from . import sar
from .iota import Iota
from .kriging import Kriging
from .model import Model
from .sample import Sample
from .sampler import Sampler

# print everything
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ==============================================================================
# functions

def make_filename(name, clazz):
    return f'{name}_{clazz}.json'

def load_measured_sample(filename):
    """
    Returns the sample with the mpe and sar columns defined in the given csv file.
    No z-variable is loaded.
    """
    xvar = ['frequency', 'power', 'par', 'bandwidth', 'distance', 'angle', 'x', 'y']
    sample = Sample.from_csv(filename, xvar, [])
    return sample

def add_zvar(sample, mass='10g'):
    """Computes sard and mpe from sar and U resp. and add them to sample."""
    sar.add_sard_mpe(sample, mass=mass)

def save_sample(sample, filename='sample.csv'):
    """Saves to csv file the given sample."""
    sample.to_csv(filename)


# ==============================================================================
# class Work

class Work:
    """
    Each instance of this class contains the state of the study and serves as an
    API layer on top of the class level code. The average user is meant to perform all
    operations from here. A work instance will keep all data relevant to the last
    procedural step.
    """
    def __init__(self, name=''):
        # is a dict
        self.data = {'name':name}
        # is a const list
        self.xvar = ['frequency', 'power', 'par', 'bandwidth', 'distance', 'angle', 'x', 'y']
        # is a string
        self.zvar = None

    def clear(self):
        name = self.data['name']
        self.data = {'name':name}
        self.zvar = None

    def clear_zvar(self):
        self.zvar = None

    # ==========================================================================
    # part 1: sampling

    def generate_sample(self, size=400, xmax=40, ymax=80, show=False, save_to=None):
        """
        Generates a sample of size elements.

        The elements are 8-dimensional latin hypercube elements that have meaningful
        sar x-values along the following dimensions:

        ['frequency', 'power', 'par', 'bandwidth', 'distance', 'angle', 'x', 'y']

        The returned sample contains meaningful values in such a way that it is
        both evenly spread and locally random and . This ensures the returned sample
        can be used as both an initial set or a test set.
        """
        sampler = Sampler(xmax=xmax, ymax=ymax)
        sample = sampler.sample(size)
        self.data['sample'] = sample
        if show:
            print(sample)
        if save_to is not None:
            sample.to_csv(save_to)

    def set_sample(self, sample):
        self.data['sample'] = sample

    def clear_sample(self):
        self.data['sample'] = None

    def print_sample(self):
        """Prints the last generated sample."""
        sample = self.data.get('sample')
        if sample is not None:
            print(sample)

    def save_sample(self, filename='sample.csv'):
        """Saves to csv file the last generated sample."""
        sample = self.data.get('sample')
        if sample is not None:
            sample.to_csv(filename)

    # ==========================================================================
    # part 2: modeling

    def load_init_sample(self, filename, zvar):
        """
        Loads into self the initial sample defined in the given csv file, using the
        given z-variable.

        The csv file must contain all 8 x-variables (in any order):

        ['frequency', 'power', 'par', 'bandwidth', 'distance', 'angle', 'x', 'y']

        Only one z-variable can be loaded; its exact label is defined by the zvar
        argument.
        """
        if (self.zvar is not None) and (zvar != self.zvar):
            raise RuntimeError('invalid zvar')
        sample = Sample.from_csv(filename, self.xvar, [zvar])
        self.data['initsample'] = sample
        self.zvar = zvar
        return sample

    def extract_outliers(self, iqrfactor=5, filename=None, show=False):
        """
        Detects and extracts potential outliers that are outside the range defined by
        iqrfactor * inter-quantile range. These values might occur as measurement errors
        and are to be double checked by the user. A value is considered to be an outlier
        if it does not belong to the closed interval

          [m - f*r/2, m + f*r/2]

        for m the mean, r the inter quartile range, of the z-values, and for f the chosen
        iqrfactor (5 is most often a good iqrfactor and should be the default).
        """
        sample = self.data.get('initsample')
        if sample is None:
            raise RuntimeError('no existing initial sample')
        outliers = sample.outliers(iqrfactor=iqrfactor)
        self.data['outliers'] = outliers
        if filename is not None:
            outliers.to_csv(filename)
        if show:
            print('outliers:')
            print(outliers)
        return outliers

    def print_outliers(self):
        """Prints the last generated outliers."""
        outliers = self.data.get('outliers')
        if outliers is not None:
            print('outliers:')
            print(outliers)

    def save_outliers(self, filename='outliers.csv'):
        """Saves to csv file the last generated outliers."""
        outliers = self.data.get('outliers')
        if outliers is not None:
            outliers.to_csv(filename)

    def make_model(self, iqrfactor=5, show=False):
        """
        Builds a model, outputs the empirical (blue) and theoretical (red) semivariogram
        after rescaling to an isotropic space.

        Nothing is to be done by the user. The system analyses geostatistical properties
        along each direction in the data space, computes an invertible mapping that
        converts the space into an isotropic one. A global multi-directional semi-variogram
        is then built on the transformed space. The blue values represent empirical
        variances computed as a function of distance between points. In red, a gaussian
        variogram curve is then fitted to the empirical values: it defines the variance
        kernel used for all subsequent interpolations. The histogram below provides the
        distribution of all distances between points.
        """
        sample = self.data.get('initsample')
        if sample is None:
            raise RuntimeError('no existing initial sample')

        dir_vkwargs = {'n_lags':20, 'model':'gaussian', 'use_nugget':True}
        iso_vkwargs = {'n_lags':60, 'model':'gaussian', 'use_nugget':True}
        model = Model(sample, iqrfactor=iqrfactor, dirvg_kwargs=dir_vkwargs, isovg_kwargs=iso_vkwargs)
        self.data['model'] = model

        # set kriging function
        kkwargs = {'min_points':1, 'max_points':50}
        kg = Kriging(model, kriging_kwargs=kkwargs)
        self.data['kriging'] = kg

        # plot variogram
        if show:
            model.plot_variogram()
            plt.show()

        return model

    def model_metadata(self):
        """
        Returns a ref to the current model's meta data.
        """
        model = self.data.get('model')
        if model is None:
            return None
        return model.metadata

    def clear_model(self):
        self.data['kriging'] = None
        self.data['model'] = None

    def plot_model(self):
        """Plot model and return the corresponding figure."""
        fig = None
        model = self.data.get('model')
        if model is not None:
            fig, ax = plt.subplots(2, 1, figsize=(12, 9))
            plt.subplots_adjust(left=0.07, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
            fig = model.plot_variogram(ax=ax)
        return fig

    def load_model(self, filename='model.json'):
        """Deserializes the model from the given json file."""
        # load model
        model = Model.load(filename)
        if (self.zvar is not None) and (model.sample.zvar[0] != self.zvar):
            raise RuntimeError('invalid zvar')
        self.data['model'] = model
        self.zvar = model.sample.zvar[0]
        # set kriging function
        kkwargs = {'min_points':1, 'max_points':50}
        kg = Kriging(model, kriging_kwargs=kkwargs)
        self.data['kriging'] = kg

    def save_model(self, filename='model.json'):
        """Serializes the last model to file."""
        model = self.data.get('model')
        if model is not None:
            model.save(filename)

    def model_to_json(self):
        """Serializes the last model to a json object."""
        js = None
        model = self.data.get('model')
        if model is not None:
            js = model.to_json()
        return js

    # ==========================================================================
    # part 3: validation

    def recalibrate(self, loc=0., scale=1., show=False):
        """
        Based on normalized residuals of the test sample, recalibrates the model by
        shifting the z-values location and rescaling the kriging errors.

        The normalized residuals of the test sample (in blue) are plotted against
        the standard normal distribution. A linear regression (in red) is performed
        and is compared against the theoretical standard distribution (in black).
        A measure of how well the test sample is normally distributed is given by
        how well aligned the blue points are (with respect to their linear regression
        line in red). The qq plot scale and location assert how standard
        (location = 0, scale = 1) that normal distribution is. When normality is
        good but the location (the average kriging value) and scale (the average
        kriging error) are off, the tester may recalibrate the model on the fly to
        better fit the test values.
        """
        tsample = self.data.get('testsample')
        if tsample is None:
            raise RuntimeError('no existing test sample')
        kg = self.data.get('kriging')
        if kg is None:
            raise RuntimeError('no existing kriging object')

        # recalibrate
        kg.recalibrate(loc, scale)
        self.data['model'] = kg.model

        # compute residuals
        nresid, resid, err = kg.residuals(tsample)
        nresid = nresid[~np.isnan(nresid)]

        # plot qq
        if show:
            ut.qqplot(nresid)
            plt.show()

    def goodfit_validate(self, show=False):
        """
        Performs the good fit test: passes if the NRMSE is below 25%.

        This test measures the quality of the variogram fit: how well the red curve
        fits the blue values. The statistic used is the normalized root mean square
        error (NRMSE) of the variances along distances: it is equal to the RMSE of
        the residuals divided by the mean of the variances. Unlike the RMSE, the NRMSE
        does not depend on the scale of the model and provides a more robust evaluation
        of the goodness of fit. A NRMSE above 0.25 means the variogram model does not
        fit the empirical variances well enough. The last histogram shows the
        distribution of the absolute values of residuals.
        """
        model = self.data.get('model')
        if model is None:
            raise RuntimeError('no existing model')

        # perform good fit test
        gfres = ut.goodfittest(model)
        gfstr = f'gf test: pass = {gfres[0]}, nrmse = {gfres[1]:.3f}'
        print(gfstr)

        # plot good fit
        if show:
            ut.goodfitplot(model)
            plt.show()

    def goodfit_test(self):
        """
        Performs the good fit test.
        """
        model = self.data.get('model')
        if model is None:
            raise RuntimeError('no existing model')
        return ut.goodfittest(model)

    def goodfit_plot(self):
        """
        Plots the result of the good fit test.
        """
        fig = None
        model = self.data.get('model')
        if model is not None:
            fig = ut.goodfitplot(model)
        return fig

    def load_test_sample(self, filename, zvar):
        """Loads csv file data as the last test sample."""
        model = self.data.get('model')
        if model is None:
            raise RuntimeError('no existing model')
        xvar = model.sample.xvar
        zvar = [zvar]
        tsample = Sample.from_csv(filename, xvar, zvar)
        self.data['testsample'] = tsample
        return tsample

    def clear_test_sample(self):
        self.data['testsample'] = None

    def resid_validate(self, show=False):
        """
        The model is being confirmed by performing statistical tests that ascertain
        that the residuals between the model and the measured data are distributed
        according to the expected probability distribution. Those residuals are
        normalized into a distribution that needs to be as close as possible to the
        standard normal distribution. The test results are presented in terms of:

        i) the Shapiro-Wilk hypothesis p-value, which must be at least equal to
        0.05 for the normality test to pass.

        ii) the qq location and scale which need to be in the range of [-1, 1]
        and [0.5, 1.5] respectively for the normality to be standard enough.

        The test is successful if both i) and ii) pass.
        """
        tsample = self.data.get('testsample')
        if tsample is None:
            raise RuntimeError('no existing test sample')
        kg = self.data.get('kriging')
        if kg is None:
            raise RuntimeError('no existing kriging object')

        # compute residuals
        nresid, resid, err = kg.residuals(tsample)
        nresid = nresid[~np.isnan(nresid)]
        nresid = ut.interquantile(nresid, 0.025, 0.975)

        # perform sw test
        swres = ut.swtest(nresid)
        swstr = f'sw test: pass = {swres[0]}, pval = {swres[1]:.3f}'
        print(swstr)

        # perform qq test
        qqres = ut.qqtest(nresid)
        qqstr = f'qq test: pass = {qqres[0]}, loc = {qqres[1]:.3f}, scale = {qqres[2]:.3f}'
        print(qqstr)

        # plot qq
        if show:
            ut.qqplot(nresid)
            plt.show()

    def compute_resid(self):
        """
        Compute residuals.
        """
        tsample = self.data.get('testsample')
        if tsample is None:
            raise RuntimeError('no existing test sample')
        kg = self.data.get('kriging')
        if kg is None:
            raise RuntimeError('no existing kriging object')

        # compute residuals
        nresid, resid, err = kg.residuals(tsample)
        nresid = nresid[~np.isnan(nresid)]
        nresid = ut.interquantile(nresid, 0.025, 0.975)
        return nresid

    def resid_test(self, nresid):
        """
        Performs the residuals tests.
        """
        swres = ut.swtest(nresid)
        qqres = ut.qqtest(nresid)
        return swres, qqres

    def resid_plot(self, nresid):
        """
        Plots the result of the residuals test.
        """
        swres = ut.swtest(nresid)
        swstr = f'SW p-value = {swres[1]:.3f}'
        fig = ut.qqplot(nresid)
        ax0 = fig.axes[0]
        ax0.set_title(swstr + ', ' + ax0.get_title())
        return fig

    # ==========================================================================
    # part 4: exploration

    def init_critsample(self):
        xvar = [
            'antenna',
            'frequency',
            'par',
            'bandwidth',
            'distance',
            'power',
            'angle',
            'x',
            'y',
            'sard',
            'err',
            # 'mpe',
            # 'passl',
            # 'passu',
            'pass',
        ]
        df = pd.DataFrame(columns=xvar, index=[0])
        csample = Sample(df, xvar=xvar)
        self.data['critsample'] = csample
        return csample

    def explore(self, maxsize=None, sens=0.1, niter=8, snap=True, show=False, save_to=None):
        """
        Performs space exploration using at most maxsize trajectories and outputs to
        file the most critical regions.

        Now that the model is deemed valid, it can be used to explore the entire data
        space for potential regions that exceed the most permissible error (mpe).
        This is done by a hybrid search trajectory and population-based algorithm
        where a population of a computed number of search trajectories evolve through
        a predetermined number of iterations (generations) in such a way that:

        i) the elements of the population are pulled towards the most extreme regions
        of the data space,

        ii) the elements of the population exert a repulsive force on each other.
        This ensure not all trajectories will be lead to the same locations, but
        insted will evenly cover a region deemed critical,

        iii) the resulting values have meaningful SAR coordinates,

        iv) the trajectories converge as rapidly as possible.

        The resulting coordinates, with the computed z-values and associated probabilities
        to pass the mpe value are outputed as a csv file whose name is to be provided by
        the user. The population usually stabilizes after only 8 iterations.
        """
        model = self.data.get('model')
        if model is None:
            raise RuntimeError('no existing model')
        kg = self.data.get('kriging')
        if kg is None:
            raise RuntimeError('no existing kriging object')

        minsize = 50
        if maxsize is not None:
            maxsize = max(minsize, maxsize)
        else:
            maxsize = 500

        sdf = model.initial_search_sample(minsize=minsize, maxsize=maxsize)
        cdf = kg.search(sdf, sens=sens, niter=niter, snap=snap)
        cdf = kg.add_prob(cdf)
        cdf = cdf.drop(columns=['mpe', 'passl', 'passu'])
        cdf = cdf.round(decimals = 4)
        csample = Sample(cdf, xvar=self.xvar, zvar=[self.zvar])
        self.data['critsample'] = csample
        if show:
            print(f'critical sample:')
            print(csample)
        if save_to is not None:
            csample.to_csv(save_to)
        return csample

    def print_crit_sample(self):
        """Prints the last found critical sample."""
        csample = self.data.get('critsample')
        if csample is not None:
            print(f'critical sample:')
            print(csample)

    def save_crit_sample(self, filename):
        """Saves to file the last found cirtical sample."""
        csample = self.data.get('critsample')
        if csample is not None:
            csample.to_csv(filename)


