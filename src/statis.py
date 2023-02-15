import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from numpy import nan

# ============================================================================
# General numpy based statistical utility functions.

def subplot_marginal(ax, df, var, val, title=None):
    grp = grp_by(df, var, val)
    xspan = np.ptp(grp[var].to_numpy())
    yspan = np.ptp(grp['mean'].to_numpy())
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(f'{var} (span = {xspan})')
    ax.set_ylabel(f'{val} (span = {yspan:.3f})')
    ax.axhline(y=0, color='k', linestyle='-')
    ax.errorbar(grp[var], grp['mean'], yerr=grp['std'], linestyle='None', marker='o', color='blue')

def subplot_correlation(ax, df, var, val, title=None):
    grp = grp_by(df, var, val)
    grp = grp[var + ['mean']]
    grp.rename(columns={'mean': val}, inplace=True)
    if title is not None:
        ax.set_title(title)
    heatmap = sns.heatmap(grp.corr(method='kendall'), annot=True, cbar=False, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)

def interquartile_range(data, iqrfactor=4):
    """
    Returns the range q1 - (f * iqr), q3 + (f * iqr) for iqr the inter quartile range and
    f = (iqrfactor - 1) / 2.
    """
    q1, q3 = np.quantile(data, [0.25, 0.75])
    iqr = q3 - q1
    f = (iqrfactor - 1) / 2.
    return q1 - (f * iqr), q3 + (f * iqr)

def interquantile(data, lq=0.05, uq=0.95):
    """
    Returns the elements of data that are within quantile intervall [lq, uq].
    """
    data = np.asarray(data)
    mn = np.quantile(data, lq) 
    mx = np.quantile(data, uq) 
    return data[(data >= mn) & (data <= mx)]

def outerquantile(data, lq=0.05, uq=0.95):
    """
    Returns the elements of data that are not within the quantile intervall [lq, uq].
    """
    data = np.asarray(data)
    mn = np.quantile(data, lq) 
    mx = np.quantile(data, uq) 
    return data[(data < mn) | (data > mx)]

def goodfittest(model, alpha=0.25):
    """
    Performs the good fit test illustrated by the goodfitplot function.
    """
    nrmse = model.nrmse()
    passes = (nrmse <= alpha)
    return passes, nrmse

def goodfitplot(model, axes=None):
    # plt.rcParams.update({'font.size': 16})
    """
    Make a plot of the variogram model vs the empirical variogram.
    """
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.07, top=0.95, wspace=0.2, hspace=0.3)
    naxes = min(3, len(axes))
    if naxes == 1:
        model.plot_variogram(ax=axes[0])
    else:
        model.plot_variogram(ax=axes[0:2])
    if naxes == 3:
        ax = axes[2]
        vgbins, vgerr = model.variogram_errors()
        vgerr = vgerr[~np.isnan(vgerr)]
        ax.hist(np.abs(vgerr), bins=25, color='blue')
        ax.set_xlabel('abs error')
        ax.set_ylabel('count')
    rmse = model.rmse()
    nrmse = model.nrmse()
    axes[0].set_title(f'rmse = {rmse:.6f}, nrmse = {nrmse:.6f}')
    return axes[0].figure

# perform the qq test
def qqtest(data, alpha=(0.5, 1.5, 1.)):
    """
    Performs the qq test illustrated by the qqplot function.
    """
    qs, fit = stats.probplot(data, dist='norm')
    scale = fit[0]
    loc = fit[1]
    passes = (scale >= alpha[0]) and (scale <= alpha[1]) and (abs(loc) <= alpha[2])
    return passes, loc, scale

def qqplot(data, ax=None):
    # plt.rcParams.update({'font.size': 16})
    """
    Make a qq plot of data order statistics versus quantiles of N(0, 1).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(left=0.07, right=0.95, bottom=0.07, top=0.95, wspace=0.2, hspace=0.2)
    qs, fit = stats.probplot(data, dist='norm')
    slope = fit[0]
    inter = fit[1]
    ax.plot(qs[0], qs[1], 'bo')
    ax.plot(qs[0], inter + slope * qs[0], 'r--')
    ax.plot(qs[0], qs[0], 'k--')
    ax.plot([0], [0], 'ko', mfc='none',  markersize=10)
    ax.set_xlabel('std norm')
    ax.set_ylabel('order stat')
    ax.set_title(f'location = {inter:.3f}, scale = {slope:.3f}')
    return ax.figure

def kstest(data, alpha=0.05):
    """
    Performs kolmogorov-smirnov hypothesis testing on data against standard normality 
    and return the triple (passes, pval, stat) where passes is boolean with respect 
    to tolerance alpha.
    """
    data = np.asarray(data).ravel()
    stat = nan
    pval = nan
    passes = False
    if len(data) >= 2:
        stat, pval = stats.kstest(data, 'norm')
        passes = (stat >= 0) & (pval >= alpha)
    return passes, pval, stat

def swtest(data, alpha=0.05):
    """
    Performs shapiro-wilk hypothesis testing on data and return the triple 
    (passes, pval, stat) where passes is boolean with respect to tolerance alpha.
    """
    data = np.asarray(data).ravel()
    stat = nan
    pval = nan
    passes = False
    if len(data) >= 2:
        stat, pval = stats.shapiro(data)
        passes = (stat >= 0) & (pval >= alpha)
    return passes, pval, stat

