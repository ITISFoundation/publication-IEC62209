# plot utilities.

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools as it
from .sample import Sample

# ==============================================================================
# functions

# corresponding values in columns vals
def _vals_by(df, ids, vals): 
    return pd.melt(df, id_vars=ids, value_vars=vals, var_name='device').drop(['device'], axis=1)

# return a new df of columns of grouped values [[ids], min, max, mean, std]
def _grp_by(df, ids, vals):
    grp = _vals_by(df, ids, vals).groupby(ids) 
    x = grp.min().reset_index()
    x.rename(columns={'value':'min'}, inplace=True)
    x['max'] = grp.max().reset_index()['value']
    x['mean'] = grp.mean().reset_index()['value']
    x['std'] = grp.std().reset_index()['value']
    x['std'] = x['std'].fillna(0)
    return x

def _subplot_marginal(ax, df, var, val, title=None, var_label=None, val_label=None):
    grp = _grp_by(df, var, val)
    xspan = np.ptp(grp[var].to_numpy())
    yspan = np.ptp(grp['mean'].to_numpy())
    fsize = 11
    if title is not None:
        ax.set_title(title)
    if var_label is not None:
        ax.set_xlabel(var_label, fontsize=fsize)
    else:
        ax.set_xlabel(f'{var}', fontsize=fsize)
    if val_label is not None:
        ax.set_ylabel(val_label, fontsize=fsize)
    else:
        ax.set_ylabel(f'{val}', fontsize=fsize)
    ax.axhline(y=0, color='k', linestyle='-')
    ax.errorbar(grp[var], grp['mean'], yerr=grp['std'], linestyle='None', marker='o', color='blue')

def plot_sample_distribution(sample):
    df = sample.data
    combs = list(it.combinations(['frequency', 'power', 'angle', 'x', 'y',], 2))
    last = len(combs)-1
    rows = 4
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(11, 12))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.05, top=0.95, wspace=0.36, hspace=0.28)
    for i, c in enumerate(combs):
        if i == last:
            i = i+1
        ax = axes[i//cols, i%cols]
        xlabel = c[0]
        ylabel = c[1]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.scatter(df[xlabel], df[ylabel], s=5)
    axes[3, 0].axis('off')
    axes[3, 2].axis('off')
    return fig

def plot_sample_marginals(sample, mass='10g'):
    df = sample.data
    zvar = 'sard' + mass
    labels = ['frequency', 'power', 'par', 'bandwidth', 'distance', 'angle', 'x', 'y'] 
    zlabel = r'$\Delta SAR_{{{mass}}}$ (dB)'.format(mass=mass)
    nrows = 4
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=[15, 12])
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, wspace=0.2, hspace=0.4)
    for i, v in enumerate(labels):
        ax = axes[i//ncols, i%ncols]
        _subplot_marginal(ax, df, v, zvar, val_label=zlabel)
        ax.set_ylim([-2, 2])
    return fig

def plot_sample_deviations(sample, mass='10g'):
    df = sample.data.copy()
    index = 'index'
    df[index] = range(1, len(df) + 1)
    sardlab = 'sard' + mass
    mpelab = 'mpe' + mass
    fig, ax = plt.subplots(1, 1, figsize=[12, 9])
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, wspace=0.2, hspace=0.4)
    bright_red, dark_blue, grey = '#f0240a', '#030764', '#54504f'
    ax.plot( df[mpelab], color=grey, marker = 'None', linestyle = '--')
    ax.plot(-df[mpelab], color=grey, marker = 'None', linestyle = '--')
    ax.fill_between(df[index], -df[mpelab], df[mpelab], alpha=0.07)
    accepted = [abs(df[sardlab][i]) <= df[mpelab][i] for i in range(len(df))]
    refused = [abs(df[sardlab][i]) > df[mpelab][i] for i in range(len(df))]
    data_inside = df.loc[accepted]
    data_outside = df.loc[refused]
    ax.plot(data_inside[index], data_inside[sardlab], color=dark_blue, marker = 'o', linestyle = 'None')
    ax.plot(data_outside[index], data_outside[sardlab], color=bright_red, marker = 'o', linestyle = 'None')
    mx = math.ceil(max(df[sardlab].abs().max(), df[mpelab].abs().max()))
    ax.set_ylim(-mx, mx)
    ax.set_yticks(np.arange(-mx, mx+0.1, 0.5))
    plt.axhline(0, color='k')
    plt.grid(axis='both', color='0.9')
    fsize = 12
    ax.set_xlabel('test', fontsize=fsize)
    ax.set_ylabel(r'$\Delta SAR_{{{mass}}}$ (dB)'.format(mass=mass), fontsize=fsize)
    return fig

