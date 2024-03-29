# Sar utilities.

import math
import numpy as np
import pandas as pd
from .sample import Sample

# ==============================================================================
# functions

def add_sard_mpe(sample, mass='10g'):
    sarcol = 'sar' + mass
    # sample inner dataframe
    sample_df = sample.data
    # check nec cols are there
    if not pd.Series(['antenna', 'frequency', 'power', 'distance', sarcol]).isin(sample_df.columns).all():
        raise RuntimeError('invalid sample dataframe')

    # targets dataframe
    target_df = pd.DataFrame([
        ['D300', 300, 30, 15, 2.85, 1.94], 
        ['D300', 300, 30, 25, 1.92, 1.39], 
        ['D450', 450, 30, 15, 4.58, 3.06], 
        ['D450', 450, 30, 25, 3.15, 2.25], 
        ['D750', 750, 20, 15, 0.849, 0.555], 
        ['D750', 750, 20, 25, 0.515, 0.359], 
        ['D835', 835, 30, 15, 9.56, 6.22], 
        ['D835', 835, 30, 5, 11.176, 5.776], 
        ['D835', 835, 30, 25, 5.501, 3.826], 
        ['D900', 900, 17, 15, 0.547, 0.35], 
        ['D900', 900, 17, 5, 0.667, 0.342], 
        ['D900', 900, 17, 25, 0.3, 0.205], 
        ['D1450', 1450, 13, 10, 0.583, 0.324], 
        ['D1450', 1450, 13, 5, 0.706, 0.349], 
        ['D1450', 1450, 13, 25, 0.143, 0.092], 
        ['D1750', 1750, 10, 10, 0.364, 0.193], 
        ['D1750', 1750, 10, 5, 0.473, 0.223], 
        ['D1750', 1750, 10, 25, 0.069, 0.042], 
        ['D1950', 1950, 24, 10, 10.17, 5.25], 
        ['D1950', 1950, 3, 5, 0.13, 0.059], 
        ['D1950', 1950, 30, 25, 6.91, 4.15], 
        ['D2300', 2300, 20, 10, 4.87, 2.33], 
        ['D2300', 2300, 20, 5, 8.37, 3.566], 
        ['D2300', 2300, 20, 25, 0.657, 0.373], 
        ['D2450', 2450, 30, 10, 51.4, 23.8], 
        ['D2450', 2450, 3, 5, 0.188, 0.078], 
        ['D2450', 2450, 20, 25, 0.717, 0.393], 
        ['D2600', 2600, 20, 10, 5.53, 2.46], 
        ['D2600', 2600, 20, 5, 11.138, 4.329], 
        ['D2600', 2600, 20, 25, 0.675, 0.361], 
        ['D3700', 3700, 10, 10, 0.674, 0.242], 
        ['D3700', 3700, 10, 5, 1.86, 0.568], 
        ['D3700', 3700, 10, 25, 0.068, 0.031], 
        ['D4200', 4200, 30, 10, 66.4, 22.2], 
        ['D4200', 4200, 30, 5, 216.9, 59.0], 
        ['D4200', 4200, 30, 25, 6.6, 2.9], 
        ['D4600', 4600, 30, 10, 66.7, 21.5], 
        ['D4600', 4600, 30, 5, 224.4, 59.2], 
        ['D4600', 4600, 30, 25, 7.0, 3.0], 
        ['D4900', 4900, 30, 10, 68.4, 21.2], 
        ['D5200', 5200, 7, 10, 0.379, 0.107], 
        ['D5200', 5200, 7, 5, 1.38, 0.285], 
        ['D5200', 5200, 7, 25, 0.049, 0.02], 
        ['D5500', 5500, 7, 10, 0.417, 0.117], 
        ['D5500', 5500, 7, 5, 1.54, 0.319], 
        ['D5500', 5500, 7, 25, 0.057, 0.024], 
        ['D5600', 5600, 27, 10, 40.1, 11.3], 
        ['D5600', 5600, 27, 5, 149.0, 31.0], 
        ['D5600', 5600, 27, 25, 5.683, 2.353], 
        ['D5800', 5800, 10, 10, 0.78, 0.219], 
        ['D5800', 5800, 0, 5, 0.274, 0.057], 
        ['D5800', 5800, 20, 25, 1.48, 0.57], 
        ['V750', 750, 24, 2, 3.25, 0.97], 
        ['V835', 835, 24, 2, 3.32, 0.96], 
        ['V1950', 1950, 24, 2, 2.18, 0.9], 
        ['V3700', 3700, 24, 2, 2.48, 1.04], 
        ['C2450A', 2450, 24, 7, 1.7, 0.806], 
        ['C2450B', 2450, 24, 7, 1.34, 0.661]], 
        columns=['antenna', 'frequency', 'power', 'distance', 'sar1g_target', 'sar10g_target'])

    df = sample_df.merge(target_df, how='left', on=['antenna', 'frequency', 'distance'])
    df = df.rename({'power_x': 'power', 'power_y': 'power_target'}, axis=1)

    # compute sar targets
    sartgcol1 = 'sar' + mass + '_target'
    sartgcol = 'sartg' + mass
    df[sartgcol] = df[sartgcol1] * 10**((df['power'] - df['power_target'])/10) # normalize to measured power

    # compute sar deviations
    sardcol = 'sard' + mass
    df[sardcol] = 10*np.log10(df[sarcol]/df[sartgcol])

    # add a small amount of noise to avoid degenerate variograms
    lim = 0.11
    while df[sardcol].std() < lim:
        df[sardcol] = df[sardcol] + np.random.normal(0, lim/2, len(df))

    # compute mpe
    ucol = 'u' + mass
    mpecol = 'mpe' + mass
    df[mpecol] = 10*np.log10(1 + sample_df[ucol] + 0.15)

    # add sar deviations value as z-variable
    sample.set_zvar(sardcol, vals=df[sardcol].round(2))

    # add mpe column
    sample.data[mpecol] = df[mpecol].round(2)

