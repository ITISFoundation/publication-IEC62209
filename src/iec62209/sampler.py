# Sampler class.

import numbers
import os

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from numpy import nan

from .defs import DATA_PATH, OUT_PATH
from .lhs import lhs
from .sample import Sample

# ==============================================================================
# functions

def latin_cube(dom, size, method=None, seed=None):
    """
    Latin hypercube sample generator.

    Generates size uniformly distributed points as latin hypercube sample taken
    from domain dom and return the result as a pd.DataFrame of len(dom) columns
    and size rows.

    If dom is a list, the domain is defined by lists as the elements of dom,
    each of which is an increasing list of values: if there are two values
    these are the min and max of the domain on that dimension; if there are 3
    or more values these are the discrete possible values for that dimension.

    If dom is a dictionary it must be of the form dict = {str:list(list)},
    whose key become the returned DataFrame column labels and values are list
    of domain values as in the list case.
    """
    if isinstance(dom, numbers.Number):
        cube = np.asarray(lhs(dom, size, method=method, seed=seed))
        cols = [str(x) for x in range(0,dom)]
        return pd.DataFrame(cube, columns=cols)
    if isinstance(dom, list):
        n = len(dom)
        cols = [str(x) for x in range(0, n)]
        cube = np.asarray(lhs(n, size, method=method, seed=seed))
        scale = [d[-1]-d[0] for d in dom]
        loc = [d[0] for d in dom]
        rect = scale * cube + loc
        df = pd.DataFrame(rect, columns=cols)
        for c, d in zip(cols, dom):
            if len(d) > 2:
                ind = pd.cut(df[c], d, labels=False, include_lowest=True)
                df[c] = [d[i] for i in ind]
        return df
    if isinstance(dom, dict):
        keys = list(dom.keys())
        vals = list(dom.values())
        n = len(keys)
        cube = np.asarray(lhs(n, size, method=method, seed=seed))
        scale = [v[-1]-v[0] for v in vals]
        loc = [v[0] for v in vals]
        rect = scale * cube + loc
        df = pd.DataFrame(rect, columns=keys)
        for c, d in zip(keys, vals):
            if len(d) > 2:
                ind = pd.cut(df[c], d, labels=False, include_lowest=True)
                df[c] = [d[i] for i in ind]
        return df


# ============================================================================
# Sampler class:

class Sampler:
    # the object attributes initialized here can be modified, yet if they are changed
    # this must be done in a meaningful way (they are not independent from each other)
    def __init__(self, xmax=40, ymax=80):
        xmax = min(max(xmax, 40), 500)
        ymax = min(max(ymax, 80), 500)
        self.xdom = [-xmax, xmax]
        self.ydom = [-ymax, ymax]
        self.angdom = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
            180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
        self.antdom = [
            ('D300' , 300),
            ('D450' , 450),
            ('D750' , 750),
            ('D835' , 835),
            ('D900' , 900),
            ('D1450',1450),
            ('D1750',1750),
            ('D1950',1950),
            ('D2300',2300),
            ('D2450',2450),
            ('D2600',2600),
            ('D3700',3700),
            ('D4200',4200),
            ('D4600',4600),
            ('D5200',5200),
            ('D5500',5500),
            ('D5600',5600),
            ('D5800',5800),
            ('V750' , 750),
            ('V835' , 835),
            ('V1950',1950),
            ('V3700',3700),
            ('C2450A',2450),
            ('C2450B',2450),
        ]

        self.freqdom = [
            300, 450, 750, 835, 900, 1450, 1750, 1950, 2300, 2450,
            2600, 3700, 4200, 4600, 5200, 5500, 5600, 5800,
        ]
        self.moddom = {
                '0': ( 'M1',     0,    0),
            '10010': ( 'M2', 10.00,    0),
            '10647': ( 'M3', 11.96,  0.2),
            '10904': ( 'M4',  5.68,  0.4),
            '10801': ( 'M5',  7.89,  0.4),
            '10803': ( 'M6',  7.93,  0.4),
            '10835': ( 'M7',  7.70,  0.8),
            '10145': ( 'M8',  5.76,  1.4),
            '10146': ( 'M9',  6.41,  1.4),
            '10227': ('M10', 10.26,  1.4),
            '10142': ('M11',  5.73,    3),
            '10144': ('M12',  6.65,    3),
            '10011': ('M13',  2.91,    5),
            '10097': ('M14',  3.98,    5),
            '10110': ('M15',  5.75,    5),
            '10111': ('M16',  6.44,    5),
            '10112': ('M17',  6.59,   10),
            '10100': ('M18',  5.67,   20),
            '10066': ('M19',  9.38,   20),
            '10067': ('M20', 10.12,   20),
            '10782': ('M21',  8.43,   25),
            '10696': ('M22',  8.91,   40),
            '10724': ('M23',  8.90,   80),
            '10974': ('M24', 10.28,  100),
        }
        self.moddom_pulse = {
                '0': ( 'M1',     0, 0),
            '10010': ( 'M2', 10.00, 0),
            '10659': ( 'M3P',  6.99, 0),
            '10660': ( 'M4P',  3.98, 0),
            '10661': ( 'M5P',  2.22, 0),
            '10662': ( 'M6P',  0.97, 0),
        }
        self.modtab = {
            'D300' : ['0', '10010'],
            'D450' : ['0', '10010', '10142', '10111', '10145', '10144', '10146', '10110',],
            'D750' : ['0', '10010', '10112', '10647', '10144', '10146', '10100', '10097',],
            'D835' : ['0', '10010', '10142', '10011', '10145', '10144', '10100', '10097',],
            'D900' : ['0', '10010', '10142', '10111', '10145', '10144', '10146', '10110',],
            'D1450': ['0', '10010', '10112', '10142', '10801', '10835', '10110', '10100',],
            'D1750': ['0', '10010', '10142', '10011', '10145', '10146', '10100', '10097',],
            'D1950': ['0', '10010', '10227', '10011', '10111', '10145', '10100', '10097',],
            'D2300': ['0', '10010', '10112', '10647', '10904', '10835', '10110',],
            'D2450': ['0', '10010', '10696', '10227', '10801', '10803', '10724',],
            'D2600': ['0', '10010', '10974', '10011', '10803', '10904', '10100',],
            'D3700': ['0', '10010', '10782', '10974', '10801', '10904', '10835',],
            'D4200': ['0', '10010', '10782', '10974', '10801', '10803', '10835',],
            'D4600': ['0', '10010', '10782', '10974', '10801', '10803', '10904', '10835',],
            'D5200': ['0', '10010', '10696', '10066', '10647', '10067', '10724',],
            'D5500': ['0', '10010', '10696', '10066', '10647', '10067', '10724',],
            'D5600': ['0', '10010', '10696', '10066', '10647', '10067', '10724',],
            'D5800': ['0', '10010', '10696', '10066', '10647', '10067', '10724',],
            'V750' : ['0', '10010', '10112', '10111', '10144', '10146', '10110', '10097',],
            'V835' : ['0', '10010', '10112', '10142', '10011', '10145', '10144', '10146',],
            'V1950': ['0', '10010', '10112', '10227', '10011', '10111', '10100', '10097',],
            'V3700': ['0', '10010', '10782', '10974', '10111', '10803', '10904', '10835',],
            'C2450A':['0', '10010', '10696', '10227', '10801', '10803', '10724',],
            'C2450B':['0', '10010', '10696', '10227', '10801', '10803', '10724',],
        }
        self.distdom = [
            2, 5, 7, 10, 15, 25,
        ]
        # dist = distmat[i][j] for k in [0,6) where:
        # i == k % 1 and j == 0 if CPIFA
        # i == k % 1 and j == 1 if VPIFA
        # i == k % 2 and j == 2 if dipole and freq < 800
        # i == k % 3 and j == 3 if dipole and  800 <= freq < 1000
        # i == k % 3 and j == 4 if dipole and  1000 <= freq
        self.distmat = [
            (  7,   2,  15, 15, 10),
            (nan, nan,  25, 25, 25),
            (nan, nan, nan,  5,  5),
        ]
        self.powlen = 21
        self.powrow = {
            'D300' : 0,
            'D450' : 1,
            'D750' : 2,
            'D835' : 3,
            'D900' : 4,
            'D1450': 5,
            'D1750': 6,
            'D1950': 7,
            'D2300': 8,
            'D2450': 9,
            'D2600':10,
            'D3700':11,
            'D4200':12,
            'D4600':13,
            'D5200':14,
            'D5500':15,
            'D5600':16,
            'D5800':17,
            'V750' :18,
            'V835' :19,
            'V1950':20,
            'V3700':21,
            'C2450A':22,
            'C2450B':23,
        }
        self.powcol = {
             7: 0,
             2: 1,
             5: 2,
            10: 3,
            15: 4,
            25: 5,
        }
        self.powmat = np.array([
            [nan, nan, nan, nan,  16,  17], #D0300
            [nan, nan, nan, nan,  14,  15], #D0450
            [nan, nan, nan, nan,  11,  13], #D0750
            [nan, nan,  10, nan,  10,  13], #D0835
            [nan, nan,   9, nan,  10,  12], #D0900
            [nan, nan,   5,   6, nan,  12], #D1450
            [nan, nan,   4,   5, nan,  12], #D1750
            [nan, nan,   2,   4, nan,  12], #D1950
            [nan, nan,   1,   3, nan,  12], #D2300
            [nan, nan,   0,   3, nan,  12], #D2450
            [nan, nan,   0,   3, nan,  12], #D2600
            [nan, nan,  -2,   2, nan,  12], #D3700
            [nan, nan,  -3,   2, nan,  12], #D4200
            [nan, nan,  -3,   2, nan,  11], #D4600
            [nan, nan,  -4,   2, nan,  10], #D5200
            [nan, nan,  -5,   1, nan,  10], #D5500
            [nan, nan,  -4,   1, nan,  10], #D5600
            [nan, nan,  -4,   1, nan,   8], #D5800
            [nan,   9, nan, nan, nan, nan], #V0750
            [nan,   9, nan, nan, nan, nan], #V0835
            [nan,  11, nan, nan, nan, nan], #V1950
            [nan,  11, nan, nan, nan, nan], #V3700
            [ 12, nan, nan, nan, nan, nan], #C2450A
            [ 12, nan, nan, nan, nan, nan], #C2450B
        ])

    # ==========================================================================
    # sample generator

    def sample(self, size, seed=None, devtype=None, modtype=None, index=False):
        """
        Sar sample generator.

        Outputs a dataframe of size sample points that conform to the standard and
        that are uniformly iid across the sar data space.
        """
        # static local data
        angind = [int(i) for i in range(0, len(self.angdom)+1)]
        angmap = {i:v for i, v in enumerate(self.angdom)}
        antind = [int(i) for i in range(0, len(self.antdom)+1)]
        antmap = {i:v for i, v in enumerate(self.antdom)}
        distind = [int(i) for i in range(0, 7)]
        distmap = {i:v for i, v in enumerate(self.distmat)}
        powind = [int(i) for i in range(0, self.powlen+1)]
        powinc = [i for i in range(0, self.powlen)]

        df = None
        modind = []
        modmatrix = {}
        if modtype == 'pulse':
            modkeys = list(self.moddom_pulse.keys())
            modperant = len(modkeys)
            modind = [int(i) for i in range(0, modperant + 1)]
            modmatrix['all'] = modkeys
        else:
            # must: modperant > 2
            modperant = 5
            modind = [int(i) for i in range(0, modperant + 1)]
            for k, v in self.modtab.items():
                # repeat the list in case modperant > len(v)
                if len(v) < modperant:
                    v = v * ((modperant-1) // len(v) + 1)
                # we pull M1, M2 with prob 0.05 each, the rest is uniform
                # we know that len(v) > 2 since modperant > 2
                modlen = len(v) - 2
                modprob = 0.9 / modlen
                weights = [0.05, 0.05] + [modprob] * modlen
                modmatrix[k] = np.random.choice(v, size=modperant, p=weights, replace=False)
        dom = {
            'antenna_index':antind,
            'modulation_index':modind,
            'power_index':powind,
            'distance_index':distind,
            'angle_index':angind,
        }
        if devtype != 'dasy':
            dom['x'] = self.xdom
            dom['y'] = self.ydom

        df = latin_cube(dom, size, method='m', seed=seed)

        # build main dataframe
        def tuple_map(row, colname, tupleind):
            return row[colname][tupleind]

        # antenna and frequency
        df['antenna_set'] = df['antenna_index'].map(antmap)
        df['antenna'] = df.apply(lambda r: tuple_map(r, 'antenna_set', 0), 1)
        df['frequency'] = df.apply(lambda r: tuple_map(r, 'antenna_set', 1), 1)

        # modulation
        # apply after antenna is set
        def mod_map(row, antname, modindname):
            ind = row[modindname]
            if modtype == 'pulse':
                uid = modmatrix['all'][ind]
                return self.moddom_pulse[uid]
            else:
                ant = row[antname]
                uid = modmatrix[ant][ind]
                return self.moddom[uid]
        df['modulation_set'] = df.apply(lambda r: mod_map(r, 'antenna', 'modulation_index'), 1)
        df['modulation'] = df.apply(lambda r: tuple_map(r, 'modulation_set', 0), 1)
        df['par'] = df.apply(lambda r: tuple_map(r, 'modulation_set', 1), 1)
        df['bandwidth'] = df.apply(lambda r: tuple_map(r, 'modulation_set', 2), 1)

        # distance
        # apply after frequency and antenna are set
        def distance_map(row, antname, freqname, distind):
            ant = row[antname]
            freq = row[freqname]
            ind = row[distind]
            if ant.startswith('D'):
                if freq < 800:
                    return self.distmat[ind % 2][2]
                elif freq < 1000:
                    return self.distmat[ind % 3][3]
                else:
                    return self.distmat[ind % 3][4]
            elif ant.startswith('V'):
                return self.distmat[0][1]
            else:
                return self.distmat[0][0]
        df['distance'] = df.apply(lambda r: distance_map(r, 'antenna', 'frequency', 'distance_index'), 1)

        # power
        # apply after antenna, modulation and distance are set
        def power_map(row, antname, modname, distname, powindname):
            i = self.powrow[row[antname]]
            j = self.powcol[row[distname]]
            p = self.powmat[i][j] + powinc[row[powindname]]
            if row[modname] == 'M1':
                p += 9
            return p
        df['power'] = df.apply(lambda r: power_map(r, 'antenna', 'modulation', 'distance', 'power_index'), 1)

        # angle
        df['angle'] = df['angle_index'].map(angmap)

        # round x, y values
        df = df.round({'x':0, 'y':0, 'power':1,})

        # columns to keep
        cols = [
            'antenna_index',
            'antenna',
            'frequency',
            'power_index',
            'power',
            'modulation_index',
            'modulation',
            'par',
            'bandwidth',
            'distance_index',
            'distance',
            'angle_index',
            'angle',
        ]
        # rows sorting by columns
        sortby = [
            'antenna_index',
            'power_index',
            'modulation_index',
            'distance',
            'angle_index']

        if devtype != 'dasy':
            cols += ['x', 'y']
            sortby += ['x', 'y']
        df = df[cols]
        df.sort_values(by=sortby, inplace=True)
        df = df.reset_index(drop=True)

        # eventually drop index columns
        if not index:
            df.drop(columns=['antenna_index', 'power_index', 'modulation_index',
                'distance_index', 'angle_index'], inplace=True)

        # those 8 dims that are xvalues
        xvar = [
            'frequency',
            'par',
            'bandwidth',
            'distance',
            'power',
            'angle',
        ]
        if devtype != 'dasy':
            xvar += ['x', 'y']

        mdata = {}
        if devtype != 'dasy':
            mdata['xmax'] = self.xdom[1]
            mdata['ymax'] = self.ydom[1]

        return Sample(df, xvar=xvar, metadata=mdata)

    # ==========================================================================
    # snap elements in sample to valid values

    def _snap(self, col, grid):
        if (np.ndim(col) <= 1):
            col = np.asarray(col).reshape(-1, 1)
        if (np.ndim(grid) <= 1):
            grid = np.asarray(grid).reshape(-1, 1)
        dists = dist.cdist(col, grid)
        inds = np.argmin(dists, axis=1)
        return np.take(grid, inds, 0)

    def _snap_ind(self, col, grid):
        if (np.ndim(col) <= 1):
            col = np.asarray(col).reshape(-1, 1)
        if (np.ndim(grid) <= 1):
            grid = np.asarray(grid).reshape(-1, 1)
        dists = dist.cdist(col, grid)
        inds = np.argmin(dists, axis=1)
        return np.take(grid, inds, 0), inds

    def _snap_frequency(self, df, freq_name='frequency', **kwargs):
        if freq_name in df:
            grid = self.freqdom
            col = df[freq_name].to_numpy()
            df[freq_name] = self._snap(col, grid)
        return df

    def _snap_distance(self, df, dist_name='distance', **kwargs):
        if dist_name in df:
            grid = self.distdom
            col = df[dist_name].to_numpy()
            df[dist_name] = self._snap(col, grid)
        return df

    def _add_antenna(self, df, ant_name='antenna',
            freq_name='frequency', dist_name='distance', **kwargs):
        vfreqs = set([750, 835, 1950, 3700])
        cfreqs = set([2450])
        if freq_name in df and dist_name in df:
            def antenna(row):
                freq = row[freq_name]
                dist = row[dist_name]
                if (dist == 7) & (freq in cfreqs):
                    ab = ['A', 'B']
                    suff = ab[np.random.randint(0, high=2)]
                    return f'C{int(freq)}{suff}'
                if (dist == 2) & (freq in vfreqs):
                    return f'V{int(freq)}'
                if (
                    ((dist == 5) & (freq >= 800)) or \
                    ((dist == 10) & (freq >= 1000)) or \
                    ((dist == 15) & (freq < 800)) or \
                    (dist == 25)
                    ):
                    return f'D{int(freq)}'
                return nan
            df[ant_name] = df.apply(lambda r: antenna(r), 1)
            df.dropna(subset=[ant_name], inplace=True)
        return df

    def _snap_power(self, df, pow_name='power', ant_name='antenna',
            dist_name='distance', **kwargs):
        if pow_name in df:
            df[pow_name] = df[pow_name].round(decimals=0)
            if ant_name in df and dist_name in df:
                def power_in_range(row):
                    i = self.powrow[row[ant_name]]
                    j = self.powcol[row[dist_name]]
                    pmin = self.powmat[i][j]
                    pmax = pmin + 20
                    p = row[pow_name]
                    # return (pmin <= p) & (p <= pmin + 20)
                    return min(max(p, pmin), pmax)
                inrange = df.apply(lambda r: power_in_range(r), 1)
                df[pow_name] = inrange
                # df = df[inrange]
        return df

    def _snap_modulation(self, df, mod_name='modulation', par_name='par', bw_name='bandwidth', **kwargs):
        mods = [tup[0] for tup in self.moddom.values()]
        grid = np.array([
            [tup[1], tup[2]] for tup in self.moddom.values()
        ])
        mod = [par_name, bw_name]
        if (par_name in df) and (bw_name in df):
            cols = df[mod].to_numpy()
            df[mod], inds = self._snap_ind(cols, grid)
            df[mod_name] = [mods[i] for i in inds]
        return df

    def _snap_angle(self, df, ang_name='angle', **kwargs):
        if ang_name in df:
            df[ang_name] = df[ang_name] % 360
            grid = self.angdom + [360]
            col = df[ang_name].to_numpy()
            df[ang_name] = self._snap(col, grid) % 360
        return df

    def _snap_location(self, df, x_name='x', y_name='y', **kwargs):
        if x_name in df:
            df[x_name] = df[x_name].round(decimals=0)
        if y_name in df:
            df[y_name] = df[y_name].round(decimals=0)
        return df

    def snap(self, sample, **kwargs):
        """
        Snaps the x-values of sample to meaningfull coordinates.

        The the size of the returned sample might be lower that the sample argument.

        """
        default = {
            'ant_name': 'antenna',
            'freq_name': 'frequency',
            'pow_name': 'power',
            'mod_name': 'modulation',
            'par_name': 'par',
            'bw_name': 'bandwidth',
            'dist_name': 'distance',
            'ang_name': 'angle',
            'x_name': 'x',
            'y_name': 'y',
        }
        names = {k:( kwargs[k] if k in kwargs else default[k] ) for k in default}
        df = sample.data.copy()
        df = self._snap_frequency(df, **names)
        df = self._snap_distance(df, **names)
        df = self._add_antenna(df, **names)
        df = self._snap_power(df, **names)
        df = self._snap_modulation(df, **names)
        df = self._snap_angle(df, **names)
        df = self._snap_location(df, **names)
        df = df.reset_index(drop=True)
        df = df[[
            names['ant_name'],
            names['freq_name'],
            names['pow_name'],
            names['mod_name'],
            names['par_name'],
            names['bw_name'],
            names['dist_name'],
            names['ang_name'],
            names['x_name'],
            names['y_name'],
        ] + sample.zvar]
        mdata = sample.metadata()

        return Sample(df, xvar=sample.xvar, zvar=sample.zvar, metadata=mdata)
