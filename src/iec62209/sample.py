# Sample class:

import json as js

import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import statis as st

# ============================================================================
# Sample class

class Sample:
    # while df is not copied into self, xvar and zvar lists are.
    def __init__(self, df, xvar = [], zvar = [], metadata = None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError
        # metadata has extra information that does not directly affect the class mechanisms
        self.mdata = metadata
        self.data = df
        self.xvar = list(xvar)
        self.zvar = list(zvar)

    def __str__(self):
        return self.data.to_string()

    def copy(self, xdata=None, zdata=None, xvar=None, zvar=None, metadata=None):
        """
        Returns a deep copy of self.

        The returned copy's content is restricted to all conditions defined by
        non None arguments.
        """
        xv = None
        zv = None
        if xvar is None:
            xv = self.xvar
        else:
            if isinstance(xvar, str):
                xvar = [xvar]
            xv = [x for x in xvar if x in list(self.data)]
        if zvar is None:
            zv = self.zvar
        else:
            if isinstance(zvar, str):
                zvar = [zvar]
            zv = [z for z in zvar if z in list(self.data)]
        data = self.data.copy()
        if xdata is not None:
            if (xdata.shape[0] != data.shape[0]) or (xdata.shape[1] != len(xv)):
                raise ValueError('invalid shape for xdata')
            df = pd.DataFrame(xdata, columns=xv)
            data.loc[:, xv] = df[xv]
        if zdata is not None:
            if (zdata.shape[0] != data.shape[0]) or (zdata.shape[1] != len(zv)):
                raise ValueError('invalid shape for zdata')
            df = pd.DataFrame(zdata, columns=zv)
            data.loc[:, zv] = df[zv]
        md = None
        if metadata is None:
            if self.mdata is not None:
                md = copy.deepcopy(self.mdata)
        else:
            md = copy.deepcopy(metadata)
        return Sample(data, xvar=xv, zvar=zv, metadata=md)

    def isrealvalued(self):
        """Returns true iff self has exactly 1 z-variable and at least 1 x-variable."""
        return (len(self.xvar) > 0) and (len(self.zvar) == 1)

    def metadata(self):
        """Returns a copy of the metadata."""
        return copy.deepcopy(self.mdata)

    def contains(self, sample):
        """Returns True iff self domain contains sample domain."""
        return (self.mdata['xmax'] >= sample.mdata['xmax']) and (self.mdata['ymax'] >= sample.mdata['ymax'])

    def xshape(self):
        """Returns the shape of the sub-dataframe of x-variables."""
        return (self.data.shape[0], len(self.xvar))

    def zshape(self):
        """Returns the shape of the sub-dataframe of z-variables."""
        return (self.data.shape[0], len(self.zvar))

    def size(self):
        """Returns the number of points in this sample."""
        return self.data.shape[0]

    def add_df(self, df):
        """Appends the content of df into self."""
        self.data[list(df)] = df.to_numpy()

    def get_var(self, var):
        """Returns the column associated to var."""
        if var not in list(self.data):
            raise ValueError('var does not exist')
        return self.data[var]

    def set_xvar(self, xvar, vals=np.nan):
        """Sets the values of the x-variable xvar to vals."""
        if not isinstance(xvar, str):
            raise ValueError('xvar not a str')
        if xvar not in list(self.data):
            self.xvar.append(xvar)
        self.data[xvar] = vals

    def set_zvar(self, zvar, vals=np.nan):
        """Sets the values of the z-variable zvar to vals."""
        if not isinstance(zvar, str):
            raise ValueError('zvar not a str')
        if zvar not in list(self.data):
            self.zvar.append(zvar)
        self.data[zvar] = vals

    def rm_xvar(self, xvar):
        """Drops x-variable column xvar."""
        self.data = self.data.drop(xvar, axis=1)
        self.xvar = [x for x in self.xvar if x not in xvar]

    def rm_zvar(self, zvar):
        """Drops z-variable column zvar."""
        self.data = self.data.drop(zvar, axis=1)
        self.zvar = [z for z in self.zvar if z not in zvar]

    def xdata(self):
        """Returns a numpy array of all x-variables values."""
        return self.data[self.xvar].to_numpy()

    def zdata(self):
        """Returns a numpy array of all z-variables values."""
        return self.data[self.zvar].to_numpy()

    def outliers(self, iqrfactor=4):
        """Returns the outliers that are not contained in the iqrfactor*iqr range."""
        data = self.data
        zvar = self.zvar
        dfs = [pd.DataFrame(columns=data.columns)]
        for zv in zvar:
            lb, ub = st.interquartile_range(data[zv], iqrfactor=iqrfactor)
            dfs.append(data[(data[zv] < lb) | (data[zv] > ub)])
        df = pd.concat(dfs, axis=0).drop_duplicates()
        return Sample(df, xvar=self.xvar, zvar=self.zvar)

    def inliers_on(self, var, iqrfactor=4):
        """Returns the inliers of column var that are within the iqrfactor*iqr range."""
        data = self.data
        lb, ub = st.interquartile_range(data[var], iqrfactor=iqrfactor)
        indata = data[(data[var] >= lb) & (data[var] <= ub)]
        return Sample(indata, xvar=self.xvar, zvar=self.zvar)

    def split(self, test_size, seed=None):
        """Splits self into a train sample and a test sample."""
        if seed is not None:
            seed = int(seed)
        df_train, df_test = train_test_split(self.data, test_size=test_size, random_state=seed)
        return Sample(df_train, xvar=self.xvar, zvar=self.zvar), Sample(df_test, xvar=self.xvar, zvar=self.zvar)

    def to_json(self):
        """Returns a json object (a dict) representation of self."""
        md = None
        if self.mdata is not None:
            md = copy.deepcopy(self.mdata)
        return {'metadata':md, 'xvar':list(self.xvar), 'zvar':list(self.zvar), 'data':dict(self.data.to_dict('list'))}

    @classmethod
    def from_json(cls, json):
        """Builds a Sample for the given json object."""
        if isinstance(json, (str, bytes, bytearray)):
            json = js.loads(json)
        mdata = json['metadata']
        data = pd.DataFrame(pd.DataFrame.from_dict(json['data']))
        xvar = [x for x in json['xvar'] if x in list(data)]
        zvar = [z for z in json['zvar'] if z in list(data)]
        return Sample(data, xvar=xvar, zvar=zvar, metadata=mdata)

    def to_csv(self, filename):
        """Saves self's dataframe to csv (without the xvar, zvar defintions)."""
        self.data.to_csv(filename, float_format='%.6g')

    @classmethod
    def from_csv(cls, filename, xvar, zvar):
        """Builds a Sample from the provided csv file."""
        df = pd.read_csv(filename)
        var = list(df)
        xn = [x for x in xvar if x in var]
        zn = [z for z in zvar if z in var]
        if len(xn) < len(xvar) or len(zn) < len(zvar):
            raise ValueError('non existent columns')
        return Sample(df, xvar=xn, zvar=zn)

    def save(self, filename):
        """Serializes self to file."""
        with open(filename, 'w', encoding='utf8') as outfile:
            js.dump(self.to_json(), outfile, ensure_ascii=False, separators=(',', ':'))

    @classmethod
    def load(cls, filename):
        """Deserializes self from file."""
        with open(filename) as infile:
            return cls.from_json(js.load(infile))




