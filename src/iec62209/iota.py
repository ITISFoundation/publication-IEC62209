import json as js

import numpy as np

from .sample import Sample

# ============================================================================

class Iota:
    """
    Iota class.

    Represents an invertible map on R^n needed to transform the data space to
    an isotropic one. In the present form it is a function of the form
    iota(x) = sigma * x + mu.
    """
    def __init__(self, sigma, mu=0):
        """Constructs iota(x) = sigma * x + mu."""
        sigma = np.atleast_2d(sigma)
        if (sigma.shape[0] != sigma.shape[1]):
            raise ValueError('sigma not square')
        dim = sigma.shape[0]
        if np.isscalar(mu):
            mu = np.zeros(dim) + mu
        if (mu.shape[0] != sigma.shape[0]):
            raise ValueError('mu and sigma dimensions do not match')
        self.mu = mu
        self.sigma = sigma
        self.dim = dim

    def __str__(self):
        return repr(self.sigma) + 'x + ' + repr(self.mu)

    def __call__(self, x):
        """
        Applies self to x.

        x must be a Sample or array like with shape k x n for n the number
        of dimensions and k the number of elements.
        """
        if isinstance(x, Sample):
            xdata = x.xdata()
            xdata = np.einsum('ij,kj->ki', self.sigma, xdata) + self.mu
            return x.copy(xdata=xdata)
        else:
            return np.einsum('ij,kj->ki', self.sigma, x) + self.mu

    def compose(self, iota):
        """Function composition x -> iota(self(x))."""
        if (self.dim != iota.dim):
            raise ValueError('dimensions do not match')
        mu = (iota.sigma @ self.mu) + iota.mu
        sigma = iota.sigma @ self.sigma
        return Iota(sigma, mu)

    def inv(self):
        """Returns the inverse of self."""
        sigmainv = np.linalg.inv(self.sigma)
        return Iota(sigmainv, -sigmainv @ self.mu)

    def to_json(self):
        """Returns a json serialization object of self."""
        return {'sigma': self.sigma.tolist(), 'mu': self.mu.tolist()}

    @classmethod
    def from_json(cls, json):
        """Builds a iota instance from the given json object."""
        if isinstance(json, (str, bytes, bytearray)):
            json = js.loads(json)
        return Iota(np.array(json['sigma']), np.array(json['mu']))

    def save(self, filename):
        """Serializes self to file."""
        with open(filename, 'w', encoding='utf8') as outfile:
            js.dump(self.to_json(), outfile, ensure_ascii=False, separators=(',', ':'))

    @classmethod
    def load(cls, filename):
        """Deserializes self from file."""
        with open(filename) as infile:
            return cls.from_json(js.load(infile))
