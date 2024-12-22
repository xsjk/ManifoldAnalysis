import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from dataclasses import dataclass
from typing import Iterable, Optional
from sympy.utilities import lambdify
from functools import cache
from scipy.stats import uniform as __uniform
uniform_gen = __uniform.__class__

class Distribution:
    '''
    A distribution class

    Note
    ----
    - The `rvs` method will be automatically implemented if `sampler` is set.
    - `pdf`, `ppf` and `cdf` methods should be implemented by subclasses if needed.
        - `pdf` is needed for Acceptance-Rejection method
        - `ppf` is needed for Inverse-Transform method
    '''
    _sampler = None
    @property
    def sampler(self):
        return self._sampler
    @sampler.setter
    def sampler(self, f):
        self._sampler = f
    def __call__(self, x):
        return self.pdf(x)
    def rvs(self, *args, **kwargs):
        '''
        Generate random samples from the distribution
        '''
        return self.sampler(*args, **kwargs)
    
    def pdf(self, x: float | Iterable[float]):
        '''
        Probability density function
        '''
        raise NotImplementedError
    
    def ppf(self, x: float | Iterable[float]):
        '''
        Percent point function (inverse of `cdf`)
        '''
        raise NotImplementedError
    
    def cdf(self, x: float | Iterable[float]):
        '''
        Cumulative distribution function
        '''
        raise NotImplementedError
    
    @staticmethod
    @cache
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    
    
class Sampler:
    def __call__(self, *args, **kwargs):
        return self.rvs(*args, **kwargs)

# rejection sampling method
@dataclass
class AcceptanceRejection(Sampler):
    f: Distribution
    g: Distribution
    c: float = None
    def __post_init__(self):
        if self.c is None:
            X = self.g.rvs(100000)
            self.c = np.max(self.f.pdf(X) / self.g.pdf(X)) * 1.1
        self.n_sampled = 0
        self.n_accepted = 0
        
    def rvs(self, n: int, /, range: Optional[tuple[float, float]] = None):
        n_samples = int(n * ((self.n_sampled + 1) / (self.n_accepted + 1)))
        X = self.g.rvs(n_samples, range)
        X = X[np.random.rand(n_samples) < self.f.pdf(X) / (self.c * self.g.pdf(X))]
        self.n_sampled += n_samples
        self.n_accepted += len(X)
        return X[:n] if len(X) >= n else np.append(X, self.rvs(n-len(X), range))
    
# importance sampling method
@dataclass
class ImportanceSampling(Sampler):
    f: Distribution
    g: Distribution
    def rvs(self, n: int, /, range: Optional[tuple[float, float]] = None):
        X = self.g.rvs(n, range)
        return X, self.f.pdf(X) / self.g.pdf(X)
    
# inverse transform sampling method
@dataclass
class InverseTransformSampling(Sampler):
    f: Distribution
    def rvs(self, n: int, /, range: Optional[tuple[float, float]] = None):
        if range:
            a, b = range
            u =np.random.uniform(self.f.cdf(a), self.f.cdf(b), n)
        else:
            u = np.random.rand(n)
        return self.f.ppf(u)

class SineDisribution(Distribution):
    '''
    A distribution with PDF proportional to `sin(x)^d` on `[0, pi]`
    '''
    range = (0, np.pi)

    @cache
    def __init__(self, d: int = 1):
        '''
        Parameters
        ----------
        d: int
            The degree of sin(x), default to 1
        '''
        assert d >= 0, "d must be non-negative"
        self.d = d

        from sympy.abc import x
        self.pdf = sp.sin(x) ** d
        self.cdf = sp.integrate(self.pdf, (x, 0, x)).simplify()
        c = self.cdf.subs({x: sp.pi})
        self.pdf = lambdify(x, self.pdf / c)
        self.cdf = lambdify(x, self.cdf / c)
        self.sampler = InverseTransformSampling(self) if d in (0, 1) else \
            AcceptanceRejection(self, g:=SineDisribution(1), self.pdf(np.pi / 2) / g.pdf(np.pi / 2))
    
    def ppf(self, x: float | Iterable[float]):
        if self.d == 0:
            return np.pi * x
        if self.d == 1:
            return np.arccos(1 - 2 * x)
        else:
            raise NotImplementedError
        
class PolyDistribution(Distribution):
    '''
    A distribution with PDF proportional to `x^d` on `[0, 1]`
    '''
    range = (0, 1)

    @cache
    def __init__(self, d: int = 1):
        '''
        Parameters
        ----------
        d: int
            The degree of x, default to 1
        '''
        assert d >= 0, "d must be non-negative"
        self.d = d
        self.sampler = InverseTransformSampling(self)

    def pdf(self, x: float | Iterable[float]):
        return x ** self.d * (self.d + 1)
    
    def cdf(self, x: float | Iterable[float]):
        return x ** (self.d + 1)
    
    def ppf(self, x: float | Iterable[float]):
        return x ** (1 / (self.d + 1))
    
if __name__ == "__main__":

    s = SineDisribution(d=10)
    plt.figure(figsize=(10, 3))
    plt.hist(s.rvs(100000), bins=50, density=True)
    plt.plot(x:=np.linspace(*s.range, 1000), s.pdf(x))
    plt.axis('equal')
    plt.show()