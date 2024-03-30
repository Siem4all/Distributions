#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import pickle

import numpy as np
import scipy.stats as stats
from printf import printf


# Generating normal distributed random variables with mean 0 and standard deviation 1
# data_normal = norm.rvs(size=1000, loc=0, scale=1)


class DistrBase:
    def __init__(self, params_dict, range_min, range_max, *args, **kwargs):
        self.params_dict = params_dict
        assert range_max >= range_min
        self.range_min = range_min
        self.range_max = range_max

    def pdf(self, x):
        raise NotImplementedError()

    def sample(self, shape):
        raise NotImplementedError()

    def print(self):
        raise NotImplementedError()


class ClippedGaussDistr(DistrBase):
    """
    The normal distribution density function simply accepts a data point along with a mean value and a standard deviation
     and throws a value which we call probability density. A standard normal distribution is just
     similar to a normal distribution with mean = 0 and standard deviation = 1.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mu = self.params_dict["mu"]
        sigma = self.params_dict["sigma"]
        self.point_mass_range_min = stats.norm.cdf(self.range_min, loc=mu, scale=sigma)
        self.point_mass_range_max = 1.0 - stats.norm.cdf(self.range_max, loc=mu, scale=sigma)

    def print(self):
        print(
            "Gaussian distr ",
            ", mu = ",
            self.params_dict["mu"],
            ", sigma = ",
            self.params_dict["sigma"],
            " clipped at [",
            self.range_min,
            ",",
            self.range_max,
            "]",
        )

    def cdf(self, x_np):
        """
        The cumulative distribution function (CDF) of the standard normal distribution
        """
        p = stats.norm.cdf(x, self.params_dict["mu"], self.params_dict["sigma"])
        return p

    def pdf(self, x):
        """The simplest case of a normal distribution is known as the standard normal distribution or unit normal distribution.
         This is a special case when mu =0 and sigma =1, and it is described by this probability density function (or density)
        """
        p = stats.norm.pdf(x, self.params_dict["mu"], self.params_dict["sigma"])
        return p

    def inverse_cdf(self, x):
        """
        Compute the inverse of cdf values evaluated at the probability values in p for the normal distribution with mean mu and standard deviation sigma.
        """
        res = stats.norm.ppf(x, loc=self.params_dict["mu"], scale=self.params_dict["sigma"])
        return res

    def sample(self, shape):
        r = np.random.normal(
            loc=self.params_dict["mu"], scale=self.params_dict["sigma"], size=shape
        )
        r = np.clip(r, self.range_min, self.range_max)
        return r


class ClippedStudentTDistr(DistrBase):
    """
    The t-distribution is used for estimation and hypothesis testing of a population mean (average).
    The t-distribution is adjusted for the extra uncertainty of estimating the mean.

    If the sample is small, the t-distribution is wider. If the sample is big, the t-distribution is narrower.

    The bigger the sample size is, the closer the t-distribution gets to the standard normal distribution.
    https://www.geo.fu-berlin.de/en/v/soga-py/Basics-of-statistics/Continous-Random-Variables/Students-t-Distribution/Students-t-Distribution-in-Python/index.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nu = self.params_dict["nu"]
        self.point_mass_range_min = stats.t.cdf(self.range_min, nu)
        self.point_mass_range_max = 1.0 - stats.t.cdf(self.range_max, nu)

    def print(self):
        print(
            "Student's-t distr",
            ", nu = ",
            self.params_dict["nu"],
            " clipped at [",
            self.range_min,
            ",",
            self.range_max,
            "]",
        )

    def pdf(self, x):
        """
        Calculates PDF of the Student's t-distribution at point 'x'

        Parameters
        ==========
        x: float
        x value of the function
        nu: int or float
        number of degrees of freedom

        Returns
        =======
        (gamma((self.nu + 1.) / 2.) / (np.sqrt(np.pi * self.nu) * gamma(self.nu / 2.))) * np.power(
            (1. + x ** 2 / self.nu), -(self.nu + 1.) / 2.)
        """
        p = stats.t.pdf(x, self.params_dict["nu"])
        return p

    def cdf(self, x):
        p = stats.t.cdf(x, self.params_dict["nu"])
        return p

    def inverse_cdf(self, x):
        res = stats.t.ppf(x, self.params_dict["nu"])
        return res

    def sample(self, shape):
        r = np.random.standard_t(self.params_dict["nu"], size=shape)
        r = np.clip(r, self.range_min, self.range_max)
        return r


class UniformDistr(DistrBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pd = 1 / (self.range_max - self.range_min)

    def print(self):
        print("Uniform distribution on [", self.range_min, ",", self.range_max, "]")

    def pdf(self, x):
        pd=[]
        for i in range(len(x)):
            if self.range_min<=x[i] and x[i]>=self.range_max:
               pd.append(1 / (self.range_max - self.range_min))
            else:
                pd.append(0)
        return pd

    def cdf(self, x):
        return (x - self.range_min) * self.pd

    def sample(self, shape):
        return np.random.uniform(self.range_min, self.range_max, shape)


def dumpDictToPcl(dict):
    """
    Dump a single dict of data into pclOutputFile
    """
    fileName="distributions"
    pclOutputFile = open(f'res/pcl_files/{fileName}.pcl', 'ab+')  # the path where we keep the the pcl file
    pickle.dump(dict, pclOutputFile)


def writeDictToResFile(dict):
    """
    Write a single dict of data into resOutputFile
    """
    fileName="distributions"
    resFile = open(f'res/{fileName}.res', 'a+')
    printf(resFile, f'{dict}\n\n')


if __name__ == "__main__":
    distr_list = {"uniform": UniformDistr(range_min=-1.0, range_max=1.0, params_dict={}),
                  "Gauss": ClippedGaussDistr(params_dict={"mu": 0.0, "sigma": 1.0}, range_min=-10.0, range_max=10.0),
                  "student": ClippedStudentTDistr(params_dict={"nu": 8.0}, range_min=-100.0, range_max=100.0)}
    # We want the distributions to appear on the plot from -3 to 3
    x = np.linspace(-10, 10, 200)
    dict = {"floatInput":x}
    for key, value in distr_list.items():
        print("*" * 80)
        dict[key]=value.pdf(x)
    dumpDictToPcl(dict)
    writeDictToResFile(dict)
