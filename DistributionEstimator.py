import pickle

import numpy as np
import scipy.stats as stats
from fitter import Fitter
import matplotlib.pyplot as plt
from printf import printf


class DistributionEstimator:
    def __init__(self, params):
        self.params = params

    def estimate_pdf(self, distribution, x):
        if distribution == 'gamma':
            pdf = stats.gamma.pdf(x, *self.params['gamma'])
        elif distribution == 'lognorm':
            pdf = stats.lognorm.pdf(x, *self.params['lognorm'])
        elif distribution == 'beta':
            pdf = stats.beta.pdf(x, self.params['beta'][0], self.params['beta'][1])
        elif distribution == 'norm':
            pdf = stats.norm.pdf(x, *self.params['norm'])
        elif distribution == 'f':
            pdf = stats.f.pdf(x, *self.params['f'])
        elif distribution == 't':
            pdf = stats.t.pdf(x, *self.params['t'])
        else:
            raise ValueError("Invalid distribution")

        return pdf

    def estimate_cdf(self, distribution, x):
        if distribution == 'gamma':
            cdf = stats.gamma.cdf(x, *self.params['gamma'])
        elif distribution == 'lognorm':
            cdf = stats.lognorm.cdf(x, *self.params['lognorm'])
        elif distribution == 'beta':
            cdf = stats.beta.cdf(x, self.params['beta'][0], self.params['beta'][1])
        elif distribution == 'norm':
            cdf = stats.norm.cdf(x, *self.params['norm'])
        elif distribution == 'f':
            cdf = stats.f.cdf(x, *self.params['f'])
        elif distribution == 't':
            cdf = stats.t.cdf(x, *self.params['t'])
        else:
            raise ValueError("Invalid distribution")

        return cdf
def dump_dict_to_pkl(dict):
    """
    Dump a single dict of data into pclOutputFile
    """
    fileName="fillters"
    pclOutputFile = open(f'res/pcl_files/{fileName}.pcl', 'ab+')  # the path where we keep the the pcl file
    pickle.dump(dict, pclOutputFile)


def write_dict_to_res_file(dict):
    """
    Write a single dict of data into resOutputFile
    """
    fileName="fillters"
    resFile = open(f'res/{fileName}.res', 'a+')
    printf(resFile, f'{dict}\n\n')

# Example usage
def run_simulations():
    # Data
    values = np.array([15, 4, 2, 11, 1, -2, -1, -14, -39, -3])
    values_lo = np.array([7, 16, 6, 3, 6, 5, 10, 6, 11, 13])
    values_hi = np.array([7, 15, 8, 3, 6, 6, 10, 7, 14, 14])

    # Generating samples
    n_data = len(values)
    samples = []
    for i in range(n_data):
        u = np.random.normal(size=400)
        v = values[i] + np.where(u < 0, u * values_lo[i], u * values_hi[i])
        samples.append(v)

    dataset = np.array(samples)

    # Fitting distributions
    f = fitter.Fitter(dataset, distributions=['norm'])
    f.fit()
    best_distribution = f.get_best(method='sumsquare_error')

    estimator = DistributionEstimator(f.fitted_param)
    distribution_name = next(iter(best_distribution))

    # Generate x values for plotting
    x = np.linspace(np.min(dataset), np.max(dataset), 100)
    # Estimate PDF
    pdf = estimator.estimate_pdf(distribution_name, x)
    # Estimate CDF
    cdf = estimator.estimate_cdf(distribution_name, x)
    data = {"distribution_name": distribution_name, "input_value": x, "pdf": pdf, "cdf": cdf}
    dump_dict_to_pkl(data)
    write_dict_to_res_file(data)

if __name__ == '__main__':
    run_simulations()
