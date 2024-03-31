import numpy as np
from scipy.stats import norm, t


class DistributionEstimator:
    def __init__(self, vector):
        self.vector = vector

    def estimate_pdf(self, dist, stdev=None, df=None):
        """
        Estimate the probability density function (PDF) of the given distribution.

        Parameters:
        - dist: The distribution type ('Uniform', 'Gaussian', or 'Student').
        - stdev: The standard deviation of the distribution (optional, used for 'Gaussian' and 'Student').
        - df: The degrees of freedom of the distribution (optional, used for 'Student').

        Returns:
        - pdf: The estimated PDF.
        """
        pdf = None

        if dist == 'Uniform':
            min_val = np.min(self.vector)
            max_val = np.max(self.vector)
            pdf = 1.0 / (max_val - min_val)
        elif dist == 'Gaussian':
            mean = np.mean(self.vector)
            std_dev = stdev if stdev is not None else np.std(self.vector)
            pdf = norm.pdf(self.vector, loc=mean, scale=std_dev)
        elif dist == 'Student':
            mean = np.mean(self.vector)
            std_dev = stdev if stdev is not None else np.std(self.vector)
            df = df if df is not None else 5  # default degrees of freedom
            pdf = t.pdf(self.vector, df, loc=mean, scale=std_dev)

        return pdf

    def estimate_cdf(self, dist, stdev=None, df=None):
        """
        Estimate the cumulative distribution function (CDF) of the given distribution.

        Parameters:
        - dist: The distribution type ('Uniform', 'Gaussian', or 'Student').
        - stdev: The standard deviation of the distribution (optional, used for 'Gaussian' and 'Student').
        - df: The degrees of freedom of the distribution (optional, used for 'Student').

        Returns:
        - cdf: The estimated CDF.
        """
        cdf = None

        if dist == 'Uniform':
            cdf = (self.vector - np.min(self.vector)) / (np.max(self.vector) - np.min(self.vector))
        elif dist == 'Gaussian':
            mean = np.mean(self.vector)
            std_dev = stdev if stdev is not None else np.std(self.vector)
            cdf = norm.cdf(self.vector, loc=mean, scale=std_dev)
        elif dist == 'Student':
            mean = np.mean(self.vector)
            std_dev = stdev if stdev is not None else np.std(self.vector)
            df = df if df is not None else 5  # default degrees of freedom
            cdf = t.cdf(self.vector, df, loc=mean, scale=std_dev)

        return cdf

    def estimate_likelihood(self, dist, stdev=None, df=None):
        pdf = self.estimate_pdf(dist, stdev, df)
        likelihood = np.prod(pdf)
        return likelihood


def main():
    # https://johannesbuchner.github.io/UltraNest/example-outliers.html
    values = np.array([15, 4, 2, 11, 1, -2, -1, -14, -39, -3])
    values_lo = np.array([7, 16, 6, 3, 6, 5, 10, 6, 11, 13])
    values_hi = np.array([7, 15, 8, 3, 6, 6, 10, 7, 14, 14])

    n_data = len(values)
    samples = []
    for i in range(n_data):
        # draw normal random points
        u = np.random.normal(size=400)
        v = values[i] + np.where(u < 0, u * values_lo[i], u * values_hi[i])
        samples.append(v)

    input_vector = np.array(samples)
    estimator = DistributionEstimator(input_vector)

    distributions = [
        {'dist': 'Uniform', 'stdev': None, 'df': None},
        {'dist': 'Gaussian', 'stdev': 1.0, 'df': None},
        {'dist': 'Student', 'stdev': 1.0, 'df': 5}
    ]

    for dist in distributions:
        likelihood = estimator.estimate_likelihood(dist['dist'], dist['stdev'], dist['df'])
        dist['likelihood'] = likelihood

    print(distributions)


if __name__ == "__main__":
    main()
