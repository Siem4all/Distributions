import inline as inline
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Creating a series of data of in range of 1-50.
x = np.linspace(-5, 5, 10000)
mean = np.mean(x)
sd = np.std(x)
# colors
color_map = plt.get_cmap('tab10').colors

def normalPDF(x, mu, sigma):
    """ Calculates PDF of the normal distribution at point 'x'

    Parameters
    ==========

    x: float
        x value of the function
    mu: float
        expectation value of the normal distribution
    sigma: float
        standard deviation of the normal distribution

    Returns
    =======

    val: float
        Value of the PDF at point 'x'
    """

    return 1. / np.sqrt(2. * np.pi * sigma ** 2) * np.exp(-0.5 * ((x - mu) ** 2) / (sigma ** 2))


def studentPDF(x, nu):
    """ Calculates PDF of the Student's t-distribution at point 'x'

    Parameters
    ==========

    x: float
        x value of the function
    nu: int or float
        number of degrees of freedom

    Returns
    =======

    val: float
        Value of the PDF at point 'x'
    """

    return (gamma((nu + 1.) / 2.) / (np.sqrt(np.pi * nu) * gamma(nu / 2.))) * np.power((1. + x ** 2 / nu),
                                                                                       -(nu + 1.) / 2.)


plt.plot(x, studentPDF(x, nu=3.), lw=2.5, c=color_map[0], label="Student's: v=3")
plt.plot(x, normalPDF(x, mu=0, sigma=1), lw=2.5, c=color_map[6], dashes=[2, 2], label="Normal: $N(0,1)$")
plt.legend(loc='best', fontsize=15)
plt.show()
