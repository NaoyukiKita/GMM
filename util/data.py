import numpy as np
from scipy.stats import multivariate_normal as mnormal
from numpy.random import default_rng

def make_data(sizes, means, vars, rng=default_rng()):
    sample = []
    for k in range(len(sizes)):
        sample.append(mnormal.rvs(means[k], vars[k], sizes[k]))
    sample = np.concatenate(sample)
    sample = rng.permutation(sample, axis=0)

    return sample
