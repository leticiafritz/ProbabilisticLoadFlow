import os
import sys

from copulae import NormalCopula
import numpy as np
import pandas as pd

load_det = pd.read_csv(os.path.dirname(sys.argv[0]) + "/curve_load.csv", header=0, sep=';')
print(load_det.shape)
#np.random.seed(8)
#data = np.random.normal(size=(300, 8))
cop = NormalCopula(68)
cop.fit(load_det)

print(cop.random(10))  # simulate random number

# getting parameters
p = cop.params

# cop.params = ...  # you can override parameters too, even after it's fitted!

# get a summary of the copula. If it's fitted, fit details will be present too
print(cop.summary())

# overriding parameters, for Elliptical Copulae, you can override the correlation matrix
cop[:] = np.eye(8)  # in this case, this will be equivalent to an Independent Copula