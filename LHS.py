import numpy as np
from scipy.stats import qmc

sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
sample = sampler.random(n=10)
l_bounds = [1,0]
u_bounds = [15, 25]
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
sample_disc= qmc.discrepancy(sample)


print(sample_scaled, sample_disc)