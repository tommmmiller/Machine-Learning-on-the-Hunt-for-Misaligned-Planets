import pandas as pd
import numpy as np
from scipy.stats import qmc

# Load metadata
meta_df = pd.read_csv("metadata.csv")

# Extract relevant parameters
params = meta_df[["inc", "mass", "azimuth", "phi"]].drop_duplicates()

# Latin Hypercube Sampling
sampler = qmc.LatinHypercube(d=params.shape[1])
lhs_sample = sampler.random(n=5)   # pick 5 sims

# Scale to parameter ranges
lhs_scaled = qmc.scale(lhs_sample, params.min(), params.max())

# Find closest actual simulations
chosen_sims = []
for row in lhs_scaled:
    dists = np.linalg.norm(params.values - row, axis=1)
    chosen_sims.append(params.iloc[dists.argmin()])

print("Held-out simulations:")
print(pd.DataFrame(chosen_sims))

