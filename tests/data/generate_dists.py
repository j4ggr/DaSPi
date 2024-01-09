import pandas as pd

from scipy import stats
from pathlib import Path

savedir = Path(__file__).parent

params = dict(loc=2, scale=1, random_state=72)
dists = {
    'norm': params,
    'chi2': params | {'df': 2},
    'foldnorm': params | {'c': 1.5},
    'rayleigh': params,
    'gamma': params | {'a': 1.0},
    'wald': params,
    'expon': params,
    'logistic': params, 
    'lognorm': params | {'s': 0.5}}

size = 10
df = pd.DataFrame()
for dist, kw in dists.items():
    df[dist] = getattr(stats, dist).rvs(size=size, **kw)
df.to_csv(savedir/f'dists_{size}-samples.csv', index=False)

size = 25
df = pd.DataFrame()
for dist, kw in dists.items():
    df[dist] = getattr(stats, dist).rvs(size=size, **kw)
df.to_csv(savedir/f'dists_{size}-samples.csv', index=False)
