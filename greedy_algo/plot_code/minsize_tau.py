import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import get_params

expName = os.path.basename(__file__)[:-3]
taus = []
min_ds = []
for file in os.listdir(f"../logs/{expName}"):
    file_path = os.path.join(f"../logs/{expName}", file)
    file_params = get_params(file)
    taus.append(file_params['threshTau'])
    stats = json.load(open(file_path, 'r'))
    min_ds.append(stats['Num-Time-Slots'])

taus = np.array(taus)
min_ds = np.array(min_ds)
arg_sorted = np.argsort(taus)
taus, min_ds = taus[arg_sorted], min_ds[arg_sorted]

plt.figure()
plt.plot(taus, min_ds, marker='o')
plt.xlabel('Parameter tau')
plt.ylabel('Minimized Dataset Size (out of 1000)')
plt.title(f"Variation in size of minimized dataset\ntrain_len={file_params['train_len']}, collectTill={file_params['threshCollectTill']}, stochastic_cnt={file_params['stochastic_value']}")
plt.savefig(f'../plots/nll/{expName}.png')
plt.close('all')