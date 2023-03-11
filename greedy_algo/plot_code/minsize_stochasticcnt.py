import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import get_params

expName = os.path.basename(__file__)[:-3]
stochastic_cnts = []
min_ds = []
for file in os.listdir(f"../logs/{expName}"):
    file_path = os.path.join(f"../logs/{expName}", file)
    file_params = get_params(file)
    stochastic_cnts.append(file_params['stochastic_value'])
    stats = json.load(open(file_path, 'r'))
    min_ds.append(stats['Num-Time-Slots'])

stochastic_cnts = np.array(stochastic_cnts)
min_ds = np.array(min_ds)
arg_sorted = np.argsort(stochastic_cnts)
stochastic_cnts, min_ds = stochastic_cnts[arg_sorted], min_ds[arg_sorted]

plt.figure()
plt.plot(stochastic_cnts, min_ds, marker='o')
plt.xlabel('Stochastic Sampling Size')
plt.ylabel('Minimized Dataset Size')
plt.title(f"Variation in size of minimized dataset\ntau={file_params['threshTau']}, collectTill={file_params['threshCollectTill']}, train_len={file_params['train_len']}")
plt.savefig(f'../plots/nll/{expName}.png')
plt.close('all')