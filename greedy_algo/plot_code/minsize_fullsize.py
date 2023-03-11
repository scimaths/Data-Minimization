import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import get_params

expName = os.path.basename(__file__)[:-3]
full_ds = []
min_ds = []
for file in os.listdir(f"../logs/{expName}"):
    file_path = os.path.join(f"../logs/{expName}", file)
    file_params = get_params(file)
    full_ds.append(file_params['train_len'])
    stats = json.load(open(file_path, 'r'))
    min_ds.append(stats['Num-Time-Slots'])

full_ds = np.array(full_ds)
min_ds = np.array(min_ds)
arg_sorted = np.argsort(full_ds)
full_ds, min_ds = full_ds[arg_sorted], min_ds[arg_sorted]

plt.figure()
plt.plot(full_ds, min_ds, marker='o')
plt.xlabel('Total Train Dataset Size')
plt.ylabel('Minimized Dataset Size')
plt.title(f"Variation in size of minimized dataset\ntau={file_params['threshTau']}, collectTill={file_params['threshCollectTill']}, stochastic_cnt={file_params['stochastic_value']}")
plt.savefig('../plots/nll/minsize_fullsize.png')
plt.close('all')

plt.figure()
plt.plot(full_ds, min_ds/full_ds, marker='o')
plt.xlabel('Total Train Dataset Size')
plt.ylabel('Minimized Dataset Size (Ratio)')
plt.title(f"Variation in size of minimized dataset\ntau={file_params['threshTau']}, collectTill={file_params['threshCollectTill']}, stochastic_cnt={file_params['stochastic_value']}")
plt.savefig('../plots/nll/minsize_fullsize_ratio.png')
plt.close('all')