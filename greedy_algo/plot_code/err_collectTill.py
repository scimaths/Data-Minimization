import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import get_params

expName = os.path.basename(__file__)[:-3]
collectTills_1 = []
collectTills_2 = []
for file in os.listdir(f"../logs/{expName}"):
    file_path = os.path.join(f"../logs/{expName}", file)
    file_params = get_params(file)
    stats = json.load(open(file_path, 'r'))
    if file_params['mode'] == 1:
        collectTills_1.append((file_params['threshCollectTill'], stats['Error']/len(stats['Actual'])))
    else:
        collectTills_2.append((file_params['threshCollectTill'], stats['Error']/len(stats['Actual'])))

collectTills = np.array(list(map(lambda x: x[0], sorted(collectTills_1))))
errors_1 = np.array(list(map(lambda x: x[1], sorted(collectTills_1))))
errors_2 = np.array(list(map(lambda x: x[1], sorted(collectTills_2))))

plt.figure()
plt.plot(collectTills, errors_1, marker='o', color='red', label='Using original set')
plt.plot(collectTills, errors_2, marker='o', color='blue', label='Using minimized set')
plt.xlabel('Parameter collectTill')
plt.ylabel('Error in Prediction')
plt.legend()
plt.title(f"Variation in prediction error\ntrain_len={file_params['train_len']}, tau={file_params['threshTau']}, stochastic_cnt={file_params['stochastic_value']}")
plt.savefig(f'../plots/nll/{expName}.png')
plt.close('all')