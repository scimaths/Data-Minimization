import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import get_params

expName = os.path.basename(__file__)[:-3]
stochastic_cnts_1 = []
stochastic_cnts_2 = []
for file in os.listdir(f"../logs/{expName}"):
    file_path = os.path.join(f"../logs/{expName}", file)
    file_params = get_params(file)
    stats = json.load(open(file_path, 'r'))
    if file_params['mode'] == 1:
        stochastic_cnts_1.append((file_params['stochastic_value'], stats['Error']/len(stats['Actual'])))
    else:
        stochastic_cnts_2.append((file_params['stochastic_value'], stats['Error']/len(stats['Actual'])))

stochastic_cnts = np.array(list(map(lambda x: x[0], sorted(stochastic_cnts_1))))
errors_1 = np.array(list(map(lambda x: x[1], sorted(stochastic_cnts_1))))
errors_2 = np.array(list(map(lambda x: x[1], sorted(stochastic_cnts_2))))

plt.figure()
plt.plot(stochastic_cnts, errors_1, marker='o', color='red', label='Using original set')
plt.plot(stochastic_cnts, errors_2, marker='o', color='blue', label='Using minimized set')
plt.xlabel('Stochastic Sampling Size')
plt.ylabel('Error in Prediction')
plt.legend()
plt.title(f"Variation in prediction error\ntau={file_params['threshTau']}, collectTill={file_params['threshCollectTill']}, train_len={file_params['train_len']}")
plt.savefig(f'../plots/nll/{expName}.png')
plt.close('all')