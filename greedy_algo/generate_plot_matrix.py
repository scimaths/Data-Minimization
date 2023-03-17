import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import argparse
import os, pickle

budgets = [0.2, 0.4, 0.6, 0.8, 1.0]
coll_till = [0.25, 0.5, 0.75, 1.0]
train_len = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
stoch_val = [50, 100]
mode = [2, 1]
for_back = ['forward', 'backward']

def get_params(fileName):
    hyphen_idx = [idx for idx in range(len(fileName)) if fileName[idx] == "-"]
    print(fileName, hyphen_idx)
    return {
        'nll': 'backward' if fileName[0] == 'r' else 'forward',
        'mode': int(fileName[hyphen_idx[0]+1:hyphen_idx[1]]),
        'stochastic_gradient': True if fileName[hyphen_idx[2]+1:hyphen_idx[3]] == "True" else False,
        'stochastic_value': int(fileName[hyphen_idx[4]+1:hyphen_idx[5]]),
        'threshCollectTill': float(fileName[hyphen_idx[6]+1:hyphen_idx[7]]),
        'budget': float(fileName[hyphen_idx[8]+1:hyphen_idx[9]]),
        'threshTau': int(fileName[hyphen_idx[10]+1:hyphen_idx[11]]),
        'train_len': int(fileName[hyphen_idx[12]+1:hyphen_idx[13]]),
        'test_len': int(fileName[hyphen_idx[14]+1:-5])
    }

err_array = np.zeros((len(budgets), len(coll_till), len(train_len), len(stoch_val), len(mode), len(for_back)))
len_array = np.zeros((len(budgets), len(coll_till), len(train_len), len(stoch_val), len(mode), len(for_back)))
pred_array = np.zeros((len(budgets), len(coll_till), len(train_len), len(stoch_val), len(mode), len(for_back), 500))
true_array = np.zeros((len(budgets), len(coll_till), len(train_len), len(stoch_val), len(mode), len(for_back), 500))
err_array[:] = np.nan
len_array[:] = np.nan
pred_array[:] = np.nan
true_array[:] = np.nan

file_loc = "/mnt/home/ashwinr/ashwinr-abirde/Data-Minimization/greedy_algo/logs"
for file_name in os.listdir(file_loc):
    file_path = os.path.join(file_loc, file_name)
    file_data = json.load(open(file_path, 'r'))
    file_params = get_params(file_name)
    
    bud_index = budgets.index(file_params['budget'])
    collect_index = coll_till.index(file_params['threshCollectTill'])
    train_len_index = train_len.index(file_params['train_len'])
    stoch_val_index = stoch_val.index(file_params['stochastic_value'])
    mode_index = mode.index(file_params['mode'])
    for_back_index = for_back.index(file_params['nll'])
    err_array[bud_index, collect_index, train_len_index, stoch_val_index, mode_index, for_back_index] = file_data['Error']
    len_array[bud_index, collect_index, train_len_index, stoch_val_index, mode_index, for_back_index] = file_data['Degree of Minimization']
    pred_array[bud_index, collect_index, train_len_index, stoch_val_index, mode_index, for_back_index, :] = np.array(file_data['Pred'])
    true_array[bud_index, collect_index, train_len_index, stoch_val_index, mode_index, for_back_index, :] = np.array(file_data['Actual'])

with open('plots/err.pkl', 'wb') as f:
    pickle.dump(err_array, f)
with open('plots/len.pkl', 'wb') as f:
    pickle.dump(len_array, f)
with open('plots/pred.pkl', 'wb') as f:
    pickle.dump(pred_array, f)
with open('plots/true.pkl', 'wb') as f:
    pickle.dump(true_array, f)