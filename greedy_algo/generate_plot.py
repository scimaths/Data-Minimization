import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import itertools

variables = ['budget', 'coll_till', 'train_len', 'stoch_val', 'mode', 'for_back']
budgets = [0.2, 0.4, 0.6, 0.8, 1.0]
coll_till = [0.25, 0.5, 0.75, 1.0]
train_len = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
stoch_val = [50, 100]
mode = [2, 1]
for_back = ['forward', 'backward']

parser = argparse.ArgumentParser()
parser.add_argument("--plot_type", default="Error")
parser.add_argument("--plot_var", required=True, choices=variables)
parser.add_argument("--budget", nargs="+", default=[0.6], type=float, choices=budgets)
parser.add_argument("--coll_till", nargs="+", default=[0.5], type=float, choices=coll_till)
parser.add_argument("--train_len", nargs="+", default=[1000], type=int, choices=train_len)
parser.add_argument("--stoch_val", nargs="+", default=[100], type=int, choices=stoch_val)
parser.add_argument("--mode", nargs="+", default=[2], type=int, choices=mode)
parser.add_argument("--for_back", nargs="+", default=["forward"], type=str, choices=for_back)

args = parser.parse_args()

choices = [budgets, coll_till, train_len, stoch_val, mode, for_back]
args_var = [args.budget, args.coll_till, args.train_len, args.stoch_val, args.mode, args.for_back]

data_pickle = pickle.load(open('plots/err.pkl', 'rb'))

ref_strings = []
indices = []
variable_idx = variables.index(args.plot_var)
length = len(choices[variable_idx])
for idx, var in enumerate(variables):
    if var != args.plot_var:
        indices.append([choices[idx].index(val) for val in args_var[idx]])
        if len(args_var[idx]) == 1:
            ref_strings.append(f"{var}={args_var[idx][0]}".replace(" ", ""))

plt.figure()
for combination in itertools.product(*indices):
    this_indices = [[val for _ in range(length)] for val in combination]
    label_str = []
    for idx in range(len(combination)+1):
        if len(args_var[idx]) <= 1:
            continue
        if idx < variable_idx:
            label_str.append(f"{variables[idx]}={choices[idx][combination[idx]]}")
        elif idx > variable_idx:
            label_str.append(f"{variables[idx]}={choices[idx][combination[idx-1]]}")
    this_indices = this_indices[:variable_idx] + [list(range(length))] + this_indices[variable_idx:]
    values = data_pickle[tuple(this_indices)]
    plt.plot(choices[variable_idx], values, label=",".join(label_str), marker='o')
plt.xlabel(f'{args.plot_var}')
plt.ylabel('error')
plt.legend()
plt.title(f'Plot of error vs {args.plot_var}')
plt.savefig(f"plots/{args.plot_var}_{'_'.join(ref_strings)}.png")
plt.close('all')