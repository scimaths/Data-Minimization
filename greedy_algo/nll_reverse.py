from __future__ import annotations
import torch
import numpy as np
from nll import History, seed_everything

class ReverseModel(torch.nn.Module):
    def __init__(self, history: History, omega, specific_indices=None, mu=None, alpha=None, final_T=None, lambda_mu=10, lambda_alpha=15, device='cuda:0'):
        super().__init__()
        self.history = np.sort(history.time_slots).reshape(1, -1)
        self.num_time_slots = len(specific_indices) if specific_indices is not None else len(history.time_slots)
        self.history_length = self.history.shape[1] - 1
        epsilon = 1
        self.final_T = final_T
        if not final_T:
            self.final_T = np.max(history.time_slots) + epsilon

        self.times = torch.Tensor(np.tile(history.time_slots.reshape((1, -1)), (self.history_length + 1))). \
                    flatten()[1:].view(self.history_length, self.history_length + 2)[:,:-1].reshape(self.history_length + 1, self.history_length).to(device)
        if specific_indices is not None:
            self.times = self.times[specific_indices, :]
        assert self.times.shape == (self.num_time_slots, self.history_length)

        if mu is None:
            self.mu = torch.nn.Parameter(
                torch.Tensor([[1]] * self.num_time_slots))
        else:
            self.mu = torch.nn.Parameter(torch.Tensor(mu))

        if alpha is None:
            self.alpha = torch.nn.Parameter(
                torch.Tensor([[1]] * self.num_time_slots))
        else:
            self.alpha = torch.nn.Parameter(torch.Tensor(alpha))
        self.omega = omega
        self.lambda_mu = lambda_mu
        self.lambda_alpha = lambda_alpha

        times_delta = self.times.unsqueeze(2) - self.times.unsqueeze(1)
        times_delta[times_delta <= 0] = np.inf
        times_delta *= self.omega
        if torch.sum(times_delta < 0) > 0:
            print(torch.sum(times_delta < 0))
        assert times_delta.shape == (
            self.num_time_slots, self.history_length, self.history_length)

        self.times_delta_exp = torch.exp(-times_delta).sum(dim=2)

    def forward(self):
        summed = torch.log(self.mu + self.alpha *
                           self.times_delta_exp).sum(dim=1).unsqueeze(1)
        alpha_term = (self.alpha/self.omega)*(1-torch.exp(-self.omega *
                                                          (self.final_T-self.times))).sum(dim=1).unsqueeze(1)
        values = summed - alpha_term - (self.mu*self.final_T)
        return -values


class Setting1(torch.nn.Module):
    def __init__(self, stochastic_elements, threshRemoveTill, threshTau, final_T, omega=2, init_num_epochs=300, lr=1e-4, epoch_decay=0.6, budget=0.5, device='cuda:0'):
        super().__init__()
        self.omega = omega
        self.num_epochs = init_num_epochs
        self.epoch_decay = epoch_decay
        self.final_T = final_T
        self.stochastic_elements = stochastic_elements
        self.threshRemoveTill = threshRemoveTill
        self.threshTau = threshTau
        self.budget = budget
        self.likelihood_data = []
        self.mu_data = []
        self.alpha_data = []
        self.lr = lr
        self.device = device

    def do_forward(self, history: History, specific_elements=None, mu=None, alpha=None):
        model = ReverseModel(history, omega=self.omega, specific_indices=specific_elements,  mu=mu, alpha=alpha, final_T=self.final_T).to(self.device)
        optim = torch.optim.Adam(
            model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        
        last_mu = model.mu.data.to('cpu')
        last_alpha = model.alpha.data.to('cpu')

        idx = 0
        while (idx < self.num_epochs):
            optim.zero_grad()
            output = model.forward()
            output.sum().backward()
            optim.step()
            
            model.alpha.data = torch.maximum(torch.nan_to_num(model.alpha.data), torch.Tensor([1e-3]).to(self.device))
            model.mu.data = torch.maximum(torch.nan_to_num(model.mu.data), torch.Tensor([1e-3]).to(self.device))

            last_mu = model.mu.data.to('cpu')
            last_alpha = model.alpha.data.to('cpu')
            idx += 1
        return last_mu, last_alpha, output.to('cpu')

    def greedy_algo(self, history: History, stochastic_gradient):
        b = history.time_slots.shape[0]
        current_history = history

        _, _, curr_history_score = self.do_forward(History(np.concatenate(([0], history.time_slots), axis=0)), specific_elements=[0])
        last_score = curr_history_score[0].item()
        last_mu = None
        last_alpha = None

        while (current_history.time_slots.shape[0] > max(1, (1 - self.budget) * b)):
            if stochastic_gradient:
                stochastic_idxs = np.random.choice(
                    current_history.time_slots.shape[0], self.stochastic_elements)

                mu, alpha, output = self.do_forward(current_history, stochastic_idxs, last_mu, last_alpha)
            else:
                mu, alpha, output = self.do_forward(current_history, None, last_mu, last_alpha)

            if (output.min() > last_score + self.threshTau) and (current_history.time_slots.shape[0] < (1 - self.threshRemoveTill * self.budget) * b):
                # print("Rejected", pending_history[current_history.time_slots.argmin(
                # )], 'Score: ', output.min().item())
                break

            self.likelihood_data.append(output.min().item())
            self.mu_data.append(mu[:,0])
            self.alpha_data.append(alpha[:,0])

            print(current_history.time_slots.shape[0], output.min().item())

            last_score = output.min().item()
            last_mu = mu
            last_alpha = alpha
            self.num_epochs = max(self.num_epochs * self.epoch_decay, 100)

            if stochastic_gradient:
                current_history.time_slots = np.concatenate(
                    (current_history.time_slots[:stochastic_idxs[output.argmin()]], current_history.time_slots[stochastic_idxs[output.argmin()]+1:]), axis=0)
            else:
                current_history.time_slots = np.concatenate(
                    (current_history.time_slots[:output.argmin()], current_history.time_slots[output.argmin()+1:]), axis=0)
            
        return current_history

    def predict(self, mu, alpha, history: History):
        last_time = history.time_slots[-1]
        lambda_max = mu + alpha * \
            np.exp((last_time - history.time_slots) * -1 * self.omega).sum()
        while (True):
            u = np.random.uniform(0, 1)
            last_time = last_time - (np.log(1-u) / lambda_max)
            u2 = np.random.uniform(0, 1)
            value = (mu + alpha * np.exp((last_time - history.time_slots) * -1 * self.omega).sum()) / lambda_max
            if u2 <= value:
                break
        return last_time

    def mode_1(self, history: History, next_time_slot):
        mu, alpha, _ = self.do_forward(History(np.concatenate(([0], history.time_slots), axis=0)), [0])
        curr = 0
        error = 0
        actual = []
        pred = []
        np.random.seed(0)
        while (curr < next_time_slot.shape[0]):
            value = self.predict(mu, alpha, history)
            actual.append(next_time_slot[curr])
            pred.append(value.item())
            error += (value-next_time_slot[curr])**2
            history.add(next_time_slot[curr])
            curr += 1

        return error, actual, pred

    def mode_2(self, history: History, next_time_slot, stochastic_gradient):
        new_history: History = self.greedy_algo(history, stochastic_gradient)
        new_history.time_slots = np.sort(new_history.time_slots)
        new_history_len = new_history.__len__()
        mu, alpha, _ = self.do_forward(History(np.concatenate(([0], new_history.time_slots), axis=0)), [0])
        curr = 0
        error = 0
        actual = []
        pred = []
        np.random.seed(0)
        while (curr < next_time_slot.shape[0]):
            value = self.predict(mu, alpha, new_history)
            actual.append(next_time_slot[curr])
            pred.append(value.item())
            error += (value-next_time_slot[curr])**2
            new_history.add(next_time_slot[curr])
            curr += 1

        return error, actual, pred, new_history_len

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Mode")
    parser.add_argument("--StocGrad")
    parser.add_argument("--StocValue")
    parser.add_argument("--TrainLen")
    parser.add_argument("--TestLen")
    parser.add_argument("--Budget")
    parser.add_argument("--ThreshTau")
    parser.add_argument("--ThreshRemoveTill")
    args = parser.parse_args()

    seed_everything(0)
    
    train_len = int(args.TrainLen)
    test_len = int(args.TestLen)

    train_data = None
    with open('data_exp16_train.npy', 'rb') as f:
        train_data = list(np.load(f))
    train_data_history = History(train_data[:train_len])

    new_history_len = train_len

    test_data = None
    with open('data_exp16_test.npy', 'rb') as f:
        test_data = list(np.load(f))
    test_data_history = History(train_data[train_len:test_len+train_len])

    setting1 = Setting1(int(args.StocValue), float(args.ThreshRemoveTill), float(args.ThreshTau), final_T=np.max(train_data_history.time_slots), budget=float(args.Budget))

    mode = int(args.Mode)
    stochastic_gradient = bool(args.StocGrad)
    
    if mode == 1:
        # Mode 1
        # Without Data Minimization
        # Use complete training data to get function params
        # And then predict on test data
        # Parameters not updated in between predictions
        error, actual, pred = setting1.mode_1(
            train_data_history, test_data_history.time_slots)

    else:
        # Mode 2
        # With Data Minimization
        # Use complete training data to get function params and minimized history
        # And then predict on test data
        # Parameters not updated in between predictions
        error, actual, pred, new_history_len = setting1.mode_2(
            train_data_history, test_data_history.time_slots, stochastic_gradient)

        # with open("likelihood-mode-2-rev-0.6", 'a') as f:
        #     f.write("\n".join(map(str, setting1.likelihood_data)))

        # with open("indexes-added-mode-2-rev", 'a') as f:
        #     f.write("\n".join(map(str, setting1.indexes_added)))

        # with open("alpha-mode-2", 'w') as f:
        #     f.write("\n".join(map(str, setting1.alpha_data)))

        # with open("mu-mode-2", 'w') as f:
        #     f.write("\n".join(map(str, setting1.mu_data)))

        # with open(f'degree-of-minimization-rev', 'a') as f:
        #     f.write(f'{train_len}, {new_history_len}, {error}\n')

    import json
    with open(f'logs/reverse_mode-{mode}-stochastic_gradient-{stochastic_gradient}-stochastic_value-{args.StocValue}-threshRemoveTill-{args.ThreshRemoveTill}-budget-{args.Budget}-threshTau-{args.ThreshTau}-train_len-{train_len}-test_len-{test_len}.json', 'wb') as f:
        data = {"Error": error.item(), "Actual": actual, "Pred": pred, "Degree of Minimization": new_history_len}
        obj = json.dumps(data) + "\n"
        json_bytes = obj.encode('utf-8')
        f.write(json_bytes)