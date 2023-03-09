from __future__ import annotations
import torch
import numpy as np
from nll import History

class ReverseModel(torch.nn.Module):
    def __init__(self, history: History, omega, specific_indices=None, final_T=None, lambda_mu=10, lambda_alpha=15):
        super().__init__()
        self.history = np.sort(history.time_slots).reshape(1, -1)
        self.num_time_slots = len(specific_indices) if specific_indices is not None else len(history.time_slots)
        self.history_length = self.history.shape[1] - 1
        epsilon = 1
        self.final_T = final_T
        if not final_T:
            self.final_T = np.max(history.time_slots) + epsilon

        self.times = torch.Tensor(np.tile(history.time_slots.reshape((1, -1)), (self.history_length + 1))). \
                    flatten()[1:].view(self.history_length, self.history_length + 2)[:,:-1].reshape(self.history_length + 1, self.history_length)
        if specific_indices is not None:
            self.times = self.times[specific_indices, :]
        assert self.times.shape == (self.num_time_slots, self.history_length)

        self.mu = torch.nn.Parameter(torch.Tensor([[1]] * self.num_time_slots))
        self.alpha = torch.nn.Parameter(torch.Tensor([[1]] * self.num_time_slots))
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
    def __init__(self, stochastic_elements, threshRemoveTill, threshTau, final_T):
        super().__init__()
        self.omega = 2
        self.num_epochs = 3
        self.final_T = final_T
        self.stochastic_elements = stochastic_elements
        self.threshRemoveTill = threshRemoveTill
        self.threshTau = threshTau
        self.likelihood_data = []
        self.mu_data = []
        self.alpha_data = []

    def do_forward(self, history: History, specific_elements=None):
        model = ReverseModel(history, omega=self.omega, specific_indices=specific_elements, final_T=self.final_T)
        optim = torch.optim.Adam(
            model.parameters(), lr=1e-4, betas=(0.9, 0.999))
        last_mu = torch.zeros_like(model.mu.data)
        last_alpha = torch.zeros_like(model.alpha.data)

        idx = 0
        while (idx < self.num_epochs):
            optim.zero_grad()
            output = model.forward()
            output.sum().backward()
            optim.step()
            change = (last_mu-model.mu)**2 + (last_alpha-model.alpha)**2
            
            model.alpha.data = torch.maximum(torch.nan_to_num(model.alpha.data), torch.Tensor([1e-3]))
            model.mu.data = torch.maximum(torch.nan_to_num(model.mu.data), torch.Tensor([1e-3]))

            last_mu = model.mu.data
            last_alpha = model.alpha.data
            idx += 1
        return model.mu.data, model.alpha.data, output

    def greedy_algo(self, history: History, stochastic_gradient):
        b = history.time_slots.shape[0]
        current_history = history

        _, _, curr_history_score = self.do_forward(History(np.concatenate(([0], history.time_slots), axis=0)), specific_elements=[0])
        last_score = curr_history_score[0].item()

        while (current_history.time_slots.shape[0] > 1):
            if stochastic_gradient:
                stochastic_idxs = np.random.choice(
                    current_history.time_slots.shape[0], self.stochastic_elements)

                mu, alpha, output = self.do_forward(current_history, stochastic_idxs)
            else:
                mu, alpha, output = self.do_forward(current_history)

            if (output.min() > last_score + self.threshTau) and (current_history.time_slots.shape[0] < self.threshRemoveTill * b):
                # print("Rejected", pending_history[current_history.time_slots.argmin(
                # )], 'Score: ', output.min().item())
                break

            self.likelihood_data.append(output.min().item())
            self.mu_data.append(mu[:,0])
            self.alpha_data.append(alpha[:,0])

            last_score = output.min().item()

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
        mu, alpha, _ = self.do_forward(History(np.concatenate(([0], new_history.time_slots), axis=0)), [0])
        curr = 0
        error = 0
        actual = []
        pred = []
        while (curr < next_time_slot.shape[0]):
            value = self.predict(mu, alpha, new_history)
            actual.append(next_time_slot[curr])
            pred.append(value.item())
            error += (value-next_time_slot[curr])**2
            new_history.add(next_time_slot[curr])
            curr += 1

        return error, actual, pred

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Mode")
    parser.add_argument("--StocGrad")
    parser.add_argument("--StocValue")
    parser.add_argument("--TrainLen")
    parser.add_argument("--TestLen")
    parser.add_argument("--ThreshTau")
    parser.add_argument("--ThreshRemoveTill")
    args = parser.parse_args()
    
    train_len = int(args.TrainLen)
    test_len = int(args.TestLen)

    train_data = None
    with open('data_exp16_train.npy', 'rb') as f:
        train_data = list(np.load(f))
    train_data_history = History(train_data[:train_len])

    test_data = None
    with open('data_exp16_test.npy', 'rb') as f:
        test_data = list(np.load(f))
    test_data_history = History(train_data[train_len:test_len+train_len])

    setting1 = Setting1(int(args.StocValue), float(args.ThreshRemoveTill), float(args.ThreshTau), final_T=np.max(train_data_history.time_slots))

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
        error, actual, pred = setting1.mode_2(
            train_data_history, test_data_history.time_slots, stochastic_gradient)

        # with open("likelihood-mode-2", 'w') as f:
        #     f.write("\n".join(map(str, setting1.likelihood_data)))

        # with open("alpha-mode-2", 'w') as f:
        #     f.write("\n".join(map(str, setting1.alpha_data)))

        # with open("mu-mode-2", 'w') as f:
        #     f.write("\n".join(map(str, setting1.mu_data)))

    import json
    with open(f'logs/reverse_mode-{mode}-stochastic_gradient-{stochastic_gradient}-stochastic_value-{args.StocValue}-threshRemoveTill-{args.ThreshRemoveTill}-threshTau-{args.ThreshTau}-train_len-{train_len}-test_len-{test_len}.json', 'wb') as f:
        data = {"Error": error.item(), "Actual": actual, "Pred": pred}
        obj = json.dumps(data) + "\n"
        json_bytes = obj.encode('utf-8')
        f.write(json_bytes)