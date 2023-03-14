from __future__ import annotations
import torch
import numpy as np


class History:
    def __init__(self, time_slots: list[int] = []):
        self.time_slots = np.array(time_slots)

    def __len__(self):
        return len(self.time_slots)

    def add(self, time_slot: int):
        self.time_slots = np.concatenate(
            (self.time_slots, [time_slot]), axis=0)


class Model(torch.nn.Module):
    def __init__(self, history: History, next_time_slot: np.ndarray, omega, mu, alpha, final_T=None, lambda_mu=10, lambda_alpha=15):
        super().__init__()

        self.history = np.sort(history.time_slots).reshape(1, -1)
        self.next_time_slot = np.sort(next_time_slot).reshape(-1, 1)

        self.num_time_slots = self.next_time_slot.shape[0]
        self.history_length = self.history.shape[1] + 1
        self.omega = omega
        self.lambda_mu = lambda_mu
        self.lambda_alpha = lambda_alpha

        epsilon = 1
        self.final_T = final_T
        if not final_T:
            self.final_T = np.max(self.next_time_slot) + epsilon

        self.times = torch.Tensor(np.concatenate(
            (np.tile(self.history, (self.num_time_slots, 1)), self.next_time_slot), axis=1))
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
    def __init__(self, num_stochastic_elements: int, threshCollectTill: float, threshTau: float, final_T: float, omega=2, init_num_epochs=300, lr=1e-4, epoch_decay=0.6, budget=0.5):
        super().__init__()
        self.omega = omega
        self.num_epochs = init_num_epochs
        self.epoch_decay = epoch_decay
        self.lr = lr

        self.final_T = final_T
        self.num_stochastic_elements = num_stochastic_elements
        self.threshCollectTill = threshCollectTill
        self.threshTau = threshTau
        self.budget = budget

        self.likelihood_data = []
        self.mu_data = []
        self.alpha_data = []
        self.indexes_added = []

    def do_forward(self, history: History, next_time_slot, mu=None, alpha=None):
        device = 'cuda:0'
        model = Model(history, next_time_slot,
                      self.omega, mu, alpha, final_T=self.final_T).to(device)
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

            # Prevent negative values
            model.alpha.data = torch.maximum(torch.nan_to_num(
                model.alpha.data), torch.Tensor([1e-3])).to(device)
            model.mu.data = torch.maximum(torch.nan_to_num(
                model.mu.data), torch.Tensor([1e-3])).to(device)

            last_mu = model.mu.data.to('cpu')
            last_alpha = model.alpha.data.to('cpu')
            idx += 1
        return last_mu, last_alpha, output.to('cpu')

    def greedy_algo(self, history: History, next_time_slot: np.ndarray, stochastic_gradient: bool):
        total_num_time_slots: int = history.time_slots.shape[0] + \
            next_time_slot.shape[0]
        current_history: History = history
        pending_history: np.ndarray = next_time_slot

        _, _, curr_history_score = self.do_forward(
            History(history.time_slots[:-1]), history.time_slots[-1:])
        last_score = curr_history_score.min().item()
        last_mu = None
        last_alpha = None

        while (pending_history.shape[0] > 0):
            if stochastic_gradient:
                # :TODO fix random.choice to be random.sample
                new_pending_history_indxs = np.random.choice(
                    pending_history.shape[0], self.num_stochastic_elements)

                mu, alpha, output = self.do_forward(
                    current_history, pending_history[new_pending_history_indxs], last_mu, last_alpha)
            else:
                mu, alpha, output = self.do_forward(
                    current_history, pending_history, last_mu, last_alpha)

            if (current_history.time_slots.shape[0] > self.threshCollectTill * self.budget * total_num_time_slots) and (output.min() > last_score + self.threshTau):
                break

            last_score = output.min().item()
            last_mu = mu
            last_alpha = alpha
            self.num_epochs = max(100, self.num_epochs * self.epoch_decay)

            self.likelihood_data.append(output.min().item())
            self.mu_data.append(mu[:, 0])
            self.alpha_data.append(alpha[:, 0])
            self.indexes_added.append(new_pending_history_indxs[output.argmin()])

            if stochastic_gradient:
                current_history.time_slots = np.concatenate(
                    (current_history.time_slots, [pending_history[new_pending_history_indxs][output.argmin()]]), axis=0)
                pending_history = np.concatenate(
                    [pending_history[:new_pending_history_indxs[output.argmin()]], pending_history[new_pending_history_indxs[output.argmin()]+1:]])
            else:
                current_history.time_slots = np.concatenate(
                    (current_history.time_slots, [pending_history[output.argmin()]]), axis=0)
                pending_history = np.concatenate(
                    [pending_history[:output.argmin()], pending_history[output.argmin()+1:]])

            if current_history.time_slots.shape[0] >= total_num_time_slots * self.budget:
                break

        return current_history

    def predict(self, mu, alpha, history: History):
        last_time = history.time_slots[-1]
        lambda_max = mu + alpha * \
            np.exp((last_time - history.time_slots) * -1 * self.omega).sum()
        while (True):
            u = np.random.uniform(0, 1)
            last_time = last_time - (np.log(1-u) / lambda_max)
            u2 = np.random.uniform(0, 1)
            value = (mu + alpha * np.exp((last_time - history.time_slots)
                     * -1 * self.omega).sum()) / lambda_max
            if u2 <= value:
                break
        return last_time

    def mode_1(self, history: History, next_time_slot: np.ndarray):
        mu, alpha, _ = self.do_forward(
            History(history.time_slots[:-1]), history.time_slots[-1:])

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

    def mode_2(self, history: History, next_time_slot, stochastic_gradient: bool):
        new_history: History = self.greedy_algo(
            History(history.time_slots[:1]), history.time_slots[1:], stochastic_gradient)

        new_history.time_slots = np.sort(new_history.time_slots)
        new_history_len = new_history.__len__()

        mu, alpha, _ = self.do_forward(
            History(new_history.time_slots[:-1]), new_history.time_slots[-1:])

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

        return error, actual, pred, new_history_len


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Mode")
    parser.add_argument("--StocGrad")
    parser.add_argument("--StocValue")
    parser.add_argument("--TrainLen")
    parser.add_argument("--TestLen")
    parser.add_argument("--ThreshTau")
    parser.add_argument("--ThreshCollectTill")
    parser.add_argument("--Budget")
    args = parser.parse_args()

    train_len = int(args.TrainLen)
    new_history_len = train_len
    test_len = int(args.TestLen)
    mode = int(args.Mode)
    stochastic_gradient = bool(args.StocGrad)
    thresh_collect_till = float(args.ThreshCollectTill)
    stochastic_elements = int(args.StocValue)
    thresh_tau = float(args.ThreshTau)
    budget = float(args.Budget)

    omega = 2
    init_num_epochs = 300
    lr = 0.0001
    epoch_decay = 0.6

    train_data = None
    with open('data_exp16_train.npy', 'rb') as f:
        train_data = list(np.load(f))
    train_data_history = History(train_data[:train_len])

    test_data = None
    with open('data_exp16_test.npy', 'rb') as f:
        test_data = list(np.load(f))
    test_data_history = History(train_data[train_len:test_len+train_len])

    final_T = np.max(train_data_history.time_slots)
    setting1 = Setting1(stochastic_elements,
                        thresh_collect_till, thresh_tau, final_T, omega, init_num_epochs, lr, epoch_decay, budget)

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

        # with open("likelihood-mode-2-0.6", 'a') as f:
        #     f.write("\n".join(map(str, setting1.likelihood_data)))

        # with open("indexes-added-mode-2", 'a') as f:
        #     f.write("\n".join(map(str, setting1.indexes_added)))

        # with open("alpha-mode-2", 'w') as f:
        #     f.write("\n".join(map(str, setting1.alpha_data)))

        # with open("mu-mode-2", 'w') as f:
        #     f.write("\n".join(map(str, setting1.mu_data)))

        # with open(f'degree-of-minimization', 'a') as f:
        #     f.write(f'{train_len}, {new_history_len}, {error}\n')

    import json
    with open(f'logs/mode-{mode}-stochastic_gradient-{stochastic_gradient}-stochastic_value-{args.StocValue}-threshCollectTill-{args.ThreshCollectTill}-budget-{args.Budget}-threshTau-{args.ThreshTau}-train_len-{train_len}-test_len-{test_len}.json', 'wb') as f:
        data = {"Error": error.item(), "Actual": actual, "Pred": pred, "Degree of Minimization": new_history_len}
        obj = json.dumps(data) + "\n"
        json_bytes = obj.encode('utf-8')
        f.write(json_bytes)