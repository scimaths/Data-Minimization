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


class Model_Sensitivity(torch.nn.Module):
    def __init__(self, history, val_history, next_time_slot, omega, mu, alpha, final_T=None, lambda_mu=10, lambda_alpha=15, device='cuda:0', arg_min=None):
        super().__init__()
        
        # self.history = history.time_slots.reshape(1, -1) # (1, H)
        # self.next_time_slot = next_time_slot.reshape(-1, 1) # (T, 1) 
        self.history = np.sort(history.time_slots).reshape(1, -1)
        self.next_time_slot = np.sort(next_time_slot).reshape(-1, 1)

        self.num_time_slots = self.next_time_slot.shape[0] # T
        self.history_length = self.history.shape[1] + 1 # (H+1)
        self.omega = omega
        self.lambda_mu = lambda_mu
        self.lambda_alpha = lambda_alpha

        epsilon = 1
        self.final_T = final_T
        if not final_T:
            self.final_T = np.max(self.next_time_slot) + epsilon

        # Comments made for T=2; H=2;

        self.times = np.concatenate(
            (np.tile(self.history, (self.num_time_slots, 1)), self.next_time_slot), axis=1) # (T, H) -> (T, H+1)
        
        # H1 H2 T1
        # H1 H2 T2

        self.times = np.repeat(
            self.times, self.history_length * self.num_time_slots, axis=0) # (T*((H+1)*T), H+1)
        
        # H1 H2 T1
        # H1 H2 T1
        # H1 H2 T1
        # H1 H2 T1
        # H1 H2 T1
        # H1 H2 T1
        # H1 H2 T2
        # H1 H2 T2
        # H1 H2 T2
        # H1 H2 T2
        # H1 H2 T2
        # H1 H2 T2

        row_pos = np.arange(len(self.times)) # (T*(H+1)*T,)
        col_pos = np.tile(np.arange(self.history_length),
                          self.num_time_slots**2) # (T*(H+1)*T,)


        self.times[tuple([row_pos, col_pos])] = np.tile(
            np.repeat(self.next_time_slot[:, 0], self.history_length), self.num_time_slots) # (T*(H+1)*T,H+1)

        # T1 H2 T1
        # H1 T1 T1
        # H1 H2 T1
        # T2 H2 T1
        # H1 T2 T1
        # H1 H2 T2
        # T1 H2 T2
        # H1 T1 T2
        # H1 H2 T1
        # T2 H2 T2
        # H1 T2 T2
        # H1 H2 T2
        
        self.times = np.sort(self.times, axis=1) 
        difference = self.times[:, 1:] - self.times[:, :-1]
        min_diff = np.min(difference, axis=1)
        self.times = torch.Tensor(
            # self.times[np.where(self.times[:, -1] - self.times[:, -2] > 1e-3)])#.to(device) # ()
            self.times[np.where(min_diff > 1e-5)])#.to(device) # ()

        # T1 H2 T1 x
        # H1 T1 T1 x
        # H1 H2 T1
        # T2 H2 T1
        # H1 T2 T1
        # H1 H2 T2
        # T1 H2 T2
        # H1 T1 T2
        # H1 H2 T1
        # T2 H2 T2 x
        # H1 T2 T2 x
        # H1 H2 T2

        self.num_params = ((self.num_time_slots-1)*self.history_length+1)*self.num_time_slots
        assert self.times.shape == (self.num_params, self.history_length)

        if mu is None:
            self.mu = torch.nn.Parameter(torch.Tensor([[1]] * self.num_params))
        else:
            self.mu = torch.nn.Parameter(torch.Tensor([[mu[arg_min].item()]] * self.num_params))
            # self.mu = torch.nn.Parameter(torch.Tensor(mu))
        
        if alpha is None:
            self.alpha = torch.nn.Parameter(torch.Tensor([[1]] * self.num_params))
        else:
            self.alpha = torch.nn.Parameter(torch.Tensor([[alpha[arg_min].item()]] * self.num_params))
            # self.alpha = torch.nn.Parameter(torch.Tensor(alpha))
        self.omega = omega
        self.lambda_mu = lambda_mu
        self.lambda_alpha = lambda_alpha

        times_delta = self.times.unsqueeze(2) - self.times.unsqueeze(1)
        times_delta[times_delta <= 0] = np.inf
        times_delta *= self.omega
        self.times = self.times.to(device)

        assert times_delta.shape == (
            self.num_params, self.history_length, self.history_length)
        
        self.times_delta_exp = torch.exp(-times_delta).sum(dim=2)
        self.times_delta_exp = self.times_delta_exp.to(device)

    def forward(self):
        summed = torch.log(self.mu + self.alpha * self.times_delta_exp).sum(dim=1).unsqueeze(1)
        alpha_term = (self.alpha/self.omega)*(1-torch.exp(-self.omega *
                                                          (self.final_T-self.times))).sum(dim=1).unsqueeze(1)
        values = summed - alpha_term - (self.mu*self.final_T)
        return -values + (self.alpha**2)*self.lambda_alpha + (self.mu**2)*self.lambda_mu
        # return -values

class Setting1(torch.nn.Module):
    def __init__(self, sensitivity_weight=0.5):
        super().__init__()
        self.omega = 2
        self.num_epochs = 100
        self.sensitivity_weight = sensitivity_weight

    def do_forward(self, history: History, next_time_slot):
        device = 'cuda:3'
        model = Model_Sensitivity(history, next_time_slot,
                      self.omega, mu, alpha, final_T=self.final_T, device=device).to(device)
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
                model.alpha.data), torch.Tensor([1e-3]).to(device))
            model.mu.data = torch.maximum(torch.nan_to_num(
                model.mu.data), torch.Tensor([1e-3]).to(device))

            last_mu = model.mu.data.to('cpu')
            last_alpha = model.alpha.data.to('cpu')
            idx += 1
        
        num_time_slots = next_time_slot.shape[0]
        history_len = len(history) + 1
        output = output.cpu().detach().numpy()[:, 0]
        actual_indices = np.arange(
            0, len(output), history_len * num_time_slots + 1)

        actual_out = output[actual_indices]
        other_out = np.delete(output, actual_indices).reshape(
            num_time_slots, history_len * (num_time_slots - 1))
        actual_other_delta = np.expand_dims(actual_out, axis=1) - other_out

        total_score = actual_out + \
            actual_other_delta.sum(axis=1) * self.sensitivity_weight
        return last_mu, last_alpha, total_score

    def greedy_algo(self, history: History, next_time_slot):
        current_history = history
        pending_history = next_time_slot

        _, _, curr_history_score = self.do_forward(
            History(history.time_slots[:-1]), history.time_slots[-1:])
        last_score = curr_history_score[0].item()

        while (pending_history.shape[0] > 0):
            _, _, output = self.do_forward(
                current_history, pending_history)
            if output.min() > last_score:
                print("Rejected", pending_history[current_history.time_slots.argmin(
                )], 'Score: ', output.min().item())
                break
            last_score = output.min().item()
            current_history.time_slots = np.concatenate(
                (current_history.time_slots, [pending_history[output.argmin()]]), axis=0)
            pending_history = np.concatenate(
                [pending_history[:output.argmin()], pending_history[output.argmin()+1:]])
        return current_history

    def predict(self, mu, alpha, history: History):
        last_time = history.time_slots[-1]
        lambda_max = mu + alpha * \
            np.exp((last_time - history.time_slots) * -1 * self.omega).sum()
        while (True):
            u = np.random.uniform(0, 1)
            last_time = last_time - (np.log(1-u) / lambda_max)
            u2 = np.random.uniform(0, 1)
            if u2 <= (mu + alpha * np.exp((last_time - history.time_slots) * -1 * self.omega).sum()) / lambda_max:
                break
        return last_time

    def mode_1(self, history: History, next_time_slot):
        mu, alpha, _ = self.do_forward(
            History(history.time_slots[:-1]), history.time_slots[-1:])
        print(mu, alpha)
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

        print(error)
        print(actual)
        print(pred)

    def mode_2(self, history: History, next_time_slot):
        new_history = self.greedy_algo(
            History(history.time_slots[:2]), history.time_slots[2:])
        print(new_history.time_slots[-1])
        mu, alpha, _ = self.do_forward(
            History(new_history.time_slots[:-1]), new_history.time_slots[-1:])
        print(mu, alpha)
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

        print(error)
        print(actual)
        print(pred)


if __name__ == '__main__':
    setting1 = Setting1()
    train_data = None
    with open('data_exp16_train.npy', 'rb') as f:
        train_data = list(np.load(f))
    train_data_history = History(train_data[:100])

    test_data = None
    with open('data_exp16_test.npy', 'rb') as f:
        test_data = list(np.load(f))
    test_data_history = History(train_data[100:120])

    # Mode 1
    # Without Data Minimization
    # Use complete training data to get function params
    # And then predict on test data
    # Parameters not updated in between predictions
    setting1.mode_1(
        train_data_history, test_data_history.time_slots)

    # Mode 2
    # With Data Minimization
    # Use complete training data to get function params and minimized history
    # And then predict on test data
    # Parameters not updated in between predictions
    setting1.mode_2(
        train_data_history, test_data_history.time_slots)
