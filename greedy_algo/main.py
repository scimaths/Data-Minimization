from __future__ import annotations
import torch
import numpy as np


class History:
    def __init__(self, time_slots: list[int] = []):
        self.time_slots = torch.Tensor([time_slots])


class Model(torch.nn.Module):
    def __init__(self, history, next_time_slot, omega):
        super().__init__()
        self.history = history.time_slots.reshape((1, -1))
        self.next_time_slot = next_time_slot.reshape((-1, 1))
        self.num_time_slots = self.next_time_slot.shape[0]
        self.history_length = self.history.shape[1] + 1
        self.times = torch.cat((self.history.expand(
            self.next_time_slot.size(0), -1), self.next_time_slot), dim=1)
        assert self.times.shape == (self.num_time_slots, self.history_length)
        self.mu = torch.nn.Parameter(torch.Tensor(
            [[1] * self.num_time_slots]))
        self.alpha = torch.nn.Parameter(torch.Tensor(
            [[1] * self.num_time_slots]))
        self.omega = omega

    def forward(self):
        times_delta = torch.maximum(self.times.unsqueeze(
            2) - self.times.unsqueeze(1), torch.Tensor([0])).reshape((self.num_time_slots, self.history_length, self.history_length))
        times_delta[times_delta == 0] = np.inf
        times_delta *= self.omega
        mu_sq = self.mu**2
        alpha_sq = self.alpha**2
        summed = torch.log(mu_sq.T + alpha_sq.T *
                           torch.exp(-times_delta).sum(dim=1)).sum(dim=1).reshape((self.num_time_slots, 1))
        values = summed - (alpha_sq.T/self.omega)*(1-torch.exp(-self.omega *
                                                               (self.next_time_slot-self.times))).sum(dim=1).reshape((-1, 1)) - (mu_sq.T*self.next_time_slot)
        return -values


class Setting1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.Tensor([1]))
        self.alpha = torch.nn.Parameter(torch.Tensor([1]))
        self.omega = 20
        self.num_epochs = 3

    def excitation_intensity(self, history: History, t: int):
        exponents = (t - history.time_slots)
        return self.mu + self.alpha * torch.sum((torch.exp(exponents(exponents > 0) * self.omega)))

    def do_forward(self, history: History, next_time_slot):
        size = next_time_slot.shape[1]
        model = Model(history, next_time_slot, self.omega)
        output = model.forward()
        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        last_mu = torch.Tensor([0.5] * size).reshape((-1, 1))
        last_alpha = torch.Tensor([0.5] * size).reshape((-1, 1))

        idx = 0
        while (idx < self.num_epochs):
            sgd.zero_grad()
            output = model.forward()
            output.backward(gradient=torch.tensor([[1.] * size]).T)
            sgd.step()
            change = (last_mu-model.mu)**2 + (last_alpha-model.alpha)**2
            last_mu = model.mu
            last_alpha = model.alpha
            idx += 1
        return model.mu, model.alpha, output

    def test_do_forward(self):
        history = History([0, 0.25, 0.5, 0.75, 1])
        print('history', history.time_slots, history.time_slots.shape)
        next_time_slot = torch.Tensor([[1.01, 1.5, 1.75, 2.0]])
        print('next_time_slot', next_time_slot, next_time_slot.shape)
        mu, alpha, output = self.do_forward(history, next_time_slot)
        print('chosen time_slot', output)

    def greedy_algo(self, history: History, next_time_slot):
        current_history = history
        pending_history = next_time_slot

        while (pending_history.shape[1] > 0):
            mu, alpha, output = self.do_forward(
                current_history, pending_history)
            current_history.time_slots = torch.cat(
                (current_history.time_slots, pending_history[:, output.argmin()].reshape((1, -1))), dim=1)
            pending_history = pending_history[:, output.argmin()+1:]
            print(current_history.time_slots, pending_history)

    def test_greedy_algo(self):
        history = History([0, 0.25, 0.5, 0.75, 1])
        print('history', history.time_slots, history.time_slots.shape)
        next_time_slot = torch.Tensor([[1.01, 1.75, 2.0]])
        print('next_time_slot', next_time_slot, next_time_slot.shape)
        self.greedy_algo(history, next_time_slot)


if __name__ == '__main__':
    setting1 = Setting1()
    setting1.test_do_forward()
