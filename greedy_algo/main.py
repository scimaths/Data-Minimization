from __future__ import annotations
import torch
import numpy as np


class History:
    def __init__(self, time_slots: list[int] = []):
        self.time_slots = torch.Tensor(time_slots)


class Model(torch.nn.Module):
    def __init__(self, history, next_time_slot, omega, lambda_mu=0.5, lambda_alpha=0.5):
        super().__init__()
        self.history = history.time_slots.reshape((1, -1))
        self.next_time_slot = next_time_slot.reshape((-1, 1))
        self.num_time_slots = self.next_time_slot.shape[0]
        self.history_length = self.history.shape[1] + 1
        
        self.times = torch.cat((self.history.expand(self.num_time_slots, -1), self.next_time_slot), dim=1)
        assert self.times.shape == (self.num_time_slots, self.history_length)
        
        self.mu = torch.nn.Parameter(torch.Tensor([[1]] * self.num_time_slots))
        self.alpha = torch.nn.Parameter(torch.Tensor([[1]] * self.num_time_slots))
        self.omega = omega
        self.lambda_mu = 10
        self.lambda_alpha = 10

    def forward(self):
        times_delta = self.times.unsqueeze(2) - self.times.unsqueeze(1)
        times_delta[times_delta <= 0] = np.inf
        times_delta *= self.omega
        assert times_delta.shape == (self.num_time_slots, self.history_length, self.history_length)

        summed = torch.log(self.mu + self.alpha * torch.exp(-times_delta).sum(dim=2)).sum(dim=1).unsqueeze(1)
        alpha_term = (self.alpha/self.omega)*(1-torch.exp(-self.omega*(self.next_time_slot-self.times))).sum(dim=1).unsqueeze(1)
        values = summed - alpha_term - (self.mu*self.next_time_slot)
        return -values + (self.alpha**2)*self.lambda_alpha + (self.mu**2)*self.lambda_mu

class Setting1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.Tensor([1]))
        self.alpha = torch.nn.Parameter(torch.Tensor([1]))
        self.omega = 2
        self.num_epochs = 50

    def excitation_intensity(self, history: History, t: int):
        exponents = (t - history.time_slots)
        return self.mu + self.alpha * torch.sum((torch.exp(exponents(exponents > 0) * self.omega)))

    def do_forward(self, history: History, next_time_slot):
        size = next_time_slot.shape[0]
        model = Model(history, next_time_slot, self.omega)
        sgd = torch.optim.SGD(model.parameters(), lr=0.005)
        last_mu = torch.Tensor([0.5] * size).reshape((-1, 1))
        last_alpha = torch.Tensor([0.5] * size).reshape((-1, 1))

        idx = 0
        while (idx < self.num_epochs):
            sgd.zero_grad()
            output = model.forward()
            print(output.tolist())
            # output.backward(gradient = torch.ones(output.shape))
            output.sum().backward()
            sgd.step()
            change = (last_mu-model.mu)**2 + (last_alpha-model.alpha)**2
            last_mu = model.mu
            last_alpha = model.alpha
            idx += 1
        return model.mu, model.alpha, output

    def test_do_forward(self):
        history = History(np.arange(10, 0.5))
        print('history', history.time_slots, history.time_slots.shape)
        next_time_slot = torch.Tensor([10.1, 10.2, 10.25])
        print('next_time_slot', next_time_slot, next_time_slot.shape)
        mu, alpha, output = self.do_forward(history, next_time_slot)
        print('chosen time_slot', output)

    def greedy_algo(self, history: History, next_time_slot):
        current_history = history
        pending_history = next_time_slot

        while (pending_history.shape[0] > 0):
            mu, alpha, output = self.do_forward(current_history, pending_history)
            current_history.time_slots = torch.cat((current_history.time_slots, torch.Tensor([pending_history[output.argmin()]])), dim=0)
            pending_history = pending_history[output.argmin()+1:]
            # print("-"*50)
            # print(mu.data.tolist(), output.tolist())
            # print("-"*50)
            # print(len(current_history.time_slots), len(pending_history))

    def test_greedy_algo(self):
        history = History(np.arange(0, 50, 10))
        # print('history', history.time_slots, history.time_slots.shape)
        next_time_slot = torch.Tensor(np.arange(50, 70, 10))
        # print('next_time_slot', next_time_slot, next_time_slot.shape)
        self.greedy_algo(history, next_time_slot)

if __name__ == '__main__':
    setting1 = Setting1()
    setting1.test_greedy_algo()
