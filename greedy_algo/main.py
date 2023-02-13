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
            self.next_time_slot.size(0), -1), self.next_time_slot), dim=1).reshape((self.num_time_slots, self.history_length))
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

    def get_gradient(self):
        times_delta = torch.maximum(self.times.unsqueeze(
            2) - self.times.unsqueeze(1), torch.Tensor([0])).reshape((self.num_time_slots, self.history_length, self.history_length))
        times_delta[times_delta == 0] = np.inf
        times_delta *= self.omega
        mu_sq = self.mu**2
        alpha_sq = self.alpha**2
        summed = (1/torch.log(mu_sq.T + alpha_sq.T *
                           torch.exp(-times_delta).sum(dim=1))).sum(dim=1).reshape((self.num_time_slots, 1))
        dmu = summed - (self.next_time_slot)


        summed = (torch.exp(-times_delta).sum(dim=1)/torch.log(mu_sq.T + alpha_sq.T *
                           torch.exp(-times_delta).sum(dim=1))).sum(dim=1).reshape((self.num_time_slots, 1))
        dalpha = summed - (1/self.omega)*(1-torch.exp(-self.omega *
                                                               (self.next_time_slot-self.times))).sum(dim=1).reshape((-1, 1)) 
        
        return torch.cat((dmu, dalpha), dim=1)

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
            output.backward(gradient=torch.tensor([[1.] * size]).T)#model.get_gradient())
            sgd.step()
            change = (last_mu-model.mu)**2 + (last_alpha-model.alpha)**2
            last_mu = model.mu
            last_alpha = model.alpha
            # print(idx, last_alpha, last_mu, change, output)
            idx += 1
        # print(output)
        # print(model.mu, model.alpha)
        return model.mu, model.alpha, output

    # def likelihood_maximum(self, history: History, next_time_slot: torch.Tensor, mu, alpha):
    #     mu = mu.reshape((-1, 1))
    #     alpha = alpha.reshape((-1, 1))

    #     exponent = self.alpha / self.omega * \
    #         torch.sum(torch.exp((next_time_slot.T - history.time_slots)
    #                   * -1 * self.omega), 1).reshape((-1, 1))
    #     updates = -1 * self.mu * next_time_slot.T + exponent + \
    #         torch.log(self.mu + self.omega * exponent)
    #     print(updates.T[0])
    #     return next_time_slot[torch.argmax(updates.T[0], dim=0)]

    # def test_likelihood_maximum(self):
    #     history = History([0, 0.25, 0.5, 0.75, 1])
    #     print('history', history.time_slots, history.time_slots.shape)
    #     next_time_slot = torch.Tensor([[1.01, 1.15, 1.25, 1.5, 1.75, 2.0]])
    #     print('next_time_slot', next_time_slot, next_time_slot.shape)
    #     print('chosen time_slot', self.likelihood_maximum(history, next_time_slot))

    def test_do_forward(self):
        history = History([0, 0.25, 0.5, 0.75, 1])
        print('history', history.time_slots, history.time_slots.shape)
        next_time_slot = torch.Tensor([[1.01, 1.15, 1.25, 1.5, 1.75, 2.0]])
        print('next_time_slot', next_time_slot, next_time_slot.shape)
        mu, alpha, output = self.do_forward(history, next_time_slot)
        print('chosen time_slot', output)


if __name__ == '__main__':
    setting1 = Setting1()
    setting1.test_do_forward()
