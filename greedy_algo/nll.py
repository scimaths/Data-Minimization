from __future__ import annotations
import torch
import numpy as np
# from tick.plot import plot_point_process
# from tick.hawkes import SimuHawkes, HawkesKernelSumExp


class History:
    def __init__(self, time_slots: list[int] = []):
        self.time_slots = np.array(time_slots)


class Model(torch.nn.Module):
    def __init__(self, history, next_time_slot, omega, lambda_mu=10, lambda_alpha=15):
        super().__init__()
        self.history = history.time_slots.reshape(1, -1)
        self.next_time_slot = next_time_slot.reshape(-1, 1)
        self.num_time_slots = self.next_time_slot.shape[0]
        self.history_length = self.history.shape[1] + 1
        epsilon = 1
        self.final_T = np.max(self.next_time_slot) + epsilon

        self.times = torch.Tensor(np.concatenate(
            (np.tile(self.history, (self.num_time_slots, 1)), self.next_time_slot), axis=1))
        assert self.times.shape == (self.num_time_slots, self.history_length)

        self.mu = torch.nn.Parameter(torch.Tensor([[1]] * self.num_time_slots))
        self.alpha = torch.nn.Parameter(
            torch.Tensor([[1]] * self.num_time_slots))
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
        return -values #+ (self.alpha**2)*self.lambda_alpha + (self.mu**2)*self.lambda_mu


class Setting1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.Tensor([1]))
        self.alpha = torch.nn.Parameter(torch.Tensor([1]))
        self.omega = 1
        self.num_epochs = 50

    def excitation_intensity(self, history: History, t: int):
        exponents = (t - history.time_slots)
        return self.mu + self.alpha * torch.sum((torch.exp(exponents(exponents > 0) * self.omega)))

    def do_forward(self, history: History, next_time_slot):
        size = next_time_slot.shape[0]
        model = Model(history, next_time_slot, self.omega)
        optim = torch.optim.Adam(
            model.parameters(), lr=0.1, betas=(0.9, 0.999))
        last_mu = torch.zeros_like(model.mu.data)
        last_alpha = torch.zeros_like(model.alpha.data)

        idx = 0
        while (idx < self.num_epochs):
            optim.zero_grad()
            output = model.forward()
            # output.backward(gradient = torch.ones(output.shape))
            output.sum().backward()
            optim.step()
            change = (last_mu-model.mu)**2 + (last_alpha-model.alpha)**2
            last_mu = model.mu
            last_alpha = model.alpha
            idx += 1
        return model.mu.data, model.alpha.data, output

    def test_do_forward(self, history, next_time_slot):
        # print('history', history.time_slots, history.time_slots.shape)
        print('next_time_slot', next_time_slot, next_time_slot.shape)
        mu, alpha, output = self.do_forward(history, next_time_slot)
        print('chosen time_slot', output)
        print('mu', mu, 'alpha', alpha)

    def greedy_algo(self, history: History, next_time_slot):
        current_history = history
        pending_history = next_time_slot

        _, _, curr_history_score = self.do_forward(
            History(history.time_slots[:-1]), history.time_slots[-1:])
        last_score = curr_history_score[0].item()
        print(history.time_slots, last_score)

        while (pending_history.shape[0] > 0):
            mu, alpha, output = self.do_forward(
                current_history, pending_history)
            print(mu, alpha)
            if output.min() > last_score:
                print("Rejected", pending_history[current_history.time_slots.argmin(
                )], 'Score: ', output.min().item())
                break
            last_score = output.min().item()
            current_history.time_slots = np.concatenate(
                (current_history.time_slots, [pending_history[output.argmin()]]), axis=0)
            pending_history = np.concatenate(
                [pending_history[:output.argmin()], pending_history[output.argmin()+1:]])
            print(current_history.time_slots, last_score)
        return current_history

    def test_greedy_algo(self, history, next_time_slot):
        # print('history', history.time_slots, history.time_slots.shape)
        # print('next_time_slot', next_time_slot, next_time_slot.shape)
        return self.greedy_algo(history, next_time_slot)

    def simulate_hawkes(self, mu, alpha, omega, num_time_stamps, run_time):
        # mu = 2
        # alpha_times_omega = alpha*omega
        # hawkes = SimuHawkes(end_time=run_time, verbose=False, baseline=np.array(
        #     [mu]), seed=1398, max_jumps=num_time_stamps)
        # kernel = HawkesKernelSumExp([alpha_times_omega], [omega])
        # hawkes.set_kernel(0, 0, kernel)

        # dt = 0.01
        # hawkes.track_intensity(dt)
        # hawkes.simulate()
        # timestamps = hawkes.timestamps
        timestamps = np.load('timestamps.npy')
        print(timestamps.size)
        return timestamps


if __name__ == '__main__':
    setting1 = Setting1()
    timestamps = setting1.simulate_hawkes(
        mu=0.1, alpha=1, omega=1, num_time_stamps=1000, run_time=4000)
    # intensity = hawkes.tracked_intensity
    # intensity_times = hawkes.intensity_tracked_times
    history = History(timestamps[:-1])
    next_time_slot = timestamps[-1:]
    # next_time_slot = np.arange(50, 71, 10)
    setting1.test_do_forward(history, next_time_slot)
