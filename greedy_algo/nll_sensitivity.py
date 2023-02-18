from __future__ import annotations
import torch
import numpy as np
from tick.plot import plot_point_process
from tick.hawkes import SimuHawkes, HawkesKernelSumExp
import Hawkes as hk


class History:
    def __init__(self, time_slots: list[int] = []):
        self.time_slots = np.array(time_slots)
    
    def len(self):
        return len(self.time_slots)


class Model(torch.nn.Module):
    def __init__(self, history, next_time_slot, omega, lambda_mu=0.5, lambda_alpha=0.5):
        super().__init__()
        self.history = history.time_slots.reshape(1, -1)
        self.next_time_slot = next_time_slot.reshape(-1, 1)
        self.num_time_slots = self.next_time_slot.shape[0]
        self.history_length = self.history.shape[1] + 1
        
        self.times = np.concatenate((np.tile(self.history, (self.num_time_slots, 1)), self.next_time_slot), axis=1)
        self.times = np.repeat(self.times, self.history_length * self.num_time_slots, axis=0)
        row_pos = np.arange(len(self.times))
        col_pos = np.tile(np.arange(self.history_length), self.num_time_slots**2)
        self.times[tuple([row_pos, col_pos])] = np.tile(np.repeat(self.next_time_slot[:,0], self.history_length), self.num_time_slots)
        self.times = np.sort(self.times, axis=1)
        self.times = torch.Tensor(self.times[np.where(self.times[:, -1] - self.times[:, -2] > 1e-5)])
        
        self.num_params = ((self.num_time_slots-1)*self.history_length + 1)*self.num_time_slots
        assert self.times.shape == (self.num_params, self.history_length)
        
        self.mu = torch.nn.Parameter(torch.Tensor([[1]] * self.num_params))
        self.alpha = torch.nn.Parameter(torch.Tensor([[1]] * self.num_params))
        self.omega = omega
        self.lambda_mu = 10
        self.lambda_alpha = 10

    def forward(self):
        times_delta = self.times.unsqueeze(2) - self.times.unsqueeze(1)
        times_delta[times_delta <= 0] = np.inf
        times_delta *= self.omega
        assert times_delta.shape == (self.num_params, self.history_length, self.history_length)

        final_T = self.times[:, -1].unsqueeze(dim=1)
        summed = torch.log(self.mu + self.alpha * torch.exp(-times_delta).sum(dim=2)).sum(dim=1).unsqueeze(1)
        alpha_term = (self.alpha/self.omega)*(1-torch.exp(-self.omega*(final_T-self.times))).sum(dim=1).unsqueeze(1)
        values = summed - alpha_term - (self.mu*final_T)
        return -values + (self.alpha**2)*self.lambda_alpha + (self.mu**2)*self.lambda_mu

class Setting1(torch.nn.Module):
    def __init__(self, sensitivity_weight=0.5):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.Tensor([1]))
        self.alpha = torch.nn.Parameter(torch.Tensor([1]))
        self.omega = 2
        self.num_epochs = 50
        self.sensitivity_weight = sensitivity_weight

    def excitation_intensity(self, history: History, t: int):
        exponents = (t - history.time_slots)
        return self.mu + self.alpha * torch.sum((torch.exp(exponents(exponents > 0) * self.omega)))

    def do_forward(self, history: History, next_time_slot):
        model = Model(history, next_time_slot, self.omega)
        optim = torch.optim.Adam(model.parameters(), lr=0.05, betas=(0.9, 0.999))
        last_mu = torch.zeros_like(model.mu.data)
        last_alpha = torch.zeros_like(model.alpha.data)

        num_time_slots = next_time_slot.shape[0]
        history_len = history.len() + 1

        idx = 0
        while (idx < self.num_epochs):
            optim.zero_grad()
            output = model.forward()
            # output.backward(gradient = torch.ones(output.shape))
            output.sum().backward()
            optim.step()
            change = (last_mu-model.mu)**2 + (last_alpha-model.alpha)**2
            last_mu = model.mu.data
            last_alpha = model.alpha.data
            idx += 1

        output = output.detach().numpy()[:,0]
        actual_indices = np.arange(0, len(output), history_len * num_time_slots + 1)
        
        actual_out = output[actual_indices]
        other_out = np.delete(output, actual_indices).reshape(num_time_slots, history_len * (num_time_slots - 1))
        actual_other_delta = np.expand_dims(actual_out, axis=1) - other_out
        
        total_score = actual_out + actual_other_delta.sum(axis=1) * self.sensitivity_weight
        return total_score

    def test_do_forward(self, history, next_time_slot):
        print('history', history.time_slots, history.time_slots.shape)
        print('next_time_slot', next_time_slot, next_time_slot.shape)
        mu, alpha, output = self.do_forward(history, next_time_slot)
        print('chosen time_slot', output)
        print('mu', mu, 'alpha', alpha)

    def greedy_algo(self, history: History, next_time_slot):
        current_history = history
        pending_history = next_time_slot
        num_time_slots = len(next_time_slot)

        while (pending_history.shape[0] > 0):
            scores = self.do_forward(current_history, pending_history)
            current_history.time_slots = np.concatenate((current_history.time_slots, np.array([pending_history[scores.argmin()]])), axis=0)
            pending_history = pending_history[scores.argmin()+1:]
            print("-"*50)
            print(scores.tolist())
            print("-"*50)
            print(len(current_history.time_slots), len(pending_history))

    def test_greedy_algo(self, history, next_time_slot):
        print('history', history.time_slots, history.time_slots.shape[0])
        print('next_time_slot', next_time_slot, next_time_slot.shape[0])
        self.greedy_algo(history, next_time_slot)
    
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
        para = {'mu':mu, 'alpha':alpha, 'beta':omega}
        model = hk.simulator().set_kernel('exp').set_baseline('const').set_parameter(para)
        itv = [0,run_time] # the observation interval
        T = model.simulate(itv)
        timestamps = np.array(T[:num_time_stamps])
        return timestamps

if __name__ == '__main__':
    setting1 = Setting1()
    timestamps = setting1.simulate_hawkes(mu = 0.1, alpha = 1, omega = 1, num_time_stamps = 1000, run_time = 4000)
    # intensity = hawkes.tracked_intensity
    # intensity_times = hawkes.intensity_tracked_times
    history = History(timestamps[:-1])
    next_time_slot = timestamps[-1:]
    # next_time_slot = np.arange(50, 71, 10)
    setting1.test_do_forward(history, next_time_slot)
