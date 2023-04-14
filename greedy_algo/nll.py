from __future__ import annotations
import torch
import pickle
import numpy as np
from nll_sensitivity import Model_Sensitivity
from nll_reverse import ReverseModel

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class History:
    def __init__(self, time_slots: list[int] = []):
        self.time_slots = np.array(time_slots)

    def __len__(self):
        return len(self.time_slots)

    def add(self, time_slot: int):
        self.time_slots = np.concatenate(
            (self.time_slots, [time_slot]), axis=0)

class Model(torch.nn.Module):
    def __init__(self, history: History, val_history: History, next_time_slot: np.ndarray, omega, mu, alpha, final_T=None, lambda_mu=10, lambda_alpha=15, device='cuda:1', arg_min=None):
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
        
        if val_history is not None:
            self.times = torch.Tensor(np.concatenate(
                (np.tile(self.history, (self.num_time_slots, 1)), self.next_time_slot), axis=1))
            self.times = torch.Tensor(np.concatenate(
                (self.times, np.tile(val_history.time_slots, (self.num_time_slots, 1))), axis=1)).to(device)
            assert self.times.shape == (self.num_time_slots, self.history_length + len(val_history))
        else:
            self.times = torch.Tensor(np.concatenate(
                (np.tile(self.history, (self.num_time_slots, 1)), self.next_time_slot), axis=1)).to(device)
            assert self.times.shape == (self.num_time_slots, self.history_length)

        if mu is None:
            self.mu = torch.nn.Parameter(
                torch.Tensor([[1]] * self.num_time_slots))
        else:
            self.mu = torch.nn.Parameter(torch.Tensor([[mu[arg_min].item()]] * self.num_time_slots))
            # self.mu = torch.nn.Parameter(torch.Tensor(mu))

        if alpha is None:
            self.alpha = torch.nn.Parameter(
                torch.Tensor([[1]] * self.num_time_slots))
        else:
            # self.alpha = torch.nn.Parameter(torch.Tensor(alpha))
            self.alpha = torch.nn.Parameter(torch.Tensor([[alpha[arg_min].item()]] * self.num_time_slots))
        self.omega = omega

        times_delta = self.times.unsqueeze(2) - self.times.unsqueeze(1)
        times_delta[times_delta <= 0] = np.inf
        times_delta *= self.omega 

        if val_history is not None:
            assert times_delta.shape == (
                self.num_time_slots, self.history_length + len(val_history), self.history_length + len(val_history))
        else:
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
    def __init__(self, num_stochastic_elements: int, threshCollectTill: float, threshTau: float, final_T: float, omega=1, init_num_epochs=300, lr=1e-4, epoch_decay=0.6, budget=0.5, sensitivity=0.1):
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
        self.sensitivity_weight = sensitivity

        self.likelihood_data = []
        self.mu_data = []
        self.alpha_data = []
        self.indexes_added = []

    def do_forward(self, history: History, val_history: History, next_time_slot, mu=None, alpha=None, arg_min=None):
        device = 'cuda:1'
        model = Model(history, val_history, next_time_slot,
                      self.omega, mu, alpha, final_T=self.final_T, device=device, arg_min=arg_min).to(device)
        optim = torch.optim.Adam(
            model.parameters(), lr=self.lr, betas=(0.9, 0.999),)
        # optim = torch.optim.SGD(model.parameters(), lr=self.lr)
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
        return last_mu, last_alpha, output.to('cpu')

    def do_forward_reverse(self, history: History, val_history: History, specific_elements=None, mu=None, alpha=None, arg_min=None):
        device = 'cuda:1'
        model = ReverseModel(history, val_history, specific_elements, 
                             self.omega, mu, alpha, final_T=self.final_T, device=device, arg_min=arg_min).to(device)
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
            
            model.alpha.data = torch.maximum(torch.nan_to_num(
                model.alpha.data), torch.Tensor([1e-3]).to(device))
            model.mu.data = torch.maximum(torch.nan_to_num(
                model.mu.data), torch.Tensor([1e-3]).to(device))

            last_mu = model.mu.data.to('cpu')
            last_alpha = model.alpha.data.to('cpu')
            idx += 1
        
        # prob_log = torch.zeros((len(last_mu),1))
        # mu = last_mu
        # alpha = last_alpha
        # omega = model.omega
        # train_data = model.times.to('cpu')
        # curr = train_data[:,0].reshape((-1,1))

        # for i in range(1, len(train_data)):
        #     inter = np.exp((train_data[:, i].reshape((-1,1)) - curr) * -1 * omega).reshape((-1,i))
        #     inter = inter.sum(axis=1).reshape((-1,1))
        #     prob_log += torch.log(mu + alpha * inter)
        #     prob_log -= mu * train_data[:, i].reshape((-1,1))
        #     prob_log += alpha/omega * (torch.exp(-1 * omega * (train_data[:, i].reshape((-1,1)) - curr[:, -1].reshape((-1,1)))) - 1) 
        #     curr = torch.cat((curr, train_data[:, i].reshape((-1,1))), axis=1)
        # likelihood_end_vals = -prob_log
        
        return last_mu, last_alpha, output.to('cpu') #torch.Tensor(likelihood_end_vals)

    def do_forward_sensitivity(self, history: History, val_history: History, next_time_slot, mu=None, alpha=None, arg_min=None):
        device = 'cuda:1'
        model = Model_Sensitivity(history, val_history, next_time_slot,
                      self.omega, mu, alpha, final_T=self.final_T, device=device, arg_min=arg_min).to(device)
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
        
        num_time_slots = len(next_time_slot)
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
        # print(total_score)
        
        return last_mu, last_alpha, total_score

    def greedy_algo(self, history: History, val_history: History, next_time_slot: np.ndarray, stochastic_gradient: bool):
        total_num_time_slots: int = history.time_slots.shape[0] + \
            next_time_slot.shape[0]
        current_history: History = history
        pending_history: np.ndarray = next_time_slot

        if len(history) == 0:
            last_score = np.inf
        else:
            _, _, curr_history_score = self.do_forward(
                History(history.time_slots[:-1]), None, history.time_slots[-1:])
            last_score = curr_history_score.min().item()
        last_mu = None
        arg_min = None
        last_alpha = None

        while (pending_history.shape[0] > 0):
            if stochastic_gradient:
                # :TODO fix random.choice to be random.sample
                new_pending_history_indxs = np.random.choice(
                    pending_history.shape[0], min(pending_history.shape[0], self.num_stochastic_elements), replace=False)

                mu, alpha, output = self.do_forward(
                    current_history, val_history, pending_history[new_pending_history_indxs], last_mu, last_alpha, arg_min)
            else:
                mu, alpha, output = self.do_forward(
                    current_history, val_history, pending_history, last_mu, last_alpha, arg_min)

            if (current_history.time_slots.shape[0] > self.threshCollectTill * self.budget * total_num_time_slots) and (output.min() > last_score + self.threshTau):
                break

            last_score = output.min().item()
            # print(last_score)
            last_mu = mu
            last_alpha = alpha
            arg_min = output.argmin()
            self.num_epochs = max(100, self.num_epochs * self.epoch_decay)

            self.likelihood_data.append(output.min().item())
            self.mu_data.append(mu[:, 0])
            self.alpha_data.append(alpha[:, 0])
            # self.indexes_added.append(new_pending_history_indxs[output.argmin()])

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
    
    def greedy_algo_with_tests(self, history: History, val_history: History, next_time_slot: np.ndarray, stochastic_gradient: bool):
        total_num_time_slots: int = history.time_slots.shape[0] + \
            next_time_slot.shape[0]
        current_history: History = history
        pending_history: np.ndarray = next_time_slot

        _, _, curr_history_score = self.do_forward(
            History(history.time_slots[:-1]), None, history.time_slots[-1:])
        last_score = curr_history_score.min().item()
        last_mu = None
        last_alpha = None
        arg_min = None
        histories = []

        while (pending_history.shape[0] > 0):
            if stochastic_gradient:
                # :TODO fix random.choice to be random.sample
                new_pending_history_indxs = np.random.choice(
                    pending_history.shape[0], self.num_stochastic_elements, replace=False)

                mu, alpha, output = self.do_forward_sensitivity(
                    current_history, val_history, pending_history[new_pending_history_indxs], last_mu, last_alpha, arg_min)
            else:
                mu, alpha, output = self.do_forward_sensitivity(
                    current_history, val_history, pending_history, last_mu, last_alpha, arg_min)

            if (current_history.time_slots.shape[0] > self.threshCollectTill * self.budget * total_num_time_slots) and (output.min() > last_score + self.threshTau):
                break

            last_score = output.min().item()
            # print(last_score)
            last_mu = mu
            last_alpha = alpha
            print(mu[output.argmin()], alpha[output.argmin()])
            arg_min = output.argmin()
            self.num_epochs = max(100, self.num_epochs * self.epoch_decay)

            self.likelihood_data.append(output.min().item())
            self.mu_data.append(mu[:, 0])
            self.alpha_data.append(alpha[:, 0])
            # self.indexes_added.append(new_pending_history_indxs[output.argmin()])

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

            if current_history.time_slots.shape[0] % 5 == 0:
                histories.append(current_history.time_slots)
                # print(current_history.time_slots.shape[0])

            if current_history.time_slots.shape[0] >= total_num_time_slots * self.budget:
                break
            
        with open('nll_sens_histories.pkl', 'wb') as f:
            pickle.dump(histories, f)

        return current_history

    def greedy_algo_with_reverse(self, history: History, val_history: History, stochastic_gradient: bool):

        total_num_time_slots: int = history.time_slots.shape[0]
        current_history: History = history

        _, _, curr_history_score = self.do_forward(
            History(history.time_slots[:-1]), None, history.time_slots[-1:])
        last_score = curr_history_score.min().item()
        last_mu = None
        last_alpha = None
        arg_min = None

        while (current_history.time_slots.shape[0] > max(1, (1 - self.budget) * total_num_time_slots)):
            if stochastic_gradient:
                # :TODO fix random.choice to be random.sample
                stochastic_idxs = np.random.choice(
                    current_history.time_slots.shape[0], self.num_stochastic_elements, replace=False)

                mu, alpha, output = self.do_forward_reverse(
                    current_history, val_history, stochastic_idxs, last_mu, last_alpha, arg_min)
            else:
                mu, alpha, output = self.do_forward_reverse(
                    current_history, None, None, last_mu, last_alpha, arg_min)

            if (current_history.time_slots.shape[0] < ( 1 - self.threshCollectTill * self.budget) * total_num_time_slots) and (output.min() > last_score + self.threshTau):
                break

            last_score = output.min().item()
            # print(last_score)
            last_mu = mu
            last_alpha = alpha

            model_temp = ReverseModel(current_history, val_history, None, mu=last_mu, alpha=last_alpha, final_T=val_history.time_slots[-1], arg_min=None)
            output = model_temp.forward()
            
            print(mu[output.argmin()], alpha[output.argmin()], output.argmin())
            print(mu[mu.argmin()], alpha[alpha.argmin()], mu.argmin(), alpha.argmin())
            print(mu[mu.argmax()], alpha[alpha.argmax()], mu.argmax(), alpha.argmax())
            arg_min = output.argmin()
            self.num_epochs = max(100, self.num_epochs * self.epoch_decay)

            self.likelihood_data.append(output.min().item())
            self.mu_data.append(mu[:, 0])
            self.alpha_data.append(alpha[:, 0])

            if stochastic_gradient:
                current_history.time_slots = np.concatenate(
                    (current_history.time_slots[:stochastic_idxs[output.argmin()]], current_history.time_slots[stochastic_idxs[output.argmin()] + 1:]), axis=0)
            else:
                current_history.time_slots = np.concatenate(
                    (current_history.time_slots[:output.argmin()], current_history.time_slots[output.argmin()+1:]), axis=0)
                # index_to_be_removed = np.random.randint(0, len(current_history.time_slots))
                # current_history.time_slots = np.concatenate(
                #     (current_history.time_slots[:index_to_be_removed], current_history.time_slots[index_to_be_removed+1:]), axis=0)
                # print(output.argmin(), output.min().item(), output[0].item())

            
            if current_history.time_slots.shape[0] >= total_num_time_slots * self.budget:
                break
            
        return current_history

    def predict(self, mu, alpha, history: History):
        last_time = history.time_slots[-1] + 1e-10
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

    def mode_1(self, history: History, next_time_slot: np.ndarray, seed=0):
        mu, alpha, _ = self.do_forward(
            History(history.time_slots[:-1]), history.time_slots[-1:])

        curr = 0
        error = 0
        actual = []
        pred = []

        np.random.seed(seed)
        while (curr < next_time_slot.shape[0]):
            value = self.predict(mu, alpha, history)
            actual.append(next_time_slot[curr])
            pred.append(value.item())
            error += (value-next_time_slot[curr])**2
            history.add(next_time_slot[curr])
            curr += 1

        return error, actual, pred
    
    def mode_3(self, history: History, next_time_slot: np.ndarray, seed=0):
        original_history_len = len(history.time_slots)
        mu, alpha, _ = self.do_forward(
            History(history.time_slots[-int(self.budget * len(history.time_slots)):-1]), history.time_slots[-1:])

        # print("Mu", mu, "\nAlpha", alpha)
        curr = 0
        error = 0
        actual = []
        pred = []

        np.random.seed(seed)
        while (curr < next_time_slot.shape[0]):
            value = self.predict(mu, alpha, history)
            actual.append(next_time_slot[curr])
            pred.append(value.item())
            error += (value-next_time_slot[curr])**2
            history.add(next_time_slot[curr])
            curr += 1

        return error, actual, pred, int(self.budget * original_history_len)

    def mode_4(self, history: History, next_time_slot, stochastic_gradient: bool, seed=0):
        original_history_len = len(history.time_slots)
        random_slots = history.time_slots[np.random.choice(list(range(original_history_len)), size=(int(original_history_len * self.budget),), replace=False)]
        mu, alpha, _ = self.do_forward(
            History(random_slots[:-1]), random_slots[-1:])

        # print("Mu", mu, "\nAlpha", alpha)
        curr = 0
        error = 0
        actual = []
        pred = []

        np.random.seed(seed)
        while (curr < next_time_slot.shape[0]):
            value = self.predict(mu, alpha, history)
            actual.append(next_time_slot[curr])
            pred.append(value.item())
            error += (value-next_time_slot[curr])**2
            history.add(next_time_slot[curr])
            curr += 1

        return error, actual, pred, int(self.budget * original_history_len)

    def mode_2(self, history: History, next_time_slot, stochastic_gradient: bool, seed=0):
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
        
        np.random.seed(seed)
        while (curr < next_time_slot.shape[0]):
            value = self.predict(mu, alpha, new_history)
            actual.append(next_time_slot[curr])
            pred.append(value.item())
            error += (value-next_time_slot[curr])**2
            new_history.add(next_time_slot[curr])
            curr += 1

        return error, actual, pred, new_history_len

    def test_histories(self, history_file, test_time_slots, seed=0):
        all_histories = pickle.load(open(history_file, 'rb'))
        for np_history in all_histories:
            history = History(np.sort(np_history))
            mu, alpha, _ = self.do_forward(History(history.time_slots[:-1]), history.time_slots[-1:])
            curr = 0
            error = 0
            actual = []
            pred = []

            np.random.seed(seed)
            while (curr < test_time_slots.shape[0]):
                value = self.predict(mu, alpha, history)
                actual.append(test_time_slots[curr])
                pred.append(value.item())
                error += (value-test_time_slots[curr])**2
                history.add(test_time_slots[curr])
                curr += 1
            print(error[0,0].item()/len(test_time_slots), len(np_history))

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
    parser.add_argument("--Seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.Seed)

    train_len = int(args.TrainLen)
    new_history_len = train_len
    test_len = int(args.TestLen)
    mode = int(args.Mode)
    stochastic_gradient = bool(args.StocGrad)
    thresh_collect_till = float(args.ThreshCollectTill)
    stochastic_elements = int(args.StocValue)
    thresh_tau = float(args.ThreshTau)
    budget = float(args.Budget)/    1000

    omega = 2
    init_num_epochs = 300
    lr = 0.01
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
            train_data_history, test_data_history.time_slots, args.Seed)
        print(error[0,0].item()/500)

    elif mode == 3:
        error, actual, pred, new_history_len = setting1.mode_3(
            train_data_history, test_data_history.time_slots, args.Seed)
        print(error[0,0].item()/500, new_history_len)
    
    elif mode == 4:
        error, actual, pred, new_history_len = setting1.mode_4(
            train_data_history, test_data_history.time_slots, args.Seed
        )
        print(error[0,0].item()/500, new_history_len)

    elif mode == 2:
        # Mode 2
        # With Data Minimization
        # Use complete training data to get function params and minimized history
        # And then predict on test data
        # Parameters not updated in between predictions
        # error, actual, pred, new_history_len = setting1.mode_2(
        #     train_data_history, test_data_history.time_slots, stochastic_gradient, args.Seed)
        # print(error[0,0].item()/500, new_history_len)

        setting1.greedy_algo_with_tests(History(train_data_history.time_slots[:2]), train_data_history.time_slots[2:], stochastic_gradient)

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

    elif mode == 5:
        setting1.test_histories('nll_sens_histories.pkl', test_data_history.time_slots, seed=args.Seed)

    # import json
    # with open(f'logs/mode-{mode}-stochastic_gradient-{stochastic_gradient}-stochastic_value-{args.StocValue}-threshCollectTill-{args.ThreshCollectTill}-budget-{args.Budget}-threshTau-{args.ThreshTau}-train_len-{train_len}-test_len-{test_len}.json', 'wb') as f:
        # data = {"Error": error.item(), "Actual": actual, "Pred": pred, "Degree of Minimization": new_history_len}
        # obj = json.dumps(data) + "\n"
        # json_bytes = obj.encode('utf-8')
        # f.write(json_bytes)