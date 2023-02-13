import torch
import numpy as np
from torch import nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.omega = 20
        mu = torch.Tensor([1])
        self.mu = nn.Parameter(mu)
        alpha = torch.Tensor([1])
        self.alpha = nn.Parameter(alpha)

    def forward(self,  times, max_time=1):
        times_delta = torch.maximum(times.unsqueeze(1) - times.unsqueeze(0), torch.Tensor([0]))
        times_delta[times_delta == 0] = np.inf
        times_delta *= self.omega
        mu_sq = self.mu**2
        alpha_sq = self.alpha**2
        summed = torch.log(mu_sq + alpha_sq * torch.exp(-times_delta).sum(dim=1)).sum()
        values = summed - (alpha_sq/self.omega)*(1-torch.exp(-self.omega*(max_time-times))).sum() - (mu_sq*max_time)
        return -values[0]

if __name__ == "__main__":
    size = 1
    times = torch.Tensor(np.arange(0.0, 1.000001, 1/size))
    model = MyModel()
    output = model.forward(times)
    sgd = torch.optim.SGD(model.parameters(), lr = 0.1)
    last_mu = 0.5
    last_alpha = 0.5
    idx = 0
    while (1):
        sgd.zero_grad()
        output = model.forward(times)
        output.backward()
        sgd.step()
        change = (last_mu-model.mu[0])**2 + (last_alpha-model.alpha[0])**2
        # if change < 1e-8:
        #     break
        last_mu = model.mu[0].item()
        last_alpha = model.alpha[0].item()
        if idx % 10 == 0:
            print(idx, last_alpha, last_mu, change.item(), output.item())
        idx += 1
    print(output.item())
    print(model.mu[0].item(), model.alpha[0].item())
    # values = torch.maximum(times.unsqueeze(1) - times.unsqueeze(0), torch.Tensor([0]))
    # values[values < 0.005] = -np.inf
    # summed = torch.sum(torch.exp(values), dim=1)
    # print(summed)