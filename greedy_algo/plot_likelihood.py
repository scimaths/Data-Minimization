import os
import sys
import numpy as np
import json

x = []
y = []
for file in ['likelihood-mode-2']:
    with open(file) as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            y.append(float(line))
            x.append(i)
            i += 1

import matplotlib.pyplot as plt
plt.plot(x, y, label="with Data Minimization")
plt.xlabel("Progression")
plt.ylabel("Negative Likelihood Value")
plt.legend()
# plt.show()
plt.savefig("plot.png")
