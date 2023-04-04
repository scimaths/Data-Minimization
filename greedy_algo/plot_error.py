import matplotlib.pyplot as plt
import numpy as np

mode_2_error = open("mode_2.txt").read().strip().split("\n")
mode_4_error = open("mode_4.txt").read().strip().split("\n")
xticks = list(range(5, 1001, 5))
errors_2 = list(map(lambda x: float(x.split()[0]), mode_2_error))
errors_4 = list(map(lambda x: float(x.split()[0]), mode_4_error))
plt.figure()
plt.plot(xticks[10:], errors_2[10:], label='Max-LL')
plt.plot(xticks[10:], errors_4[10:], label='Random')
plt.xlabel('Number of timestamps taken')
plt.ylabel('Error (MSE, Average over 500 test timestamps)')
plt.legend()
plt.title('Comparing Max-LL vs Random')
plt.savefig('error_random_vs_ll.png')