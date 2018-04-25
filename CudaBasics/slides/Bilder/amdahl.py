# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np

def f(ts, tp, n):
    """Amdahl's law shows that the maximum speedup is limited by the fraction of serial code."""
    return (ts + tp) / (ts + tp / n)

plt.figure(figsize = (3,2), dpi=200)
n = 2 ** np.arange(0, 13)

for p in [0.5, 0.25, 0.1, 0.01]:
	plt.plot(n, f(p, 1 - p, n), label="%s%%" % (1 - p) * 100)
plt.xlim([0, 4100])
plt.xticks([1, 1024, 2048, 4096])
#legend(loc = "best")
savefig("amdahl.png", dpi=200)
