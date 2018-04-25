import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

datafile = "ddot_bench.dat"
posFlops = 4
n = np.array([10**i for i in range(3, 10 )])
data = np.array([float(l.split()[posFlops]) for l in open(datafile)])
# Convert into FLOP/s
flop = (2 * n - 1) * 1e-6
plt.loglog(n, flop / data[::2], 'o-', label="CPU")
plt.loglog(n, flop / data[1::2],'s-',  label="GPU")
plt.legend(loc='upper left')
plt.xlabel("Length of Vector")
plt.ylabel("MFLOP/s")
plt.title("DDot Benchmark")
plt.savefig("ddot_bench.png")
plt.savefig("ddot_bench.svg")
plt.savefig("ddot_bench.pdf")
# plt.show()
