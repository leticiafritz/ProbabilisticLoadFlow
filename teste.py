import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 6, 200)
MU_EV, SD_EV = 0.418749176686875, 0.859455801705594  # 30, 5
dist = ss.lognorm([SD_EV], loc=MU_EV)
#y1 = beta.pdf(x, 4.5, 2.8, scale=8, loc=0)
plt.plot(x, dist.pdf(x))
plt.show()