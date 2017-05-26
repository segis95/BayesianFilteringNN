import numpy as np

import matplotlib.pyplot as plt
    

def mufun(u):
    if (0 <= u) and (u < 1):
        return np.sign(np.exp(8-3*u)*(-u*(u*(u+3) + 5) + 5*np.exp(u) - 5))
    if (1 <= u) and (u < 2):
        return np.sign(np.exp(8-3*u)*(np.exp(1)*(u**3 + 2*u + 2)-u*(u*(u+3) + 5) - 5))
    if (2 <= u) and (u < 3):
        return np.sign(np.exp(6-3*u)*(np.exp(3)*(u**3 + 2 * u + 2) - 35 * np.exp(u)))
    
x = np.linspace(0, 3, 1000)
y = [mufun(u) for u in x]
plt.ylim(-2, 2)
plt.xlim(-1, 4)
p = plt.plot(x,y, -4,4)

plt.show()


