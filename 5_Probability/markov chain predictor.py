import numpy as np

#%%
P = np.array([.8, .15, .05, .075, .9, .025, .25, .25, .5]).reshape(3, 3)
x = np.array([1, 0, 0])

x1 = np.dot(x,P)
x2 = np.dot(x1, P)
x3 = np.dot(x2, P)
print(x3)

# %%
import numpy as np

# Transition matrix
# Bull | Bear | Stagnant
W = np.array([
    [0.9, 0.075, 0.025],
    [0.15, 0.8, 0.05],
    [0.25, 0.25, 0.5]
])

# State vector
T0 = np.array([0,1,0])

# Calculate next states
T1 = np.dot(T0, W)
T2 = np.dot(T1, W)
T3 = np.dot(T2, W)

for t, T in enumerate([T0, T1, T2, T3]):
    print(f"At T={t}, the market is: {T[0]:.1%} bullish | {T[1]:.1%} bearish | {T[2]:.1%} stagnant")