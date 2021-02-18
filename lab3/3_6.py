# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import importlib
import hopfieldNetwork as hn
import matplotlib.pyplot as plt
import matplotlib


# %%
importlib.reload(hn)

# %% [markdown]
# ### 3.6.1

# %%
p = 0.1
data = np.random.binomial(1,p,(300,100))


# %%
biases = np.linspace(0,10,50)
capacity = list()
for bias in biases:
    hop_net2 = hn.HopfieldNetwork(data[:50,:],sparse=True, bias=bias)
    capacity.append(hop_net2.check_capacity())


# %%
plt.plot(biases,capacity)
plt.xlabel('bias')
plt.ylabel('capacity')
plt.title(r'capacity given different biases and $\rho = {}$'.format(p))
plt.show()

# %% [markdown]
# ### without self connection

# %%
p = 0.1
data = np.random.binomial(1,p,(300,100))


# %%
biases = np.linspace(0,10,50)
capacity = list()
for bias in biases:
    hop_net2 = hn.HopfieldNetwork(data[:50,:],sparse=True, bias=bias)
    hop_net2.zero_self_connection()
    capacity.append(hop_net2.check_capacity())


# %%
plt.plot(biases,capacity)
plt.xlabel('bias')
plt.ylabel('capacity')
plt.title(r'capacity given different biases and $\rho = {}$'.format(p))
plt.show()

# %% [markdown]
# ### 3.6.2
# %% [markdown]
# ### p = 0.05

# %%
p = 0.05
data = np.random.binomial(1,p,(300,100))


# %%
biases = np.linspace(0,10,50)
capacity = list()
for bias in biases:
    hop_net2 = hn.HopfieldNetwork(data[:50,:],sparse=True, bias=bias)
    hop_net2.zero_self_connection()
    capacity.append(hop_net2.check_capacity())


# %%
plt.plot(biases,capacity)
plt.xlabel('bias')
plt.ylabel('capacity')
plt.title(r'capacity given different biases and $\rho = {}$'.format(p))
plt.show()

# %% [markdown]
# ### p = 0.01

# %%
p = 0.01
data = np.random.binomial(1,p,(300,100))


# %%
biases = np.linspace(0,10,50)
capacity = list()
for bias in biases:
    hop_net2 = hn.HopfieldNetwork(data[:50,:],sparse=True, bias=bias)
    hop_net2.zero_self_connection()
    capacity.append(hop_net2.check_capacity())


# %%
plt.plot(biases,capacity)
plt.xlabel('bias')
plt.ylabel('capacity')
plt.title(r'capacity given different biases and $\rho = {}$'.format(p))
plt.show()


# %%



