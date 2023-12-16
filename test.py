import pandas as pd
import numpy as np
from functools import reduce
import torch
import matplotlib.pyplot as plt
from snntorch import spikegen
import snntorch.spikeplot as splt


def leaky_integrate_neuron(U, time_step=1e-3, I=0, R=5e7, C=1e-10):
  tau = R*C
  U = U + (time_step/tau)*(-U + I*R)
  return U

num_steps = 100
U = 0.9
U_trace = []  # keeps a record of U for plotting

for step in range(num_steps):
  U_trace.append(U)
  U = leaky_integrate_neuron(U)  # solve next step of U

def plot_mem(mem, title=None):
  if title:
    plt.title(title)
  plt.plot(mem)
  plt.xlabel("Time step")
  plt.ylabel("Membrane Potential")
  plt.xlim([0, 50])
  plt.ylim([0, 1])
  plt.show()

plot_mem(U_trace, "Leaky Neuron Model")

mem = torch.ones(1) * 0.9  # U=0.9 at t=0
print([mem])
