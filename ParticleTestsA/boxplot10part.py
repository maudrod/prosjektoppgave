#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:27:21 2020

@author: maudrodsmoen
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

A_0 = np.load("./10_particles/A_10particles_dataset0.npy")
A_1 = np.load("./10_particles/A_10particles_dataset1.npy")
A_2 = np.load("./10_particles/A_10particles_dataset2.npy")


mean10 = np.mean(A_0)
var10 = np.var(A_0)

plt.boxplot(A_0)

cases = [0,1,2]

alleA = [list(A_0),list(A_1),list(A_2)]


mpl.style.use(['seaborn-darkgrid'])

fig, ax = plt.subplots()
ax.boxplot(alleA)
ax.set_title('Boxplots for values of A, last 1300 iterations')
ax.set_xticklabels(cases)
ax.set_xlabel('Number of Particles')
ax.set_ylabel('Value of A')