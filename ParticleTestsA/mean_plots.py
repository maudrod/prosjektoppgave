#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:13:51 2020

@author: maudrodsmoen
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
"""
cases = [10,50,100,500,1000,5000]

for case in cases:
    strcase = str(case)
    print(strcase)

A_10 = np.load("./10_particles/A_10.npy")[200:]
A_50 = np.load("./50_particles/A_50.npy")[200:]
A_100 = np.load("./100_particles/A_100.npy")[200:]
A_500 = np.load("./500_particles/A_500.npy")[200:]
A_1000 = np.load("./1000_particles/A_1000.npy")[200:]
A_5000 = np.load("./5000_particles/A_5000.npy")[200:]



mean10 = np.mean(A_10)
var10 = np.var(A_10)

plt.boxplot(A_10)

alleA = [list(A_10),list(A_50),list(A_100),list(A_500),list(A_1000),list(A_5000)]


mpl.style.use(['seaborn-darkgrid'])

fig, ax = plt.subplots()
ax.boxplot(alleA)
ax.set_title('Boxplots for values of A, last 1300 iterations')
ax.set_xticklabels(cases)
ax.set_xlabel('Number of Particles')
ax.set_ylabel('Value of A')

"""


# For each particle amount
# Average the A value for each dataset
# So we have 20 mean values now, take the mean and std of these, then plot
# for each value of particles. See fig 5.8 Emil

# We also want the maps plotted.........

particleCandidates = [10,50,100,500,1000,5000]

mean_vals = [0,0,0,0,0,0]
var_vals = [0,0,0,0,0,0]

for i in range(0,len(particleCandidates)):
    
    foldername = './'+str(particleCandidates[i])+'_particles/'
    
    means = np.zeros(10)
    
    for k in range(0,10):
        pathname = foldername + 'A_'+str(particleCandidates[i])+'particles'+'_dataset'+str(k)+'.npy'
        print(pathname)
        
        mean_dataset_k = np.mean(np.load(pathname))
        
        means[k] = mean_dataset_k
    
    print(means)
    
    mean_vals[i] = np.mean(means)
    var_vals[i] = np.var(means)
    
mpl.style.use(['seaborn-darkgrid'])

x = [1,2,3,4,5,6]
ticksss = ['10','50','100','500','1000','5000']

plt.figure()
plt.title('Mean of posterior means - 20 datasets')
plt.xlabel('Particles')
plt.ylabel('Ap estimation')
plt.ylim([0.004,0.007])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(6):
    if i == 1:
        plt.errorbar(x[i], mean_vals[i], yerr = var_vals[i],c=[0.4,0.3,0.9],marker = 'o',label='Std of means',ecolor=[0.4,0.3,0.9])
        #plt.errorbar(x[i], means[i], yerr = varrs[i],marker = 'o',label='Std of samples',barsabove=(True))
    else:
        plt.errorbar(x[i], mean_vals[i], yerr = var_vals[i],c=[0.4,0.3,0.9],marker = 'o',ecolor=[0.4,0.3,0.9])#,label='Std of means')
        #plt.errorbar(x[i], means[i], yerr = varrs[i],marker = 'o')#,label='Std of samples')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
  
