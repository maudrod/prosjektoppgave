#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:13:51 2020

@author: maudrodsmoen
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# For each particle amount
# Average the A value for each dataset
# So we have 20 mean values now, take the mean and std of these, then plot
# for each value of particles. See fig 5.8 Emil

# We also want the maps plotted.........

particleCandidates = [10,50,100,500,1000,5000]

mean_vals = [0,0,0,0,0,0]
var_vals = [0,0,0,0,0,0]

for i in range(0,len(particleCandidates)):
    print(particleCandidates[i])
    number_datasets = 10
    
    if i == 5:
        number_datasets = 6
    
    foldername = './'+str(particleCandidates[i])+'_particles/'
    
    means = np.zeros(number_datasets)
    
    for k in range(0,number_datasets):
        pathname = foldername + 'A_'+str(particleCandidates[i])+'particles'+'_dataset'+str(k)+'.npy'
        #print(pathname)
        
        mean_dataset_k = np.mean(np.load(pathname))
        
        means[k] = mean_dataset_k
    
    print(means)
    
    mean_vals[i] = np.mean(means)
    
    print(mean_vals[i])
    
    var_vals[i] = np.var(means)
    
    print(var_vals[i])
    
mpl.style.use(['seaborn-darkgrid'])

x = [1,2,3,4,5,6]
ticksss = ['10','50','100','500','1000','5000']

plt.figure()
plt.title('Mean of posterior means - 10 datasets')
plt.xlabel('Particles')
plt.ylabel('Ap estimation')
plt.ylim([0.00475,0.0051])
plt.xlim([0,7])
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
plt.savefig('mean.pdf')
plt.show()
  
