#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:42:12 2020

@author: emilam
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rv_histogram as hist

Datasets1 = np.load('ApInferenceData1to4.npy')
Datasets2 = np.load('ApInferenceData5to8.npy')
Datasets3 = np.load('ApInferenceData9to12.npy')
Datasets4 = np.load('ApInferenceData13to16.npy')
Datasets5 = np.load('ApInferenceData17to20.npy')

Datasets = [Datasets1,Datasets2,Datasets3,Datasets4,Datasets5]

maps1 = np.load("Maps0.0001.npy")
maps2 = np.load("Maps0.0005.npy")
maps3 = np.load("Maps0.001.npy")
maps4 = np.load("Maps0.003.npy")
maps5 = np.load("Maps0.005.npy")
### CALCULATED WITH R
maps = [maps1,maps2,maps3,maps4,maps5]

means = []
medians = []
varrs = []
count = 0
for i in range(5):
    for j in range(4):
        means_temp = []
        varrs_temp = []
        maps_temp =[]
        medians_temp =[]
        count += 1
        for k in range(5):
            means_temp.append(np.mean(Datasets[i][j][k][300:]))
            medians_temp.append(np.median(Datasets[i][j][k][300:]))
            varrs_temp.append(np.var(Datasets[i][j][k][300:]))
        medians.append(medians_temp) 
        varrs.append(varrs_temp)
        means.append(means_temp)
        
means = np.asarray(means).mean(0)
medians = np.asarray(medians).mean(0)
maps = np.asarray(maps).mean(1)
varrs = np.asarray(varrs).mean(0)
varrs = np.sqrt(varrs)

x = [1,2,3,4,5]
ticksss = ['0.0001','0.0005','0.001','0.003','0.005']

plt.figure()
plt.title('Mean of posterior mean - 20 datasets')
plt.xlabel('Noise')
plt.ylabel('Ap estimation')
plt.ylim([0.004,0.007])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], means[i], yerr = varrs[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Mean of MAPs - 20 datasets')
plt.xlabel('Noise')
plt.ylabel('Ap estimation')
plt.ylim([0.004,0.007])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], maps[i], yerr = varrs[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Mean of medians - 20 datasets')
plt.xlabel('Noise')
plt.ylabel('Ap estimation')
plt.ylim([0.004,0.007])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], medians[i], yerr = varrs[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
            
'''    
HistAp1 = np.histogram(Datasets[1][0][0][300:],10000)
DensAp1= hist(HistAp1)


X = np.linspace(0,0.1,10000)
Map = 0
Map_x = 0
for x in X:
    if DensAp1.pdf(x) > Map:
        Map = DensAp1.pdf(x)
        Map_x = x
                    
plt.title("PDF from Template")
plt.xlim([0.001,0.008])
#plt.hist(HistAp1, density=True, bins=1000)
plt.plot(X, DensAp1.pdf(X), label='PDF')
plt.show()
'''
