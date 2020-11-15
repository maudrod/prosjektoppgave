#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:11:10 2020

@author: emilam
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


Taus = np.load('Taus.npy')
w0ests2ms = np.load('w0estimates_2ms.npy')
w0ests5ms = np.load('w0estimates_5ms.npy')
w0ests1ms = np.load('w0estimates_2ms.npy')


loglikesTau2ms = np.load('loglikes_2ms.npy')
loglikesTau5ms= np.load('Loglikes_5ms.npy')
loglikesTau1ms = np.load('loglikes_1ms_m3.5.npy')

indexes = np.linspace(0,len(loglikesTau2ms)-1,len(loglikesTau2ms))
indexes1ms =np.linspace(0,len(loglikesTau1ms)-1,len(loglikesTau1ms))


w0highs2ms = np.where(w0ests2ms > 1.1)
w0lows2ms = np.where(w0ests2ms < 0.9)

w0highs1ms = np.where(w0ests1ms >1.1)
w0lows1ms = np.where(w0ests1ms < 0.9)

w0highs5ms = np.where(w0ests5ms > 1.1)
w0lows5ms = np.where(w0ests5ms<0.9)

indexesgood2ms = np.delete(indexes,np.concatenate((w0highs2ms[0],w0lows2ms[0])))

indexesgood5ms = np.delete(indexes,np.concatenate((w0highs5ms[0],w0lows5ms[0])))

indexesgood1ms = np.delete(indexes,np.concatenate((w0highs1ms[0],w0lows1ms[0])))


Tau2msgoods = []
for i in range(len(indexesgood2ms)):
    Tau2msgoods.append(loglikesTau2ms[indexesgood2ms[i].astype(int)])
    
Tau2mshighs = []
for i in range(len(w0highs2ms[0])):
    Tau2mshighs.append(loglikesTau2ms[w0highs2ms[0][i].astype(int)])


Tau2mslows = []

for i in range(len(w0lows2ms[0])):
    Tau2mslows.append(loglikesTau2ms[w0lows2ms[0][i].astype(int)])

Tau5msgoods = []

for i in range(len(indexesgood5ms)):
    Tau5msgoods.append(loglikesTau5ms[indexesgood5ms[i].astype(int)])

    
Tau5mshighs = []
for i in range(len(w0highs5ms[0])):
    Tau5mshighs.append(loglikesTau5ms[w0highs5ms[0][i].astype(int)])
    
Tau5mslows = []
for i in range(len(w0lows5ms[0])):
    Tau5mslows.append(loglikesTau5ms[w0lows5ms[0][i].astype(int)])
    
    
Tau1msgoods = []
for i in range(len(indexesgood1ms)):
    Tau1msgoods.append(loglikesTau1ms[indexesgood1ms[i].astype(int)])
    
Tau1mshighs = []
for i in range(len(w0highs1ms[0])):
    Tau1mshighs.append(loglikesTau1ms[w0highs1ms[0][i].astype(int)])


Tau1mslows = []

for i in range(len(w0lows1ms[0])):
    Tau1mslows.append(loglikesTau1ms[w0lows1ms[0][i].astype(int)])
'''
plt.figure()
plt.title('Tau loglikelihood - Good estimate w0 - 2ms')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tau2msgoods).mean(0),label='Mean 100 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Tau loglikelihood - Good estimate w0 - 5ms')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tau5msgoods).mean(0),label='Mean 100 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Tau loglikelihood - High estimate w0 - 2ms')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tau2mshighs).mean(0),label='Mean 100 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Tau loglikelihood - High estimate w0 - 5ms')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tau5mshighs).mean(0),label='Mean 100 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
    
plt.figure()
plt.title('Tau loglikelihood - Low estimate w0 - 2ms')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tau2mslows).mean(0),label='Mean 100 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Tau loglikelihood - Low estimate w0 - 5ms')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tau5mslows).mean(0),label='Mean 100 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''
'''
plt.figure()
plt.title('Tau loglikelihood - Good estimate w0 - 1ms')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tau1msgoods).mean(0),label='Mean 100 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Tau loglikelihood - High estimate w0 - 1ms')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tau1mshighs).mean(0),label='Mean 100 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''
plt.figure()
plt.title('Tau loglikelihood - Low estimate w0 - 1ms')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tau1mslows).mean(0),label='Mean 100 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()