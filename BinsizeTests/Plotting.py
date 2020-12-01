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

plt.figure()
plt.title('Tau loglikelihood - Low estimate w0 - 1ms')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tau1mslows).mean(0),label='Mean 100 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''
##### 2MS TAU INFERENCE ######
'''
Tau1 = np.load('Tau0.0001noise2msB275.npy')
Tau2 = np.load('Tau0.001noise2msB275.npy')
Tau3 = np.load('Tau0.003noise2msB275.npy')
Tau4 = np.load('Tau0.005noise2msB275.npy')


Tau5 = np.load('Tau0.0001noise2msB31.npy')
Tau6 = np.load('Tau0.001noise2msB31.npy')
Tau7 = np.load('Tau0.003noise2msB31.npy')
Tau8 = np.load('Tau0.005noise2msB31.npy')

Taumean1 = np.mean(Tau1[300:])
Tauvar1 = np.var(Tau1[300:])
Taumean2 = np.mean(Tau2[300:])
Tauvar2 = np.var(Tau2[300:])
Taumean3 = np.mean(Tau3[300:])
Tauvar3 = np.var(Tau3[300:])
Taumean4 = np.mean(Tau4[300:])
Tauvar4 = np.var(Tau4[300:])
Taumean5 = np.mean(Tau5[300:])


Tauvar5 = np.var(Tau5[300:])
Taumean6 = np.mean(Tau6[300:])
Tauvar6 = np.var(Tau6[300:])
Taumean7 = np.mean(Tau7[300:])
Tauvar7 = np.var(Tau7[300:])
Taumean8 = np.mean(Tau7[300:])
Tauvar8 = np.var(Tau7[300:])

Taumeans275 = [Taumean1,Taumean2,Taumean3,Taumean4]
Taumeans31 = [Taumean5,Taumean6,Taumean7,Taumean8]
Taustds275 = [np.sqrt(Tauvar1),np.sqrt(Tauvar2),np.sqrt(Tauvar3),np.sqrt(Tauvar4)] 
Taustds31 = [np.sqrt(Tauvar5),np.sqrt(Tauvar6),np.sqrt(Tauvar7),np.sqrt(Tauvar8)]

x = [1,2,3,4]
ticksss = ['0.0001','0.001','0.003','0.005']
plt.figure()
plt.title('Tau sensitivity of noise - 2ms - $b_1 = b_2$ = -3.1')
plt.xlabel('Noise')
plt.ylabel('Tau estimation')
plt.ylim([-0.06,0.1])
plt.xlim([0,5])
plt.xticks(x,labels = ticksss)
for i in range(4):
    plt.errorbar(x[i], Taumeans31[i], yerr = Taustds31[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''
'''
plt.figure()
plt.title('Loglikelihoods good $w^0$ estimation - 1ms binsize ')
plt.xlabel('Tau')
plt.ylabel(r'log$P(s_2^{(0:T)}$ given $\tau)$')
plt.plot(Taus,np.asarray(Tau1msgoods).mean(0),label='Mean 100 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''
plt.figure()
plt.title('Loglikelihoods for a high $w^0$ estimation - 1ms binsize ')
plt.xlabel('Tau')
plt.ylabel(r'log$P(s_2^{(0:T)}$ given $\tau)$')
plt.plot(Taus,np.asarray(Tau5mshighs[27]),label='Mean 100 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
#data = np.transpose(np.asarray([Tau5msgoods2,Taus2]))

#df = pd.DataFrame(data, columns =['log', 'tau'])


Tau1ms1 = np.load('Tau0.0001noise1msB35.npy')
Tau1ms2 = np.load('Tau0.001noise2msB35.npy')
Tau1ms3 = np.load('Tau0.003noise2msB35.npy')
Tau1ms4 = np.load('Tau0.005noise2msB35.npy')

Tau1 = np.load('Tau0.0001noise2msB275.npy')
Tau2 = np.load('Tau0.001noise2msB275.npy')
Tau3 = np.load('Tau0.003noise2msB275.npy')
Tau4 = np.load('Tau0.005noise2msB275.npy')
'''
plt.figure()
sns.displot(Tau1[300:], kde=True,bins=100)
#plt.xlim([0.004,0.007])
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.xlabel('Tau')
plt.title('Posterior distribution Tau - $\sigma = 0.0001$ - Binsize 2ms')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title('Posterior distribution Tau - 0.0001 noise')
#plt.axvline(np.mean(Theta1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()
plt.show()

plt.figure()
sns.displot(Tau2[300:], kde=True,bins=100)
#plt.xlim([0.004,0.007])
plt.xlabel('Tau')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.title('Posterior distribution Tau - $\sigma = 0.001$ - Binsize 2ms')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title('Posterior distribution Tau - 0.001 noise')
#plt.axvline(np.mean(Theta1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()
plt.show()

plt.figure()
sns.displot(Tau3[300:], kde=True,bins=100)
#plt.xlim([0.004,0.007])
plt.xlabel('Tau')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.title('Posterior distribution Tau - $\sigma = 0.003$ - Binsize 2ms')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title('Posterior distribution Tau - 0.003 noise')
#plt.axvline(np.mean(Theta1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()
plt.show()

plt.figure()
sns.displot(Tau4[300:], kde=True,bins=100)
#plt.xlim([0.004,0.007])
plt.xlabel('Tau')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.title('Posterior distribution Tau - $\sigma = 0.005$ - Binsize 2ms')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title('Posterior distribution Tau - 0.005 noise')
#plt.axvline(np.mean(Theta1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()
plt.show()
'''
'''
Taumean1 = np.mean(Tau1[300:])
Tauvar1 = np.var(Tau1[300:])
Taumean2 = np.mean(Tau2[300:])
Tauvar2 = np.var(Tau2[300:])
Taumean3 = np.mean(Tau3[300:])
Tauvar3 = np.var(Tau3[300:])
Taumean4 = np.mean(Tau4[300:])
Tauvar4 = np.var(Tau4[300:])

Taumean1ms1 = np.mean(Tau1ms1[300:])
Tauvar1ms1 = np.var(Tau1ms1[300:])
Taumean1ms2 = np.mean(Tau1ms2[300:])
Tauvar1ms2 = np.var(Tau1ms2[300:])
Taumean1ms3 = np.mean(Tau1ms3[300:])
Tauvar1ms3 = np.var(Tau1ms3[300:])
Taumean1ms4 = np.mean(Tau1ms4[300:])
Tauvar1ms4 = np.var(Tau1ms4[300:])

Taumeans275 = [Taumean1,Taumean2,Taumean3,Taumean4]
Taumeans1ms = [Taumean1ms1,Taumean1ms2,Taumean1ms3,Taumean1ms4]
Taustds275 = [np.sqrt(Tauvar1),np.sqrt(Tauvar2),np.sqrt(Tauvar3),np.sqrt(Tauvar4)] 
Taustds1ms = [np.sqrt(Tauvar1ms1),np.sqrt(Tauvar1ms2),np.sqrt(Tauvar1ms3),np.sqrt(Tauvar1ms4)]

x = [1,2,3,4]
ticksss = ['0.0001','0.001','0.003','0.005']
plt.figure()
plt.title('Tau sensitivity of noise - Binsize 2ms')
plt.xlabel('Noise')
plt.ylabel('Tau estimation')
plt.ylim([-0.06,0.1])
plt.xlim([0,5])
plt.xticks(x,labels = ticksss)
for i in range(4):
    plt.errorbar(x[i], Taumeans275[i], yerr = Taustds275[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Tau sensitivity of noise - Binsize 1ms')
plt.xlabel('Noise')
plt.ylabel('Tau estimation')
plt.ylim([-0.06,0.1])
plt.xlim([0,5])
plt.xticks(x,labels = ticksss)
for i in range(4):
    plt.errorbar(x[i], Taumeans1ms[i], yerr = Taustds1ms[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''