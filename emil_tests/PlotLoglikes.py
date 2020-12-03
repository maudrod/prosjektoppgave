#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:39:03 2020

@author: emilam
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

Aps = np.load('Aps.npy')
Taus = np.load('Taus.npy')
w0ests1pr = np.load('w0estimates_1prosent_norminit.npy')
w0ests10pr = np.load('w0estimates_10prosent_norminit.npy')
w0estsDelta = np.load('w0estimates_1prosent_deltainit.npy')

loglikesAp1pr=np.load('loglikes_Ap_1pr_norminit.npy')
loglikesTau1pr = np.load('Loglikes_Tau_1pr_norminit.npy')

loglikesAp10pr= np.load('loglikes_Ap_10pr_norminit.npy')
loglikesTau10pr = np.load('Loglikes_Tau_10pr_norminit.npy')

loglikesApDelta = np.load('loglikes_Ap_1pr_deltainit.npy')
loglikesTauDelta = np.load('Loglikes_Tau_1pr_deltainit.npy')

indexes = np.linspace(0,199,200)

w0highs1pr = np.where(w0ests1pr > 1.1)
w0lows1pr  = np.where(w0ests1pr < 0.9)

w0highs10pr = np.where(w0ests10pr > 1.1)
w0lows10pr = np.where(w0ests10pr<0.9)

w0highsDelta = np.where(w0estsDelta > 1.1)
w0lowsDelta = np.where(w0estsDelta < 0.9)

indexesgood1pr = np.delete(indexes,np.concatenate((w0highs1pr[0],w0lows1pr[0])))

indexesgood10pr = np.delete(indexes,np.concatenate((w0highs10pr[0],w0lows10pr[0])))

indexesgoodDelta = np.delete(indexes,np.concatenate((w0highsDelta[0],w0lowsDelta[0])))

ApPeaks1pr = []
ApPeaks10pr = []
ApPeaksDelta = []
for i in range(200):
    ApPeaks1pr.append(np.where(loglikesAp1pr[i] == np.amax(loglikesAp1pr[i])))
    ApPeaks10pr.append(np.where(loglikesAp10pr[i] == np.amax(loglikesAp10pr[i])))
    ApPeaksDelta.append(np.where(loglikesApDelta[i] == np.amax(loglikesApDelta[i])))
    

####### 1 PROSENT LOGLIKES ########
Aploglikesgood1pr = []
for i in range(len(indexesgood1pr)):
    Aploglikesgood1pr.append(loglikesAp1pr[indexesgood1pr[i].astype(int)])


Tauloglikesgood1pr = []
for i in range(len(indexesgood1pr)):
    Tauloglikesgood1pr.append(loglikesTau1pr[indexesgood1pr[i].astype(int)])
    
Aploglikeshigh1pr = []
for i in range(len(w0highs1pr[0])):
    Aploglikeshigh1pr.append(loglikesAp1pr[w0highs1pr[0][i].astype(int)])

Tauloglikeshigh1pr = []
for i in range(len(w0highs1pr[0])):
    Tauloglikeshigh1pr.append(loglikesTau1pr[w0highs1pr[0][i].astype(int)])

    
Aploglikeslow1pr = []
for i in range(len(w0lows1pr[0])):
    Aploglikeslow1pr.append(loglikesAp1pr[w0lows1pr[0][i].astype(int)])
    

Tauloglikeslow1pr = []
for i in range(len(w0lows1pr[0])):
    Tauloglikeslow1pr.append(loglikesTau1pr[w0lows1pr[0][i].astype(int)])
    
####### 10 PROSENT LOGLIKES #######

Aploglikesgood10pr = []
for i in range(len(indexesgood10pr)):
    Aploglikesgood10pr.append(loglikesAp10pr[indexesgood10pr[i].astype(int)])

Tauloglikesgood10pr = []
for i in range(len(indexesgood10pr)):
    Tauloglikesgood10pr.append(loglikesTau10pr[indexesgood10pr[i].astype(int)])

Aploglikeshigh10pr = []
for i in range(len(w0highs10pr[0])):
    Aploglikeshigh10pr.append(loglikesAp10pr[w0highs10pr[0][i].astype(int)])

Tauloglikeshigh10pr = []
for i in range(len(w0highs10pr[0])):
    Tauloglikeshigh10pr.append(loglikesTau10pr[w0highs10pr[0][i].astype(int)])
    
Aploglikeslow10pr = []
for i in range(len(w0lows10pr[0])):
    Aploglikeslow10pr.append(loglikesAp10pr[w0lows10pr[0][i].astype(int)])

Tauloglikeslow10pr = []
for i in range(len(w0lows10pr[0])):
    Tauloglikeslow10pr.append(loglikesTau10pr[w0lows10pr[0][i].astype(int)])

##### DELTA INIT LOGLIKES #####

AploglikesgoodDelta = []
for i in range(len(indexesgoodDelta)):
    AploglikesgoodDelta.append(loglikesApDelta[indexesgoodDelta[i].astype(int)])
    
TauloglikesgoodDelta = []
for i in range(len(indexesgoodDelta)):
    TauloglikesgoodDelta.append(loglikesTauDelta[indexesgoodDelta[i].astype(int)])


AploglikeshighDelta = []

for i in range(len(w0highsDelta[0])):
    AploglikeshighDelta.append(loglikesApDelta[w0highsDelta[0][i].astype(int)])

TauloglikeshighDelta = []

for i in range(len(w0highsDelta[0])):
    TauloglikeshighDelta.append(loglikesTauDelta[w0highsDelta[0][i].astype(int)])

    
AploglikeslowDelta = []
for i in range(len(w0lowsDelta[0])):
    AploglikeslowDelta.append(loglikesApDelta[w0lowsDelta[0][i].astype(int)])
    
TauloglikeslowDelta = []
for i in range(len(w0lowsDelta[0])):
    TauloglikeslowDelta.append(loglikesTauDelta[w0lowsDelta[0][i].astype(int)])
    
    ##### A PLOTTING ######
'''
plt.figure()
plt.title('Ap loglikelihood - Good estimate for w0 - $w0 \sim Norm(w0,w0/100)$')
plt.xlabel('Ap')
plt.ylabel('Loglikelihood')
plt.plot(Aps,np.asarray(Aploglikesgood1pr).mean(0),label='mean')
plt.show()

plt.figure()
plt.title('Ap loglikelihood - Good estimate for w0 - $w0 \sim Norm(w0,w0/10)$')
plt.xlabel('Ap')
plt.ylabel('Loglikelihood')
plt.plot(Aps,np.asarray(Aploglikesgood10pr).mean(0),label='mean')
plt.show()

plt.figure()
plt.title('Ap loglikelihood - Good estimate w0 - $w0 \sim \delta(w_0)$')
plt.xlabel('Ap')
plt.ylabel('Loglikelihood')
plt.plot(Aps,np.asarray(AploglikesgoodDelta).mean(0),label='Mean 200 datasets')
plt.axvline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Ap loglikelihood - Low estimate for w0 - $w0 \sim Norm(w0,w0/100)$')
plt.xlabel('Ap')
plt.ylabel('Loglikelihood')
plt.plot(Aps,np.asarray(Aploglikeslow1pr).mean(0),label='mean')
plt.show()

plt.figure()
plt.title('Ap loglikelihood - Low estimate for w0 - $w0 \sim Norm(w0,w0/10)$')
plt.xlabel('Ap')
plt.ylabel('Loglikelihood')
plt.plot(Aps,np.asarray(Aploglikeslow10pr).mean(0),label='mean')
plt.show()

plt.figure()
plt.title('Ap loglikelihood - Low estimate w0 - $w0 \sim \delta(w_0)$')
plt.xlabel('Ap')
plt.ylabel('Loglikelihood')
plt.plot(Aps,np.asarray(AploglikeslowDelta).mean(0),label='Mean 200 datasets')
plt.axvline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Ap loglikelihood - High estimate for w0 - $w0 \sim Norm(w0,w0/100)$')
plt.xlabel('Ap')
plt.ylabel('Loglikelihood')
plt.plot(Aps,np.asarray(Aploglikeshigh1pr).mean(0),label='mean')
plt.show()

plt.figure()
plt.title('Ap loglikelihood - High estimate for w0 - $w0 \sim Norm(w0,w0/10)$')
plt.xlabel('Ap')
plt.ylabel('Loglikelihood')
plt.plot(Aps,np.asarray(Aploglikeshigh10pr).mean(0),label='mean')
plt.show()

plt.figure()
plt.title('Ap loglikelihood - High estimate w0 - $w0 \sim \delta(w_0)$')
plt.xlabel('Ap')
plt.ylabel('Loglikelihood')
plt.plot(Aps,np.asarray(AploglikeshighDelta).mean(0),label='Mean 200 datasets')
plt.axvline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

'''
sns.set_style('darkgrid')
PeakApsDelta = []
for i in range(200):
    PeakApsDelta.append(Aps[ApPeaksDelta[i][0].astype(int)][0])
PeakApsDelta = np.asarray(PeakApsDelta)

df = pd.DataFrame(data={'$w^0$ estimate': w0estsDelta, '$A_+$: max $P(s_2^{(0:T)}$ given $A_+$)': np.asarray(PeakApsDelta)})


sns.jointplot(
    data=df,
    x="$w^0$ estimate", y="$A_+$: max $P(s_2^{(0:T)}$ given $A_+$)",
    kind="kde",xlim=(0.4,1.6),color='b',marginal_kws={'lw':3, 'color':'red'})
#plt.plot(1,0.005,'ro',label='True Value')
plt.legend()
plt.show()
'''


##### TAU PLOTTING #####

plt.figure()
plt.title('Tau loglikelihood - Good estimate for w0 - $w0 \sim Norm(w0,w0/100)$')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tauloglikesgood1pr).mean(0),label='Mean 200 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Tau loglikelihood - Good estimate for w0 - $w0 \sim Norm(w0,w0/10)$')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tauloglikesgood10pr).mean(0),label='mean')
plt.show()

plt.figure()
plt.title('Tau loglikelihood - Good estimate w0 - $w0 \sim \delta(w_0)$')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(TauloglikesgoodDelta).mean(0),label='Mean 200 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Tau loglikelihood - Low estimate for w0 - $w0 \sim Norm(w0,w0/100)$')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tauloglikeslow1pr).mean(0),label='mean')
plt.show()

plt.figure()
plt.title('Tau loglikelihood - Low estimate for w0 - $w0 \sim Norm(w0,w0/10)$')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tauloglikeslow10pr).mean(0),label='mean')
plt.show()

plt.figure()
plt.title('Tau loglikelihood - Low estimate w0 - $w0 \sim \delta(w_0)$')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(TauloglikeslowDelta).mean(0),label='Mean 200 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Tau loglikelihood - High estimate for w0 - $w0 \sim Norm(w0,w0/100)$')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tauloglikeshigh1pr).mean(0),label='mean')
plt.show()

plt.figure()
plt.title('Tau loglikelihood - High estimate for w0 - $w0 \sim Norm(w0,w0/10)$')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(Tauloglikeshigh10pr).mean(0),label='mean')
plt.show()

plt.figure()
plt.title('Tau loglikelihood - High estimate w0 - $w0 \sim \delta(w_0)$')
plt.xlabel('Tau')
plt.ylabel('Loglikelihood')
plt.plot(Taus,np.asarray(TauloglikeshighDelta).mean(0),label='Mean 200 datasets')
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''
