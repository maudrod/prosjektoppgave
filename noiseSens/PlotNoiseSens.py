#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:00:30 2020

@author: emilam
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


'''
Ap analysis
'''
'''
Ap1 = np.load('Ap0.0001noise.npy')
Ap2 = np.load('Ap0.0005noise.npy')
Ap3 = np.load('Ap0.001noise.npy')
Ap4 = np.load('Ap0.002noise.npy')
Ap5 = np.load('Ap0.003noise.npy')
Ap6 = np.load('Ap0.004noise.npy')
Ap7 = np.load('Ap0.005noise.npy')

mean1 = np.mean(Ap1[300:])
var1 = np.var(Ap1[300:])
mean2 = np.mean(Ap2[300:])
var2 = np.var(Ap2[300:])
mean3 = np.mean(Ap3[300:])
var3 = np.var(Ap3[300:])
mean4 = np.mean(Ap4[300:])
var4 = np.var(Ap4[300:])
mean5 = np.mean(Ap5[300:])
var5 = np.var(Ap5[300:])
mean6 = np.mean(Ap6[300:])
var6 = np.var(Ap6[300:])
mean7 = np.mean(Ap7[300:])
var7 = np.var(Ap7[300:])

x = [1,2,3,4,5,6,7]
ticksss = ['0.0001','0.0005','0.001','0.002','0.003','0.004','0.005']


means = [mean1,mean2,mean3,mean4,mean5,mean6,mean7]
stds = [np.sqrt(var1),np.sqrt(var2),np.sqrt(var3),np.sqrt(var4),np.sqrt(var5),np.sqrt(var6),np.sqrt(var7)]

plt.figure()
plt.title('Sensitivity of noise')
plt.xlabel('Noise')
plt.ylabel('Ap estimation')
plt.ylim([0.002,0.008])
plt.xlim([0,8])
plt.xticks(x,labels = ticksss)
for i in range(7):
    plt.errorbar(x[i], means[i], yerr = stds[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()



plt.figure()
plt.xlim([0.002,0.008])
sns.displot(Ap2[300:], kde=True)
plt.axvline(0.001,color='r',linestyle='--',label='True Value')
plt.legend()
plt.title('Posterior distribution Ap - 0.005 noise')
plt.show()
'''


#Tau analysis
'''
x = [1,2,3,4,5,6,7]
ticksss = ['0.0001','0.0005','0.001','0.002','0.003','0.004','0.005']

Tau1 = np.load('Tau0.0001noise.npy')
Tau2 = np.load('Tau0.0005noise.npy')
Tau3 = np.load('Tau0.001noise.npy')
Tau4 = np.load('Tau0.002noise.npy')
Tau5 = np.load('Tau0.003noise.npy')
Tau6 = np.load('Tau0.004noise.npy')
Tau7 = np.load('Tau0.005noise.npy')

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

Taumeans = [Taumean1,Taumean2,Taumean3,Taumean4,Taumean5,Taumean6,Taumean7]
Taumaps = np.load('MapsTau5ms.npy')
Taustds = [np.sqrt(Tauvar1),np.sqrt(Tauvar2),np.sqrt(Tauvar3),np.sqrt(Tauvar4),np.sqrt(Tauvar5),np.sqrt(Tauvar6),np.sqrt(Tauvar7)]

plt.figure()
plt.title(r'Sensitivity of noise - Binsize: 5ms ')
plt.xlabel('Noise')
plt.ylabel('Tau estimation')
plt.ylim([-0.06,0.1])
plt.xlim([0,8])
plt.xticks(x,labels = ticksss)
for i in range(7):
    plt.errorbar(x[i], Taumaps[i], yerr = Taustds[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
sns.displot(Tau1[300:], kde=True,bins=100)
plt.axvline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.title('Posterior distribution Tau - 0.0001 noise')
plt.show()

plt.figure()
plt.title('Tau inference - 0.0001 noise')
plt.xlabel('Iteration')
plt.ylabel('Tau')
plt.ylim([0,0.08])
plt.plot(np.linspace(1,1500,1500),Tau1,'ko')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''
'''
Simultaneous
'''
'''
Sim1 = np.load('Sim0.0001noise.npy')
Sim2 = np.load('Sim0.0005noise.npy')
Sim3 = np.load('Sim0.001noise.npy')
Sim4 = np.load('Sim0.002noise.npy')
Sim5 = np.load('Sim0.003noise.npy')
Sim6 = np.load('Sim0.004noise.npy')
Sim7 = np.load('Sim0.005noise.npy')

SimTaumean1 = np.mean(Sim1[300:,1])
SimTauvar1 = np.var(Sim1[300:,1])
SimTaumean2 = np.mean(Sim2[300:,1])
SimTauvar2 = np.var(Sim2[300:,1])
SimTaumean3 = np.mean(Sim3[300:,1])
SimTauvar3 = np.var(Sim3[300:,1])
SimTaumean4 = np.mean(Sim4[300:,1])
SimTauvar4 = np.var(Sim4[300:,1])
SimTaumean5 = np.mean(Sim5[300:,1])
SimTauvar5 = np.var(Sim5[300:,1])
SimTaumean6 = np.mean(Sim6[300:,1])
SimTauvar6 = np.var(Sim6[300:,1])
SimTaumean7 = np.mean(Sim7[300:,1])
SimTauvar7 = np.var(Sim7[300:,1])

SimApmean1 = np.mean(Sim1[300:,0])
SimApvar1 = np.var(Sim1[300:,0])
SimApmean2 = np.mean(Sim2[300:,0])
SimApvar2 = np.var(Sim2[300:,0])
SimApmean3 = np.mean(Sim3[300:,0])
SimApvar3 = np.var(Sim3[300:,0])
SimApmean4 = np.mean(Sim4[300:,0])
SimApvar4 = np.var(Sim4[300:,0])
SimApmean5 = np.mean(Sim5[300:,0])
SimApvar5 = np.var(Sim5[300:,0])
SimApmean6 = np.mean(Sim6[300:,0])
SimApvar6 = np.var(Sim6[300:,0])
SimApmean7 = np.mean(Sim7[300:,0])
SimApvar7 = np.var(Sim7[300:,0])
x = [1,2,3,4,5,6,7]
ticksss = ['0.0001','0.0005','0.001','0.002','0.003','0.004','0.005']
TaumeansSim = [SimTaumean1,SimTaumean2,SimTaumean3,SimTaumean4,SimTaumean5,SimTaumean6,SimTaumean7]
TaustdsSim = [np.sqrt(SimTauvar1),np.sqrt(SimTauvar2),np.sqrt(SimTauvar3),np.sqrt(SimTauvar4),np.sqrt(SimTauvar5),np.sqrt(SimTauvar6),np.sqrt(SimTauvar7)]

ApmeansSim = [SimApmean1,SimApmean2,SimApmean3,SimApmean4,SimApmean5,SimApmean6,SimApmean7]
ApstdsSim = [np.sqrt(SimApvar1),np.sqrt(SimApvar2),np.sqrt(SimApvar3),np.sqrt(SimApvar4),np.sqrt(SimApvar5),np.sqrt(SimApvar6),np.sqrt(SimApvar7)]

plt.figure()
sns.displot(Alt1True[300:,0], kde=True,bins=100)
plt.xlim([0.004,0.0075])
plt.xlabel('$A_+$')
plt.axvline(0.005,color='r',linestyle='--',label='True Value')
plt.title('Posterior distribution $A_+$ - $\sigma = 0.0002$')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title('Posterior distribution Tau - 0.001 noise')
#plt.axvline(np.mean(Theta1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()
plt.show()


plt.figure()
plt.title('Sensitivity of noise - Tau')
plt.xlabel('Noise')
plt.ylabel('Tau estimation')
plt.ylim([-0.06,0.1])
plt.xlim([0,8])
plt.xticks(x,labels = ticksss)
for i in range(7):
    plt.errorbar(x[i], TaumeansSim[i], yerr = TaustdsSim[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()


plt.figure()
plt.title('Sensitivity of noise - $A_+$')
plt.xlabel('Noise')
plt.ylabel('$A_+$ estimation')
plt.ylim([0.002,0.008])
plt.xlim([0,8])
plt.xticks(x,labels = ticksss)
for i in range(7):
    plt.errorbar(x[i], ApmeansSim[i], yerr = ApstdsSim[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''

'''
Alternating analaysis
'''
'''
Alt1True = np.load('Tau0.02Ap0.005Alt0.0001noise.npy')
Alt2True = np.load('Tau0.02Ap0.005Alt0.0005noise.npy')
Alt3True  = np.load('Tau0.02Ap0.005Alt0.001noise.npy')
Alt4True = np.load('Tau0.02Ap0.005Alt0.002noise.npy')
Alt5True = np.load('Tau0.02Ap0.005Altm0.003noise.npy')
Alt6True = np.load('Tau0.02Ap0.005Alt0.004noise.npy')
Alt7True = np.load('Tau0.02Ap0.005Alt0.005noise.npy')

AltTaumeanT1 = np.mean(Alt1True[300:,1])
AltTauvarT1 = np.var(Alt1True[300:,1])
AltTaumeanT2 = np.mean(Alt2True[300:,1])
AltTauvarT2 = np.var(Alt2True[300:,1])
AltTaumeanT3 = np.mean(Alt3True[300:,1])
AltTauvarT3 = np.var(Alt3True[300:,1])
AltTaumeanT4 = np.mean(Alt4True[300:,1])
AltTauvarT4 = np.var(Alt4True[300:,1])
AltTaumeanT5 = np.mean(Alt5True[300:,1])
AltTauvarT5 = np.var(Alt5True[300:,1])
AltTaumeanT6 = np.mean(Alt6True[300:,1])
AltTauvarT6 = np.var(Alt6True[300:,1])
AltTaumeanT7 = np.mean(Alt7True[300:,1])
AltTauvarT7 = np.var(Alt7True[300:,1])

AltApmeanT1 = np.mean(Alt1True[300:,0])
AltApvarT1 = np.var(Alt1True[300:,0])
AltApmeanT2 = np.mean(Alt2True[300:,0])
AltApvarT2 = np.var(Alt2True[300:,0])
AltApmeanT3 = np.mean(Alt3True[300:,0])
AltApvarT3 = np.var(Alt3True[300:,0])
AltApmeanT4 = np.mean(Alt4True[300:,0])
AltApvarT4 = np.var(Alt4True[300:,0])
AltApmeanT5 = np.mean(Alt5True[300:,0])
AltApvarT5 = np.var(Alt5True[300:,0])
AltApmeanT6 = np.mean(Alt6True[300:,0])
AltApvarT6 = np.var(Alt6True[300:,0])
AltApmeanT7 = np.mean(Alt7True[300:,0])
AltApvarT7 = np.var(Alt7True[300:,0])

TaumeansAltT = [AltTaumeanT1,AltTaumeanT2,AltTaumeanT3,AltTaumeanT4,AltTaumeanT5,AltTaumeanT6,AltTaumeanT7]
TaustdsAltT = [np.sqrt(AltTauvarT1),np.sqrt(AltTauvarT2),np.sqrt(AltTauvarT3),np.sqrt(AltTauvarT4),np.sqrt(AltTauvarT5),np.sqrt(AltTauvarT6),np.sqrt(AltTauvarT7)]

ApmeansAltT = [AltApmeanT1,AltApmeanT2,AltApmeanT3,AltApmeanT4,AltApmeanT5,AltApmeanT6,AltApmeanT7]
ApstdsAltT= [np.sqrt(AltApvarT1),np.sqrt(AltApvarT2),np.sqrt(AltApvarT3),np.sqrt(AltApvarT4),np.sqrt(AltApvarT5),np.sqrt(AltApvarT6),np.sqrt(AltApvarT7)]

#plt.figure()
#plt.plot(Alt7True[300:,1],Alt7True[300:,0],'ko')
#plt.show()

Alt1Big = np.load('Tau0.01Ap0.0075Alt0.0001noise.npy')
Alt2Big = np.load('Tau0.01Ap0.0075Alt0.0005noise.npy')
Alt3Big  = np.load('Tau0.01Ap0.0075Alt0.001noise.npy')
Alt4Big = np.load('Tau0.01Ap0.0075Alt0.002noise.npy')
Alt5Big = np.load('Tau0.01Ap0.0075Altm0.003noise.npy')
Alt6Big = np.load('Tau0.01Ap0.0075Alt0.004noise.npy')
Alt7Big = np.load('Tau0.01Ap0.0075Alt0.005noise.npy')

AltTaumeanB1 = np.mean(Alt1Big[300:,1])
AltTauvarB1 = np.var(Alt1Big[300:,1])
AltTaumeanB2 = np.mean(Alt2Big[300:,1])
AltTauvarB2 = np.var(Alt2Big[300:,1])
AltTaumeanB3 = np.mean(Alt3Big[300:,1])
AltTauvarB3 = np.var(Alt3Big[300:,1])
AltTaumeanB4 = np.mean(Alt4Big[300:,1])
AltTauvarB4 = np.var(Alt4Big[300:,1])
AltTaumeanB5 = np.mean(Alt5Big[300:,1])
AltTauvarB5 = np.var(Alt5Big[300:,1])
AltTaumeanB6 = np.mean(Alt6Big[300:,1])
AltTauvarB6 = np.var(Alt6Big[300:,1])
AltTaumeanB7 = np.mean(Alt7Big[300:,1])
AltTauvarB7 = np.var(Alt7Big[300:,1])

AltApmeanB1 = np.mean(Alt1Big[300:,0])
AltApvarB1 = np.var(Alt1Big[300:,0])
AltApmeanB2 = np.mean(Alt2Big[300:,0])
AltApvarB2 = np.var(Alt2Big[300:,0])
AltApmeanB3 = np.mean(Alt3Big[300:,0])
AltApvarB3 = np.var(Alt3Big[300:,0])
AltApmeanB4 = np.mean(Alt4Big[300:,0])
AltApvarB4 = np.var(Alt4Big[300:,0])
AltApmeanB5 = np.mean(Alt5Big[300:,0])
AltApvarB5 = np.var(Alt5Big[300:,0])
AltApmeanB6 = np.mean(Alt6Big[300:,0])
AltApvarB6 = np.var(Alt6Big[300:,0])
AltApmeanB7 = np.mean(Alt7Big[300:,0])
AltApvarB7 = np.var(Alt7Big[300:,0])

TaumeansAltB = [AltTaumeanB1,AltTaumeanB2,AltTaumeanB3,AltTaumeanB4,AltTaumeanB5,AltTaumeanB6,AltTaumeanB7]
TaustdsAltB = [np.sqrt(AltTauvarB1),np.sqrt(AltTauvarB2),np.sqrt(AltTauvarB3),np.sqrt(AltTauvarB4),np.sqrt(AltTauvarB5),np.sqrt(AltTauvarB6),np.sqrt(AltTauvarB7)]

ApmeansAltB = [AltApmeanB1,AltApmeanB2,AltApmeanB3,AltApmeanB4,AltApmeanB5,AltApmeanB6,AltApmeanB7]
ApstdsAltB = [np.sqrt(AltApvarB1),np.sqrt(AltApvarB2),np.sqrt(AltApvarB3),np.sqrt(AltApvarB4),np.sqrt(AltApvarB5),np.sqrt(AltApvarB6),np.sqrt(AltApvarB7)]

x = [1,2,3,4,5,6,7]
ticksss = ['0.0001','0.0005','0.001','0.002','0.003','0.004','0.005']

plt.figure()
sns.displot(Alt7True[300:,0], kde=True)
plt.xlabel('$A_+$')
plt.axvline(0.005,color='r',linestyle='--',label='True Value')
#plt.xlim([0.003,0.007])
plt.legend()
plt.title('Posterior distribution $A_+$ - 0.005 noise')
plt.show()


plt.figure()
plt.title('Sensitivity of noise - Tau')
plt.xlabel('Noise')
plt.ylabel('Tau estimation')
plt.ylim([-0.06,0.1])
plt.xlim([0,8])
plt.xticks(x,labels = ticksss)
for i in range(7):
    plt.errorbar(x[i], TaumeansAltT[i], yerr = TaustdsAltT[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()


plt.figure()
plt.title('Sensitivity of noise - $A_+$')
plt.xlabel('Noise')
plt.ylabel('$A_+$ estimation')
plt.ylim([0.002,0.008])
plt.xlim([0,8])
plt.xticks(x,labels = ticksss)
for i in range(7):
    plt.errorbar(x[i], ApmeansAltT[i], yerr = ApstdsAltT[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''

#AltVsim SAMME DATASETT

Alt1Same = np.load('Alt0.0001noiseSame.npy')
Alt2Same = np.load('Alt0.0005noiseSame.npy')
Alt3Same = np.load('Alt0.001noiseSame.npy')
Alt4Same = np.load('Alt0.002noiseSame.npy')
Alt5Same = np.load('Alt0.003noiseSame.npy')
Alt6Same = np.load('Alt0.004noiseSame.npy')
Alt7Same = np.load('Alt0.005noiseSame.npy')

Sim1Same = np.load('Sim0.0001noiseSame.npy')
Sim2Same = np.load('Sim0.0005noiseSame.npy')
Sim3Same = np.load('Sim0.001noiseSame.npy')
Sim4Same = np.load('Sim0.002noiseSame.npy')
Sim5Same = np.load('Sim0.003noiseSame.npy')
Sim6Same = np.load('Sim0.004noiseSame.npy')
Sim7Same = np.load('Sim0.005noiseSame.npy')

meansASim = [np.mean(Sim1Same[300:,0]),np.mean(Sim2Same[300:,0]),np.mean(Sim3Same[300:,0]),np.mean(Sim4Same[300:,0]),np.mean(Sim5Same[300:,0]),np.mean(Sim6Same[300:,0]),np.mean(Sim7Same[300:,0])]
meansAAlt = [np.mean(Alt1Same[300:,0]),np.mean(Alt2Same[300:,0]),np.mean(Alt3Same[300:,0]),np.mean(Alt4Same[300:,0]),np.mean(Alt5Same[300:,0]),np.mean(Alt6Same[300:,0]),np.mean(Alt7Same[300:,0])]

meansTauSim = [np.mean(Sim1Same[300:,1]),np.mean(Sim2Same[300:,1]),np.mean(Sim3Same[300:,1]),np.mean(Sim4Same[300:,1]),np.mean(Sim5Same[300:,1]),np.mean(Sim6Same[300:,1]),np.mean(Sim7Same[300:,1])]
meansTauAlt = [np.mean(Alt1Same[300:,1]),np.mean(Alt2Same[300:,1]),np.mean(Alt3Same[300:,1]),np.mean(Alt4Same[300:,1]),np.mean(Alt5Same[300:,1]),np.mean(Alt6Same[300:,1]),np.mean(Alt7Same[300:,1])]

plt.figure()
plt.title('Tau inference - 0.005 noise - Alt')
plt.xlabel('Iteration')
plt.ylabel('Tau')
#plt.ylim([0,0.0])
plt.plot(np.linspace(1,1500,1500),Alt7Same[:,1],'ko')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()


