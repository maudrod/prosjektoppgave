#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:00:30 2020

@author: emilam
"""
import numpy as np
import matplotlib.pyplot as plt


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

means = [mean1,mean2,mean3,mean4,mean5,mean6,mean7]
stds = [np.sqrt(var1),np.sqrt(var2),np.sqrt(var3),np.sqrt(var4),np.sqrt(var5),np.sqrt(var6),np.sqrt(var7)]
x = [1,2,3,4,5,6,7]
ticksss = ['0.0001','0.0005','0.001','0.002','0.003','0.004','0.005']
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
'''
'''
Tau analysis
'''
'''
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
Taustds = [np.sqrt(Tauvar1),np.sqrt(Tauvar2),np.sqrt(Tauvar3),np.sqrt(Tauvar4),np.sqrt(Tauvar5),np.sqrt(Tauvar6),np.sqrt(Tauvar7)]
plt.figure()
plt.title('Sensitivity of noise')
plt.xlabel('Noise')
plt.ylabel('Tau estimation')
plt.ylim([-0.06,0.1])
plt.xlim([0,8])
plt.xticks(x,labels = ticksss)
for i in range(7):
    plt.errorbar(x[i], Taumeans[i], yerr = Taustds[i],marker = 'o')
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

TaumeansSim = [SimTaumean1,SimTaumean2,SimTaumean3,SimTaumean4,SimTaumean5,SimTaumean6,SimTaumean7]
TaustdsSim = [np.sqrt(SimTauvar1),np.sqrt(SimTauvar2),np.sqrt(SimTauvar3),np.sqrt(SimTauvar4),np.sqrt(SimTauvar5),np.sqrt(SimTauvar6),np.sqrt(SimTauvar7)]

ApmeansSim = [SimApmean1,SimApmean2,SimApmean3,SimApmean4,SimApmean5,SimApmean6,SimApmean7]
ApstdsSim = [np.sqrt(SimApvar1),np.sqrt(SimApvar2),np.sqrt(SimApvar3),np.sqrt(SimApvar4),np.sqrt(SimApvar5),np.sqrt(SimApvar6),np.sqrt(SimApvar7)]


plt.figure()
plt.title('Sensitivity of noise')
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
plt.title('Sensitivity of noise')
plt.xlabel('Noise')
plt.ylabel('Ap estimation')
plt.ylim([0.002,0.008])
plt.xlim([0,8])
plt.xticks(x,labels = ticksss)
for i in range(7):
    plt.errorbar(x[i], ApmeansSim[i], yerr = ApstdsSim[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''
plt.figure()
plt.title('Ap inference - 0.005 noise')
plt.xlabel('Iteration')
plt.ylabel('Ap')
plt.ylim([0,0.01])
plt.plot(np.linspace(1,1500,1500),Ap7,'ko')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
