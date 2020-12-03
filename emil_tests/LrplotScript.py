#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:07:03 2020

@author: emilam
"""

import numpy as np
import matplotlib.pyplot as plt

def lr1(s2,s1,Ap,delta,taup):
    return s2*s1*Ap*np.exp(-delta/taup)

def lr2(s1,s2,Am,delta,taum):
    return -s1*s2*Am*np.exp(delta/taum)
 
deltas = np.linspace(0,0.1,10000)
deltas2 = np.linspace(-0.1,0,10000)   
lrs1 = lr1(1,1,0.005,deltas,0.02)
lrs2 = lr2(1,1,0.005,deltas2,0.02) 

lrs3 = lr1(1,1,0.0075,deltas,0.02)
lrs4 = lr2(1,1,0.0075,deltas2,0.02) 

deltass = np.concatenate((deltas2,deltas))
lrss = np.concatenate((lrs2,lrs1))
lrss2 = np.concatenate((lrs4,lrs3))
x = [-0.1,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.1]
labels = [r'-5$\tau_{\pm}$',r'-4$\tau_{\pm}$',r'-3$\tau_{\pm}$',r'-2$\tau_{\pm}$',r'-$\tau_{\pm}$','0',r'$\tau_{\pm}$',r'2$\tau_{\pm}$',r'3$\tau_{\pm}$',r'4$\tau_{\pm}$',r'5$\tau_{\pm}$']
plt.figure()
plt.title('Learning rule')
plt.plot(deltass,lrss,c='r',label=r'$A_+=A_-: 0.005$'+'\n'+r'$\tau_{\pm}: 20ms$')
#plt.plot(deltass,lrss2,label=r'$A_+=A_-: 0.0075$'+'\n'+r'$\tau_{\pm}: 20ms$')
plt.xlabel('$\Delta t$')
plt.ylabel('$\Delta w$')
plt.xticks(x,labels)
plt.axhline(0,color='k',linestyle='--')
plt.legend()
plt.show()