#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:58:27 2020

@author: emilam
"""
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from scipy import stats
'''
w02_275 = np.load('w0estimates2ms_bm2.75.npy')
w02_31= np.load('w0estimates2ms_bm3.1.npy')

w05 = np.load('w0estimates5ms.npy')

ci1 = stats.norm.interval(0.95,np.mean(w02_275),np.sqrt(np.var(w02_275)))
ci2 = stats.norm.interval(0.95,np.mean(w02_31),np.sqrt(np.var(w02_31)))
ci3 = stats.norm.interval(0.95,np.mean(w05),np.sqrt(np.var(w05)))

b1_275 = np.load('B1estimates2ms_bm2.75.npy')
b1_31= np.load('B1estimates2ms_bm3.1.npy')

b105 = np.load('B1estimates5ms.npy')

b1ci1 = stats.norm.interval(0.95,np.mean(b1_275),np.sqrt(np.var(b1_275)))
b1ci2 = stats.norm.interval(0.95,np.mean(b1_31),np.sqrt(np.var(b1_31)))
b1ci3 = stats.norm.interval(0.95,np.mean(b105),np.sqrt(np.var(b105)))


b2_275 = np.load('B2estimates2ms_bm2.75.npy')
b2_31= np.load('B2estimates2ms_bm3.1.npy')

b205 = np.load('B2estimates5ms.npy')

b2ci1 = stats.norm.interval(0.95,np.mean(b2_275),np.sqrt(np.var(b2_275)))
b2ci2 = stats.norm.interval(0.95,np.mean(b2_31),np.sqrt(np.var(b2_31)))
b2ci3 = stats.norm.interval(0.95,np.mean(b205),np.sqrt(np.var(b205)))


plt.figure()
plt.title('W0 estimation, 2ms gridsize, $b_1 = b_2$ = -2.75')
sns.set_style('darkgrid')
sns.distplot(w02_275, norm_hist = True)
plt.axvline(1,color='r',linestyle='--',label='True Value')
plt.axvline(ci1[0],color='g',linestyle='--')
plt.axvline(ci1[1],color='g',linestyle='--',label='95% CI')
plt.legend()
plt.show()

plt.figure()
plt.title('W0 estimation, 2ms gridsize, $b_1 = b_2$ = -3.1')
sns.set_style('darkgrid')
sns.distplot(w02_31, norm_hist = True)
plt.axvline(1,color='r',linestyle='--',label='True Value')
plt.axvline(ci2[0],color='g',linestyle='--')
plt.axvline(ci2[1],color='g',linestyle='--',label='95% CI')
plt.legend()
plt.show()

plt.figure()
plt.title('W0 estimation, 5ms gridsize')
sns.set_style('darkgrid')
sns.distplot(w05, norm_hist = True)
plt.axvline(1,color='r',linestyle='--',label='True Value')
plt.axvline(ci3[0],color='g',linestyle='--')
plt.axvline(ci3[1],color='g',linestyle='--',label='95% CI')
plt.legend()
plt.show()

plt.figure()
plt.title('$b_1$ estimation, 2ms gridsize, $b_1 = b_2$ = -2.75')
sns.set_style('darkgrid')
sns.distplot(b1_275, norm_hist = True)
plt.axvline(-2.75,color='r',linestyle='--',label='True Value')
plt.axvline(b1ci1[0],color='g',linestyle='--')
plt.axvline(b1ci1[1],color='g',linestyle='--',label='95% CI')
plt.legend()
plt.show()

plt.figure()
plt.title('$b_1$ estimation, 2ms gridsize, $b_1 = b_2$ = -3.1')
sns.set_style('darkgrid')
sns.distplot(b1_31, norm_hist = True)
plt.axvline(-3.1,color='r',linestyle='--',label='True Value')
plt.axvline(b1ci2[0],color='g',linestyle='--')
plt.axvline(b1ci2[1],color='g',linestyle='--',label='95% CI')
plt.legend()
plt.show()

plt.figure()
plt.title('$b_1$ estimation, 5ms gridsize')
sns.set_style('darkgrid')
sns.distplot(b105, norm_hist = True)
plt.axvline(-2,color='r',linestyle='--',label='True Value')
plt.axvline(b1ci3[0],color='g',linestyle='--')
plt.axvline(b1ci3[1],color='g',linestyle='--',label='95% CI')
plt.legend()
plt.show()

plt.figure()
plt.title('$b_2$ estimation, 2ms gridsize, $b_1 = b_2$ = -2.75')
sns.set_style('darkgrid')
sns.distplot(b2_275, norm_hist = True)
plt.axvline(-2.75,color='r',linestyle='--',label='True Value')
plt.axvline(b2ci1[0],color='g',linestyle='--')
plt.axvline(b2ci1[1],color='g',linestyle='--',label='95% CI')
plt.legend()
plt.show()

plt.figure()
plt.title('$b_2$ estimation, 2ms gridsize, $b_1 = b_2$ = -3.1')
sns.set_style('darkgrid')
sns.distplot(b2_31, norm_hist = True)
plt.axvline(-3.1,color='r',linestyle='--',label='True Value')
plt.axvline(b2ci2[0],color='g',linestyle='--')
plt.axvline(b2ci2[1],color='g',linestyle='--',label='95% CI')
plt.legend()
plt.show()
'''
plt.figure()
plt.title('$b_2$ estimation, 5ms gridsize')
sns.set_style('darkgrid')
sns.distplot(b205, norm_hist = True)
plt.axvline(-2,color='r',linestyle='--',label='True Value')
plt.axvline(b2ci3[0],color='g',linestyle='--')
plt.axvline(b2ci3[1],color='g',linestyle='--',label='95% CI')
plt.legend()
plt.show()


