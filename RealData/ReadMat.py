#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:45:07 2020

@author: emilam
"""
import numpy as np
from mat4py import loadmat
import matplotlib.pyplot as plt

#data = loadmat('shortTermPlasticity1.mat')
'''
data18102 = loadmat('camkii10_180102.spikes.cellinfo.mat')

data18104 = loadmat('camkii10_180104.spikes.cellinfo.mat')

data18220 = loadmat('camkii13_181220.spikes.cellinfo.mat')

data18103 = loadmat('camkii10_180103.spikes.cellinfo.mat')
data18105 = loadmat('camkii10_180105.spikes.cellinfo.mat')
data18125 = loadmat('camkii10_180125.spikes.cellinfo.mat')
data18126 = loadmat('camkii10_180126.spikes.cellinfo.mat')
data18221 = loadmat('camkii13_181221.spikes.cellinfo.mat')

data18228 = loadmat('camkii13_181228.spikes.cellinfo.mat')

data18229 = loadmat('camkii13_181229.spikes.cellinfo.mat')

data18230 = loadmat('camkii13_181230.spikes.cellinfo.mat')

data18231 = loadmat('camkii13_181231.spikes.cellinfo.mat')
#data3= loadmat('camkii10_180123_CellParams.mat')

data19109 = loadmat('camkii13_190109.spikes.cellinfo.mat')
data19110 = loadmat('camkii13_190110.spikes.cellinfo.mat')
data19112 = loadmat('camkii13_190112.spikes.cellinfo.mat')
data19113 = loadmat('camkii13_190113.spikes.cellinfo.mat')
data19114 = loadmat('camkii13_190114.spikes.cellinfo.mat')
data19115 = loadmat('camkii13_190115.spikes.cellinfo.mat')
'''

#st1 = np.concatenate(data19112['spikes']['times'][2]).round(1)
#st2 = np.concatenate(data19112['spikes']['times'][3]).round(1)
#st3 = np.concatenate(data19112['spikes']['times'][11]).round(1)
#st4 = np.concatenate(data19112['spikes']['times'][25]).round(1)

#st1 = np.concatenate(data19110['spikes']['times'][3]).round(1)
#st2 = np.concatenate(data19110['spikes']['times'][11]).round(1)
#st3 = np.concatenate(data19110['spikes']['times'][23]).round(1)
#st4 = np.concatenate(data19110['spikes']['times'][24]).round(1)

#st1 = np.concatenate(data19113['spikes']['times'][2]).round(1)
#st2 = np.concatenate(data19113['spikes']['times'][3]).round(1)
#st3 = np.concatenate(data19113['spikes']['times'][5]).round(1)
#st4 = np.concatenate(data19113['spikes']['times'][21]).round(1)
#st5 = np.concatenate(data19113['spikes']['times'][24]).round(1)
#st6 = np.concatenate(data19113['spikes']['times'][11]).round(1)

st1 = np.concatenate(data18231['spikes']['times'][2]).round(1)
st2 = np.concatenate(data18231['spikes']['times'][4]).round(1)
st3 = np.concatenate(data18231['spikes']['times'][7]).round(1)
st4 = np.concatenate(data18231['spikes']['times'][12]).round(1)
st5 = np.concatenate(data18231['spikes']['times'][15]).round(1)
st6 = np.concatenate(data18231['spikes']['times'][16]).round(1)
st7 = np.concatenate(data18231['spikes']['times'][17]).round(1)
st8 = np.concatenate(data18231['spikes']['times'][20]).round(1)
st9 = np.concatenate(data18231['spikes']['times'][21]).round(1)
st10 = np.concatenate(data18231['spikes']['times'][30]).round(1)
st11 = np.concatenate(data18231['spikes']['times'][32]).round(1)

end = int(max(st1[-1],st2[-1],st3[-1],st4[-1],st5[-1],st6[-1],st7[-1],st8[-1],st9[-1],st10[-1],st11[-1]))

binsize = 0.1 #milliseconds
bins = int((end)/binsize)
timesteps = np.linspace(0,end-binsize,bins)

s1,s2,s3,s4,s5,s6 = np.zeros(bins),np.zeros(bins),np.zeros(bins),np.zeros(bins),np.zeros(bins),np.zeros(bins)
s7,s8,s9,s10,s11 = np.zeros(bins),np.zeros(bins),np.zeros(bins),np.zeros(bins),np.zeros(bins)
for i in range(bins):
    if timesteps[i] in st1:
        s1[i] = 1
    if timesteps[i] in st2:
        s2[i] = 1
    if timesteps[i] in st3:
        s3[i] = 1
    if timesteps[i] in st4:
        s4[i] = 1
    if timesteps[i] in st5:
        s5[i] = 1
    if timesteps[i] in st6:
        s6[i] = 1
    if timesteps[i] in st7:
        s7[i] = 1
    if timesteps[i] in st8:
        s8[i] = 1
    if timesteps[i] in st9:
        s9[i] = 1
    if timesteps[i] in st10:
        s10[i] = 1
    if timesteps[i] in st11:
        s11[i] = 1



plt.figure()
plt.title('s10 vs s11')
plt.xcorr(s10,s11,usevlines=True, maxlags=50, normed=True, lw=2)
plt.show()


'''
Potential interesting connections so far?
Data19110: index 11v23, index 11v24
Data19109: index 2v4
Data19113: index 11v3!!!
Data18231 index 4v21?, GOD KANTIDAT: 7v32 (fra bin 30000:)??
'''

'''
Ekempler på IKKE korrelerte:
Data19109 index 2v5
Data18231: index 7v17, index 15v16
'''

