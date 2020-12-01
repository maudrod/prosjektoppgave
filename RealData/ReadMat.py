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
data18102 = loadmat('camkii10_180102.spikes.cellinfo.mat')
#data18231 = loadmat('camkii13_181231.spikes.cellinfo.mat')
#st1 = np.concatenate(data19112['spikes']['times'][7])
#st2 = np.concatenate(data19112['spikes']['times'][11])
#st3 = np.concatenate(data19112['spikes']['times'][3])
#st4 = np.concatenate(data19112['spikes']['times'][2])

#st1 = np.concatenate(data19110['spikes']['times'][3]).round(1)
#st2 = np.concatenate(data19110['spikes']['times'][11]).round(1)
#st3 = np.concatenate(data19110['spikes']['times'][23]).round(1)
#st4 = np.concatenate(data19110['spikes']['times'][24]).round(1)

#st1 = np.concatenate(data19113['spikes']['times'][2])
#st2 = np.concatenate(data19113['spikes']['times'][3])
#st3 = np.concatenate(data19113['spikes']['times'][5])
#st4 = np.concatenate(data19113['spikes']['times'][21])
#st5 = np.concatenate(data19113['spikes']['times'][24])
#st6 = np.concatenate(data19113['spikes']['times'][11])

st1 = np.concatenate(data18102['spikes']['times'][0])
st2 = np.concatenate(data18102['spikes']['times'][1])
st3 = np.concatenate(data18102['spikes']['times'][8])
st4 = np.concatenate(data18102['spikes']['times'][9])
st5 = np.concatenate(data18102['spikes']['times'][10])#*1000
st6 = np.concatenate(data18102['spikes']['times'][14])
st7 = np.concatenate(data18102['spikes']['times'][15])
st8 = np.concatenate(data18102['spikes']['times'][20])
st9 = np.concatenate(data18102['spikes']['times'][21])#*1000

#st1 = np.concatenate(data18231['spikes']['times'][2])
#st2 = np.concatenate(data18231['spikes']['times'][4])
#st3 = np.concatenate(data18231['spikes']['times'][7])
#st4 = np.concatenate(data18231['spikes']['times'][12])
#st5 = np.concatenate(data18231['spikes']['times'][15])
#st6 = np.concatenate(data18231['spikes']['times'][16]).round(1)
#st6 = np.concatenate(data18231['spikes']['times'][17])
#st8 = np.concatenate(data18231['spikes']['times'][20]).round(1)
#st9 = np.concatenate(data18231['spikes']['times'][21]).round(1)
#st10 = np.concatenate(data18231['spikes']['times'][30]).round(1)
#st11 = np.concatenate(data18231['spikes']['times'][32]).round(1)
#spiketrains = [st1[7769:9548],st2[1625:1686],st3[1662:1785],st4[1527:1557]]#,st5,st6]
spiketrains = [st1[:2000],st2[:2000],st3[:2000],st4[:2000],st5[:2000],st6[:2000],st7[:2000],st8[:2000],st9[:2000]]
end = int(max(st1[-1],st2[-1],st4[-1],st5[-1],st6[-1]))#,st7[-1],st8[-1],st9[-1],st10[-1],st11[-1]))

#st5 = st5.round()
#st9 = st9.round()
start = int(1350*1000)
end = int(1488 * 1000)
binsize = 1
bins = int((end-start)/binsize)
timesteps = np.linspace(start,end-binsize,bins)

#s5,s9 = np.zeros(bins),np.zeros(bins)
#s1,s2,s3,s4,s5,s6 = np.zeros(bins),np.zeros(bins),np.zeros(bins),np.zeros(bins),np.zeros(bins),np.zeros(bins)
#s7,s8,s9,s10,s11 = np.zeros(bins),np.zeros(bins),np.zeros(bins),np.zeros(bins),np.zeros(bins)
#for i in range(bins):
#    if timesteps[i] in st5:
#        s5[i] = 1
#    if timesteps[i] in st9:
#        s9[i] = 1
#    if timesteps[i] in st3:
#        s3[i] = 1
#    if timesteps[i] in st4:
#        s4[i] = 1
#    if timesteps[i] in st5:
#        s5[i] = 1
#    if timesteps[i] in st6:
#        s6[i] = 1
    #if timesteps[i] in st7:
    #    s7[i] = 1
    #if timesteps[i] in st8:
    #    s8[i] = 1
    #if timesteps[i] in st9:
    #    s9[i] = 1
    #if timesteps[i] in st10:
    #    s10[i] = 1
    #if timesteps[i] in st11:
    #    s11[i] = 1


#plt.figure()
#plt.title('Cross-correlation of non-connected neurons')
#plt.xcorr(s5,s9,usevlines=True, maxlags=10, normed=True, lw=2)
#plt.xticks(x,labels = ms)
#plt.xlabel('Timelag (ms)')
#plt.show()

colorCodes = np.array([[0, 0, 0],

                        [1, 0, 0],

                        [0, 1, 0],

                        [0, 0, 1],

                        [1, 1, 0],

                        [0, 1, 1]])
lineSize = [0.4, 0.4, 0.4, 0.4, 0.4,0.4,0.4,0.4,0.4]
plt.figure()
plt.title('Spike Trains of selected neurons')
plt.xlabel('Time (seconds)')
plt.ylabel('Neuron')
plt.eventplot(spiketrains,linelengths=lineSize)#,colors= colorCodes)
plt.show()

'''
Potential interesting connections so far?
Data19110: index 11v23, index 11v24
Data19109: index 2v4
Data19113: index 11v3!!! (første test)
Data18231 index 4v21?, GOD KANTIDAT: 7v32 (fra bin 30000:)??
'''

'''
Ekempler på IKKE korrelerte:
Data19109 index 2v5
Data18231: index 7v17, index 15v16
'''

