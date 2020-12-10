#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:45:07 2020

@author: emilam
"""
import numpy as np
from mat4py import loadmat
import matplotlib.pyplot as plt
import scipy.stats as stats
plt.style.use('seaborn-darkgrid')
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
#data18102 = loadmat('camkii10_180102.spikes.cellinfo.mat')
#st1 = np.concatenate(data18102['spikes']['times'][13])
#st2 = np.concatenate(data18102['spikes']['times'][1])
#st3 = np.concatenate(data18102['spikes']['times'][14])
#st4 = np.concatenate(data18102['spikes']['times'][8])
#st5 = np.concatenate(data18102['spikes']['times'][9])
#st6 = np.concatenate(data18102['spikes']['times'][21])
#st1 = st1.astype(int)
#st2 = st2.astype(int)

#data19109 = loadmat('camkii13_190109.spikes.cellinfo.mat')
#st1 = np.concatenate(data19109['spikes']['times'][0])
#st2 = np.concatenate(data19109['spikes']['times'][4])
#st3 = np.concatenate(data19109['spikes']['times'][12])
#st4 = np.concatenate(data19109['spikes']['times'][14])
#st5 = np.concatenate(data19109['spikes']['times'][16])
#st6 = np.concatenate(data19109['spikes']['times'][19])


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

#st1 = np.concatenate(data18102['spikes']['times'][0])
#st2 = np.concatenate(data18102['spikes']['times'][1])
#st3 = np.concatenate(data18102['spikes']['times'][8])
#st4 = np.concatenate(data18102['spikes']['times'][9])
#st5 = np.concatenate(data18102['spikes']['times'][10])#*1000
#st6 = np.concatenate(data18102['spikes']['times'][14])
#st7 = np.concatenate(data18102['spikes']['times'][15])
#st8 = np.concatenate(data18102['spikes']['times'][20])
#st9 = np.concatenate(data18102['spikes']['times'][21])#*1000

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

#end = max(st1[-1],st2[-1])#,st7[-1],st8[-1],st9[-1],st10[-1],st11[-1]))

#st5 = st5.round()
#st9 = st9.round()
data18231 = loadmat('camkii13_181231.spikes.cellinfo.mat')

st2 = np.concatenate(data18231['spikes']['times'][8])
st5 = np.concatenate(data18231['spikes']['times'][10])
st1 = np.concatenate(data18231['spikes']['times'][1])
st3 = np.concatenate(data18231['spikes']['times'][9])
st4 = np.concatenate(data18231['spikes']['times'][0])
st6 = np.concatenate(data18231['spikes']['times'][23])

st2pos = st2[(np.where((st2 > 3900) & (st2 < 4035)))]

st5pos = st5[(np.where((st5 > 3900) & (st5 < 4035)))]



st2pos = st2pos*1000

st5pos = st5pos * 1000

st2pos = st2pos.astype(int)

st5pos = st5pos.astype(int)

def cicc(lags,significance,n):
    return stats.norm.interval(significance,0,np.sqrt(1/(n-abs(lags))))


start = min(st2pos[0],st5pos[0])
end = max(st2pos[-1],st5pos[-1])
binsize = 1
bins = int((end-start)/binsize)
timesteps = np.linspace(start,end-binsize,bins)

s1,s2 = np.zeros(bins),np.zeros(bins)
for i in range(bins):
    if (timesteps[i] in st2pos):#: or timesteps[i]+1 in st2pos):
        s1[i] = 1
    if (timesteps[i] in st5pos):# or timesteps[i]+1 in st5pos):
        s2[i] = 1

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

maxlag = 10
lags = np.linspace(-maxlag,maxlag,2*maxlag+1)
#ccov = plt.xcorr(s1 - s1.mean(), s2 - s2.mean(),maxlags=10,normed=True)
#ccor = (ccov[1]) / (len(s1) * s1.std() * s2.std())
ci = cicc(lags,0.99,len(s1))

plt.figure()
plt.title('Cross-correlation for the first 10 seconds')
plt.xcorr(s1 - s1.mean(), s2 - s2.mean(),maxlags=10,normed=True)
plt.plot(lags,ci[1],'r--',label='99% CI under $H_0$')
plt.plot(lags,ci[0],'r--')#,label='99% CI under $H_0$')
plt.ylim((-0.035,0.035))
#plt.xticks(x,labels = ms)
plt.xlabel('Timelag (ms)')
plt.legend(loc=1,fancybox = True)
plt.show()

'''
startsec = 3880
endsec = 4050
spiketrains = [st1[(np.where((st1 > startsec) & (st1 < endsec)))],st2[(np.where((st2 > startsec) & (st2 < endsec)))]\
               ,st3[(np.where((st3 > startsec) & (st3 < endsec)))],st4[(np.where((st4 > startsec) & (st4 < endsec)))]\
                   ,st5[(np.where((st5 > startsec) & (st5 < endsec)))],st6[(np.where((st6 > startsec) & (st6 < endsec)))]]
lineSize = [0.4, 0.4, 0.4, 0.4, 0.4,0.4]

spiketrain_halfs = []
for i in range(len(spiketrains)):
    temp = np.zeros(int(len(spiketrains[i])/4)+1)
    for j in range(len(spiketrains[i])):
        if j%4 == 0:
            temp[int(j/4)] = spiketrains[i][j]
    spiketrain_halfs.append(temp)
    
    
colorCodes = np.array([[0.3, 0.3, 0.4],

                        [0, 0, 1],

                       [0.3, 0.3, 0.4],

                        [0.3, 0.3, 0.4],

                        [0, 0, 1],

                        [0.3, 0.3, 0.4]])
plt.figure()
plt.title('Spike Trains of selected neurons')
plt.xlabel('Time (seconds)')
plt.xlim([startsec,endsec])
plt.ylim([-0.5,5.9])
plt.ylabel('Neuron')
plt.eventplot(spiketrain_halfs,linelengths=lineSize,colors= colorCodes)
plt.axvline(3914,color='g',linestyle='--',alpha=0.8)
plt.axvline(4034,color='g',linestyle='--',alpha=0.8,label='Possible stimulation time')
plt.legend(loc=('upper center'))
plt.show()
'''
'''
for i in range(4000,6000,100):
    startsec = i
    endsec = i + 100
    spiketrains = [st1[(np.where((st1 > startsec) & (st1 < endsec)))],st2[(np.where((st2 > startsec) & (st2 < endsec)))]\
                   ,st3[(np.where((st3 > startsec) & (st3 < endsec)))],st4[(np.where((st4 > startsec) & (st4 < endsec)))]\
                       ,st5[(np.where((st5 > startsec) & (st5 < endsec)))],st6[(np.where((st6 > startsec) & (st6 < endsec)))]]
    lineSize = [0.4, 0.4, 0.4, 0.4, 0.4,0.4]
    plt.figure()
    plt.title('Spike Trains of selected neurons')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Neuron')
    plt.eventplot(spiketrains,linelengths=lineSize)#,colors= colorCodes)
    plt.show()



colorCodes = np.array([[0, 0, 0],

                        [1, 0, 0],

                        [0, 1, 0],

                        [0, 0, 1],

                        [1, 1, 0],

                        [0, 1, 1]])
lineSize = [0.4, 0.4, 0.4, 0.4, 0.4,0.4]
'''

'''
Potential interesting connections so far?
Data18102: ind 1v9 (between seconds 1040 and 1160) PossiblePair?
Data18231 ind 0v10 between seconds 1900 and 2020 or 2660-2800 PossiblePair2?
Data18231 ind 8v10 between 3900-4035s PossiblePair3?
Data1909 : ind0v4 1360-1480s
'''

'''
Ekempler på IKKE korrelerte:
Data19109 index 2v5
Data18231: index 7v17, index 15v16
'''
'''
# DATA: 18231 ind 8v10!
test1 = st3[(np.where((st3 > 3900) & (st3 < 4035)))]

test2 = st4[(np.where((st4 > 3900) & (st4 < 4035)))]

test1halv = np.zeros(int(len(test1)/4)+1)
test2halv = np.zeros(int(len(test2)/4))
for i in range(len(test1)):
    if i%4 == 0:
        test1halv[int(i/4)] = test1[i]
        
for i in range(len(test2)):
    if i%4 == 0:
        test2halv[int(i/4)] = test2[i]

testspikes = [test1halv,test2halv]
plt.figure()
plt.title('Spike Trains of selected neurons')
plt.xlabel('Time (seconds)')
plt.ylabel('Neuron')
plt.axvline(3970,color='r',linestyle='--',label='Possible stimulus')
plt.eventplot(testspikes,linelengths=[0.2,0.2])#,colors= colorCodes)
plt.yticks([0,1],labels=['1','2'])
plt.legend()
plt.show()
'''  
    