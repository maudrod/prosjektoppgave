#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:45:07 2020

@author: emilam
"""
import numpy as np
from mat4py import loadmat

#data = loadmat('shortTermPlasticity1.mat')

data102 = loadmat('camkii10_180102.spikes.cellinfo.mat')

data104 = loadmat('camkii10_180104.spikes.cellinfo.mat')

data13 = loadmat('camkii13_181220.spikes.cellinfo.mat')

#data3= loadmat('camkii10_180123_CellParams.mat')

info104 = loadmat('camkii10_180104.sessionInfo.mat')

info102 = loadmat('camkii10_180102.sessionInfo.mat')


#spiketrains = data104[]