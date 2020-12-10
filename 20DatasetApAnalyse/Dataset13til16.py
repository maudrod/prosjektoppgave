#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:31:13 2020

@author: emilam
"""
import numpy as np              
import matplotlib.pyplot as plt 
from scipy.stats import gamma
from numba import njit
@njit

def learning_rule(s1,s2,Ap,Am,taup,taum,t,i,binsize): 
    '''
    5.8 in article (typo in article, should be negative exponent for e)
    s1,s2 : binary values for the different time bins for neuron 1 and 2 respectively
    t : numpy array with the measured time points
    Ap,Am,taup,taum : learning rule parameters 
    '''
    l = i - np.int(np.ceil(10*taup / binsize))
    return s2[i-1]*np.sum(s1[max([l,0]):i]*Ap*np.exp((t[max([l,0]):i]-max(t))/taup)) - s1[i-1]*np.sum(s2[max([l,0]):i]*Am*np.exp((t[max([l,0]):i]-max(t))/taum))

def logit(x):
    return np.log(x/(1-x))

def inverse_logit(x):
    return np.exp(x)/(1+np.exp(x))

def generative(Ap,Am,taup,taum,b1,b2,w0,std,seconds,binsize):
    '''time and binsize measured in seconds'''
    iterations = np.int(seconds/binsize)
    t,W,s1,s2 = np.zeros(iterations),np.zeros(iterations),np.zeros(iterations),np.zeros(iterations)
    W[0] = w0 #Initial value for weights
    s1[0] = np.random.binomial(1,inverse_logit(b1)) #5.4 in article, generate spike/not for neuron 1
    for i in range(1,iterations):
        s2[i] = np.random.binomial(1,inverse_logit(W[i-1]*s1[i-1]+b2)) #5.5 in article, spike/not neuron 2
        lr = learning_rule(s1,s2,Ap,Am,taup,taum,t,i,binsize)
        W[i] = W[i-1] + lr + np.random.normal(0,std) #updating weights, as in 5.8 in article
        s1[i] = np.random.binomial(1,inverse_logit(b1)) #5.4
        t[i] = binsize*i #list with times (start time of current bin)
    return(s1,s2,t,W)

def plot_gen_weight(t,W):
    plt.figure()
    plt.title('Weight trajectory')
    plt.plot(t,W)
    plt.xlabel('Time')
    plt.ylabel('Weight')
    #plt.legend()
    plt.show()

def infer_b1(s1):
    return logit(np.sum(s1)/len(s1)) #5.23 

def normalize(vp): #normalisere vekter 
    return vp/np.sum(vp)

def perplexity_func(vp_normalized,P):
    h = -np.sum(vp_normalized*np.log(vp_normalized))
    return np.exp(h)/P

def resampling(vp_normalized,wp,P):
    wp_new = np.copy(wp)
    indexes = np.linspace(0,P-1,P)
    resampling_indexes = np.random.choice(indexes,P,p=vp_normalized)
    for i in range(P):
        wp_new[i] = np.copy(wp[resampling_indexes.astype(int)[i]])
    return wp_new

def likelihood_step(s1,s2,w,b2): #p(s2 given s1,w,theta)
    return inverse_logit(w*s1 + b2)**(s2) * (1-inverse_logit(w*s1 + b2))**(1-s2)

def parameter_priors(shapes,rates):
    return np.array([(np.random.gamma(shapes[i],1/rates[i])) for i in range(len(shapes))])

def proposal_step(shapes,theta):
    return np.random.gamma(shapes,theta/shapes)

def adjust_variance(theta, U,it,shapes):
    means = theta[-U:].mean(0)
    var_new = 0
    while (var_new == 0):
        var_new = theta[-U:].var(0)*(2.4**2)
        U += 50
        if U > it:
            return shapes, np.random.gamma(shapes,theta[-1]/shapes)
    new_shapes = means**2 / var_new
    proposal = np.random.gamma(new_shapes,theta[-1]/new_shapes)
    return new_shapes,proposal
    
def ratio(prob_old,prob_next,shapes_prior,rates_prior,shapes,theta_next,theta_prior):
    spike_prob_ratio = prob_next / prob_old
    prior_ratio = gamma.pdf(theta_next,a=shapes_prior,scale=1/rates_prior)/\
        gamma.pdf(theta_prior,a=shapes_prior,scale=1/rates_prior)
    proposal_ratio = gamma.pdf(theta_prior,a=shapes,scale=theta_next/shapes)/\
        gamma.pdf(theta_next,a=shapes,scale=theta_prior/shapes)
    return spike_prob_ratio * prior_ratio * proposal_ratio


def scaled2_spike_prob(old,new):
    return np.exp(old - min(old,new)),np.exp(new - min(old,new))
    
def infer_b2_w0(s1,s2,tol):
    '''
    Fisher scoring algorithm 
    '''
    beta = [0,0] 
    x = np.array([np.ones(len(s1)-1),s1[:-1]])
    i = 0
    score = np.array([np.inf,np.inf])
    while(i < 1000 and any(abs(i) > tol for i in score)):
        eta = np.matmul(beta,x) #linear predictor
        mu = inverse_logit(eta)
        score = np.matmul(x,s2[1:] - mu)
        hessian_u = mu * (1-mu)
        hessian = np.matmul(x*hessian_u,np.transpose(x))
        delta = np.matmul(np.linalg.inv(hessian),score)
        beta = beta + delta
        i += 1
    return beta

def particle_filter(w0,b2,theta,s1,s2,std,P,binsize,seconds,tau):
    '''
    Particle filtering, (doesnt quite work yet, smth with weights vp)
    Possible to speed it up? 
    How to initiate w0 and vp?
    '''
    timesteps = np.int(seconds/binsize)
    t = np.zeros(timesteps)
    wp = np.full((P,timesteps),np.float(w0))
    vp = np.ones(P)
    log_posterior = 0
    for i in range(1,timesteps):
        v_normalized = normalize(vp)
        perplexity = perplexity_func(v_normalized,P)
        if perplexity < 0.66:
            wp = resampling(v_normalized,wp,P)
            vp = np.full(P,1/P)
            v_normalized = normalize(vp)
        lr = learning_rule(s1,s2,theta,theta*1.05,tau,tau,t,i,binsize)
        ls = likelihood_step(s1[i-1],s2[i],wp[:,i-1],b2)  
        vp = ls*v_normalized
        wp[:,i] = wp[:,i-1] + lr + np.random.normal(0,std,size = P)
        log_posterior += np.log(np.sum(vp)/P)
        t[i] = i*binsize
    v_normalized = normalize(vp)
    return log_posterior

def MHsampler2(w0,b2est,shapes_prior,rates_prior,s1,s2,std,P,binsize,seconds,U,it,tau):
    '''
    Monte Carlo sampling with particle filtering, algoritme 3
    '''
    theta_prior = 0.001
    theta = np.zeros(it)
    theta[0] = np.copy(theta_prior)
    shapes = np.copy(shapes_prior)
    old_log_post = particle_filter(w0,b2est,theta_prior,s1,s2,std,P,binsize,seconds,tau)
    for i in range(1,it):
        if (i % U == 0):
            theta_change = np.copy(theta[:i])
            shapes, theta_next = adjust_variance(theta_change,U,it,shapes)
        else:    
            theta_next = proposal_step(shapes,theta_prior)
        new_log_post = particle_filter(w0,b2est,theta_next,s1,s2,std,P,binsize,seconds,tau)
        #print('old:', theta_prior)
        #print('new:', theta_next)
        prob_old,prob_next = scaled2_spike_prob(old_log_post,new_log_post)
        r = ratio(prob_old,prob_next,shapes_prior,rates_prior,shapes,theta_next,theta_prior)
        #print('r:',r)
        choice = np.int(np.random.choice([1,0], 1, p=[min(1,r),1-min(1,r)]))
        theta_choice = [np.copy(theta_prior),np.copy(theta_next)][choice == 1]
        #print('choice:',theta_choice)
        theta[i] = theta_choice
        theta_prior = np.copy(theta_choice)
        old_log_post = [np.copy(old_log_post),np.copy(new_log_post)][choice == 1]
    return theta
        
        
'''
PARAMETERS AND RUNNING OF ALGORITHM :
'''        
std = 0.005
w0 = 1.0
b1 = -2
b2 = -2
Ap = 0.005
Am = Ap*1.05
tau = 20.0e-3
seconds = 120.0
binsize = 1/200.0
P = 1000
U = 100
it = 1500
shapes_prior = 4
rates_prior = 50


w0ests = []
s1s = []
s2s = []
b1ests = []
b2ests = []
stds = [0.0001,0.0005,0.001,0.003,0.005]
Aps = []
for i in range(4):
    w0est1 = -np.inf
    Aps_temp = []
    while (w0est1 < 0.97 or w0est1 > 1.03):
        s11,s21,t,W = generative(Ap, Am, tau, tau, b1, b2, w0, stds[0], seconds, binsize)
        b1est1 = infer_b1(s11)
        b2est1 = infer_b2_w0(s11, s21, 1e-10)[0]
        w0est1 = infer_b2_w0(s11[:2000], s21[:2000], 1e-10)[1]
    for j in range(len(stds)):
        Apest = MHsampler2(w0est1, b2est1, shapes_prior, rates_prior, s11, s21, stds[j], P, binsize, seconds, U, it, tau) 
        Aps_temp.append(Apest)
    w0ests.append(w0est1)
    s1s.append(s11)
    s2s.append(s21)
    b1ests.append(b1est1)
    b2ests.append(b2est1)
    Aps.append(Aps_temp)
    

#np.save('S1Datasets1to4Opposite',s1s)
#np.save('S2Datasets1to4Opposite',s2s)
#np.save('b1sDatasets1to4Opposite',b1ests)
#np.save('b2sDatasets1to4Opposite',b2ests)
np.save('ApInferenceData13to16Opposite',Aps)
#np.save('w0ests1to4Opposite',w0ests)
