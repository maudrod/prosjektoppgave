#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:20:28 2020

@author: emilam
"""
import numpy as np              
import matplotlib.pyplot as plt 
from scipy.stats import gamma
from tqdm import tqdm
import seaborn as sns
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

def proposal_step(shapes,theta,par_ind):
    theta_new = np.copy(theta)
    for i in par_ind:
        theta_new[i] = np.random.gamma(shapes[i],theta[i]/shapes[i])
    return theta_new

def adjust_variance(theta, U, par_ind,N,it,shapes):
    mean = np.zeros(N)
    var_new = np.zeros(N)
    theta_new = np.copy(theta[-1])
    while (any(i == 0 for i in var_new)):
        for i in range(N):
            if i == 0:
                mean[i] = theta[-U+1::2].mean(0)[i]
                var_new[i] = theta[-U+1::2].var(0)[i]*(2.4**2)
            elif i == 1:
                mean[i] = theta[-U::2].mean(0)[i]
                var_new[i] = theta[-U::2].var(0)[i]*(2.4**2)
            else:
                mean[i] = theta[-U:].mean(0)[i]
                var_new[i] = theta[-U:].var(0)[i]*(2.4**2)
        U += 100
        if U > it:
            return shapes, np.array([(np.random.gamma(shapes[i],theta[-1][i]/shapes[i])) for i in range(N)])
    new_shapes = np.array([((mean[i]**2) / var_new[i]) for i in range(N)])
    for i in par_ind:
        theta_new[i] = np.random.gamma(new_shapes[i],theta[-1][i]/new_shapes[i])
    return new_shapes,theta_new

def ratio(prob_old,prob_next,shapes_prior,rates_prior,shapes,theta_next,theta_prior,N):
    spike_prob_ratio = prob_next / prob_old
    prior_ratio, proposal_ratio = 1,1
    for i in range(N):
        prior_ratio *= gamma.pdf(theta_next[i],a=shapes_prior[i],scale=1/rates_prior[i])/\
        gamma.pdf(theta_prior[i],a=shapes_prior[i],scale=1/rates_prior[i])
        proposal_ratio *= gamma.pdf(theta_prior[i],a=shapes[i],scale=theta_next[i]/shapes[i])/\
        gamma.pdf(theta_next[i],a=shapes[i],scale=theta_prior[i]/shapes[i])
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

def particle_filter(w0,b2,theta,s1,s2,std,P,binsize,timesteps):
    '''
    Particle filtering, (doesnt quite work yet, smth with weights vp)
    Possible to speed it up? 
    How to initiate w0 and vp?
    '''
    #timesteps = np.int(seconds/binsize)
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
        lr = learning_rule(s1,s2,theta[0],theta[0]*1.05,theta[1],theta[1],t,i,binsize) 
        ls = likelihood_step(s1[i-1],s2[i],wp[:,i-1],b2)  
        vp = ls*v_normalized
        wp[:,i] = wp[:,i-1] + lr + np.random.normal(0,std,size = P)
        log_posterior += np.log(np.sum(vp)/P)
        t[i] = i*binsize
    return log_posterior

def MHsampler(w0est,b2est,shapes_prior,rates_prior,s1,s2,std,P,binsize,timesteps,U,it,N):
    '''
    Monte Carlo sampling with particle filtering, algoritme 3
    '''
    theta_prior = parameter_priors(shapes_prior,rates_prior)
    #theta_prior = np.array([0.001,0.005])
    theta = np.array([theta_prior])
    shapes = np.copy(shapes_prior)
    par_ind = np.linspace(0,N-1,N).astype(int)
    old_log_post = particle_filter(w0est,b2est,theta_prior,s1,s2,std,P,binsize,timesteps)
    for i in tqdm(range(1,it)):
        ex = [1,0][i % 2 == 0] #oddetall iterasjoner, eksluder Tau (index 1), partall: Ap (index 0)
        par_ind_temp = np.delete(par_ind,ex)
        if (i % U == 0):
            shapes, theta_next = adjust_variance(theta,U,par_ind_temp,N,it,shapes)
        else:    
            theta_next = proposal_step(shapes,theta_prior,par_ind_temp)
        new_log_post = particle_filter(w0est,b2est,theta_next,s1,s2,std,P,binsize,timesteps)
        prob_old,prob_next = scaled2_spike_prob(old_log_post,new_log_post)
        r = ratio(prob_old,prob_next,shapes_prior,rates_prior,shapes,theta_next,theta_prior,N)
        print('old theta:', theta_prior)
        print('new theta:', theta_next)
        print('r:',r)
        choice = np.int(np.random.choice([1,0], 1, p=[min(1,r),1-min(1,r)]))
        theta_choice = [np.copy(theta_prior),np.copy(theta_next)][choice == 1]
        print('choice:', theta_choice)
        theta = np.vstack((theta, theta_choice))
        theta_prior = np.copy(theta_choice)
        old_log_post = [np.copy(old_log_post),np.copy(new_log_post)][choice == 1]
    return theta
        
        
'''
PARAMETERS AND RUNNING OF ALGORITHM :
'''        
std = 0.0001
#w0 = 1.0
#b1 = -2.0
#b2 = -2.0
#Ap = 0.0025
#Am = Ap*1.05
#tau = 40.0e-3
#seconds = 120.0
#binsize = 1/200.0
P = 1000
U = 200
it = 1500
estimate_noise = False 
N = [2,3][estimate_noise == True] #number of parameters to estimate
shapes_prior = [np.array([4,5]),np.array([4,5,5])][estimate_noise == True]
rates_prior = [np.array([50,100]),np.array([50,100,350])][estimate_noise == True]

#a = np.load('PreNeur_PostNeur_Timesteps(ms).npy')
#spk_pre = a[0][40000:90000]
#spk_post = a[1][40000:90000]
#timesteps = len(spk_pre)
#binsize = (a[2][1] - a[2][0]) / 1000

#b1est = infer_b1(spk_pre)
#b2est = infer_b2_w0(spk_pre, spk_post, 1e-10)[0]
#w0est = infer_b2_w0(spk_pre[:10000], spk_post[:10000], 1e-10)[1]

#Simest1 = MHsampler(w0est, b2est, shapes_prior, rates_prior, spk_pre, spk_post, std, P, binsize, timesteps, U, it,N)
'''
plt.figure()
plt.xlim([0,0.01])
plt.plot(Simest1[300:,1],Simest1[300:,0],'ko')
plt.show()
'''
plt.figure()
sns.displot(Simest1[300:,0], kde=True,bins=10000)
plt.xlim([0.00,0.005])
#plt.axvline(0.005,color='r',linestyle='--',label='True Value')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title('Posterior distribution $A_+$ - 0.0001 noise')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()
plt.show()

plt.figure()
sns.displot(Simest1[300:,1], kde=True,bins=10000)
plt.xlim([0.00,0.007])
#plt.axvline(0.005,color='r',linestyle='--',label='True Value')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title('Posterior distribution tau - 0.0001 noise')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()
plt.show()
