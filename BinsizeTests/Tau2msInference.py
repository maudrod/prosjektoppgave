#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:27:55 2020

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

def MHsampler2(w0,b2est,shapes_prior,rates_prior,s1,s2,std,P,binsize,seconds,U,it,Ap):
    '''
    Monte Carlo sampling with particle filtering, algoritme 3
    '''
    theta_prior = 0.004
    theta = np.zeros(it)
    theta[0] = np.copy(theta_prior)
    shapes = np.copy(shapes_prior)
    old_log_post = particle_filter(w0,b2est,Ap,s1,s2,std,P,binsize,seconds,theta_prior)
    for i in range(1,it):
        if (i % U == 0):
            theta_change = np.copy(theta[:i])
            shapes, theta_next = adjust_variance(theta_change,U,it,shapes)
        else:    
            theta_next = proposal_step(shapes,theta_prior)
        new_log_post = particle_filter(w0,b2est,Ap,s1,s2,std,P,binsize,seconds,theta_next)
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
std = 0.0001
w0 = 1.0
b1 = -2.75
b2 = -2.75
Ap = 0.005
Am = Ap*1.05
tau = 20.0e-3
seconds = 120.0
binsize = 1/500.0
P = 1000
U = 100
it = 1500
shapes_prior = 5
rates_prior = 100


w0est = -np.inf
while (w0est < 0.97 or w0est > 1.03):
    s1,s2,t,W = generative(Ap, Am, tau, tau, b1, b2, w0, std, seconds, binsize)
    b1est = infer_b1(s1)
    b2est = infer_b2_w0(s1, s2, 1e-10)[0]
    w0est = infer_b2_w0(s1[:5000], s2[:5000], 1e-10)[1]

w0est2 = -np.inf
while (w0est2 < 0.97 or w0est2 > 1.03):
    s12,s22,t,W2 = generative(Ap, Am, tau, tau, b1, b2, w0, std, seconds, binsize)
    b1est2 = infer_b1(s12)
    b2est2 = infer_b2_w0(s12, s22, 1e-10)[0]
    w0est2 = infer_b2_w0(s12[:5000], s22[:5000], 1e-10)[1]

w0est3 = -np.inf
while (w0est3 < 0.97 or w0est3 > 1.03):
    s13,s23,t,W3 = generative(Ap, Am, tau, tau, b1, b2, w0, std, seconds, binsize)
    b1est3 = infer_b1(s13)
    b2est3 = infer_b2_w0(s13, s23, 1e-10)[0]
    w0est3 = infer_b2_w0(s13[:5000], s23[:5000], 1e-10)[1]

w0est4 = -np.inf
while (w0est4 < 0.97 or w0est4 > 1.03):
    s14,s24,t,W4 = generative(Ap, Am, tau, tau, b1, b2, w0, std, seconds, binsize)
    b1est4 = infer_b1(s14)
    b2est4 = infer_b2_w0(s14, s24, 1e-10)[0]
    w0est4 = infer_b2_w0(s14[:5000], s24[:5000], 1e-10)[1]

w0est5 = -np.inf
while (w0est5 < 0.97 or w0est5 > 1.03):
    s15,s25,t,W5 = generative(Ap, Am, tau, tau, b1, b2, w0, std, seconds, binsize)
    b1est5 = infer_b1(s15)
    b2est5 = infer_b2_w0(s15, s25, 1e-10)[0]
    w0est5 = infer_b2_w0(s15[:5000], s25[:5000], 1e-10)[1]

w0est6 = -np.inf
while (w0est6 < 0.97 or w0est6 > 1.03):
    s16,s26,t,W6 = generative(Ap, Am, tau, tau, b1, b2, w0, std, seconds, binsize)
    b1est6 = infer_b1(s16)
    b2est6 = infer_b2_w0(s16, s26, 1e-10)[0]
    w0est6 = infer_b2_w0(s16[:5000], s26[:5000], 1e-10)[1]

w0est7 = -np.inf
while (w0est7 < 0.97 or w0est7 > 1.03):
    s17,s27,t,W7 = generative(Ap, Am, tau, tau, b1, b2, w0, std, seconds, binsize)
    b1est7 = infer_b1(s17)
    b2est7 = infer_b2_w0(s17, s27, 1e-10)[0]
    w0est7 = infer_b2_w0(s17[:5000], s27[:5000], 1e-10)[1]

Tauest1 = MHsampler2(w0est, b2est, shapes_prior, rates_prior, s1, s2, std, P, binsize, seconds, U, it, Ap)

std = 0.0005


Tauest2 = MHsampler2(w0est2, b2est2, shapes_prior, rates_prior, s12, s22, std, P, binsize, seconds, U, it, Ap)

std = 0.001


Tauest3 = MHsampler2(w0est3, b2est3, shapes_prior, rates_prior, s13, s23, std, P, binsize, seconds, U, it, Ap)

std = 0.002

Tauest4 = MHsampler2(w0est4, b2est4, shapes_prior, rates_prior, s14, s24, std, P, binsize, seconds, U, it, Ap)


std = 0.003

Tauest5 = MHsampler2(w0est5, b2est5, shapes_prior, rates_prior, s15, s25, std, P, binsize, seconds, U, it, Ap)


std = 0.004

Tauest6 = MHsampler2(w0est6, b2est6, shapes_prior, rates_prior, s16, s26, std, P, binsize, seconds, U, it, Ap)


std = 0.005

Tauest7 = MHsampler2(w0est7, b2est7, shapes_prior, rates_prior, s17, s27, std, P, binsize, seconds, U, it, Ap)

np.save('Tau0.0001noise2msB275',Tauest1)
np.save('Tau0.0005noise2msB275',Tauest2)
np.save('Tau0.001noise2msB275',Tauest3)
np.save('Tau0.002noise2msB275',Tauest4)
np.save('Tau0.003noise2msB275',Tauest5)
np.save('Tau0.004noise2msB275',Tauest6)
np.save('Tau0.005noise2msB275',Tauest7)