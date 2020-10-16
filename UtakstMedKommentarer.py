#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 13:08:49 2020

@author: emilam
"""

import numpy as np              
import matplotlib.pyplot as plt 
from tqdm import tqdm
from scipy.stats import gamma

def learning_rule(s1,s2,Ap,Am,taup,taum,t,i): 
    '''
    5.8 in article (typo in article, should be negative exponent for e)
    s1,s2 : binary values for the different time bins for neuron 1 and 2 respectively
    t : numpy array with the measured time points
    Ap,Am,taup,taum : learning rule parameters 
    '''
    return s2[i-1]*np.sum(s1[:i]*Ap*np.exp((t[:i]-max(t))/taup)) - s1[i-1]*np.sum(s2[:i]*Am*np.exp((t[:i]-max(t))/taum))

def logit(x):
    '''
    LINK FUNCTION FOR BERNOULLI GLM (3.12)
    '''
    return np.log(x/(1-x))

def inverse_logit(x):
    '''
    INVERSE LINK FUNCTION FOR GLM, EXPRESSING THE EXPECTED VALUE THROUGH LINEAR PREDICTOR (3.12)
    '''
    return np.exp(x)/(1+np.exp(x))

def generative(Ap,Am,taup,taum,b1,b2,w0,std,time,binsize):
    '''time and binsize measured in seconds'''
    iterations = np.int(time/binsize)
    t,W,s1,s2 = np.zeros(iterations),np.zeros(iterations),np.zeros(iterations),np.zeros(iterations)
    W[0] = w0 #Initial value for weights
    s1[0] = np.random.binomial(1,inverse_logit(b1)) #5.4 in article, generate spike/not for neuron 1
    for i in tqdm(range(1,iterations)):
        lr = learning_rule(s1,s2,Ap,Am,taup,taum,t,i)
        W[i] = W[i-1] + lr + np.random.normal(0,std) #updating weights, as in 5.8 in article
        s2[i] = np.random.binomial(1,inverse_logit(W[i]*s1[i-1]+b2)) #5.5 in article, spike/not neuron 2
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
    '''
    #5.23 IN ARTICLE
    '''
    return logit(np.sum(s1)/len(s1))

def normalize(vp): 
    '''
    NORMALISERE VEKTER
    '''
    return vp/sum(vp)

def perplexity_func(vp_normalized,P):
    '''
    REGNE UT PERPLEXITY GITT VED (4.25) - BESTEMMER OM DET SKAL RESAMPLES. FEIL I FORMELEN I ARTIKKEL
    '''
    if any(vp_normalized) == 0:
        print('ERROR, WEIGHTS EQ. 0')
        return 0
    h = -np.sum(vp_normalized*np.log(vp_normalized))
    return np.exp(h)/P

def resampling(vp_normalized,wp,P):
    '''
    RESAMPLING ACCORDING TO SECTION 4.3.3
    '''
    wp_new = np.copy(wp)
    indexes = np.linspace(0,P-1,P)
    resampling_indexes = np.random.choice(indexes,P,p=vp_normalized)
    for i in range(P):
        wp_new[i] = wp[resampling_indexes.astype(int)[i]] 
    return wp_new

def likelihood_step(s1,s2,w,b2): 
    '''
    FØRSTE FAKTOR I (5.21) - TAR INN BARE TALL (W nå, S2 nå, S1 forrige), INGEN LISTER. B1,B2 CONST.
    '''
    return inverse_logit(w*s1 + b2)**(s2) * (1-inverse_logit(w*s1 + b2))**(1-s2)

def parameter_priors(shapes,rates):
    '''
    DRAW INITIAL VALUES FOR ALPHA AND TAU ACCORDING TO (5.22). np.random.gamma har definert beta fra artikkelen som 1/beta
    '''
    return np.array([(np.random.gamma(shapes[i],1/rates[i])) for i in range(len(shapes))])

def proposal_step(shapes,theta):
    '''
    PROPOSAL FUNCTION Q - DRAW NEXT VALUE FOR PARAMETERS TO BE EVALUATED. GAMMA DISTRIBUTION: (5.24)
    her er theta bare den forrige foreslåtte verdien, altså en 1*2 liste. 
    '''
    proposal = np.array([(np.random.gamma(shapes[i],theta[i]/shapes[i])) for i in range(len(shapes))])
    return proposal

def adjust_variance(theta, U):
    '''
    ADJUST VARIANCE ACCORDING TO 4.2.2.1 and 5.3.2.3
    Her er theta hele matrisen, i*2, som tar har alle i samplesene for de 2 parameterne vi estimerer
    '''
    var_new = theta[-U:].var(0)*(2.4**2)
    alphas = np.array([((theta[-1][i]**2) / var_new[i]) for i in range(len(var_new))])
    proposal = np.array([(np.random.gamma(alphas[i],theta[-1][i]/alphas[i])) for i in range(len(var_new))])
    return alphas,proposal
    
def ratio(prob_old,prob_next,shapes_prior,rates_prior,shapes,theta_next,theta_prior):
    '''
    CALUCALATES r IN ALGORITHM 3
    '''
    spike_prob_ratio = prob_next / prob_old #FACTOR 1 IN R 
    prior_ratio, proposal_ratio = 1,1 #FACTOR 2,3 IN R
    for i in range(len(shapes)):
        prior_ratio *= gamma.pdf(theta_next[i],a=shapes_prior[i],scale=1/rates_prior[i])/\
        gamma.pdf(theta_prior[i],a=shapes_prior[i],scale=1/rates_prior[i])
        proposal_ratio *= gamma.pdf(theta_prior[i],a=shapes[i],scale=theta_next[i]/shapes[i])/\
        gamma.pdf(theta_next[i],a=shapes[i],scale=theta_prior[i]/shapes[i])
    return spike_prob_ratio * prior_ratio * proposal_ratio #THESE ARE ALL THE RATIO FACTORS 

def scaled_spike_prob(fact_old,fact_new):
    '''
    METHOD 1 FOR CALCULATING P(Y_1:T GIVEN THETA) IN ALGORITHM 3. FOR BOTH THETA_NEW and OLD. (4.27)
    TAKING TWO LISTS WITH PRODUCTS FROM 4.27, SCALING BOTH LISTS WITH COMBINED MAX VALUE 
    '''   ´
    values = np.unique(np.stack((fact_old,fact_new))) #alle unike verdier stigende rekkefølge
    fact_old_scaled = fact_old/values[-1] #values[-1] = max
    fact_new_scaled = fact_new/values[-1]
    i = 2
    while(np.prod(fact_old_scaled) == 0 or np.prod(fact_new_scaled) == 0):
        fact_old_scaled /= values[-i]
        fact_new_scaled /= values[-i]
        i+=1
    return np.prod(fact_old_scaled),np.prod(fact_new_scaled)

def scaled2_spike_prob(old,new):
    '''
    METHOD 2 FOR CALCULATING P(Y_1:T GIVEN THETA) IN ALGORITHM 3. FOR BOTH THETA_NEW and OLD. (4.27)
    FINDING THE LOGARITHM AND THEN SCALE THE SUMS BY SUBTRACTING MAX OF OLD,NEW AND THEN EXPONENTIATE
    (jeg prøver bare begge metodene)
    '''
    return np.exp(old - np.max((old,new))),np.exp(new - np.max((old,new)))
    
def infer_b2_w0(s1,s2,tol):
    '''
    Fisher scoring algorithm from 3.2
    Estimating B2, W0 as described 5.3.2.2
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

def particle_filter(w0,b2,theta,s1,s2,std,P,binsize,time):
    '''
    Particle filtering, algorithm 2
    Posterior_factors og log_posterior er for de to ulike metodene for P(Y_1:T GIVEN THETA).
    Skal egt bare bruke en av de. 
    '''
    timesteps = np.int(time/binsize)
    t = np.zeros(timesteps)
    wp = np.full((P,timesteps),w0)
    vp = np.ones(P)
    posterior_factors = []
    log_posterior = 0
    for i in tqdm(range(1,500)):
        v_normalized = normalize(vp)
        perplexity = perplexity_func(v_normalized,P)
        if perplexity < 0.66:
            print('RESAMPLING:D')
            wp = resampling(v_normalized,wp,P)
            vp = np.full(P,1/P)
            v_normalized = normalize(vp)
        t[i] = i*binsize
        for p in range(P):
            lr = learning_rule(s1,s2,theta[0],theta[0]*1.05,theta[1],theta[1],t,i) 
            wp[p][i] = wp[p][i-1] + lr + np.random.normal(0,std) 
            ls = likelihood_step(s1[i-1],s2[i],wp[p][i],b2)
            vp[p] = ls * v_normalized[p]
        if any(vp == 0):
            print('Error: wrong weights, = 0')
            break
        log_posterior += np.log(np.sum(vp)/P)
        posterior_factors.append(np.sum(vp)/P)
    return wp,np.asarray(posterior_factors),t,log_posterior

            
def MHsampler(w0,b2est,shapes_prior,rates_prior,s1,s2,std,P,binsize,time,U,it):
    '''
    Monte Carlo sampling with particle filtering, algoritme 3
    '''
    theta_prior = parameter_priors(shapes_prior,rates_prior)
    theta = np.array([theta_prior])
    shapes = np.copy(shapes_prior)
    _,posterior_factors_old,_,old_log_post = particle_filter(w0,b2est,theta_prior,s1,s2,std,P,binsize,time)
    i = 0
    for i in tqdm(range(1,it)):
        if (i % U == 0):
            shapes, theta_next = adjust_variance(theta,U)
        else:    
            theta_next = proposal_step(shapes,theta_prior)
        _,posterior_factors_next,_,new_log_post = particle_filter(w0,b2est,theta_next,s1,s2,std,P,binsize,time)
        prob_old,prob_next = scaled_spike_prob(posterior_factors_old,posterior_factors_next)
        #prob_old,prob_next = scaled2_spike_prob(old_log_post,new_log_post)
        r = ratio(prob_old,prob_next,shapes_prior,rates_prior,shapes,theta_next,theta_prior)
        choice = np.int(np.random.choice([1,0], 1, p=[np.min([1,r]),1-np.min([1,r])]))
        theta_choice = [theta_prior,theta_next][choice == 1]
        theta = np.vstack((theta, theta_choice))
        theta_prior = np.copy(theta_next)
        posterior_factors_old = posterior_factors_next
    return theta
        
'''
PARAMETERS AND RUNNING OF ALGORITHM :
'''        
std = 0.001
w0 = 1
b1 = -2.25
b2 = -2.25
Ap = 0.005
Am = Ap*1.05
tau = 20e-3
time = 120
binsize = 1/200
P = 100
U = 100
it = 2
shapes_prior = [1,1] #Disse verdiene var fra artikkelen
rates_prior = [50,100]
