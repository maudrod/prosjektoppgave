#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:20:28 2020

@author: emilam
"""
import numpy as np              
import matplotlib.pyplot as plt 
from scipy.stats import gamma
#from tqdm import tqdm
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
    for i in range(1,it):
        ex = [1,0][i % 2 == 0] #oddetall iterasjoner, eksluder Tau (index 1), partall: Ap (index 0)
        par_ind_temp = np.delete(par_ind,ex)
        if (i % U == 0):
            shapes, theta_next = adjust_variance(theta,U,par_ind_temp,N,it,shapes)
        else:    
            theta_next = proposal_step(shapes,theta_prior,par_ind_temp)
        new_log_post = particle_filter(w0est,b2est,theta_next,s1,s2,std,P,binsize,timesteps)
        prob_old,prob_next = scaled2_spike_prob(old_log_post,new_log_post)
        r = ratio(prob_old,prob_next,shapes_prior,rates_prior,shapes,theta_next,theta_prior,N)
        #print('old theta:', theta_prior)
        #print('new theta:', theta_next)
        #print('r:',r)
        choice = np.int(np.random.choice([1,0], 1, p=[min(1,r),1-min(1,r)]))
        theta_choice = [np.copy(theta_prior),np.copy(theta_next)][choice == 1]
        #print('choice:', theta_choice)
        theta = np.vstack((theta, theta_choice))
        theta_prior = np.copy(theta_choice)
        old_log_post = [np.copy(old_log_post),np.copy(new_log_post)][choice == 1]
    return theta
        
        
'''
PARAMETERS AND RUNNING OF ALGORITHM :
'''        
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


spk_pre = np.load('Cand1Pre.npy')
spk_post = np.load('Cand1Post.npy')
timesteps = len(spk_pre)
binsize = 1/1000.0

b1est = infer_b1(spk_pre)
b2est = infer_b2_w0(spk_pre, spk_post, 1e-10)[0]
w0est = infer_b2_w0(spk_pre[:10000], spk_post[:10000], 1e-10)[1]

Simest1 = MHsampler(w0est, b2est, shapes_prior, rates_prior, spk_pre, spk_post, std, P, binsize, timesteps, U, it,N)

np.save('b1estReal',b1est)
np.save('b2estReal',b2est)
np.save('w0estReal',w0est)
np.save('AltSampleReal0.0001noise',Simest1)
'''
spk_pre = np.load('Cand1Pre.npy')
spk_post = np.load('Cand1Post.npy')
timesteps = len(spk_pre)
binsize = 1/1000.0
AltReal = np.load('AltSampleReal0.0001noise.npy')
SimReal = np.load('SimSampleReal0.0001noise.npy')
w0est = np.load('w0estReal.npy')


medApSim = np.median(SimReal[300:,0])
medTauSim = np.median(SimReal[300:,1])
medApAlt = np.median(AltReal[300:,0])
medTauAlt = np.median(AltReal[300:,1])

mapApSim = np.load('MapApSimReal.npy')
mapTauSim = np.load('MapTauSimReal.npy')
mapApAlt = np.load('MapApAltReal.npy')
mapTauAlt = np.load('MapTauAltReal.npy')

meanApSim = np.mean(SimReal[300:,0])
meanTauSim = np.mean(SimReal[300:,1])
meanApAlt = np.mean(AltReal[300:,0])
meanTauAlt = np.mean(AltReal[300:,1])
'''
Traj = np.zeros(timesteps)
Traj[0] = w0est
t = np.zeros(timesteps)
for i in range(1,timesteps):
    Traj[i] = Traj[i-1] + learning_rule(spk_pre,spk_post,mapApSim,mapApSim*1.05,mapTauSim,mapTauSim,t,i,binsize) + np.random.normal(0,std)
    t[i] = binsize*i
'''

sns.set_style("darkgrid")

plt.figure()
sns.displot(SimReal[300:,0],kind="kde",color=[0.2,0.4,0.5])
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
#plt.xlim([0.00,0.001])
plt.axvline(mapApAlt,color='r',linestyle='--',label='MAP: '+str(mapApAlt[0].round(3)))
plt.axvline(medApAlt,color='g',linestyle='--',label='Median: '+str(medApAlt.round(3)))
plt.axvline(meanApAlt,color='m',linestyle='--',label='Mean: '+str(meanApAlt.round(3)))
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'Posterior distribution $A_+$ - Alternating Proposals')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

plt.figure()
sns.displot(SimReal[300:,0],kde = True,bins=100,color=[0.2,0.4,0.5])
#plt.hist(SimReal[300:,0],normed=True,bins = 100)
#plt.plot(np.linspace(1,1500,1500),AltReal[:,1],'ko')
#plt.xlim([0.00,0.0006])
plt.axvline(mapApSim,color='r',linestyle='--',label='MAP: '+str(mapApSim[0].round(3)))
plt.axvline(medApSim,color='g',linestyle='--',label='Median: '+str(medApSim.round(3)))
plt.axvline(meanApSim,color='m',linestyle='--',label='Mean: '+str(meanApSim.round(3)))
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'Posterior distribution $A_+$ - Simultaneous Proposals')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()
plt.show()

'''
AmMapAlt = mapApAlt[0] * 1.05

def lr1(s2,s1,Ap,delta,taup):
    return s2*s1*Ap*np.exp(-delta/taup)

def lr2(s1,s2,Am,delta,taum):
    return -s1*s2*Am*np.exp(delta/taum)
 
deltas = np.linspace(0,0.2,100000)
deltas2 = np.linspace(-0.2,0,100000)   
lrs1 = lr1(1,1,mapApAlt[0],deltas,mapTauAlt[0])
lrs2 = lr2(1,1,AmMapAlt,deltas2,mapTauAlt[0]) 

#lrs3 = lr1(1,1,0.0075,deltas,0.02)
#lrs4 = lr2(1,1,0.0075,deltas2,0.02) 

deltass = np.concatenate((deltas2,deltas))
lrss = np.concatenate((lrs2,lrs1))
#lrss2 = np.concatenate((lrs4,lrs3))
x = [-0.1,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.1]
#labels = [r'-5$\tau_{\pm}$',r'-4$\tau_{\pm}$',r'-3$\tau_{\pm}$',r'-2$\tau_{\pm}$',r'-$\tau_{\pm}$','0',r'$\tau_{\pm}$',r'2$\tau_{\pm}$',r'3$\tau_{\pm}$',r'4$\tau_{\pm}$',r'5$\tau_{\pm}$']
plt.figure()
plt.title('Learning rule for the considered neuron pair')
plt.plot(deltass,lrss,c='r')#,label=r'$A_+=A_-: 0.025$'+'\n'+r'$\tau_{\pm}: 55ms$')
#plt.plot(deltass,lrss2,label=r'$A_+=A_-: 0.0075$'+'\n'+r'$\tau_{\pm}: 20ms$')
plt.xlabel('$\Delta t$')
plt.ylabel('$\Delta w$')
#plt.xticks(x,labels)
plt.axhline(0,color='k',linestyle='--')
plt.legend()
plt.show()


Traj = np.zeros(timesteps)
Traj[0] = w0est
t = np.zeros(timesteps)
for i in range(1,timesteps):
    Traj[i] = Traj[i-1] + learning_rule(spk_pre,spk_post,mapApAlt[0],AmMapAlt,mapTauAlt[0],mapTauAlt[0],t,i,binsize) + np.random.normal(0,0.0001)
    t[i] = binsize*i
   

plt.figure()
plt.title('Simulated synaptic weight evolution')
plt.plot(t,Traj)
plt.xticks([0,20,40,60,80,100,120,140],['3900','3920','3940','3960','3980','4000','4020','4040'])
plt.xlabel('Time (seconds)')
plt.ylabel('w')
plt.show()
'''