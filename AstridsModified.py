#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:29:54 2020

@author: maudrodsmoen
"""
import numpy.random
import numpy as np
import scipy.stats
from scipy.stats import gamma
from numba import njit
@njit

def learning_rule(t,S1_t, S2_t,S1,S2, A_p, A_m, tau_p,tau_m): #STDP learning rule
    l_p = np.sum(A_p*np.exp(-((t-S1_t)*1000)/(tau_p*t_per_second))) 
    l_m = np.sum(A_m*np.exp(-((t-S2_t)*1000)/(tau_m*t_per_second))) 
    return(S2[t]*l_p-S1[t]*l_m)

def generate(T, b1, b2, A_p_true,A_m_true,tau_true, W_0_true, std): #Generate spikes and weights
    W_T = np.zeros(T)
    S1 = np.zeros(T) 
    S2 = np.zeros(T) 
    S1_t_init = 1 
    S2_t_init = 1
    S2_t = np.array([0])
    lambda1 = 1/(1+np.exp(-b1))
    S1[0] = np.random.binomial(1,lambda1,1) 
    
    if S1[0] == 1:
        S1_t = np.array([0])
        S1_t_init = 0 
    W_T[0] = W_0_true
    
    for t in range(1,T):
        lambda2 = 1/(1+np.exp(-(b2 + (W_T[t-1]*S1[t-1]))))
        S1[t] = np.random.binomial(1,lambda1,1) 
        S2[t] = np.random.binomial(1,lambda2,1)
        
        if S1[t] == 1: 
            if S1_t_init:
                S1_t = np.array([t])
                S1_t_init = 0 
            else:
                S1_t = np.append(S1_t, t) 
        if S2[t] == 1:
            if S2_t_init:
                S2_t = np.array([t]) 
                S2_t_init = 0
            else:
                S2_t = np.append(S2_t, t)

        if t<t_constant: 
            W_T[t] = W_T[t-1]
        else:
            W_T[t] = W_T[t-1] + (learning_rule(t,S1_t, S2_t,S1,S2,A_p_true,A_m_true,tau_true,tau_true)) + np.random.normal(0,std) 
            
    return(W_T, S1, S2, S1_t, S2_t)

def learning_rule_steps(S1, S2, S1_t, S2_t, t_per_second, A, tau, T): #Compute deterministic learning rule steps given data and hyperparameters
    t = t_constant
    step_list = [learning_rule(t,S1_t[0:int(np.sum(S1[0:t+1]))], S2_t[0:int(np.sum(S2[0:t+1]))],S1,S2,A,A*1.05,tau,tau)] 
    while (t+1 < T):
        t = t+1
        step_list.append(learning_rule(t,S1_t[0:int(np.sum(S1[0:t+1]))], S2_t[0:int(np.sum(S2[0:t+1]))],S1,S2,A,A*1.05,tau,tau)) 
    return(step_list)


#Functions for particle filtering
def sample_weights(w_p, P, learning_step, std): 
    w_p_t = np.zeros(P)
    for p in range(P):
        sample_point = np.random.normal(w_p[p][-1]+learning_step,std) 
        w_p[p].append(sample_point)
        w_p_t[p] = sample_point
    return(w_p, w_p_t)



def particle_log_likelihoods(S1, S2, w_p_t, t, P, b1, b2):
    log_alpha_t = S2[t+1]*(b2+(w_p_t*S1[t])) - np.log(1+np.exp(b2+(w_p_t*S1[t]))) 
    return(log_alpha_t)


def update_particle_weights(v_p, P, log_alpha_t): 
    alpha_t_scaled = np.exp(log_alpha_t-max(log_alpha_t)) 
    v_p = v_p*alpha_t_scaled
    return(v_p)

def resample_weights(w_p,v_p, P): 
    xk = np.arange(P)
    v_sum = np.sum(v_p)
    pk = v_p/v_sum
    custm = scipy.stats.rv_discrete(name='custm', values=(xk, pk)) 
    p_r = list(custm.rvs(size=P))
    init = 1
    for it in p_r:
        if init==1:
            w_p_r = [w_p[it].copy()] 
            init = 0
        w_p_r.append(w_p[it].copy()) 
    v_p = np.repeat(1/P, P) 
    return(w_p_r, v_p)

def N_eff(v_p):
    v_sum = np.sum(v_p) 
    v_norm = v_p/v_sum 
    squared_sum = 0
    for v in v_norm:
        squared_sum = squared_sum + v**2 
    return(1/squared_sum)

def perplexity(v_p):
    v_sum = np.sum(v_p)
    v_norm = v_p/v_sum
    v_norm_log = np.log(v_norm)
    perplexity = np.exp(- np.sum(v_norm*v_norm_log)) 
    return(perplexity)


def log_lik_total(w_p, S1, S2,P, b1, b2): 
    log_lik = np.zeros(P)
    for p in range(P):
        log_lik[p] = np.sum(S2[1:]*(b2+(w_p[p][:]*S1[:-1])) - np.log(1+np.exp(b2+(w_p[p][:]*S1[:-1])))) 
    return(log_lik)


def SMC(P, T, W_0, std_initial, S1, S2, learning_step, b1, b2):
    #Particle filtering
    resampling_list = [0]
    w_p = [[W_0]] 
    for p in range(P):
        w_p.append([W_0]) 
    
    v_p = np.ones(P)
    it = 0
    for t in range(1,T-1): 
        if t<t_constant:
            for p in range(P): 
                w_p[p].append(w_p[p][-1])
        else:
            w_p, w_p_t = sample_weights(w_p, P, learning_step[it], std) 
            alpha_t = particle_log_likelihoods(S1,S2,w_p_t,t,P, b1, b2) 
            v_p = update_particle_weights(v_p, P, alpha_t)
            it = it+1
            if perplexity(v_p)<N_threshold:
                w_p, v_p = resample_weights(w_p,v_p,P)
                resampling_list.append(t)
    return(w_p, v_p)



def estimate_W_and_b2(S1, S2, w_guess, b2_guess, iterations): #Newton iterations
    par = [b2_guess,w_guess] 
    par_list = [par]
    for j in range(iterations):
        lin_pred = par[0] + (par[1]*S1)
        lambda_it = 1/(1+np.exp(-(lin_pred[:-1])))
        arr_score = [np.ones(len(S1)-1),S1[:-1]]*(S2[1:] - (np.exp(lin_pred[:-1])/(1 + np.exp(lin_pred[:-1]))))
        score = arr_score.sum(axis=1) 
        score = np.matrix(score)
        for t in range(len(S1)-1): 
            if t == 0:
                hessian = np.matrix([[1,S1[t]],[S1[t],S1[t]]])*lambda_it[t]*(1-lambda_it[t]) 
            else:
                hessian = hessian + np.matrix([[1,S1[t]],[S1[t],S1[t]]])*lambda_it[t]*(1-lambda_it[t])
                
        m = numpy.linalg.inv(hessian)*(numpy.matrix.transpose(score)) 
        m = np.array(m)
        m = np.ndarray.flatten(m)
        par = par + m 
        par_list.append(par)
    return(par_list[-1])


def estimate_b1(S1):
    lambda1_estim = np.sum(S1)/(len(S1))
    b1_estim = np.log(lambda1_estim/(1-lambda1_estim)) 
    return(b1_estim)

def adaptive_alpha(H, A_p_list): 
    A_slice = A_p_list[len(A_p_list)-H:] 
    m = np.mean(A_slice)
    v = np.var(A_slice)*(2.4**2)
    new_alpha = (m**2)/v 
    return (new_alpha)

def adaptive_alpha_tau(H, tau_list): 
    tau_slice = tau_list[len(tau_list)-H:] 
    tau_slice_2 = [x/1000 for x in tau_slice] 
    m = np.mean(tau_slice_2)
    v = np.var(tau_slice_2)*(2.4**2) 
    new_alpha = (m**2)/v 
    return (new_alpha)

def proposal_A_p(A_p, alpha): 
    return(np.random.gamma(alpha, A_p/alpha))

def proposal_tau(tau, alpha): 
    return(1000*np.random.gamma(alpha, (tau/1000)/alpha))

def A_p_prior(A_p, A_p_rate): 
    return(gamma.pdf(A_p, a = 1, scale = 1/A_p_rate))

def tau_prior(tau, tau_rate):
    return(gamma.pdf(tau/1000, a = 1, scale = 1/tau_rate))



def p_spike_train_fraction(w_p, w_p_new, S1, S2, P, b1, b2): 
    init = 1
    for p in range(P):
        log_lik_temp_new = np.sum(S2[1:]*(b2+(w_p_new[p][:]*S1[:-1])) - np.log(1+np.exp(b2+(w_p_new[p][:]*S1[:-1])))) 
        log_lik_temp = np.sum(S2[1:]*(b2+(w_p[p][:]*S1[:-1])) - np.log(1+np.exp(b2+(w_p[p][:]*S1[:-1]))))
        if init:
            init = 0
            log_lik_list = [log_lik_temp] 
            log_lik_list_new = [log_lik_temp_new]
        else:
            log_lik_list.append(log_lik_temp) 
            log_lik_list_new.append(log_lik_temp_new)
            
    min1 = min(log_lik_list) 
    min2 = min(log_lik_list_new) 
    min_tot = min(min1,min2)
    log_lik_list_scaled = [x-min_tot for x in log_lik_list] 
    log_lik_list_new_scaled = [y - min_tot for y in log_lik_list_new]
    p_old = np.sum([np.exp(a) for a in log_lik_list_scaled])
    p_new = np.sum([np.exp(b) for b in log_lik_list_new_scaled])
    return(p_new/p_old)



def particle_marginal_Metropolis_Hastings_both(A_p_start,tau_start, W_0, b1, b2, iterations,alpha, alpha_tau):
    alpha_new = alpha
    alpha_tau_new = alpha_tau
    tau = tau_start
    A_p = A_p_start
    learning_step = learning_rule_steps(S1, S2, S1_t, S2_t, t_per_second, A_p, tau, T) 
    w_p, v_p = SMC(P, T, W_0, std_initial, S1, S2, learning_step, b1, b2)
    A_list = [A_p] 
    tau_list = [tau] 
    c=0
    for i in range(iterations):
        if (i%100)==0: 
            if i>150:
                alpha_new = adaptive_alpha(H, A_list)
        if (i%100)==0: 
            if i>150:
                alpha_tau_new = adaptive_alpha_tau(H, tau_list)

        A_p_new = proposal_A_p(A_p, alpha_new)
        tau_new = proposal_tau(tau, alpha_tau_new)
        learning_step_new = learning_rule_steps(S1, S2, S1_t, S2_t, t_per_second, A_p_new, tau_new, T) 
        w_p_new, v_p_new = SMC(P, T, W_0, std_initial, S1, S2, learning_step_new, b1, b2)
        w_p_new, v_p_new = resample_weights(w_p_new,v_p_new,P)
        p_s_ratio = p_spike_train_fraction(w_p, w_p_new, S1, S2, P, b1, b2) 
        prior_new_A = A_p_prior(A_p_new, A_p_rate)
        prior_old_A = A_p_prior(A_p, A_p_rate)
        prior_ratio_A = prior_new_A/prior_old_A
        prior_new_tau = tau_prior(tau_new, tau_rate) 
        prior_old_tau = tau_prior(tau, tau_rate) 
        prior_ratio_tau = prior_new_tau/prior_old_tau
        ratio = p_s_ratio*prior_ratio_A*prior_ratio_tau*(gamma.pdf(A_p, a = 1, scale = A_p_new)/gamma.pdf(A_p_new, a = 1, scale = A_p))*(gamma.pdf(tau/1000, a = 1, scale = tau_new/1000)/gamma.pdf(tau_new/1000, a = 1, scale = tau/1000))

        if (np.random.uniform(0,1)<ratio): 
            c = c+1
            tau = tau_new
            A_p = A_p_new
            w_p = w_p_new
            v_p = v_p_new
            learning_step = learning_step_new
            
        print(A_p, tau) 
        A_list.append(A_p)
        tau_list.append(tau) 
    
    return(A_list,tau_list, c)    
                


#Parameters
W_0_true = 1 
b1_true = 2
b2_true = -2
A_p_true= 0.005 
A_m_true = A_p_true*1.05 
tau_true = 20
n_seconds = 120 
t_per_second = 200
T = n_seconds * t_per_second 
t_constant = 100
std = 0.0001 #Noise level
#Start shape parameters for gamma proposal
alpha = 5 
alpha_tau = 4
#Rate parameters for gamma proposal
A_p_rate = 50 
tau_rate = 100

H = 100
P = 100 #Number of particles
N_threshold = P*0.66 #perplexity threshold
iterations = 1500
A_p_start = A_p_true 
tau_start = tau_true
std_initial = 0.001 
w_guess = W_0_true 
b2_guess = b2_true
#Main
#Generate spikes and weight trajectory
W_T, S1, S2, S1_t, S2_t = generate(T, b1_true, b2_true, A_p_true,A_m_true,tau_true, W_0_true, std) #Estimate b2
b2 = estimate_W_and_b2(S1, S2, w_guess, b2_guess, 40)[0] #Estimate b1
b1 = estimate_b1(S1) #Estimate W0
W_0 = estimate_W_and_b2(S1[:2000], S2[:2000], w_guess, b2, 40)[1]
#Particle Metropolis-Hastings
(A_list,tau_list,c) = particle_marginal_Metropolis_Hastings_both(A_p_start,tau_start, W_0, b1, b2, iterations,alpha, alpha_tau)


np.save('A_list_Astrid',A_list)
np.save('tau_list_Astrid',tau_list)
np.save('c_Astrid',c)
np.save('w0est_Astrid',W_0)
np.save('b1est_Astrid',b1)
np.save('b2est_Astrid',b2)







