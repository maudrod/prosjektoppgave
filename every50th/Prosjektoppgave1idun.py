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

def learning_rule2(s1,s2,Ap,Am,taup,taum,t,i,binsize): 
    return s2[i-1]*np.sum(s1[:i]*Ap*np.exp((t[:i]-max(t))/taup)) - s1[i-1]*np.sum(s2[:i]*Am*np.exp((t[:i]-max(t))/taum))

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
        lr = learning_rule(s1,s2,Ap,Am,taup,taum,t,i,binsize)
        W[i] = W[i-1] + lr + np.random.normal(0,std) #updating weights, as in 5.8 in article
        s2[i] = np.random.binomial(1,inverse_logit(W[i]*s1[i-1]+b2)) #5.5 in article, spike/not neuron 2
        s1[i] = np.random.binomial(1,inverse_logit(b1)) #5.4
        t[i] = binsize*i #list with times (start time of current bin)
    return(s1,s2,t,W)

def generative2(Ap,Am,taup,taum,b1,b2,w0,std,seconds,binsize):
    '''time and binsize measured in seconds'''
    iterations = np.int(seconds/binsize)
    t,W,s1,s2 = np.zeros(iterations),np.zeros(iterations),np.zeros(iterations),np.zeros(iterations)
    W[0] = w0 #Initial value for weights
    s1[0] = np.random.binomial(1,inverse_logit(b1)) #5.4 in article, generate spike/not for neuron 1
    for i in range(1,iterations):
        lr = learning_rule2(s1,s2,Ap,Am,taup,taum,t,i,binsize)
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
    return np.array([(np.random.gamma(shapes[i],theta[i]/shapes[i])) for i in range(len(shapes))])

def adjust_variance(theta, U,it,shapes):
    var_new = np.array([0,0])
    while (any(i == 0 for i in var_new)):
        var_new = theta[-U:].var(0)*(2.4**2)
        U += 1
        if U > it:
            return shapes, np.array([(np.random.gamma(shapes[i],theta[-1][i]/shapes[i])) for i in range(len(var_new))])
    new_shapes = np.array([((theta[-1][i]**2) / var_new[i]) for i in range(len(var_new))])
    proposal = np.array([(np.random.gamma(new_shapes[i],theta[-1][i]/new_shapes[i])) for i in range(len(var_new))])
    return new_shapes,proposal
    
def ratio(prob_old,prob_next,shapes_prior,rates_prior,shapes,theta_next,theta_prior):
    spike_prob_ratio = prob_next / prob_old
    prior_ratio, proposal_ratio = 1,1
    for i in range(len(shapes)):
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

def particle_filter(w0,b2,theta,s1,s2,std,P,binsize,seconds):
    '''
    Particle filtering, (doesnt quite work yet, smth with weights vp)
    Possible to speed it up? 
    How to initiate w0 and vp?
    '''
    timesteps = np.int(seconds/binsize)
    t = np.zeros(timesteps)
    wp = np.full((P,timesteps),w0)
    vp = np.ones(P)
    log_posterior = 0
    for i in range(1,timesteps):
        v_normalized = normalize(vp)
        perplexity = perplexity_func(v_normalized,P)
        if perplexity < 0.66:
            wp = resampling(v_normalized,wp,P)
            vp = np.full(P,1/P)
            v_normalized = normalize(vp)
        t[i] = i*binsize
        for p in range(P):
            lr = learning_rule(s1,s2,theta[0],theta[0]*1.05,theta[1],theta[1],t,i,binsize) 
            wp[p][i] = wp[p][i-1] + lr + np.random.normal(0,std) 
            ls = likelihood_step(s1[i-1],s2[i],wp[p][i],b2)
            vp[p] = ls * v_normalized[p]
        log_posterior += np.log(np.sum(vp)/P)
    v_normalized = normalize(vp)
    return wp,t,log_posterior

def MHsampler(w0,b2est,shapes_prior,rates_prior,s1,s2,std,P,binsize,seconds,U,it):
    '''
    Monte Carlo sampling with particle filtering, algoritme 3
    '''
    theta_prior = parameter_priors(shapes_prior,rates_prior)
    theta = np.array([theta_prior])
    shapes = np.copy(shapes_prior)
    _,_,old_log_post = particle_filter(w0,b2est,theta_prior,s1,s2,std,P,binsize,seconds)
    for i in range(1,it):
        if (i % U == 0):
            shapes, theta_next = adjust_variance(theta,U,it,shapes)
        else:    
            theta_next = proposal_step(shapes,theta_prior)
        _,_,new_log_post = particle_filter(w0,b2est,theta_next,s1,s2,std,P,binsize,seconds)
        prob_old,prob_next = scaled2_spike_prob(old_log_post,new_log_post)
        r = ratio(prob_old,prob_next,shapes_prior,rates_prior,shapes,theta_next,theta_prior)
        choice = np.int(np.random.choice([1,0], 1, p=[min(1,r),1-min(1,r)]))
        theta_choice = [np.copy(theta_prior),np.copy(theta_next)][choice == 1]
        theta = np.vstack((theta, theta_choice))
        theta_prior = np.copy(theta_choice)
        old_log_post = [np.copy(old_log_post),np.copy(new_log_post)][choice == 1]
    return theta
        
'''
PARAMETERS AND RUNNING OF ALGORITHM :
'''        
std = 0.001
w0 = 1.0
b1 = -2.0
b2 = -2.0
Ap = 0.005
Am = Ap*1.05
tau = 20.0e-3
seconds = 120.0
binsize = 1/200.0
P = 100
U = 50
it = 1500
shapes_prior = np.array([1,1])
rates_prior = np.array([50,100])



s1,s2,t,W = generative(Ap,Am,tau,tau,b1,b2,w0,std,seconds,binsize)
b1est = infer_b1(s1)
w0est = infer_b2_w0(s1[:2000],s2[:2000],1e-10)[1]
b2est = infer_b2_w0(s1,s2,1e-10)[0]
theta = MHsampler(w0est,b2est,shapes_prior,rates_prior,s1,s2,std,P,binsize,seconds,U,it)

np.save('s1_1',s1)
np.save('s2_1',s2)
np.save('t_1',t)
np.save('w0est_1',w0est)
np.save('b1est_1',b1est)
np.save('b2est_1',b2est)
np.save('theta_1_1500it',theta)



plt.figure()
plt.title('Ap 200it')
plt.xlabel('Iterations')
plt.ylabel('Ap')
plt.plot(np.linspace(1,it,it),theta[:,0],'ro')
plt.savefig('Aptest_1500_1')

plt.figure()
plt.title('Tau 200it')
plt.xlabel('Iterations')
plt.ylabel('Tau')
plt.plot(np.linspace(1,it,it),theta[:,1],'bo')
plt.savefig('Tautest_1500_1')

'''
w10 = []
wall = []
for i in range(5):
    _,_,_,W = generative(Ap,Am,tau,tau,b1,b2,w0,std,seconds,binsize)
    _,_,_,W2 = generative2(Ap,Am,tau,tau,b1,b2,w0,std,seconds,binsize)
    w10.append(W)
    wall.append(W2)
    
np.save('5gen10tau',w10)
np.save('5genwhole',wall)
'''





    
