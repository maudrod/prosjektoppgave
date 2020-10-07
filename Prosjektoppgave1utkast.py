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
    return s2[i-1]*np.sum(s1[:i]*Ap*np.exp((t[:i]-max(t))/taup)) - s1[i-1]*np.sum(s2[:i]*Am*np.exp((t[:i]-max(t))/taum)) # *1000 for millisekunder

def logit(x):
    return np.log(x/(1-x))

def inverse_logit(x):
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
    return logit(np.sum(s1)/len(s1)) #5.23 

def normalize(vp): #normalisere vekter 
    return vp/sum(vp)

def perplexity_func(vp_normalized,P):
    if any(vp_normalized) == 0:
        print('ERROR, WEIGHTS EQ. 0')
        return 0
    h = -np.sum(vp_normalized*np.log(vp_normalized))
    return np.exp(h)/P

def resampling(vp_normalized,wp,P):
    wp_new = np.copy(wp)
    indexes = np.linspace(0,P-1,P)
    resampling_indexes = np.random.choice(indexes,P,p=vp_normalized)
    for i in range(P):
        wp_new[i] = wp[resampling_indexes.astype(int)[i]] 
    return wp_new

def likelihood_step(s1,s2,w,b2): #p(s2 given s1,w,theta)
    return inverse_logit(w*s1 + b2)**(s2) * (1-inverse_logit(w*s1 + b2))**(1-s2)

def parameter_priors(shapes,rates):
    return np.array([(np.random.gamma(shapes[i],1/rates[i])) for i in range(len(shapes))])

def proposal_step(shapes,theta):
    proposal = np.array([(np.random.gamma(shapes[i],theta[i]/shapes[i])) for i in range(len(shapes))])
    return proposal

def adjust_variance(theta, U):
    var_new = theta[-U:].var(0)*(2.4**2)
    alphas = np.array([((theta[-1][i]**2) * var_new[i]) for i in range(len(var_new))])
    proposal = np.array([(np.random.gamma(alphas[i],theta[-1][i]/alphas[i])) for i in range(len(var_new))])
    return alphas,proposal
    
def ratio(prob_prior,prob_next,shapes_prior,rates_prior,shapes,theta_next,theta_prior):
    spike_prob_ratio = prob_next / prob_prior
    prior_ratio, proposal_ratio = 1,1
    for i in range(len(shapes)):
        prior_ratio *= gamma.pdf(theta_next[i],a=shapes_prior[i],scale=rates_prior[i])/\
        gamma.pdf(theta_prior[i],a=shapes_prior[i],scale=rates_prior[i])
        proposal_ratio *= gamma.pdf(theta_prior[i],a=shapes[i],scale=theta_next[i]/shapes[i])/\
        gamma.pdf(theta_next[i],a=shapes[i],scale=theta_prior[i]/shapes[i])
    return spike_prob_ratio * prior_ratio * proposal_ratio

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

def particle_filter(w0,b2,theta,s1,s2,std,P,binsize,time):
    '''
    Particle filtering, (doesnt quite work yet, smth with weights vp)
    Possible to speed it up? 
    How to initiate w0 and vp?
    '''
    timesteps = np.int(time/binsize)
    t = np.zeros(timesteps)
    wp = np.full((P,timesteps),w0)
    vp = np.ones(P)
    spike_posterior = 1
    for i in tqdm(range(1,timesteps)):
        v_normalized = normalize(vp)
        perplexity = perplexity_func(v_normalized,P)
        if perplexity < 0.66:
            wp = resampling(v_normalized,wp,P)
        t[i] = i*binsize
        for p in range(P):
            lr = learning_rule(s1,s2,theta[0],theta[0]*1.05,theta[1],theta[1],t,i) 
            wp[p][i] = wp[p][i-1] + lr + np.random.normal(0,std) 
            ls = likelihood_step(s1[i-1],s2[i],wp[p][i],b2)
            vp[p] = ls * v_normalized[p]
        if any(vp) == 0:
            print('Error: wrong weights, = 0')
            break 
        spike_posterior *= np.sum(vp)/P
    return wp,spike_posterior,t

            
def MHsampler(w0,b2est,shapes_prior,rates_prior,s1,s2,std,P,binsize,time,U,it):
    '''
    Monte Carlo sampling with particle filtering, algoritme 3
    '''
    theta_prior = parameter_priors(shapes_prior,rates_prior)
    theta = np.array([theta_prior])
    shapes = np.copy(shapes_prior)
    _,prob_prior,_ = particle_filter(w0,b2est,theta_prior,s1,s2,std,P,binsize,time)
    i = 0
    for i in tqdm(range(1,it)):
        if (i % U == 0):
            shapes, theta_next = adjust_variance(theta,U)
        else:    
            theta_next = proposal_step(shapes,theta_prior)
        _,prob_next,_ = particle_filter(w0,b2est,theta_next,s1,s2,std,P,binsize,time)
        r = ratio(prob_prior,prob_next,shapes_prior,rates_prior,shapes,theta_next,theta_prior)
        choice = np.random.choice([1,0], 1, p=[np.min([1,r]),1-np.min([1,r])])
        theta_choice = [theta_prior,theta_next][choice == 1]
        theta = np.vstack((theta, theta_choice))
        theta_prior = np.copy(theta_next)
        prob_prior = prob_next
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
P = 1
U = 100
it = 2
shapes_prior = [1,1]
rates_prior = [50,100]

theta_test = parameter_priors(shapes_prior,rates_prior)
wp, vp, t = particle_filter(w0est,b2est,theta_test,s1,s2,std,P,binsize,time)
#s1,s2,t,W = generative(Ap,Am,tau,tau,b1,b2,w0,std,time,binsize)
#b1est = infer_b1(s1)
#w0est = infer_b2_w0(s1[:2000],s2[:2000],1e-10)[1]
#b2est = infer_b2_w0(s1,s2,1e-10)[0]
#theta = MHsampler(w0est,b2est,shapes_prior,rates_prior,s1,s2,std,P,binsize,time,U,it)

'''
wp, vp, t = particle_filter(w0,b2,[Ap,tau],s1,s2,std,P,binsize,time)
vpn = normalize(vp)
plt.figure()
for i in range(P):
    plt.title('Weight trajectory')
    plt.plot(t,wp[i])
    plt.xlabel('Time')
    plt.ylabel('Weight')
    #plt.legend()
    plt.show()
'''
