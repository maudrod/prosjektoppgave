import numpy as np              
import matplotlib.pyplot as plt 
from tqdm import tqdm
#test
def learning_rule(s1,s2,Ap,Am,taup,taum,t,i): 
    '''
    5.8 in article (typo in article, should be negative exponent for e)
    s1,s2 : binary values for the different time bins for neuron 1 and 2 respectively
    t : numpy array with the measured time points
    Ap,Am,taup,taum : learning rule parameters 
    '''
    return s2[i-1]*np.sum(s1[:i]*Ap*np.exp(1000*(t[:i]-max(t))/taup)) - s1[i-1]*np.sum(s2[:i]*Am*np.exp(1000*(t[:i]-max(t))/taum)) # *1000 for millisekunder

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
    return logit(np.sum(s1)/len(s1))

def normalize(vp):
    return vp/sum(vp)

def likelihood_step(s1,s2,w,b2): #p(s2 given s1,w,theta)
    return inverse_logit(w*s1 + b2)**(s2) * (1-inverse_logit(w*s1 + b2))**(1-s2)

def spike_posterior(vp,P): #p(y given theta)
    return np.prod(np.sum(vp, axis = 0)/P)
    

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
    '''
    timesteps = np.int(time/binsize)
    t = np.zeros(timesteps)
    wp = np.full((P,timesteps),w0)
    #vp = np.full((P,timesteps),1)
    vp = np.ones(P)
    #v_normalized = normalize(vp)
    for i in tqdm(range(1,timesteps)):
        t[i] = i*binsize
        for p in range(P):
            lr = learning_rule(s1,s2,theta[0],theta[0]*1.05,theta[1],theta[1],t,i) 
            wp[p][i] = wp[p][i-1] + lr + np.random.normal(0,std) 
            ls = likelihood_step(s1[i-1],s2[i],wp[p][i],b2)
            #print(vp[p])
            #print(ls)
            vp[p] = vp[p] * ls
            #print('i-1:', vp[p][i-1], 'i:', vp[p][i], 'ls:', ls)
            #print(vp[p])
    return wp,vp,t

            
def MHsampler():
    '''
    Monte Carlo sampling with particle filtering
    '''
    return 0  

std = 0.001
w0 = 1
b1 = -2.25
b2 = -2.25
Ap = 0.005
Am = Ap*1.05
tau = 20
time = 120
binsize = 1/200
P = 4

s1,s2,t,W = generative(Ap,Am,tau,tau,b1,b2,w0,std,time,binsize)
b1est = infer_b1(s1)
w0 = infer_b2_w0(s1[:2000],s2[:2000],1e-10)[1]
b2est = infer_b2_w0(s1,s2,1e-10)[0]

wp, vp, t = particle_filter(w0,b2,[Ap,tau],s1,s2,std,P,binsize,time)
vpn = normalize(vp)
plot_gen_weight(t,W)
plt.figure()
for i in range(P):
    plt.title('Weight trajectory')
    plt.plot(t,wp[i])
    plt.xlabel('Time')
    plt.ylabel('Weight')
    #plt.legend()
    plt.show()

