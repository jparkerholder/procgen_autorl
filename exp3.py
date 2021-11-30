

import math
import numpy as np
import matplotlib.pyplot as plt

from exp3_multiple_play.deep_round import DepRound
from scipy.optimize import minimize

def estimate_alpha(batch_size, gamma, Wc, C):

    def single_evaluation(alpha):
        denominator = sum([alpha if val > alpha else val for idx, val in enumerate(Wc)])
        rightside = (1 / batch_size - gamma / C) / (1 - gamma)
        output = np.abs(alpha / denominator - rightside)

        return output

    x_tries = np.random.uniform(0, np.max(Wc), size=(100, 1))
    y_tries = [single_evaluation(val) for val in x_tries]
    # find x optimal for init
    # print(f'ytry_len={len(y_tries)}')
    idx_min = np.argmin(y_tries)
    x_init_min = x_tries[idx_min]

    res = minimize(single_evaluation, x_init_min, method='BFGS', options={'gtol': 1e-6, 'disp': False})
    if isinstance(res, float):
        return res
    else:
        return res.x

def timevarying_compute_prob_dist_and_draw_hts(weights, gamma, batch_size, omega, pending_actions):

    # number of category
    C=len(weights)

    if batch_size<=1:
        print("batch_size needs to be >1")

    # perform some truncation here
    maxW = np.max(weights)
    eta=(1 / batch_size - gamma / C) / (1 - gamma)
    temp = np.sum(weights) * eta# (1.0 / batch_size - gamma / C) / (1 - gamma)
    if gamma < 1 and maxW >= temp and batch_size < C:
        # find a threshold alpha
        alpha = estimate_alpha(batch_size, gamma, weights, C)
     
        S0 = [idx for idx, val in enumerate(weights) if val > alpha]
        S1 = [idx for idx in pending_actions if (weights[idx] > alpha)]
        S0+=S1
        # update Wc_list
        for idx in S0:
            weights[idx]=alpha[0]
    else:
        S0 = []
        
    e_num=2.71 # this is e number
    # Compute the probability for each category
    probabilityDistribution = distrEXP3M(weights, gamma) + e_num*omega*np.sum(weights)/C
    #print("prob",np.round(probabilityDistribution,decimals=4))

    # draw a batch here
    if batch_size < C:
        # we need to multiply the prob by batch_size before providing into DepRound
        probabilityDistribution=[prob*batch_size for prob in probabilityDistribution]
        myselection = DepRound(probabilityDistribution, k=batch_size)
        
    else:      
        probabilityDistribution=np.asarray(probabilityDistribution)
        probabilityDistribution=probabilityDistribution/np.sum(probabilityDistribution)
        myselection = np.random.choice(len(probabilityDistribution), batch_size, p=probabilityDistribution)
        myselection=myselection.tolist()
        
    return myselection, probabilityDistribution, S0

def distrEXP3M(weights,gamma=0.0):
    # given the weight vector and gamma, return the distribution
    theSum = float(sum(weights))
    return [(1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights]

def exp3_get_cat(row, data, numRounds, pendingactions):
    
    arms = row.Range
    numActions=len(arms)
    batch_size=len(data[data.t==data.t.min()])
    
    pendingactions = [arms.index(x) for x in pendingactions]
        
    if batch_size<numActions:
       gamma = math.sqrt(numActions * np.log(numActions/batch_size) / ((np.e - 1) * batch_size*(numRounds/10)))
    else:
       gamma=0.2
   
    omega=1/(np.sum(numRounds)*10)

    tt = 0
       
    weights = [1.0] * numActions
    all_choice=[]
       
    min_t = data.t.min()
    max_t = data.t.max()
    
    count=0
    choice=[0]*numRounds
    all_choice=[]

    # this is just where we build the distributions...
    for tt in range(min_t, max_t+1):
        
        batch_choice, probabilityDistribution, S0 = timevarying_compute_prob_dist_and_draw_hts(weights, gamma, batch_size, omega, pendingactions)
        
        batch_choice = [arms.index(x) for x in data[data['t']==tt]['x'+str(row.name)].values]
          
        batch_choice=np.asarray(batch_choice)
           
        e_num=2.71 # e number
        right_term=e_num*omega*np.sum(weights)/numActions
        
        rewards = data[data['t']==tt].y_exp3.values
           
        for idx, val in enumerate(batch_choice):
               
            if val in S0:
                weights[val]+=right_term
            else:
                # =============================================================================
                # this estimation of the reward comes from the RL...
                # the reward should be normalized [0-1] over time for the best performance....
                theReward = rewards[idx]
                   
                estimatedReward = 1.0 * theReward / (probabilityDistribution[val]*batch_size )
                weights[val] *= np.exp(estimatedReward * gamma*batch_size / numActions) + right_term # important that we use estimated reward here!
        
           
        sum_w=np.sum(weights)
        weights=[w/sum_w for w in weights]
           
        count+=1
    
    # now we select our arm!
    
    batch_choice, probabilityDistribution, S0 = timevarying_compute_prob_dist_and_draw_hts(weights, gamma, batch_size, omega, pendingactions)

    cat_idx = DepRound(probabilityDistribution, k=1)[0]
    cat = arms[cat_idx]
    print("\nweights",np.round(weights,decimals=4))
    print(arms)
    print("\ndist",np.round(probabilityDistribution,decimals=4))    
    print(cat)
    return(cat)

