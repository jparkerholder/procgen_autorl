# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:55:10 2020

@author: Vu
"""

# demo bandit exp3

import math
import numpy as np
import matplotlib.pyplot as plt

from deep_round import DepRound
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
        
def compute_prob_dist_and_draw_hts(weights, gamma, batch_size):

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
            # update Wc_list
            for idx in S0:
                weights[idx]=alpha[0]
        else:
            S0 = []
            
        # Compute the probability for each category
        probabilityDistribution = distrEXP3M(weights, gamma)
        #print("prob",np.round(probabilityDistribution,decimals=4))

        # draw a batch here
        if batch_size < C:
            # we need to multiply the prob by batch_size before providing into DepRound
            probabilityDistribution=[prob*batch_size for prob in probabilityDistribution]
            myselection = DepRound(probabilityDistribution, k=batch_size)            
        else:            
            myselection = np.random.choice(len(probabilityDistribution), batch_size, p=probabilityDistribution)
            myselection=myselection.tolist()
            
        #print("selection",myselection)

        return myselection, probabilityDistribution, S0
    
    
def timevarying_compute_prob_dist_and_draw_hts(weights, gamma, batch_size, omega,pending_actions):

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
     
        S0 = [idx for idx, val in enumerate(weights) if (val > alpha) ]
        S1 = [idx for idx in pending_actions if (weights[idx] > alpha) ]
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

# test EXP3 Multiple Play
def run_EXP3_M():
   numRounds =1000
   
   reward_list=[0.2, 0.5,0.1,0.7] #arm 2 and arm 4 are better than arm 1 and arm 3
     
   numActions=len(reward_list)
   batch_size=5
   
   #bestUpperBoundEstimate = 2 * numRounds / 3
   
   if batch_size<numActions:
       gamma = math.sqrt(numActions * np.log(numActions/batch_size) / ((np.e - 1) * batch_size*numRounds))
   else:
       gamma=0.5

   tt = 0
   
   weights = [1.0] * numActions
   all_choice=[]
   
   # number of epochs
   for tt in range(numRounds):
       
     
       choice,probabilityDistribution,S0=compute_prob_dist_and_draw_hts(weights, gamma, batch_size)

       # the selected arms are in choice
   
       all_choice=all_choice+choice
       
       theReward=0
       choice=np.asarray(choice)
        
       for cc in range(numActions):
           idx=np.where(choice==cc)[0]
                          
           if len(idx)==0 or cc in S0:
               continue

           # this estimation of the reward comes from the RL...
           theReward = reward_list[cc]

           #theReward = np.mean( [ reward_list[cc] for ii in idx])
           #scaledReward = (theReward - rewardMin) / (rewardMax - rewardMin) # rewards scaled to 0,1
           estimatedReward = 1.0 * theReward / (probabilityDistribution[cc]*batch_size) 
           weights[cc] *= np.exp(estimatedReward * gamma*batch_size / numActions) # important that we use estimated reward here!
       print("weights",np.round(weights,decimals=4))
       sum_w=np.sum(weights)
       weights=[w/sum_w for w in weights ]
       
       
#       for cc in choice:
#           theReward = reward_list[cc]
#           #scaledReward = (theReward - rewardMin) / (rewardMax - rewardMin) # rewards scaled to 0,1
#           estimatedReward = 1.0 * theReward / probabilityDistribution[cc]
#           weights[cc] *= np.exp(estimatedReward * gamma*batch_size / numActions) # important that we use estimated reward here!
#       
#       print("weights",np.round(weights,decimals=4))
       

   # =============================== plot ==================
   width=0.2
   
   print("weights",weights)
   
   choice_count = np.bincount(all_choice)

   plt.bar(range(numActions), choice_count, width, color='r')
   plt.xlabel("Categorical Choices")
   plt.ylabel("Frequency")

   plt.title("Selection over Choices")
   plt.show()
   
# time varying
def run_TV_EXP3_M():
   numRounds = np.asarray([200, 200,400])

   numRoundsCS=np.cumsum(numRounds)
  
    
   # let assume that our time-varing function has three segments in which the behavior will (slightly) change
   # in the 1st segment, the 4th arm is the best with 0.7 reward
   # in the 2nd segment, the 1st arm is the best with 0.8 reward
   # in the 3rd segment, the 2nd arm is the best with 0.9 reward
   
   reward_list=np.asarray([[0.4, 0.5,0.1,0.7],
            [0.8,0.4,0.3,0.1],
            [0.1,0.9,0.5,0.2]])
       
   numActions=len(reward_list[0])
   batch_size=5
   
   #bestUpperBoundEstimate = 2 * numRounds / 3
   
   omega=1/np.sum(numRounds)
   if batch_size<numActions:
       gamma = math.sqrt(numActions * np.log(numActions/batch_size) / ((np.e - 1) * batch_size*numRounds))
   else:
       gamma=0.5

   tt = 0
   
   all_choice=[]
   weights = [1.0] * numActions
   choice=[0]*numRounds[0]
   ss=0
   count=0
   for tt in range(sum(numRounds)):
       
       # this is for plotting and debugging purpose. we dont have it in the real code
       if tt in numRoundsCS:
           width=0.2
           choice_count = np.bincount(all_choice)
           plt.figure()
           plt.title(ss)
           plt.bar(range(numActions), choice_count, width, color='r')
           plt.show()
           choice=[0]*numRounds[ss+1]
           count=0
           
           ss=ss+1
           
       # this is a list of pending actions (or selected categories) that we want to penalize
       pending_actions=[1,2]
        
       batch_choice,probabilityDistribution,S0=timevarying_compute_prob_dist_and_draw_hts(weights, gamma, 
                                                                  batch_size, omega,pending_actions)
       #batch_choice,probabilityDistribution,S0=compute_prob_dist_and_draw_hts(weights, gamma, batch_size,pending_actions)
       choice[count]=[0]*len(batch_choice)
       choice[count]=batch_choice
       # the selected arms are in choice
   
       all_choice=all_choice+batch_choice
       
       theReward=0
       batch_choice=np.asarray(batch_choice)
       
       e_num=2.71 # e number
       right_term=e_num*omega*np.sum(weights)/numActions
       
       for cc in range(numActions):
           idx=np.where(batch_choice==cc)[0]
           
           if len(idx)==0:
               continue       
           
           if cc in S0:
               weights[cc]+=right_term
           else:
               
               # =============================================================================
               # this estimation of the reward comes from the RL...
               # the reward should be normalized [0-1] over time for the best performance....
               theReward = reward_list[ss][cc]
               
               estimatedReward = 1.0 * theReward / (probabilityDistribution[cc]*batch_size )
               weights[cc] *= np.exp(estimatedReward * gamma*batch_size / numActions) + right_term # important that we use estimated reward here!
       #print("weights",np.round(weights,decimals=4))
       
       sum_w=np.sum(weights)
       weights=[w/sum_w for w in weights ]
       
       count+=1
       
      
   # =============================== plot ==================
   width=0.2
   choice_count = np.bincount(all_choice)
   plt.figure()
   plt.title(ss)
   plt.bar(range(numActions), choice_count, width, color='r')

   plt.xlabel("Categorical Choices")
   plt.ylabel("Frequency")

   plt.title("Selection over Choices")
   plt.show()

def test_DepRound_Algorithm(prob):
    # test DepRound algorithm
    selection=[]
    for ii in range(100):
        selection=selection+DepRound(prob,k=2)
    choice_count = np.bincount(selection)
    plt.bar(range(3), choice_count, 0.2, color='r')
    plt.show()

if __name__ == "__main__":
    np.random.seed(1)
    
    # this is the non-time-varying setting
    #run_EXP3_M() 
    
    
    # this is the time-varying setting
    run_TV_EXP3_M()

    #test_DepRound_Algorithm(prob=[0.5,0.2,0.1])


        
   