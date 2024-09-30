#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:48:53 2023

@author: tommasoravasio
"""
#%%
"""SETUP CODE"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import generate_data

def accept(delta_c, beta):
    
    if delta_c <= 0:
        return True
    
    if beta == np.inf:
        return False
    
    p = np.exp(-beta * delta_c)
    
    return np.random.rand() < p

C=generate_data.generate_data(100,"3192281")

#%%
"""Question 1"""
class Probl:
    def __init__(self,C):
        
        n=C.shape[0]
        
        self.C=C
        self.n=n
        self.x=-1 #sentinel value
        self.y=-1 #sentinel value
        
    def init_config(self,seed=None):
        if seed is not None:
            np.random.seed(seed)    
        n=self.n
        
        #start at an initial random point i,j
        x0=np.random.randint(n)
        y0=np.random.randint(n)

        self.x=x0
        self.y=y0
        
    
    def cost(self):
        
        #getting the cost just mean to get the value of the function at the point (i,j)
        C,x,y=self.C,self.x,self.y
        return C[x,x]
    
    def propose_move(self):
        x,y,n=self.x,self.y,self.n
        
        #create an array of the two new options of x and use 
        #random.choice to select randomly one of 
        #the two as proposed move
        
        x_opt=np.array([(x-1)%n,(x+1)%n])
        x_prop=np.random.choice(x_opt)
        
        #same for y
        y_opt=np.array([(y-1)%n,(y+1)%n])
        y_prop=np.random.choice(y_opt)
        
        move=(x_prop,y_prop)
        return move
    
    def compute_delta_cost(self,move):
        C,x,y=self.C,self.x,self.y
        
        #unpack the proposed move
        x_prop,y_prop=move
        
        #compute cost of old config
        old_c=C[x,y]                        
        
        #compute cost of new config
        new_c=C[x_prop,y_prop]
        
        #compute delta cost
        delta_c=new_c-old_c
        
        return delta_c
    
    def accept_move(self,move):
        
        
        #unpack the proposed move
        x_prop,y_prop=move
        
        #change i and j to the new values
        self.x=x_prop
        self.y=y_prop
        
    def copy(self):
        return deepcopy(self)
    
        
probl=Probl(C)
    
#%%

#%%
"""QUESTION 3"""

def simann_q3(probl,
           anneal_steps = 10, mcmc_steps = 100,
           beta0 = 0.1, beta1 = 10.0,
           seed = None):
   
    if seed is not None:
        np.random.seed(seed)

    
    beta_list = np.zeros(anneal_steps)
    beta_list[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
    beta_list[-1] = np.inf

    
    probl.init_config()
    c = probl.cost()
    best_c = c
    cost_list=[]

    
    for beta in beta_list:
        accepted = 0
        for t in range(mcmc_steps):
            move = probl.propose_move()
            delta_c = probl.compute_delta_cost(move)
            
            if accept(delta_c, beta):
                probl.accept_move(move)
                c += delta_c
                accepted += 1
                if c <= best_c:
                    best_c = c
        cost_list.append(best_c)
    
    return beta_list, cost_list


def get_avg_cost_q3(probl,iteration=100,anneal_steps=50):
    cost_table=np.zeros((iteration,anneal_steps))
    for i in range(iteration):
        beta_list,cost_list = simann_q3(probl,anneal_steps=anneal_steps)
        cost_table[i,:]=cost_list
    avg_cost = np.mean(cost_table, axis = 0)
    return beta_list, avg_cost

def plot_cost_q3(beta_list,avg_cost,title="average best cost for each beta"):
    plt.clf()
    plt.plot(beta_list,avg_cost)
    plt.grid()
    plt.xlabel("Beta")
    plt.ylabel("Cost")
    plt.title(title)
    

# #avg case
# beta_list,avg_cost=get_avg_cost_q3(probl)
# plot_cost_q3(beta_list,avg_cost)

#single case
beta_list,avg_cost=get_avg_cost_q3(probl,iteration=1,anneal_steps=20)
plot_cost_q3(beta_list,avg_cost,title="Best cost for each beta (single case)")

#%%
"""Question 4"""
import time 

def simann_q4(probl,
           anneal_steps = 10, mcmc_steps = 100,
           beta0 = 0.1, beta1 = 10.0,
           seed = None):
    
    if seed is not None:
        np.random.seed(seed)
        
    t0=time.time()
    time_res=0
    
    beta_list = np.zeros(anneal_steps)
    beta_list[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
    beta_list[-1] = np.inf

    probl.init_config()
    c = probl.cost()
    
    best = probl.copy()
    best_c = c

    for beta in beta_list:
        
        accepted = 0
        for t in range(mcmc_steps):
            move = probl.propose_move()
            delta_c = probl.compute_delta_cost(move)
            
            if accept(delta_c, beta):
                probl.accept_move(move)
                c += delta_c
                accepted += 1
                if c <= best_c:
                    best_c = c
                    best = probl.copy()
        


        if accepted==0:         
            t1=time.time()      
            time_res=t1-t0
            converged= -105 <= best_c <= -95 #if best cost is different less than 5(arbitrary value) from -100 we know it converged
            return best_c,time_res,converged
    t1=time.time()      
    time_res=t1-t0    
    converged= -105 <= best_c <= -95
    return best_c,time_res,converged

def get_avg_res_q4(n_list,iterations=5):
    ln=len(n_list)
    
    results_list=np.zeros((ln,2))

    
    probl_list=np.array([Probl(generate_data.generate_data(n,"3192281")) for n in n_list])
    
    for i in range(ln):
        
        res=np.zeros((iterations,2))
        
        for j in range(iterations):
            best_c, res_time, res_converged = simann_q4(probl_list[i])
            
            res[j,:]=[res_time,res_converged]
          
        results_list[i]=np.mean(res,axis=0).reshape(2,)
    
    time_list=results_list[:,0]
    converged_list=results_list[:,1]
    return time_list, converged_list

def plot_time_q4(n_list,time_list):
    plt.clf()
    plt.plot(n_list, time_list, marker='o', linestyle='-')
    plt.xlabel('size')
    plt.ylabel('Time')
    plt.title('size vs time')
    
def plot_converged_q4(n_list,converged_list):
    plt.clf()
    plt.plot(n_list, converged_list, marker='o', linestyle='-')
    plt.xlabel('size')
    plt.ylabel('Percentage of converged')
    plt.title('size vs percentage of converged')

n_list=[100, 200,400,600,800, 1000]  
time_list, converged_list=get_avg_res_q4(n_list,iterations=50)
plot_converged_q4(n_list,converged_list)

#%%
"""
Question 5
"""

def simann_q5(probl,
           anneal_steps = 10, mcmc_steps = 100,
           beta0 = 0.1, beta1 = 10.0,
           seed = None):
    
    if seed is not None:
        np.random.seed(seed)

    
    beta_list = np.zeros(anneal_steps)
    
    beta_list[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
    
    beta_list[-1] = np.inf

    
    probl.init_config()
    c = probl.cost()
    

    
    best = probl.copy()
    best_c = c

    acc_rate_list=[]    
    for beta in beta_list:
        
        accepted = 0
        
        for t in range(mcmc_steps):
            move = probl.propose_move()
            delta_c = probl.compute_delta_cost(move)
            
            if accept(delta_c, beta):
                probl.accept_move(move)
                c += delta_c
                accepted += 1
                if c <= best_c:
                    best_c = c
                    best = probl.copy()
        
        acc_rate=accepted/mcmc_steps
        acc_rate_list.append(acc_rate)
    acc_rate_list=np.array(acc_rate_list)
    
    return best, acc_rate_list

def get_avg_acc_rate_q5(probl,anneal_steps=10,iterations=10):
    avg_acc_rate=np.zeros((iterations,anneal_steps))
    for i in range(iterations):
        best, acc_rate_list=simann_q5(probl,anneal_steps=anneal_steps)
        avg_acc_rate[i]=acc_rate_list
    avg_acc_rate=np.mean(avg_acc_rate,axis=0)
    return avg_acc_rate

def plot_avg_acc_rate_q5(avg_acc_rate,title="Average accuracy rate over betas"):
    plt.clf()
    plt.plot(avg_acc_rate, linestyle='-')
    plt.xlabel('betas')
    plt.ylabel('acceptancy rate')
    plt.title(title)
    plt.show()

# #uncomment to plot the relation between avg_acc_rate and the increasing betas
# avg_acc_rate=get_avg_acc_rate_q5(probl,iterations=50, anneal_steps=40)
# plot_avg_acc_rate_q5(avg_acc_rate)

#to plot the single case
single_case_acc_rate=get_avg_acc_rate_q5(probl,iterations=1, anneal_steps=20)
plot_avg_acc_rate_q5(single_case_acc_rate,title="accuracy rate over betas (single case)")


#%%
"""
Question 6
"""
def plot_matrix2D(probl):
    plt.clf()
    C=probl.C
    plt.imshow(C,origin="lower")
    plt.colorbar(label="Cost")
    plt.title("2D plot of the matrix")

def plot_matrix3D(probl):
    x,y =np.meshgrid(np.arange(C.shape[1]),np.arange(C.shape[0]))
    
    fig=plt.figure()
    ax=fig.add_subplot(111,projection="3d")
    
    ax.plot_surface(x,y,C,cmap="viridis")
    plt.title("3D plot of the matrix")
    plt.show()

plot_matrix2D(probl)
plot_matrix3D(probl)

#%%
