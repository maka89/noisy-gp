# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:21:25 2016

@author: Aleksander
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
import math
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.optimize import minimize
from gp_regressor import *

class SATrainer:
    def __init__(self,gp,l_bounds,num_iter):
        self.init=1
        self.num_iter=num_iter
        self.gp=gp
        self.l_bounds=l_bounds
    def prob(self,e,e_old,T):
        if e<e_old:
            return 1.0
        else:
            return np.exp(-(e-e_old)/T)
            
    def neighbour(self,state,v):
        
        a=np.random.randint(3)
        retstate= np.copy(state)
        
        if a==0:
            tmp= retstate[0] + np.random.normal(0.0,v[0])
            if tmp > self.l_bounds[0,0] and tmp < self.l_bounds[0,1]:
                retstate[0]=tmp
        elif a==1:
            tmp= retstate[1] + np.random.normal(0.0,v[1])
            if tmp > self.l_bounds[1,0] and tmp < self.l_bounds[1,1]:
                retstate[1]=tmp
        elif a==2:
            tmp= retstate[2::] + np.random.normal(0.0,v[2],size=len(retstate[2::]))
            retstate[2::] = tmp*(np.logical_and(tmp>l_bounds[2::,0],tmp<l_bounds[2::,1]))
            
        return retstate,a
       
    def train(self,t_start,t_end,v):
        self.state=(self.l_bounds[:,1]-self.l_bounds[:,0])*np.random.random(len(l_bounds[:,0]))+self.l_bounds[:,0]
        en=self.gp.neg_log_marginal_likelihood(self.state)
        t0=t_start
        temp=t0
        tf=np.power(t_end/t_start,1.0/self.num_iter)
        
        accepted = np.zeros(3)
        total = np.zeros(3)
        for i in range(0,self.num_iter):
            
            statenew,a=self.neighbour(self.state,v)
            ennew=self.gp.neg_log_marginal_likelihood(statenew)
            
            if self.prob(ennew,en,temp) > np.random.rand():
                en=ennew
                self.state=statenew
                accepted[a]+=1
            total[a] +=1
            if i%100==0:
                accratio=accepted/total
                for j in range(0,3):
                    if accratio[j] < 0.3:
                        v[j]/=2
                        
                accepted=np.zeros(3)
                total=np.zeros(3)
                print en,i,v
            
            
            temp=temp*tf
            
if __name__=="__main__":
    X_train=np.random.random((100,2))*2.0-1.0
    y_train=np.sin(3*X_train[:,0])*np.exp(-0.5*np.power(X_train[:,1],2))
    X_std = np.random.random(X_train.shape)*0.2+0.01
    X_train += np.random.normal(0.0,X_std)
    
    Xcv=np.random.random((200,2))*2.0-1.0
    ycv=np.sin(3*Xcv[:,0])*np.exp(-0.5*np.power(Xcv[:,1],2))
    
    l_bounds=np.array([[0.1,2.0],[0.01,0.2],[0.1,5.0],[0.1,5.0]])
    gp=GPRegressor(1,1,num_restarts_hyper=1)
    gp.fit(X_train,y_train,X_std**2,0.0,l_bounds=l_bounds)
    yp,std=gp.predict(Xcv)
    print np.sqrt(np.average((ycv-yp)**2))
    
    trainer = SATrainer(gp,l_bounds,10000)
    trainer.train(5.0,0.01,np.array([0.05,0.01,0.3]))
    gp=GPRegressor(trainer.state[2::],trainer.state[0],num_restarts_hyper=1)
    gp.fit(X_train,y_train,X_std**2,trainer.state[1],None)
    yp,std=gp.predict(Xcv)
    print np.sqrt(np.average((ycv-yp)**2))
    