import numpy as np
from copy import deepcopy
import os
import torch
import pickle

class Batchwalk():
    def __init__(self,y,x,
                 agent,
                 env,
                 optimizer,
                 N = 5,
                 seed = 1234567):
        
        #from inputs
        self.y         = y
        self.x         = x
        self.agent     = agent
        self.env       = env
        self.optimizer = optimizer
        self.N         = N
        self.seed      = seed
        
        #interal values
        self.activeseed = self.seed
        self.gen        = 0 #attempts to solve local area of current home
        self.era        = 0 #number of home locs in env space
        self.homelist   = [self.env]
        self.envlist    = self.makeenvlist(self.env)
        
    def self.makeenvlist(self,home):
        envlist = [home]
        for i in range(self.N):
            generated = False
            attempts = 0
            seed = self.activeseed
            while not generated:
                newenv = home.__copy__()
                sizemut = False

                #chance of mutating y
                np.random.seed(seed)
                p = np.random.random()
                if p < self.getpsizemut() or attempts >= 5:
                    eap.y += 1
                    sizemut = True
                seed += 1
                np.random.seed(seed)
                p = np.random.random()
                if p < self.getpsizemut() or attempts >= 5:
                    eap.x += 1
                    sizemut = True

                seed += 1
                if sizemut:
                    eap.resetinputs() #to adjust framespercell/maxframes for new y and x
                    eap.env.mutate(eap.y,eap.x,0,0,seed)
                    newmaze = True #we will get a different maze than the parent
                else:
                    eap.env.mutate(eap.y,eap.x,eap.mazemutpower,eap.nummuts,seed)
                    newmaze = eap.env.isdifferent(self.env) #check if we get different maze than parent

                #check if new maze is different from other mutations generated
                if newmaze:
                    for child in mutations:
                        if not eap.env.isdifferent(child.env):
                            newmaze = False

                #if completely new maze append to mutations list
                if newmaze:
                    mutations.append(eap)
                    generated = True

                attempts += 1
                seed += 1

        