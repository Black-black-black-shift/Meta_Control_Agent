"""
Created on Sat 2024/8/24

@author: JT P
"""

import torch

from torch import zeros, ones, eye

import torch.distributions as dist

import numpy as np

from torch.distributions import Categorical

def von_mises_half(x, mu, kappa):

    p = (torch.exp(kappa * torch.cos(x - mu)) + torch.exp(kappa * torch.cos(-x - mu)))/ (2 * torch.pi * torch.i0(kappa))

    return p

def von_mises_half_tensor(x_tensor, mu, kappa):

    pdf = zeros(len(x_tensor))
    for i, x in enumerate(x_tensor):
        pdf[i] = von_mises_half(x, mu, kappa)

    return pdf

class MetaAgent(object):

    def __init__(self, fit, conditions=3, blocks=10, trials=10, ni =2, dt=0.01, f_prior=[0.5, 0.5, 0.5], pF = 0.5, de_F = 0.01, de_C = 0.01, d_con = True, e_con = True, update = True):
        
        self.nb = blocks
        self.nt = trials
        self.nc = conditions
        self.ni = ni
        self.nw = 2
        self.pF = pF
        self.update = update

        self.f_prior = f_prior
        
        assert fit['mlgd0'][0, 0, 0] == fit['mlgd0'][1, 0, 0] and fit['mlgd0'][0, 0, 0] == fit['mlgd0'][2, 0, 0]
        assert fit['sigmad0'][0, 0, 0] == fit['sigmad0'][1, 0, 0] and fit['sigmad0'][0, 0, 0] == fit['sigmad0'][2, 0, 0]

        self.mlgd0 = fit['mlgd0'] #(nc, ni, nw) #a small prior mean will encourage the agent to try different strategy initially 
        self.sigmad0 = fit['sigmad0'] #(nc, ni, nw)
        self.sigma = fit['sigma']
        self.beta = fit['beta']#.clone().detach().requires_grad_(True)# grad
        self.gamma = fit['gamma']#.clone().detach().requires_grad_(True)# grad
        self.lgdm_ff = fit['lgdm_ff']#.clone().detach().requires_grad_(True)#(nc) grad
        self.ds_ff = fit['ds_ff']
        self.lgdm_cf = fit['lgdm_cf']#.clone().detach().requires_grad_(True)# grad
        self.ds_cf = fit['ds_cf']
        self.kappa = fit['kappa']
        self.me0 = fit['me0'] #(nc, ni, nw)
        self.ke0 = fit['ke0'] #(nc, ni)

        self.eF_cri = fit['eF_cri']
        self.eC_cri = fit['eC_cri']

        #self.w = data['questions'] #(b, t, 2)
        #self.d = data['reaction_times'] #(b, t) #If reaction time in F-trial is not reliable, set two constants according to strategy c, f  the agent chose

        self.dt = dt
        self.de_F = de_F
        self.de_C = de_C
        self.mesam_F = torch.arange(0, torch.pi, self.de_F)
        self.mesam_C = torch.arange(0, 1, self.de_C)

        self.d_con = d_con
        self.e_con = e_con

        self.initiate_prior_plan_strategy()

        if self.d_con:
            self.initiate_md_distribution()
        if self.e_con:
            self.initiate_me_distribution()
        
        #self.initiate_pF()

    def initiate_prior_plan_strategy(self):

        self.flag = zeros(self.nc)
        self.prior_plan_str = zeros(self.nc, self.ni)
        self.plan_str = []
        for condition in torch.arange(self.nc):
            if self.ni == 2:
                self.prior_plan_str[condition] = torch.tensor([self.f_prior[condition], 1-self.f_prior[condition]])
            elif self.ni == 1:
                self.prior_plan_str[condition] = torch.tensor([1.])


    def initiate_md_distribution(self): #Also need to be done for different conditions
        
        self.evi_d_list = []
        self.belief_md = {}

        ignore_thres = np.exp(torch.max(self.mlgd0 + 3 * self.sigmad0))
        self.dsam = torch.arange(self.dt, ignore_thres, self.dt)

        for c in range(self.nc):
            for i in range(self.ni):
                 for w in range(self.nw):
                    lognorm_dist = dist.LogNormal(loc=self.mlgd0[c, i, w].clone(), scale=self.sigmad0[c, i, w].clone())
                    self.belief_md[(c, i, w)] = [lognorm_dist.log_prob(self.dsam).exp()]
                    #print(self.dt * torch.sum(self.belief_md[(c, i, w)][-1]))
                    #self.belief_md[(c, i, w)] = []
                    #self.belief_md[(c, i, w)].append(self.prior_md[(c, i, w)].clone())
                                        
        #self.belief_md_ff = []
        #lognorm_dist0 = dist.LogNormal(loc=self.mlgd0[0, 0, 0].clone(), scale=self.sigmad0[0, 0, 0].clone())
        #self.belief_md_ff.append(lognorm_dist0.log_prob(self.dsam).exp())

    def initiate_me_distribution(self): #Also need to be done for different conditions
        
        self.prior_me = {}
        self.evi_e_list = []
        self.belief_me = {}
        
        for c in range(self.nc):
            for i in range(self.ni):
                 self.belief_me[(c, i, 0)] = [von_mises_half_tensor(self.mesam_F, self.me0[c, i].clone(), self.ke0[c, i].clone())]
                 self.belief_me[(c, i, 1)] = [torch.ones(len(self.mesam_C))]

    def initiate_pF(self):
        self.Beta_paras = torch.tensor([1, 1])
        self.dp = 0.01
        self.psam = torch.arange(0, 1, self.dp)
        self.npsam = len(self.psam)
        Beta_dist_0 = dist.Beta(self.Beta_paras[0], self.Beta_paras[1])
        self.belief_pF = []
        #self.belief_pF.append(Beta_dist_0.log_prob(self.psam).exp()) #If it shows no problem, we just need a scalar rather tahn a dictionary (don't save each time) 

    def get_d_likelihood(self, d):
        normal_dist = dist.LogNormal(loc=torch.log(d), scale=self.sigma)
        self.mdsam = self.dsam
        Ad = (self.mdsam / d) * (normal_dist.log_prob(self.mdsam).exp()) #d is the random variable rather than md!

        return Ad #This is a function of md
    
    def get_e_likelihood(self, e, question):
        if question == 0:
            Ae = von_mises_half(e, self.mesam_F, self.kappa)
        elif question == 1:
            if e == 0:
                Ae = 1 - self.mesam_C
            elif e == 1:
                Ae = self.mesam_C

        return Ae

    def plan(self, b, t, condition, mean_f, mean_c, std_f, std_c):
        
        if self.flag[condition] == 0:
            self.plan_str.append(self.prior_plan_str[condition].clone())
            self.flag[condition] = 1
        elif self.flag[condition] == 1:
            
            E_md = zeros(self.ni)
            E_me = zeros(self.ni)

            if self.d_con:
                #Mean compare? All cross distribution compare?
                # score(i) = sum ((-gamma*md) pF+(-gamma*md) (1-pF)) p(pF)p(c, i, w=C)[md]
                #E_md = zeros(self.ni, self.npsam) #pF inference
                
                for i in range(self.ni):
                    E_md_F = self.dt * torch.sum(self.dsam * self.belief_md[(condition, i, 0)][-1])
                    E_md_C = self.dt * torch.sum(self.dsam * self.belief_md[(condition, i, 1)][-1])
                    #E_md[i] = self.psam * E_md_F.clone() + (1 - self.psam) * E_md_C.clone() #Considering pF inference
                    E_md[i] = E_md_C.clone()#self.pF * E_md_F.clone() + (1 - self.pF) * E_md_C.clone()#
            
            if self.e_con:
                
                for i in range(self.ni):
                    E_me_F = self.de_F * torch.sum(self.mesam_F * self.belief_me[(condition, i, 0)][-1])
                    E_me_C = self.de_C * torch.sum(self.mesam_C * self.belief_me[(condition, i, 1)][-1])
                    #E_md[i] = self.psam * E_md_F.clone() + (1 - self.psam) * E_md_C.clone() #Considering pF inference
                    E_me[i] = self.pF * (E_me_F.clone() - mean_f) / std_f + (1 - self.pF) * (E_me_C.clone() - mean_c) / std_c

            #score = - self.gamma * torch.einsum('ip,p->i', E_md, self.belief_pF[-1]) #shape to be check #Considering pF inference
            score = - self.gamma * E_md - self.beta * E_me
            #self.d_score.append(- self.gamma * E_md.clone())
            #self.e_score.append(- self.beta * E_me.clone())
            #self.score_list.append(score.clone())
            p_i = torch.softmax(score, dim=0)
            self.plan_str.append(p_i.clone())
    
    def update_beliefs(self, b, t, condition, strategy, question, reaction_time, error):
        if self.d_con:
            self.update_d_beliefs(b, t, condition, strategy, question, reaction_time)
        if self.e_con:
            self.update_e_beliefs(b, t, condition, strategy, question, error)

    def update_d_beliefs(self, b, t, condition, strategy, question, reaction_time):

        c = condition
        i = strategy
        w = question
        d = reaction_time
    
        #firstly update belief about d
        #determine the prior to be used:
        prior = self.belief_md[(c, i, w)][-1]

        if torch.isnan(d):
            posterior = prior #Skip the trial with NaN reaction time
            self.evi_d_list.append(torch.tensor(float('nan')))
        else:
            likelihood = self.get_d_likelihood(d)
            evidence = self.dt * torch.sum(likelihood * prior)
            posterior = (likelihood * prior) / evidence # normalization to be checked

            if w == 0:
                self.evi_d_list.append(torch.tensor(float('nan')))
            else:
                self.evi_d_list.append(evidence.clone())

        if self.update:
            for cc in range(self.nc):
                for ii in range(self.ni):
                    for ww in range(self.nw):
                        last_belief = self.belief_md[(cc, ii, ww)][-1]
                        if i==0 and w==0:
                            if ii == 0 and ww==0:
                                self.belief_md[(cc, ii, ww)].append(posterior.clone())
                                #self.belief_md_ff.append(posterior.clone())
                            else:
                                self.belief_md[(cc, ii, ww)].append(last_belief.clone())
                        else:
                            if cc==c and ii == i and ww==w:
                                self.belief_md[(cc, ii, ww)].append(posterior.clone())
                            else:
                                self.belief_md[(cc, ii, ww)].append(last_belief.clone())
                            #we need clone when a variable name is reused (espetially those instrumental variable in loops)
        
        
                        
    def update_e_beliefs(self, b, t, condition, strategy, question, error):

        c = condition
        i = strategy
        w = question
        e = error

        prior = self.belief_me[(c, i, w)][-1]
        
        if torch.isnan(e):
            posterior = prior #Skip the trial with NaN reaction time
            self.evi_e_list.append(torch.tensor(float('nan')))
        else:
            likelihood = self.get_e_likelihood(e, question)

            if question == 0:
                evidence = self.de_F * torch.sum(likelihood * prior)
            elif question ==1:
                evidence = self.de_C * torch.sum(likelihood * prior)

            posterior = (likelihood * prior) / evidence # normalization to be checked
            
            self.evi_e_list.append(evidence.clone())

        if self.update:
            for cc in range(self.nc):
                for ii in range(self.ni):
                    for ww in range(self.nw):
                        last_belief = self.belief_me[(cc, ii, ww)][-1]
                        if i==0 and w==0:
                            if ii == 0 and ww == 0:
                                self.belief_me[(cc, ii, ww)].append(posterior.clone())
                                #self.belief_md_ff.append(posterior.clone())
                            else:
                                self.belief_me[(cc, ii, ww)].append(last_belief.clone())
                        else:
                            if cc==c and ii == i and ww==w:
                                self.belief_me[(cc, ii, ww)].append(posterior.clone())
                            else:
                                self.belief_me[(cc, ii, ww)].append(last_belief.clone())
                            #we need clone when a variable name is reused (espetially those instrumental variable in loops)
        
        