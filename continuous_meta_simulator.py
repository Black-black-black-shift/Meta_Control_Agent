"""This module contains the Simulator class that defines  ~. It also keeps track of all generated observations
and responses. To initiate it one needs to provide the environment
class and the agent class that will be used for the experiment.
"""
import torch
import numpy as np

zeros = torch.zeros

from torch.distributions import Categorical

#__all__ = [
#        'Simulator'
#]

def get_d(agent, condition, strategy, question, reaction_time):
        if question == 1:
            d = reaction_time
        elif question == 0:
            if strategy == 0:
                d = torch.exp(torch.normal(mean=agent.lgdm_ff, std=agent.ds_ff))
            elif strategy == 1:
                d = torch.exp(torch.normal(mean=agent.lgdm_cf[condition], std=agent.ds_cf[condition]))
        return d

class Simulator_Behavior_C(object):
 
    def __init__(self, data, agents, conditions=3, blocks=1, trials=10):
        # set inital elements of the world to None
        self.data = data #data['questions'], data['reaction_times'], maybe data['errors']
        self.agents = agents
        
        self.nc = conditions
        self.nb = blocks  # number of experimental blocks
        self.nt = trials  # number of trials in each block

        #self.strs = zeros(self.nb, self.nt)

    def simulate_experiment(self):
        
        self.lglike_sum = 0
        #self.lglike_table = zeros(self.nb, self.nt)
        self.strs = []
        #self.err_f_eff = []
        #self.err_c_eff = []

        self.error_al = 0
        self.error_op = 0

        self.count_al = 0
        self.count_op = 0

        self.error_diff = 0

        #for agent, strs, data in zip(self.agents, self.strs, self.data):
        agent = self.agents
        data = self.data
        t_count = 0
        
        for b in torch.arange(self.nb):

            condition = self.data['conditions'][b]

            for t in torch.arange(self.nt):

                # update single trial
                agent.plan(b, t, int(condition), data['mean_f_error'], data['mean_c_error'], data['std_f_error'], data['std_c_error'])                
                categorical_dist = torch.distributions.Categorical(agent.plan_str[-1])
                #rand_dist = torch.distributions.Categorical(torch.tensor([0.5, 0.5]))
                strategy = categorical_dist.sample()
                #strategy = rand_dist.sample()
                #strategy = torch.tensor(1)
                
                #self.lglike_table[b, t] = agent.plan_str[-1][strategy]
                #self.lglike_sum += agent.plan_str[-1][strategy]

                self.strs.append(strategy.clone())

                question = data['questions'][b, t]
                reaction_time = get_d(agent, condition=int(condition), strategy=int(strategy), question=int(question), reaction_time=data['reaction_times'][b, t])
                error = data['errors'][b, t]
                agent.update_beliefs(b, t, condition=int(condition), strategy=int(strategy), question=int(question), reaction_time=reaction_time, error = error, mean_f=data['mean_f_error'], mean_c=data['mean_c_error'], std_f=data['std_f_error'], std_c=data['std_c_error'])
                
                #Accumulate the acurracy difference
                if ~torch.isnan(data['errors'][b, t]):
                    if question == 0:
                        #error_eff = data['errors'][b, t] / data['mean_f_error']
                        error_eff = (data['errors'][b, t] - data['mean_f_error']) / data['std_f_error']
                        #self.err_f_eff.append(error_eff)
                    elif question == 1:
                        #error_eff = data['errors'][b, t] / data['mean_c_error']
                        error_eff = (data['errors'][b, t] - data['mean_c_error']) / data['std_c_error']
                        #self.err_c_eff.append(error_eff)
                    
                    if strategy == question:
                        self.error_al += error_eff
                        self.count_al += 1
                    else:
                        self.error_op += error_eff
                        self.count_op += 1
                t_count += 1
                print(t_count)
                    
        self.error_al = self.error_al / self.count_al
        self.error_op = self.error_op / self.count_op
        self.error_diff = self.error_al - self.error_op

        
