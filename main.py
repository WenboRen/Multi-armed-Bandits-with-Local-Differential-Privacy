#!/usr/bin/env python3

import numpy as np 

from DPclass import experiment_with_algorithms, experiment_with_epsilon, plot_algorithms, plot_epsilon


def run_experiment(mu_array, iters, episodes, type_list, type, reward_type, epsilon, epsilon_list):
   
    mu = np.hstack(mu_array)
    k = len(mu) # number of arms
    
    regrets_algorithms= experiment_with_algorithms(k,mu,epsilon,type_list,reward_type,
                                                    iters,episodes)
    
    regrets_epsilon = experiment_with_epsilon(k,mu,epsilon_list,type,reward_type,iters,episodes)
    
    return [regrets_algorithms, regrets_epsilon]
    
    
  
def main():
    reward_type = 'Gaussian' # Other options include 'Mixed' and 'Gaussian'
    mu_array = [0.9*np.ones(1), 0.8*np.ones(5), 0.7*np.ones(5), 0.6*np.ones(5), 0.5*np.ones(4)]
  
    iters = 100000 
    episodes = 5
    type_list = [None, 'Bernoulli', 'Laplace']
    type = 'Laplace' # Other option is 'Bernoulli'
    epsilon = 0.5 # Privacy parameter
    epsilon_list = [1,2]
    regrets_algorithms, regrets_epsilon = run_experiment(mu_array,iters,episodes,
                            type_list, type, reward_type, epsilon, epsilon_list)
    
    
    plot_algorithms(type_list,regrets_algorithms,iters)
    plot_epsilon(epsilon_list,regrets_epsilon,iters)
    
if __name__ == "__main__":
    main()
