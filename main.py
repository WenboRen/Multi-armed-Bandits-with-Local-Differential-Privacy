
import numpy as np 

from DPclass import experiment_with_algorithms, experiment_with_epsilon, experiment_with_epsilon_min, plot_algorithms, plot_epsilon


def run_experiment(mu_array, iters, episodes, type_list, type, reward_type, epsilon, epsilon_list, epsilon_min_list):
   
    mu = np.hstack(mu_array)
    k = len(mu) # number of arms
    
    regrets_algorithms= experiment_with_algorithms(k,mu,epsilon,type_list,reward_type,
                                                    iters,episodes)
    
    regrets_epsilon = experiment_with_epsilon(k,mu,epsilon_list,type,reward_type,iters,episodes)

    regrets_epsilon_min = experiment_with_epsilon_min(k,mu,epsilon_min_list,type,reward_type,iters,episodes)
    
    return [regrets_algorithms, regrets_epsilon, regrets_epsilon_min]
    
    
  
def main():
    reward_type = 'Gaussian' # Other options include 'Mixed' and 'Gaussian'
    mu_array = [0.9*np.ones(1), 0.8*np.ones(5), 0.7*np.ones(5), 0.6*np.ones(5), 0.5*np.ones(4)]
  
    iters = 100000 
    episodes = 5
    type_list = [None, 'Bernoulli', 'Laplace']
    type = 'Laplace' # Other option is 'Bernoulli'
    epsilon = 0.5 # Privacy parameter
    epsilon_list = [1,2]
    regrets_algorithms, regrets_epsilon, regrets_epsilon_min= run_experiment(mu_array,iters,episodes,
                            type_list, type, reward_type, epsilon, epsilon_list)
    
    
    plot_algorithms(type_list,regrets_algorithms,iters)
    plot_epsilon(epsilon_list,regrets_epsilon,iters)
    plot_epsilon(epsilon_min_list,regrets_epsilon_min,iters)
    
if __name__ == "__main__":
    main()
    
    
    
    
             
    
    
    
    
