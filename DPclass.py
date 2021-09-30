
import numpy as np 
import matplotlib.pyplot as plt 
class ucb_bandit:

    '''
    Upper Confidence Bound Bandit (with DP)
    
    Inputs 
    ============================================
    k: number of arms (int)
    iters: number of steps (int)
    mu: average rewards for each of the k-arms.
    epsilon: parameter of DP
    type: DP mechanism (e.g., Laplace or Bernoulli)
    '''
    def __init__(self, k, iters, mu, epsilon = 0, type = None):
        
        # Number of arms
        self.k = k
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(k)
        # Total regret
        self.total_regret = 0
        self.regret = np.zeros(iters)
        # Empirical reward for each arm
        self.k_reward = np.zeros(k)
        # Privacy parameter
        self.epsilon = epsilon
        # Average reward for each arm
        self.mu = np.array(mu)
        # DP type
        self.type = type

    def generate_reward(self, mu):
        return np.random.binomial(1,mu,1)

    # Naive UCB
    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_reward +  np.sqrt(
                (2*np.log(self.n)) / self.k_n))
            
       
        reward = self.generate_reward(self.mu[a])
        optimal_reward = self.generate_reward(self.mu[0])
        

        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.total_regret = self.total_regret + optimal_reward - reward
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]

    # LDP-UCB-B 
    def pull_CTB(self):
        # Select action according to the Criteria
        a = np.argmax(self.k_reward + np.sqrt(
                (2*np.log(self.n)) / self.k_n))
            
        reward = self.generate_reward(self.mu[a])
        optimal_reward = self.generate_reward(self.mu[0])

        # Update total
        self.total_regret = self.total_regret + optimal_reward - reward

        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Using Bernoulli mechanism
        mu_converted = (reward*np.exp(self.epsilon)+1-reward )/(1+np.exp(self.epsilon))
        reward_converted = np.random.binomial(1,mu_converted,1)

        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward_converted - self.k_reward[a]) / self.k_n[a]

    # LDP-UCB-L
    def pull_CTL(self):
        # Select action according to the Criteria
        th = 4 * np.log(self.n + 1) # threshold
        k_n_th = [(i,el) for (i,el) in enumerate(self.k_n) if el <= th]
        if len(k_n_th) > 0:
            a = k_n_th[0][0]
        else:
            a = np.argmax(self.k_reward + np.sqrt(
                    (2*np.log(self.n)) / self.k_n) + np.sqrt(
                    (32*np.log(self.n) / (self.epsilon * self.epsilon * self.k_n))))

        reward = self.generate_reward(self.mu[a])
        optimal_reward = self.generate_reward(self.mu[0])

        # Update total
        self.total_regret = self.total_regret + optimal_reward - reward

        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Using Laplace mechanism
        
        reward_converted = reward + np.random.laplace(0,1/self.epsilon,1)

        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward_converted - self.k_reward[a]) / self.k_n[a]

    def run(self):
        for i in range(self.iters):
            if self.type == 'Laplace':
                self.pull_CTL()
            elif self.type == 'Bernoulli':
                self.pull_CTB()
            else:
                self.pull()
            self.regret[i] = self.total_regret
            
    def reset(self):

        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.total_regret = 0
        self.regret = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)
        
class ucb_bandit_heter(ucb_bandit):
    def generate_reward(self, mu):
        '''
        mu = 0.9: Bernoulli
        mu = 0.8: Beta(4,1)
        mu = 0.7: randomly choose [0.4, 1] with prob [0.5, 0.5]
        mu = 0.6: Bernoulli
        mu = 0.5: Uniform in [0,1]
        '''
        if mu == 0.9:
            return np.random.binomial(1,mu,1)
        if mu == 0.8:
            return np.random.beta(4,1,1)
        if mu == 0.7:
            return np.random.choice([0.4, 1], 1, p=[0.5, 0.5])
        if mu == 0.6:
            return np.random.binomial(1,mu,1)
        if mu == 0.5:
            return np.random.uniform(0,1,1)
        
class ucb_bandit_heLDP(ucb_bandit):
    def __init__(self, k, iters, mu, epsilon_min, type = None):
        
        # Average reward for each arm
        self.mu = np.array(mu)
        # Number of arms
        self.k = k
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Total regret
        self.total_regret = 0
        self.regret = np.zeros(iters) 
        # DP type
        self.type = type
        
        self.N = np.zeros(self.k)
        self.s = np.zeros(self.k)
        self.A = np.zeros(self.k)
        self.epsilon_min = epsilon_min
    
        
    def generate_random_epsilon(self):
        
        epsilon = np.random.choice([0.01,0.2,1,2,1000], 1)
        if epsilon > 100:
            epsilon = 100
        return epsilon
    
    def generate_normal_epsilon(self):
        
        mu = 1
        sigma = 1
        epsilon = np.random.normal(mu, sigma, 1)
        if epsilon > 100:
            epsilon = 100
        elif epsilon < 0:
            epsilon = 0.01
        return epsilon
        

    def pull_CTB(self):
        
        
        # Select action according to the Criteria
        if self.n == 1:
            a = np.random.choice(self.k, 1)
        else:
            a = np.argmax(self.s/self.N+ np.sqrt(
                (2*np.log(self.n))*self.A / (self.N**2 )))
        

        self.n += 1   
        reward = self.generate_reward(self.mu[a])
        optimal_reward = self.generate_reward(0.9)
            
        
        # Update total
        self.total_regret = self.total_regret + optimal_reward - reward
                      
        # Generates random epsilon
        epsilon = self.generate_random_epsilon()
                      
        # Using Bernoulli mechanism
        mu_converted = (reward*np.exp(epsilon)+1-reward )/(1+np.exp(epsilon))
        
        reward_converted = np.random.binomial(1,mu_converted,1)
       
            
        
        # Comptute g-value
        if reward_converted == 1:
            g_value = 1/2*(1+(np.exp(epsilon)+1)/(np.exp(epsilon)-1))
        else:
            g_value = 1/2*(1-(np.exp(epsilon)+1)/(np.exp(epsilon)-1))
            
        #Update counts
        if epsilon >= self.epsilon_min:
            self.N[a] += 1
            self.s[a] += g_value
            self.A[a] += ( (np.exp(epsilon)+1)/(np.exp(epsilon)-1) )**2
            
            
            
            
    def pull_CTL(self):
        
        th = 4*np.log(self.n)*(1/self.epsilon_min)**2
        A_th = [(i,el) for (i,el) in enumerate(self.A) if el <= th]
        
        # Select action according to the Criteria
        if self.n == 1:
            a = np.random.choice(self.k, 1)
        else:
            if len(A_th) > 0:
                temp = [i[0] for i in A_th]
                a = np.random.choice(temp, 1)
            else:
                a = np.argmax(self.s/self.N+ np.sqrt(2*np.log(self.n)/self.N) + np.sqrt(
                    (32*np.log(self.n))*self.A / (self.N**2 )))
        

        self.n += 1   
        reward = self.generate_reward(self.mu[a])
        optimal_reward = self.generate_reward(0.9)
            
        
        # Update total
        self.total_regret = self.total_regret + optimal_reward - reward
                      
        # Generates random epsilon
        epsilon = self.generate_random_epsilon()
                      
        # Using Laplace mechanism
        reward_converted = reward + np.random.laplace(0,1/epsilon,1)
       
            
            
        #Update counts
        if epsilon >= self.epsilon_min:
            self.N[a] += 1
            self.s[a] += reward_converted
            self.A[a] += ( 1/epsilon)**2
        
        
        
    
    
    def reset(self):
        

        # Resets results while keeping settings
        self.n = 1
        self.N = np.zeros(self.k)
        self.s = np.zeros(self.k)
        self.A = np.zeros(self.k)
        self.total_regret = 0
        self.regret = np.zeros(self.iters)



    
        
    
            
class ucb_bandit_gaussian(ucb_bandit):
    def generate_reward(self, mu):
  
        sigma = 1
        return np.random.normal(mu,sigma,1)
    
    # Using Sigmoid
    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_reward +  np.sqrt(
                (2*np.log(self.n)) / self.k_n))
            
       
        reward = self.generate_reward(self.mu[a])
        optimal_reward = self.generate_reward(self.mu[0])
        

        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.total_regret = self.total_regret + optimal_reward - reward
        
        sr = 1/(1+np.exp(-reward))
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            sr - self.k_reward[a]) / self.k_n[a]

    # Using Sigmoid 
    def pull_CTB(self):
        # Select action according to the Criteria
        a = np.argmax(self.k_reward + np.sqrt(
                (2*np.log(self.n)) / self.k_n))
            
        reward = self.generate_reward(self.mu[a])
        optimal_reward = self.generate_reward(self.mu[0])

        # Update total
        self.total_regret = self.total_regret + optimal_reward - reward

        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Using Sigmoid-Bernoulli mechanism
        sr = 1/(1+np.exp(-reward))
        mu_converted = (sr*np.exp(self.epsilon)+1-sr )/(1+np.exp(self.epsilon))
        reward_converted = np.random.binomial(1,mu_converted,1)

        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward_converted - self.k_reward[a]) / self.k_n[a]

    # Using Sigmoid 
    def pull_CTL(self):
        # Select action according to the Criteria
        th = 4 * np.log(self.n + 1) # threshold
        k_n_th = [(i,el) for (i,el) in enumerate(self.k_n) if el <= th]
        if len(k_n_th) > 0:
            a = k_n_th[0][0]
        else:
            a = np.argmax(self.k_reward + np.sqrt(
                    (2*np.log(self.n)) / self.k_n) + np.sqrt(
                    (32*np.log(self.n) / (self.epsilon * self.epsilon * self.k_n))))

        reward = self.generate_reward(self.mu[a])
        optimal_reward = self.generate_reward(self.mu[0])

        # Update total
        self.total_regret = self.total_regret + optimal_reward - reward

        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Using Sigmoid-Laplace mechanism
        sr = 1/(1+np.exp(-reward))
        reward_converted = sr + np.random.laplace(0,1/self.epsilon,1)

        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward_converted - self.k_reward[a]) / self.k_n[a]
    
def experiment_with_algorithms(k, mu, epsilon, type_list, reward_type = 'Bern', iters=100000, episodes=100):
    
    UCB_list = []
    for i in range(len(type_list)):
        if reward_type == 'Bern':
            UCB_list.append(ucb_bandit(k,iters, mu, epsilon, type_list[i]))
        elif reward_type == 'Mixed':
            UCB_list.append(ucb_bandit_heter(k,iters, mu, epsilon, type_list[i]))
        elif reward_type == 'Gaussian':
            UCB_list.append(ucb_bandit_gaussian(k,iters, mu, epsilon, type_list[i]))

    Regrets_list = []
    for _ in range(len(type_list)):
        Regrets_list.append(np.zeros(iters))
    for j in range(len(type_list)):
        for i in range(episodes): 
            UCB_list[j].reset()
            UCB_list[j].run()
            Regrets_list[j] = Regrets_list[j] + (
                UCB_list[j].regret - Regrets_list[j]) / (i + 1)

    return Regrets_list
    
def experiment_with_epsilon(k, mu, epsilon_list, type, reward_type = 'Bern', iters=100000, episodes=100):
    UCB_list = []
    for i in range(len(epsilon_list)):
        if reward_type == 'Bern':
            UCB_list.append(ucb_bandit(k,iters, mu, epsilon_list[i],type))
        elif reward_type == 'Mixed':
            UCB_list.append(ucb_bandit_heter(k,iters, mu, epsilon_list[i],type))
        elif reward_type == 'Gaussian':
            UCB_list.append(ucb_bandit_gaussian(k,iters, mu, epsilon_list[i],type))

    Regrets_list = []
    for _ in range(len(epsilon_list)):
        Regrets_list.append(np.zeros(iters))

    for j in range(len(epsilon_list)):
        for i in range(episodes): 
            UCB_list[j].reset()
            UCB_list[j].run()
            Regrets_list[j] = Regrets_list[j] + (
                UCB_list[j].regret - Regrets_list[j]) / (i + 1)

    return Regrets_list


def experiment_with_epsilon_min(k, mu, epsilon_min_list, type, iters=100000, episodes=100):
    UCB_list = []
    for i in range(len(epsilon_min_list)):
        UCB_list.append(ucb_bandit_heLDP(k,iters, mu, epsilon_min_list[i],type))

    Regrets_list = []
    for _ in range(len(epsilon_min_list)):
        Regrets_list.append(np.zeros(iters))

    for j in range(len(epsilon_min_list)):
        for i in range(episodes): 
            UCB_list[j].reset()
            UCB_list[j].run()
            Regrets_list[j] = Regrets_list[j] + (
                UCB_list[j].regret - Regrets_list[j]) / (i + 1)

    return Regrets_list
    
def plot_algorithms(type_list, regrets_algorithms, iters):
    fig = plt.figure()
    ax = fig.gca()
    for i in range(len(type_list)):
        type = type_list[i] if type_list[i] is not None else 'UCB'
        if type == 'Bernoulli':
            label = 'LDP-UCB-B'
        elif type == 'Laplace':
            label = 'LDP-UCB-L'
        else:
            label = 'Non-Private-UCB'
        ax.plot(regrets_algorithms[i], label=label )
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("T")
    plt.ylabel("Regret")
    plt.axis([100, iters, 1, 100000])
    plt.grid('True',linestyle = '--')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.rcParams.update({'font.size': 16})
    #plt.ticklabel_format(style='sci', axis='x',scilimits=(5,5))
    ax.legend(loc='lower right', frameon=False)
    plt.show()
    
def plot_epsilon(epsilon_list,regrets_epsilon, iters):
    fig = plt.figure()
    ax = fig.gca()
    for i in range(len(epsilon_list)):
        ep = epsilon_list[i]
        ax.plot(regrets_epsilon[i], label=r'$\epsilon = $'+str(ep) )
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("T")
    plt.ylabel("Regret")
  
    plt.axis([100, iters, 1, 100000])
    plt.grid('True',linestyle = '--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.rcParams.update({'font.size': 16})
    #plt.ticklabel_format(style='sci', axis='x',scilimits=(4,4))
    ax.legend(loc='lower right', frameon=False)
    plt.show()