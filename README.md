# Multi-armed-Bandits-with-Local-Differential-Privacy

This repo is for our work "Ren, W., Zhou, X., Liu, J., & Shroff, N. B. (2020). Multi-armed bandits with local differential privacy. arXiv preprint arXiv:2007.03121."

Any entity can use this repo as long as the above work is properly cited.

The structure of .py files is as follows.
    * DPclass.py: define all the necessary classes and functions
	-- ucb_bandit: basic class for MAB Bern reward under UCB with LDP	
	-- ucb_bandit_gaussian: child class of ucb_bandit, defined for gaussian reward
	-- ucb_bandit_heter: child class of ucb_bandit, defined for mixed reward
	-- experiment_with_algorithms: fix a LDP parameter (i.e., epsilon), test for different algorithms, defined by type_list (e.g., Non-private, LDP-UCB-L, LDP-UCB-B)
	-- experiment_with_epsilon: fix a algorithm (e.g., LDP-UCB-B), test for different LDP parameters, defined by epsilon_list (i.e., )
	-- plot_algorithms: plot the results obtained by experiment_with_algorithms
	-- plot_epsilon: plot the results obtained by experiment_with_epsilon

    * main.py: run the experiment and plot the figures.
	-- reward_type: Bern, Mixed or Gaussian as considered in the paper
	-- iters: number of time-slots, i.e., T
	-- episodes: number of runs
	-- mu_array: define the problem instance
	
	
 
