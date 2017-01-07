import numpy as np
import matplotlib.pyplot as plt
import os
import sarsa

def plot_from_file(agent_number=1, trial_number=100, test=False):
	
	w = np.zeros((agent_number, trial_number, 3, 400))
	learning_latencies = np.zeros((agent_number, trial_number, 1))
	
	for i in range(agent_number):
		j=i+1
		w_id = "data/Trial_weights_%.2d" % i + ".npy"
		lat_id = "data/Trial_latencies_%.2d" % i + ".npy"

		print(w_id)
		print(lat_id)
		if os.path.isfile(w_id) and os.path.isfile(lat_id):
			w[i] = np.load(w_id)
			learning_latencies[i] = np.load(lat_id)
		else:
			print("Undefined file(s)")
			return

	if (test):
		nb_episode_played = 10
		s = sarsa.SarsaAgent()
		step = int(trial_number/nb_episode_played)
		print(step)
		start = step + trial_number%nb_episode_played
		print(start)
		t_out = np.zeros((agent_number, nb_episode_played, 1))
		for agent in range(agent_number):
			for i, trial in enumerate(range(start-1, trial_number+step-1, step)):
				t_out[agent, i] = s.execute_trial(w[agent, trial])

		plt.figure(1)
		plt.plot(np.arange(1, nb_episode_played+1)*step, 
					np.mean(t_out, axis=0))
		plt.show()

	plt.figure(2)
	plt.errorbar(np.arange(learning_latencies.shape[1]), 
				np.mean(learning_latencies, axis=0), np.std(learning_latencies, axis=0))
	plt.show()
	

	w_mean_agents = np.mean(w, axis=0)
	w_mean_neurons = np.mean(w_mean_agents, axis=2)
	plt.figure(3)
	plt.plot(np.arange(learning_latencies.shape[1]), w_mean_neurons)
	plt.show()
