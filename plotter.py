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

	# ******* TEST AN AGENT AT DIFFERENT EPOCHS OF LEARNING *****************
	if (test):
		nb_episode_played = 100
		nb_runs = 10
		s = sarsa.SarsaAgent()
		step = int(trial_number/nb_episode_played)
		start = step - 1 + trial_number%nb_episode_played
		t_out = np.zeros((agent_number, nb_episode_played+1, 2))
		for agent in range(agent_number):
			t_out[agent, 0] = s.execute_trial(w[agent, 0], nb_runs)
			for i, trial in enumerate(range(start, trial_number, step)):
				print(trial)
				t_out[agent, i+1] = s.execute_trial(w[agent, trial], nb_runs)

		np.save('test_escape_smallTau.npy', t_out)
		print("escape time saved!")

		# Plot results of testing different agents at different epochs of learning
		plt.figure(1)
		plt.errorbar(np.arange(0, nb_episode_played+1)*step, 
					np.mean(t_out, axis=0)[:, 0], np.std(t_out, axis=0)[:, 0])
		plt.show()

	# ********* PLOT DATA FROM LEARNING ITSELF ********************************
	# latencies
	plt.figure(2)
	plt.errorbar(np.arange(learning_latencies.shape[1]), 
				 np.mean(learning_latencies, axis=0), 
				 np.std(learning_latencies, axis=0))
	plt.show()
	
	# weight evolution
	w_mean_agents = np.mean(w, axis=0)
	w_mean_neurons = np.mean(w_mean_agents, axis=2)
	plt.figure(3)
	plt.plot(np.arange(learning_latencies.shape[1]), w_mean_neurons)
	plt.show()
