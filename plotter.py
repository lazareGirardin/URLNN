import numpy as np
import matplotlib.pyplot as plt
import os
import sarsa

def plot_from_file(agent_number=1, trial_number=100, test=False):
	
	# Plot the moving average over epochs
	smoothing = False

	w = np.zeros((agent_number, trial_number, 3, 400))
	learning_latencies = np.zeros((agent_number, trial_number, 1))
	
	for i in range(agent_number):
		j=i+1
		w_id = "data/weights20_%.2d" % i + ".npy"
		lat_id = "data/latencies20_%.2d" % i + ".npy"

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
		nb_episode_played = 50
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

		np.save('test_escape_e002.npy', t_out)
		print("escape time saved!")

		# Plot results of testing different agents at different epochs of learning
		plt.figure(1)
		plt.title("Mean escape time as a function of epochs", fontsize=30)
		plt.errorbar(np.arange(0, nb_episode_played+1)*step, 
					np.mean(t_out, axis=0)[:, 0], np.std(t_out, axis=0)[:, 0])
		plt.xlabel("Epoch", fontsize = 20)
		plt.ylabel("Escape time", fontsize = 20)
		plt.rc('xtick', labelsize=20) 
		plt.rc('ytick', labelsize=20)
		plt.grid()
		plt.show()

	# ********* PLOT DATA FROM LEARNING ITSELF ********************************
	# latencies

	n=3
	if (smoothing):
		learning_latencies[0, learning_latencies[0, :]>9000] = np.mean(learning_latencies, axis=1)
		# Moving average size 10
		cheat = np.cumsum(learning_latencies)
		cheat[n:] = cheat[n:] - cheat[:-n]
		mvg_avg = cheat[n - 1:]/n
		plt.figure(4)
		plt.title("Moving average latencies during learning", fontsize=30)
		plt.plot(np.arange(1, learning_latencies.shape[1]-1), mvg_avg)
		plt.xlabel("Epoch", fontsize = 20)
		plt.ylabel("Latency", fontsize = 20)
		plt.rc('xtick', labelsize=20) 
		plt.rc('ytick', labelsize=20)
		plt.grid()
		plt.show()
	
	
	plt.figure(2)
	plt.title("Mean latencies during learning", fontsize=30)
	plt.errorbar(np.arange(learning_latencies.shape[1]), 
				 np.mean(learning_latencies, axis=0), 
				 np.std(learning_latencies, axis=0))
	#plt.plot(np.arange(1, learning_latencies.shape[1]-1), mvg_avg)
	plt.xlabel("Epoch", fontsize = 20)
	plt.ylabel("Latency", fontsize = 20)
	plt.rc('xtick', labelsize=20) 
	plt.rc('ytick', labelsize=20)
	plt.grid()
	plt.show()
	
	# weight evolution
	w_mean_agents = np.mean(w, axis=0)
	w_mean_neurons = np.mean(w_mean_agents, axis=2)
	w_norm = np.linalg.norm(w_mean_agents, axis=2)
	plt.figure(3)
	plt.title("Mean norm of weights for each neuron during learning", fontsize=30)
	plt.plot(np.arange(learning_latencies.shape[1]), w_norm)
	plt.xlabel("Epoch", fontsize = 20)
	plt.ylabel("Norm of weights", fontsize = 20)
	plt.rc('xtick', labelsize=20) 
	plt.rc('ytick', labelsize=20)
	plt.grid()
	plt.show()
