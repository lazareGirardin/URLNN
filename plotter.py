"""
	These functions can be used to retrieve the weights and escape latencies during
	the learning of an agent. Plots or runs are used to assess performance of an agent at
	a given trial.


"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sarsa


def compare_latencies(agent_number=10, trial_number=300):
	"""
	Compare the escape latencies of different agents trained with different 
	parameters.

	"""

	# ***************** Load the Data ****************************************
	folder_1 = "Tau001"
	folder_2 = "Tau04"
	folder_3 = "Tau1"
	folder_4 = "oneWeights"

	#w1 = np.zeros((agent_number, trial_number, 3, 400))
	t1 = np.zeros((agent_number, trial_number, 1))
	#w2 = np.zeros((agent_number, trial_number, 3, 400))
	t2 = np.zeros((agent_number, trial_number, 1))
	#w3 = np.zeros((agent_number, trial_number, 3, 400))
	t3 = np.zeros((agent_number, trial_number, 1))
	t4 = np.zeros((agent_number, trial_number, 1))

	for i in range(agent_number):
		j=i+1
		#w_id_1 = "ALLdata/zeroWeights/weights_%.2d" % i + ".npy"
		lat_id_1 = "ALLdata/"+ folder_1 +"/latencies_%.2d" % i + ".npy"

		#w_id_2 = "ALLdata/weightsNoEllig_%.2d" % i + ".npy"
		lat_id_2 = "ALLdata/"+ folder_2 +"/latencies_%.2d" % i + ".npy"

		lat_id_3 = "ALLdata/"+ folder_3 +"/latencies_%.2d" % i + ".npy"

		lat_id_4 = "ALLdata/"+ folder_4 +"/latencies_%.2d" % i + ".npy"

		if os.path.isfile(lat_id_1) and os.path.isfile(lat_id_2) and os.path.isfile(lat_id_3) and os.path.isfile(lat_id_4):
			t1[i] = np.load(lat_id_1)
			t2[i] = np.load(lat_id_2)
			t3[i] = np.load(lat_id_3)
			t4[i] = np.load(lat_id_4)
		else:
			print("Undefined file(s)")
			return


	title = "Exploration Parameter"
	label_t1 = 'tau = 0.01'
	label_t2 = 'tau = 0.4'
	label_t3 = 'tau = 1'
	label_t4 = 'Decaying Tau'

	# *********** AVERAGE OVER NUMBER OF AGENTS *******************
	plt.figure(1)
	plt.title(title, fontsize=30)
	plt.plot(np.arange(trial_number), np.mean(t1, axis=0), label=label_t1, color='b')
	plt.plot(np.arange(trial_number), np.mean(t2, axis=0), label=label_t2, color='r')
	plt.plot(np.arange(trial_number), np.mean(t3, axis=0), label=label_t3, color='g')
	plt.plot(np.arange(trial_number), np.mean(t4, axis=0), label=label_t4, color='m')
	plt.legend(fontsize=20)
	plt.xlabel("Epoch", fontsize = 20)
	plt.ylabel("Latencies", fontsize = 20)
	plt.rc('xtick', labelsize=20) 
	plt.rc('ytick', labelsize=20)
	plt.grid()
	plt.show()

	# ************* MOVING AVERAGE *********************************
	n=5
	# Moving average size n
	cm = np.cumsum(np.mean(t1, axis=0))
	cm[n:] = cm[n:] - cm[:-n]
	mvg_avg1 = cm[n - 1:]/n

	cm = np.cumsum(np.mean(t2, axis=0))
	cm[n:] = cm[n:] - cm[:-n]
	mvg_avg2 = cm[n - 1:]/n

	cm = np.cumsum(np.mean(t3, axis=0))
	cm[n:] = cm[n:] - cm[:-n]
	mvg_avg3 = cm[n - 1:]/n

	cm = np.cumsum(np.mean(t4, axis=0))
	cm[n:] = cm[n:] - cm[:-n]
	mvg_avg4 = cm[n - 1:]/n

	plt.figure(2)
	plt.title(title +" (moving average)", fontsize=30)
	plt.plot(np.arange(1, trial_number-n+2), mvg_avg1, label=label_t1, color='b')
	plt.plot(np.arange(1, trial_number-n+2), mvg_avg2, label=label_t2, color='r')
	plt.plot(np.arange(1, trial_number-n+2), mvg_avg3, label=label_t3, color='g')
	plt.plot(np.arange(1, trial_number-n+2), mvg_avg4, label=label_t4, color='m')
	plt.legend(fontsize= 20)
	plt.xlabel("Epoch", fontsize = 20)
	plt.ylabel("Latencies", fontsize = 20)
	plt.rc('xtick', labelsize=20) 
	plt.rc('ytick', labelsize=20)
	plt.grid()
	plt.show()



def plot_from_file(agent_number=1, trial_number=100, test=False):

	"""
	Plot data from training. Can be used to run trials at a certain stage of training obtain
	the mean escape time over of certain amount of runs without training.

	"""
	
	# Plot the moving average over epochs
	smoothing = False

	
	# ***************** Load the Data ****************************************
	w = np.zeros((agent_number, trial_number, 3, 400))
	learning_latencies = np.zeros((agent_number, trial_number, 1))
	
	for i in range(agent_number):
		j=i+1
		w_id = "data/weightsNoEllig_%.2d" % i + ".npy"
		lat_id = "data/latenciesNoEllig_%.2d" % i + ".npy"

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
		nb_episode_played = 10
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
	#plt.errorbar(np.arange(100), 
	#			 np.mean(learning_latencies, axis=0)[0:100], 
	#			 np.std(learning_latencies, axis=0)[0:100])
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