import numpy as np
import matplotlib.pyplot as plt
import os

def plot_from_file(agent_number=1, trial_number=200):
	
	w = np.zeros((agent_number, trial_number, 3, 400))
	latencies = np.zeros((agent_number, trial_number, 1))
	
	for i in range(1, agent_number):
		w_id = "data/Trial_weights_%.1d" % i + ".npy"
		lat_id = "data/Trial_latencies_%.1d" % i + ".npy"

		print(w_id)
		print(lat_id)
		if os.path.isfile(w_id) and os.path.isfile(lat_id):
			w[i] = np.load(w_id)
			latencies[i] = np.load(lat_id)	
		else:
			print("Undefined file(s)")
			return

	plt.figure(1)
	plt.plot(np.arange(latencies.shape[1]), np.mean(latencies, axis=0))
	plt.show()
