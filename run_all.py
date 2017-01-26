import sys

import pylab as plb
import matplotlib.pyplot as plt
import numpy as np
import mountaincar
import sarsa

def create_all_weights():

	trial_number = 300
	agents = 10

	# ******************** WEIGHTS ***********
	# --- Tau decay --- weights zero ----
	print(" Testing weights: ZEROS")
	for i in range(agents):
		d = sarsa.SarsaAgent(decay=True, tau=1., w_init='zero')
		print("AGENT ", i+1)
		w, latencies = d.learn(epochs=trial_number)

		w_id = 'ALLdata/zeroWeights/weights_%.2d' % i +'.npy'
		lat_id = 'ALLdata/zeroWeights/latencies_%.2d' %i + '.npy'
		np.save(w_id, w)
		np.save(lat_id, latencies)

	# --- Tau decay --- weights ones ----
	print(" Testing weights: ONES")
	for i in range(agents):
		d = sarsa.SarsaAgent(decay=True, tau=1., w_init='one')
		print("AGENT ", i+1)
		w, latencies = d.learn(epochs=trial_number)

		w_id = 'ALLdata/oneWeights/weights_%.2d' % i +'.npy'
		lat_id = 'ALLdata/oneWeights/latencies_%.2d' %i + '.npy'
		np.save(w_id, w)
		np.save(lat_id, latencies)

	print(" Testing weights: RANDOM")
	# --- Tau decay --- weights random ----
	for i in range(agents):
		d = sarsa.SarsaAgent(decay=True, tau=1., w_init='rand')
		print("AGENT ", i+1)
		w, latencies = d.learn(epochs=trial_number)

		w_id = 'ALLdata/randWeights/weights_%.2d' % i +'.npy'
		lat_id = 'ALLdata/randWeights/latencies_%.2d' %i + '.npy'
		np.save(w_id, w)
		np.save(lat_id, latencies)

	# ******************** TAU *******************
	# decay tau -> any other from above

	# --- Tau 0.01 ---
	print(" Testing tau: 0.01")
	for i in range(agents):
		d = sarsa.SarsaAgent(decay=False, tau=0.01, w_init='variance')
		print("AGENT ", i+1)
		w, latencies = d.learn(epochs=trial_number)

		w_id = 'ALLdata/Tau001/weights_%.2d' % i +'.npy'
		lat_id = 'ALLdata/Tau001/latencies_%.2d' %i + '.npy'
		np.save(w_id, w)
		np.save(lat_id, latencies)

	# --- Tau 0.4 ---
	print(" Testing tau: 0.4")
	for i in range(agents):
		d = sarsa.SarsaAgent(decay=False, tau=0.4, w_init='variance')
		print("AGENT ", i+1)
		w, latencies = d.learn(epochs=trial_number)

		w_id = 'ALLdata/Tau04/weights_%.2d' % i +'.npy'
		lat_id = 'ALLdata/Tau04/latencies_%.2d' %i + '.npy'
		np.save(w_id, w)
		np.save(lat_id, latencies)

	# --- Tau 1 ---
	print(" Testing tau: 1")
	for i in range(agents):
		d = sarsa.SarsaAgent(decay=False, tau=1., w_init='variance')
		print("AGENT ", i+1)
		w, latencies = d.learn(epochs=trial_number)

		w_id = 'ALLdata/Tau1/weights_%.2d' % i +'.npy'
		lat_id = 'ALLdata/Tau1/latencies_%.2d' %i + '.npy'
		np.save(w_id, w)
		np.save(lat_id, latencies)

	# ********* ELIGIBILITY ***********************
	
	# --- lambda 0.95
	print(" Testing lambda: 0.95")
	for i in range(agents):
		d = sarsa.SarsaAgent(decay=True, tau=1., w_init='variance', lambda_=0.95)
		print("AGENT ", i+1)
		w, latencies = d.learn(epochs=trial_number)

		w_id = 'ALLdata/l095/weights_%.2d' % i +'.npy'
		lat_id = 'ALLdata/l095/latencies_%.2d' %i + '.npy'
		np.save(w_id, w)
		np.save(lat_id, latencies)
	
	# --- lambda 0
	print(" Testing lambda: 0")
	for i in range(agents):
		d = sarsa.SarsaAgent(decay=True, tau=1., w_init='variance', lambda_=0.)
		print("AGENT ", i+1)
		w, latencies = d.learn(epochs=trial_number)

		w_id = 'ALLdata/l0/weights_%.2d' % i +'.npy'
		lat_id = 'ALLdata/l0/latencies_%.2d' %i + '.npy'
		np.save(w_id, w)
		np.save(lat_id, latencies)

	# ******* ETA ***************
	
	# eta=0.05 -> any other data from above

	# --- eta 0.5
	print("Testing eta: 0.5")
	for i in range(3):
		d = sarsa.SarsaAgent(decay=True, tau=1., w_init='variance', eta=0.5)
		print("AGENT ", i+1)
		w, latencies = d.learn(epochs=100)

		w_id = 'ALLdata/eta05/weights_%.2d' % i +'.npy'
		lat_id = 'ALLdata/eta05/latencies_%.2d' %i + '.npy'
		np.save(w_id, w)
		np.save(lat_id, latencies)

	# --- eta 0.9
	print(" Testing eta: 0.9")
	for i in range(3):
		d = sarsa.SarsaAgent(decay=True, tau=1., w_init='variance', eta=0.9)
		print("AGENT ", i+1)
		w, latencies = d.learn(epochs=100)

		w_id = 'ALLdata/eta09/weights_%.2d' % i +'.npy'
		lat_id = 'ALLdata/eta09/latencies_%.2d' %i + '.npy'
		np.save(w_id, w)
		np.save(lat_id, latencies)

