"""
All function belonging to the sarsa agents.
List of the functions:

	-  visulalize_trial():	Visualize the run of an agant
	-  learn():				Train an agent to perform the task
	-  execute_trial():		Execute a run given weights w without learning.
							Can be used to assess performance at a given epoch of learning
	-  action_field():		Plot a vector field of the actions for each 
							position in the position-speed space.
	- _activity_r():		Compute the neurons response to a state s [position, speed]
	- _activity_Q():		Compute the Q-value of a state s
	- _choose_action():		Choose an action based on a softmax policy

Lazare Girardin - URLNN Mini-project 2
"""
import sys

import pylab as plb
import matplotlib.pyplot as plt
import numpy as np
import mountaincar

class SarsaAgent():
	"""A not so good agent for the mountain-car task.
	"""

	def __init__(self, mountain_car = None, grid_size=20, eta=0.05, 
					gamma=0.95, lambda_=0.4, tau=1., decay=True, w_init='variance'):
		
		if mountain_car is None:
			self.mountain_car = mountaincar.MountainCar()
		else:
			self.mountain_car = mountain_car

		self.N = grid_size

		# Learning rate
		self.eta = eta
		# Discount Factor
		self.gamma = gamma
		# Decay factor of elligibility trace
		self.lambda_ = lambda_
		# Exploration parameter
		self.tau = tau

		self.decay = decay
		self.w_init = w_init

		# Gaussian width is the distance between centers
		self.sig_x = 180./(self.N-1)
		self.sig_phi = 30./(self.N-1)
		# The centers are placed along the intervals
		x_centers = np.linspace(-150.0, 30.0, self.N)
		phi_centers = np.linspace(15.0, -15.0, self.N)
		# Create a meshgrid of the neurons centers
		self.grid_x, self.grid_phi = np.meshgrid(x_centers, phi_centers)

	def visualize_trial(self, w, n_steps = 200):
		"""Do a trial without learning, with display.

		Parameters
		----------
		n_steps -- number of steps to simulate for
		"""
		
		# prepare for the visualization
		plb.ion()
		plb.pause(0.0001)
		mv = mountaincar.MountainCarViewer(self.mountain_car)
		mv.create_figure(n_steps, n_steps)
		plb.show()
			
		# make sure the mountain-car is reset
		self.mountain_car.reset()

		#tau = 0.9
		end_tau = 0.001
		alpha = -n_steps/np.log(end_tau/self.tau)

		tau = self.tau
		for n in range(n_steps):
			#print("\rt =", self.mountain_car.t,
			#				sys.stdout.flush())  

			tau = self.tau*np.exp(-n/alpha)
			s = [self.mountain_car.x, self.mountain_car.x_d]
			# choose first action according to policy
			a, Q, rj = self._choose_action(w, s, end_tau)
			#Q, dummy  = self._activity_Q(w)
			#a = np.argmax(Q)
			# Simulate the action
			self.mountain_car.apply_force(a-1)
			# simulate the timestep
			self.mountain_car.simulate_timesteps(100, 0.01)

			# update the visualization
			mv.update_figure()
			plb.show()
			plb.pause(0.0001)            
			
			# check for rewards
			if self.mountain_car.R > 0.0:
				print ("\rreward obtained at t = ", self.mountain_car.t)
				break
		print("no reward :(")

	def _activity_r(self, state):
		"""
			Computes the Gaussian activities r of input neurons.
			The function returns an array of activity rj line by line of size N²
		"""
		# Compute the activity for each neurons
		rj = np.exp(-(self.grid_x   - state[0])**2/((self.sig_x)**2 )
					-(self.grid_phi - state[1])**2/((self.sig_phi)**2))
		# Return array in from top left to bottom right line after line
		return np.reshape(rj, (self.N**2))

	def _activity_Q(self, w, state):
		"""
			Compute the activity of the output neurons of a state s = (x, x_d)
			The functions returns an array of the activity for 
			each neurons (with respective weights wa)
			Input: 
					-w   : array of shape [# ouput neurons a, # input neurons j]
					-state:
						-x   : position of the car
						-x_d : speed of the car
			Ouput: 
					-Q : Q-activity of shape [# output neurons]
					-rj: Neurons response depending on the state
		"""
		rj = self._activity_r(state)
		return np.dot(w,rj), rj

	def _choose_action(self, w, state, tau):
		"""
			Choose an action depending for a state s=(x, x_d)
			Inputs:
					-tau  : Exploration temperature parameter
					-w    : Weights between input and ouput neurons [#ouput, #input]
					-state:
						-x   : position of the car
						-x_d : speed of the car
			output:
					-action : (-1) for backward, (0) for engine stop and (+1) for forward
					-Q      : Q activity at current timestep
					-rj: Neurons response depending on the state
		"""
		# Compute exponential of each Q-activity over tau
		Q, rj = self._activity_Q(w, state)
		# Avoid overflows for small tau (exp(1000 will overflow))
		#exp_action = np.exp(np.minimum(Q/tau, 500))
		exp_action = np.exp(Q/tau)
		# Compute probability of taking each actions
		prob_action = exp_action/np.sum(exp_action)

		# Choose an action depending on the probability
		action = np.random.choice(3, p=prob_action)

		return action, Q, rj

	def learn(self, epochs = 100, verbose=False):
		"""
			Learn the optimal weights for an agent to complete the task,
			following the sarsa-algorithm.
			Input parameters:
				- epochs: 	Number of trials to train on
				- verbose:	Wether or not to visulize a trial every 20 epochs
			Ouput:
				- w:			The weights of each ouput neurons (3 actions), 
								at each epoch of training.
								Size: (#epochs, #ouput neurons, #neurons)
				- latencies: 	The escape latency for each trial
		"""
		
		n = 100
		dt = 0.01
		maxTimesteps = 5000

		# Init of weights
		if (self.w_init=='zero'):
			w = np.zeros((3, self.N**2))
		elif (self.w_init=='one'):
			w = np.ones((3, self.N**2))
		elif (self.w_init=='rand'):
			w = np.random.rand(3, self.N**2)
		else:
			w = 0.01*np.random.rand(3, self.N**2)+0.5

		# Limit the decay of tau
		end_tau = 0.01
		# alpha can be used for an exponential decay parameter
		#alpha = -maxTimesteps/np.log(end_tau/self.tau)

		# Save latencies for plotting and weights for posterior evaluation
		trial_weights = np.zeros((epochs, 3, self.N**2))
		trial_latencies = np.zeros((epochs, 1))

		# ********************* EPOCHS **********************
		for trial in range(epochs):
			#init
			self.mountain_car.reset()
			e = np.zeros((3, self.N**2))			
			tau = self.tau

			# ************ LEARNING - 1 TRIAL ***************

			s = [self.mountain_car.x, self.mountain_car.x_d]
			# choose first action according to policy (when w=0, random)
			a, Q_s, rj_s = self._choose_action(w, s, tau)

			for i in range(maxTimesteps):
				# Decaying exploration parameter
				if (self.decay):
					#tau = self.tau*np.exp(-i/alpha)
					tau = np.maximum(self.tau*np.exp(-i/10.), end_tau)

				# Observe s'
				self.mountain_car.apply_force(a-1)
				self.mountain_car.simulate_timesteps(n, dt)
				s_next = [self.mountain_car.x, self.mountain_car.x_d]

				# choose next action according to policy
				a_next, Q_next, rj_next = self._choose_action(w, s_next, tau)

				# Compute Q[s, a]
				Q_s, rj_s = self._activity_Q(w, s)

				# Compute Temporal difference error
				TD_err = self.mountain_car.R - (Q_s[a] - self.gamma*Q_next[a_next])
				# Eligibility update
				e = self.gamma*self.lambda_*e
				e[a] = e[a] + rj_s
				# Weights update
				w = w + self.eta*TD_err*e
				# Update state and action
				a = a_next
				s = s_next

				if self.mountain_car.R >= 1:
					break

			print("\t TRIAL {t},  timesteps {ts}".format(t=trial+1, ts=self.mountain_car.t))

			# Visualize a run or a action vector-field
			if verbose and trial%20==0:
				#self.visualize_trial(w, 250)
				self.action_field(w)
				plb.close()

			# Save data
			trial_weights[trial] = w
			trial_latencies[trial] = self.mountain_car.t

		return trial_weights, trial_latencies

	def execute_trial(self, w, runs=10, greedy=False):
		"""
			Run a trial for an agent with neural network w.
			Input: 
				-w:    	  weights of neurons
				-runs: 	  number of runs to be tested
				-greedy:  only used to check performance with an argmax function
			Ouput:
				-mean of the escape time over the number of runs
				-standard deviation of the escape time over the number of runs
		"""
		tau = 0.01
		maxSteps = 5000
		latencies = np.zeros((runs, 1))
		for run in range(runs):
			n=0
			# make sure the mountain-car is reset
			self.mountain_car.reset()
			while self.mountain_car.R < 1 and n < maxSteps:
				n = n+1
				# choose first action according to policy
				s = [self.mountain_car.x, self.mountain_car.x_d]
				if (greedy):
					a = np.argmax(self._activity_Q(w, s))
				else:
					a, Q, rj = self._choose_action(w, s, tau)
				# Simulate the action
				self.mountain_car.apply_force(a-1)
				# simulate the timestep
				self.mountain_car.simulate_timesteps(100, 0.01)
			latencies[run] = self.mountain_car.t
			#print(latencies[run]) 

		print("Mean escape time: ", np.mean(latencies))
		return np.mean(latencies), np.std(latencies)

	def action_field(self, w):
		"""
		Print the Quiver plot (vector field) of the actions in the state graph.

		"""
		actions = np.zeros((self.N, self.N))
		Q_values = np.zeros((self.N, self.N))

		x_centers = np.linspace(-150.0, 30.0, self.N)
		phi_centers = np.linspace(15.0, -15.0, self.N)

		for i in range(self.N):
			for j in range(self.N):
				Q, r = self._activity_Q(w, [x_centers[i], phi_centers[j]])
				actions[j, i] = np.argmax(Q)-1
				Q_values[j, i] = np.max(Q)
		
		V = np.zeros((actions.shape[0], actions.shape[1]))
		U = actions
		fig, ax = plt.subplots()
		im = ax.imshow(Q_values, extent=[-150, 50, -15, 15])
		ax.quiver(self.grid_x, self.grid_phi, U, V, angles='xy', pivot='middle')
		fig.colorbar(im)
		ax.set(aspect='auto', title='Action Field')
		plt.show()

		return

if __name__ == "__main__":

	"""
		This can be used to train and try different parameters.
		Note that the file run_all.py execute similarely trainings for
		all parameters and save the results (weights, latencies) as
		npy files (numpy matrix file). The file plotter.py can be used to
		retrieve and plot or run the data that were saved.
	"""

	verbose = False
	trying_etas = False
	decay = True
	tau = 1.
	elig_decay = 0.7

	trial_number = 300
	Agents = 10

	if(trying_etas):
		etas = np.logspace(-7, 0, Agents)

	mean_lat_tests = np.zeros((10, 1))
	std_lat_tests = np.zeros((10, 1))

	for i in range(Agents):
		if(trying_etas):
			d = SarsaAgent(eta=etas[i], decay=decay, tau=tau, lambda_=elig_decay)
		else:
			d = SarsaAgent(decay=decay, tau=tau, lambda_=elig_decay)
		print("AGENT ", i+1)
		w, latencies = d.learn(trial_number, verbose)

		w_id = 'data/weights_%.2d' % i +'.npy'
		lat_id = 'data/latencies_%.2d' %i + '.npy'

		np.save(w_id, w)
		np.save(lat_id, latencies)
		#mean_lat_tests[i], std_lat_tests[i] = d.execute_trial(w[-1], 10)

	
	d.action_field(w[-1])

	if(trying_etas):
		plt.figure(1)
		plt.title("Influence of Learning Rate", fontsize=30)
		plt.errorbar(etas, mean_lat_tests, std_lat_tests)
		plt.xlabel("Eta", fontsize = 20)
		plt.ylabel("Test latencies", fontsize = 20)
		plt.xscale("log")
		plt.xlim((0.9e-5, 1.1e-1))
		plt.rc('xtick', labelsize=20) 
		plt.rc('ytick', labelsize=20)
		plt.grid()
		plt.show()
	#import pdb; pdb.set_trace()
	#d.visualize_trial(w[-1], 400)
	#plb.show()