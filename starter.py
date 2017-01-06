import sys

import pylab as plb
import numpy as np
import mountaincar

class SarsaAgent():
	"""A not so good agent for the mountain-car task.
	"""

	def __init__(self, mountain_car = None, grid_size=20):
		
		if mountain_car is None:
			self.mountain_car = mountaincar.MountainCar()
		else:
			self.mountain_car = mountain_car

		self.N = grid_size

		# Learning rate
		self.eta = 0.1
		# Discount Factor
		self.gamma = 0.95
		# Decay factor of elligibility trace
		self.lambda_ = 0.9

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
		end_tau = 0.1
		tau0 = 10
		alpha = -n_steps/np.log(end_tau/tau0)

		tau = tau0
		for n in range(n_steps):
			print("\rt =", self.mountain_car.t,
							sys.stdout.flush())  

			tau = tau0*np.exp(-n/alpha)
			# choose first action according to policy
			a, Q, rj = self._choose_action(w, tau)
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

	def _activity_r(self):
		"""
			Computes the Gaussian activities r of input neurons.
			The function returns an array of activity rj line by line of size N²
		"""
		# Compute the activity for each neurons
		rj = np.exp(-((self.grid_x - self.mountain_car.x)/self.sig_x)**2 
				    -((self.grid_phi-self.mountain_car.x_d)/self.grid_phi)**2)
		# Return array in from top left to bottom right line after line
		return np.reshape(rj, (self.N**2))

	def _activity_Q(self, w):
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
		"""
		rj = self._activity_r()
		return np.dot(w,rj), rj

	def _choose_action(self, w, tau):
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
		"""
		# Compute exponential of each Q-activity over tau
		Q, rj = self._activity_Q(w)
		exp_action = np.exp(Q/tau)
		# Compute probability of taking each actions
		prob_action = exp_action/np.sum(exp_action)

		# Choose an action depending on the probability
		
		rand = np.random.rand()
		if rand < prob_action[0]:
			action = 0
		elif rand <= prob_action[0]+prob_action[1]:
			action = 1
		else:
			action = 2

		return action, Q, rj

	def learn(self):
		
		n = 100
		dt = 0.01
		#tau = 0.9
		trail_number = 10
		maxTimesteps = 1500

		w = np.zeros((3, self.N**2))
		e = np.zeros((3, self.N**2))

		# Gaussian width is the distance between centers
		self.sig_x = 180/self.N
		self.sig_phi = 30/self.N
		# The centers are placed along the intervals
		x_centers = np.linspace(-150.0, 30.0, self.N)
		phi_centers = np.linspace(15.0, -15.0, self.N)
		# Create a meshgrid of the neurons centers
		self.grid_x, self.grid_phi = np.meshgrid(x_centers, phi_centers)

		end_tau = 0.01
		tau0 = 5
		alpha = -maxTimesteps/np.log(end_tau/tau0)

		end_eta = 0.005
		eta0 = 0.15
		beta = -maxTimesteps/np.log(end_eta/eta0)

		for trail in range(trail_number):
			#init
			self.mountain_car.reset()
			
			tau = tau0
			eta = eta0

			# choose first action according to policy
			a, Q_s, rj_s = self._choose_action(w, tau)
			# Simulate the action
			self.mountain_car.apply_force(a-1)
			# simulate the timestep
			self.mountain_car.simulate_timesteps(n, dt)

			for i in range(maxTimesteps):
				# Decaying exploration parameter
				tau = tau0*np.exp(-i/alpha)
				eta = eta0*np.exp(-i/beta)

				# choose next action according to policy
				a_next, Q_s, rj_s = self._choose_action(w, tau)
				# Simulate the action
				self.mountain_car.apply_force(a_next-1)
				self.mountain_car.simulate_timesteps(n, dt)

				# Compute Temporal difference error
				Q_next, dummy = self._activity_Q(w)

				TD_err = self.mountain_car.R - (Q_s[a] - self.gamma*Q_next[a_next])
				# Elligibility update
				e = self.gamma*self.lambda_*e
				#import pdb; pdb.set_trace()
				e[a] = e[a] + rj_s
				# Weights update
				w = w + eta*TD_err*e
				# Update action
				a = a_next

				if self.mountain_car.R >= 1:
					break

			print("TRIAL {t},  timesteps {ts}".format(t=trail+1, ts=self.mountain_car.t))	

		return w


if __name__ == "__main__":

	d = SarsaAgent()
	w = d.learn()
	import pdb; pdb.set_trace()
	d.visualize_trial(w, 400)
	plb.show()
