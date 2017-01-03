import sys

import pylab as plb
import numpy as np
import mountaincar

class SarsaAgent():
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car = None, parameter1 = 3.0, grid_size=20):
        
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.parameter1 = parameter1

        self.N = grid_size
        # Random action probability
        self.eps = 0.0
        # Learning rate
        self.eta = 0.0
        # Discount Factor
        self.gamma = 0.0
        # Eligibility trace
        self.elig = 0.0
        # Decay factor of elligibility trace
        self.lambda_ = 0.0

    def visualize_trial(self, n_steps = 200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
        
        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()
            
        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            print("\rt =", self.mountain_car.t)
            sys.stdout.flush()            
            # choose a random action
            self.mountain_car.apply_force(np.random.randint(3) - 1)
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # update the visualization
            mv.update_figure()
            plb.draw()            
            
            # check for rewards
            if self.mountain_car.R > 0.0:
                print ("\rreward obtained at t = ", self.mountain_car.t)
                break

    def learn(self):
        
        #init
        n = 100
        dt = 0.01
        e = np.zeros(3, self.N**2)

        
        # choose next action according to policy
        a_next, Q_old, rj = self._choose_action(w)
        # Simulate the action
        self.mountain_car.apply_force(a-1)
        # simulate the timestep
        self.mountain_car.simulate_timesteps(n, dt)

        # Compute Temporal difference error
        TD_err = self.R - (Q_old[a] - self.gamma * self._activity_Q(w)[a_next])
        # Elligibility update
        e = self.gamma*self.lambda_*e
        e[a_next] = e + rj
        # Weights update
        w = w + self.eta * TD_err * e
        # Update action
        a = a_next

        pass

    def _activity_r(self):
    """
        Computes the Gaussian activities r of input neurons.
        The function returns an array of activity rj line by line of size (N², 1)
    """
        # Gaussian width is the distance between centers
        sig_x = 180/self.N
        sig_phi = 30/self.N
        # The centers are placed along the intervals
        x_centers = np.linspace(-150.0, 30.0, self.N)
        phi_centers = np.linpsace(15.0, -15.0, self.N)
        # Create a meshgrid of the neurons centers
        grid_x, grid_phi = np.meshgrid(x_centers, phi_centers)
        # Compute the activity for each neurons
        rj = np.exp(-((grid_x - self.mountain_car.x)/sig_x)**2 
                   -((grid_phi-self.mountain_car.x_d)/grid_phi)**2)
        # Return array in from top left to bottom right line after line
        return np.reshape(rj, (self.N**2, 1))

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
        return np.sum(np.multiply(w,rj), axis=1), rj

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

        ##     CHANGE WEIGHTS TO SELF.W???

        # Compute exponential of each Q-activity over tau
        Q, rj = self._activity_Q(w)
        exp_action = np.exp(Q/tau)
        # Compute probability of taking each actions
        prob_action = exp_action/np.sum(exp_action)

        # Choose an action depending on the probability
        rand = numpy.random.rand()
        if rand < prob_action[0]:
            action = 0
        else if rand < prob_action[0]+prob_action[1]:
            action = 1
        # Should be useless -> else
        else if rand < prob_action[0]+prob_action[1]+prob_action[2]:
            action = 2

        return action, Q, rj


if __name__ == "__main__":
    d = SarsaAgent()
    d.visualize_trial()
    plb.show()
    import pdb; pdb.set_trace()
