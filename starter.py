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
        # This is your job!
        pass

    def _activity_r(self, x, x_d):
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
        rj = np.exp(-((grid_x - x)/sig_x)**2 
                   -((grid_phi-x_d)/grid_phi)**2)
        # Return array in from top left to bottom right line after line
        return np.reshape(rj, (self.N**2, 1))

    def _activity_Q(self, w, x, x_d):
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
        rj = self._activity_r(x, x_d)
        return np.sum(np.multiply(w,rj), axis=1)

    def _choose_action(self, w, x, x_d, tau):
    """
        Choose an action depending for a state s=(x, x_d)
        Inputs:
                -tau  : Exploration temperature parameter
                -w    : Weights between input and ouput neurons [#ouput, #input]
                -state:
                    -x   : position of the car
                    -x_d : speed of the car
    """

        ##     CHANGE WEIGHTS TO SELF.W???

        # Compute exponential of each Q-activity over tau
        exp_action = np.exp(self._activity_Q(w, x, x_d)/tau)
        # Compute probability of taking each actions
        prob_action = exp_action/np.sum(exp_action)

        ## COMPUTE CHOICE

    def _get_reward(self, x):
        return np.sign(max(0, x))

if __name__ == "__main__":
    d = SarsaAgent()
    d.visualize_trial()
    plb.show()
    import pdb; pdb.set_trace()
