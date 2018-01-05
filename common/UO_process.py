"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import numpy as np


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class UO_Process(object):
  ## TODO merge into tf graph.
  ##TODO anneal noise.
  ## mu with action_dims
  # def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
  def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
    self.theta = theta
    self.mu = mu
    self.sigma = sigma
    self.dt = dt
    self.x0 = x0
    self.x_prev = None  # must reset to give value
    self.reset()

  def sample(self):
    x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
        self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
    self.x_prev = x
    return x

  def reset(self):
    self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

  def __repr__(self):
    return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

     # noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
    # noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
    # noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)


if __name__ == "__main__":
    #test uo
    mu=np.array([0.1, 0.1, 0.1])
    # x0=np.array([0, 0.5, -0.1])
    theta=np.array([0.15, 0.15, 0.15])
    sigma=np.array([0.3, 0.3, 0.3])
    x0=np.array([-0.2, 0.0, 0.2])


    uo = UO_Process(mu,dt=1e-2,x0=x0,theta=theta,sigma=sigma)
    uo.reset()
    for i in range(2*10**5):
      uo.sample()
    for i in range(1000):
      print(uo.sample())
