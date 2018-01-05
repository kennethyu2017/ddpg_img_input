"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import tensorflow as tf
from gym_torcs.gym_torcs import TorcsEnv
from common.replay_buffer import preprocess_low_dim
from common.UO_process import UO_Process
import numpy as np
from common.common import env_step
import time
from collections import namedtuple


##TODO use tf  FLAGS.define_string, define_ing....
DDPG_CFG = tf.app.flags.FLAGS  # alias
DDPG_CFG.action_fields = ['steer', 'accel', 'brake']

noise_fields = ['mu', 'theta', 'sigma', 'dt','x0']
Noise_params= namedtuple('NoiseParams', noise_fields)

# DDPG_CFG.torcs_noise_fields = ['steer', 'accel', 'brake']

from low_dim_train.train_agent_low_dim import  train

# torcs

#TODO maybe no need when in Text mode?? still got Nan.
DDPG_CFG.torcs_relaunch_freq = 3
DDPG_CFG.learn_start= 1

# e-greedy accel
DDPG_CFG.accel_eps_max = 1./7     #once every 7 steps(140ms) give 1 accel.
DDPG_CFG.accel_eps_min = 1./10  #once every 10 steps() give 1 accel. agressive
DDPG_CFG.accel_eps_decay_steps = 1e6  #linearly decay to min after 1M steps,and keep min till training end.
# DDPG_CFG.exploration_steps=40*10**5
# DDPG_CFG.exploration_steps= 25*10**4
DDPG_CFG.greedy_steer_steps = 1*(10**5)
DDPG_CFG.greedy_accel_steps = 1*(10**5)
DDPG_CFG.greedy_brake_steps = 1*(10**5)
# DDPG_CFG.greedy_steer_steps = 1
# DDPG_CFG.greedy_accel_steps = 1
# DDPG_CFG.greedy_brake_steps = 1
# DDPG_CFG.ou_steps=6*(10**5)  # will aneal till end
DDPG_CFG.ou_steps=1  # will aneal till end


#noise we give some speed always.  #mu, theta, sigma,dt,x0
DDPG_CFG.torcs_steer_noise_params = Noise_params(mu=0.0,theta=0.15,sigma=0.3,dt=1e-2,x0=0.0)
DDPG_CFG.torcs_accel_noise_params = Noise_params(mu=0.0,theta=0.15,sigma=0.3,dt=1e-2,x0=0.7)
DDPG_CFG.torcs_brake_noise_params = Noise_params(mu=0.0,theta=0.15,sigma=0.3,dt=1e-2,x0=-0.3)

# relaunch TORCS every episodes because of the memory leak error
DDPG_CFG.policy_output_idx_steer = 0
DDPG_CFG.policy_output_idx_accel = 1
DDPG_CFG.policy_output_idx_brake = 2
# DDPG_CFG.hist_len =4 #stack 4-frames.



# x is from BN.
def scale_sigmoid(x):
  return tf.nn.sigmoid(x * 3.3)  # when x==1, result is 0.964

# x is from BN.
def scale_tanh(x):
  return tf.nn.tanh(x * 2.0)  # when x==1, result is 0.964


# Action = collections.namedtuple('Action', DDPG_CFG.action_fields)
## - [0]:steer [1]:accel [2]:brake --
DDPG_CFG.actor_output_bound_fns = [None for _ in range(len(DDPG_CFG.action_fields))]
DDPG_CFG.actor_output_bound_fns[DDPG_CFG.policy_output_idx_steer] = scale_tanh  # unit from BN to tanh
DDPG_CFG.actor_output_bound_fns[DDPG_CFG.policy_output_idx_accel] = scale_sigmoid # unit from BN to sigmoid
DDPG_CFG.actor_output_bound_fns[DDPG_CFG.policy_output_idx_brake] = scale_sigmoid # unit from BN to sigmoid
# DDPG_CFG.actor_output_bound_fns[DDPG_CFG.policy_output_idx_accel] =tf.nn.sigmoid # [0,1]
# DDPG_CFG.actor_output_bound_fns[DDPG_CFG.policy_output_idx_brake] =tf.nn.sigmoid  # [0,1]


# logging and saver
# DDPG_CFG.reward_moving_average_discount = 0.8
# DDPG_CFG.steps_moving_average_discount = 0.8
DDPG_CFG.log_dir = '/home/yuheng/PycharmProjects/rl/kenneth_ddpg/ddpg_add_low_dim/train/gym_torcs_low_dim/tf_log/'
DDPG_CFG.checkpoint_dir = '/home/yuheng/PycharmProjects/rl/kenneth_ddpg/ddpg_add_low_dim/train/gym_torcs_low_dim/chk_pnt/'
DDPG_CFG.eval_monitor_dir = '/home/yuheng/PycharmProjects/rl/kenneth_ddpg/ddpg_add_low_dim/train/gym_torcs_low_dim/eval_monitor/'
DDPG_CFG.replay_buffer_file_path = '/home/yuheng/PycharmProjects/rl/kenneth_ddpg/ddpg_add_low_dim/train/gym_torcs_low_dim/replay_buffer/'

# global var for eval
prev_eval_time = 0.0
max_avg_episode_reward = 0.0

tf.logging.set_verbosity(tf.logging.INFO)

t=0

class torcs_env_wrapper(TorcsEnv):
    def __init__(self,*args, **kwargs):
        super(torcs_env_wrapper,self).__init__(*args,**kwargs)
        self.reset_count = 0

    def make_state(self,obs):
      """
      :return: (64,64,1) grey scale
      """
      # # TODO just test what we got
      global t
      t+=1
      if t < 4:
        tf.logging.info('@@@@ obs is {}'.format(obs))

      # according to yanpanlau, we choose the following state params from named tuple obs.
      #TODO normalized is necc.
      state = np.hstack( #datas are already normalized in gym_torcs.
        (obs.angle, obs.track, obs.trackPos, obs.speedX, obs.speedY, obs.speedZ,
         obs.wheelSpinVel, obs.rpm))
      if t < 4:
        tf.logging.info('@@@@  --- state  is {}'.format(state))
      return state

    def reset(self, relaunch=False):
        # relaunch TORCS every 3 episode because of the memory leak error
        obs = self._reset(relaunch or ((self.reset_count % DDPG_CFG.torcs_relaunch_freq) == 0) )
        self.reset_count += 1
        # obs is named tuple.
        return self.make_state(obs)

    def step(self, action):
        obs, reward, term, _ = self._step(action)
        return self.make_state(obs), reward,term,_
    #TODO temp use:
    def vision_on(self):
        self.vision=True
        if self.client is not None: # after reset ,will hava snakeoil client.
          self.client.vision=True

    def vision_off(self):
        self.vision=False
        if self.client is not None:
          self.client.vision=False
    @property
    def vision_status(self):
        return self.vision


def agent_action(step, sess, actor, online_state_inputs,is_training,state,replay_buffer, noise_process,env):
    policy_output = sess.run(fetches=[actor.online_action_outputs_tensor],
          feed_dict={online_state_inputs:state,
                     is_training:False})  # must reshape to (1,11)
    policy_output=policy_output[0]

    ##add noise and bound
    stochastic_action=policy_output_to_stochastic_action(policy_output, env.action_space,noise_process)

    ## excute a_t and store Transition.
    (state, reward, terminated) = env_step(env, stochastic_action)
    # episode_reward += reward

    # if step % 20 == 0:
    if step % 2000 == 0:
      tf.logging.info(' +++++++++++++++++++ global_step:{} action:{}'
                      '  reward:{} term:{}'.format(step,stochastic_action,reward,terminated))
    # replace transition with new one.
    transition = preprocess_low_dim(action=stochastic_action,
        reward=reward,
        terminated=terminated,
        state=state)

    ##even if terminated ,we still save next_state cause FF Q network
    # will use it, but will discard Q value in the end.
    replay_buffer.store(transition)
    return transition


#
# def greedy_accel(step, policy_out):
#   epsilon = max(DDPG_CFG.accel_eps_min,
#                  DDPG_CFG.accel_eps_max - (DDPG_CFG.accel_eps_max - DDPG_CFG.accel_eps_min ) * step / DDPG_CFG.accel_eps_decay_steps)
#   out = policy_out.copy()
#   if np.random.rand() < epsilon: #uniform distrib [0,1)
#     # random accel
#     out[0][DDPG_CFG.policy_output_idx_steer] = np.random.uniform(-0.1, 0.1)
#     out[0][DDPG_CFG.policy_output_idx_accel] += np.random.uniform(0.8, 1.0)
#     out[0][DDPG_CFG.policy_output_idx_brake] += np.random.uniform(-0.9, 0.1)
#
#   return out



def greedy_function(x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn()

epsilon_greedy_steer, epsilon_greedy_accel,epsilon_greedy_brake=1,1,1
sign=1.0
epsilon_ou =1
def policy_output_to_stochastic_action(output, action_space,noise_process):
  global epsilon_greedy_steer,epsilon_greedy_accel,epsilon_greedy_brake
  global epsilon_ou
  global sign
  ## mu_s_t shape(1, action_dim), squeeze to (action_dim,)
  output = np.squeeze(output, axis=0)

  #add ou noise first. lasting till end of training.
  # epsilon_ou -= 1.0 / DDPG_CFG.ou_steps   #aneal
  # output += max(epsilon_ou, 0) * noise_process.sample()

  # greedy periodically . epsilon 1->0 ->1 ->0....
  if epsilon_greedy_steer > 0.:
    epsilon_greedy_steer -= sign / DDPG_CFG.greedy_steer_steps
  if epsilon_greedy_accel > 0.:
    epsilon_greedy_accel -= sign / DDPG_CFG.greedy_accel_steps
  if epsilon_greedy_brake > 0.:
    epsilon_greedy_brake -= sign / DDPG_CFG.greedy_brake_steps

  # we dont wanna greedy noise to aneal to 0. minimum 0.6
  # if too greedy accel, maybe we cannot move far from start line and always crash at begin.
  # we wanna explore more track segments.
  if epsilon_greedy_steer < 0.7 or epsilon_greedy_steer>1.0 :
    sign *= -1.0

  #shape(action_dim,) follow yanpanlau method. do greedy accel at beginning.
  # noise=np.array( [max(epsilon, 0) * OU_function(output[0], 0.0, 0.60, 0.30),  # steer
  #                 max(epsilon, 0) * OU_function(output[1], 0.5, 1.00, 0.10),  # accel
  #                 max(epsilon, 0) * OU_function(output[2], -0.1, 1.00, 0.05)]) # brake
  greedy_noise=np.array( [max(epsilon_greedy_steer, 0.) * greedy_function(output[0], 0.0, 0.80, 0.10),  # steer
                  max(epsilon_greedy_accel, 0.) * greedy_function(output[1], 0.8, 1.00, 0.10),  # accel
                  max(epsilon_greedy_brake, 0.) * greedy_function(output[2], -0.2, 1.00, 0.05)]) # brake

  stochastic_action = greedy_noise + output

  # add exploration
  # epsilon = max(1 - step * 1.0 / exploration_steps, 0.0)
  # stochastic_action = (1-epsilon) * output + epsilon * noise_process.sample()
  # TODO test always add noise.
  # stochastic_action = output + noise_process.sample()
  # bound to torcs scope
  bounded = np.clip(stochastic_action, action_space.low, action_space.high)
  return bounded


if __name__ == "__main__":
  tf.logging.info("@@@  start ddpg training gym_torcs @@@ start time:{}".format(time.ctime()))
  # Generate a Torcs environment
  env_train = torcs_env_wrapper(vision=False, throttle=True, gear_change=False,port=3101)
  # env_eval = torcs_env_wrapper(vision=True, throttle=True, gear_change=False,port=8888)
  #TODO rewrite.
  #steer, accel, brake .after greedy noise.
  #valid noise value can make gradients happy.
  mu = np.array([0, 0, 0])
  # x0=np.array([0, 0.5, -0.1])
  theta = np.array([0.15, 0.15, 0.15])
  sigma = np.array([0.3, 0.3, 0.3])
  # x0 = np.array([0.1, 0.3, 0.1])
  #TODO start equal exploration on steer, brake, accel.
  # x0 = np.array([-0.2, 0.0, 0.2])
  x0 = np.array([-0.2, -0.2, 0.2])
  noise_process=UO_Process(mu=mu, x0=x0, theta=theta,sigma=sigma,dt=1e-2)
  train(env_train,env_train,agent_action,noise_process)




tf.nn.conv2d_transpose()