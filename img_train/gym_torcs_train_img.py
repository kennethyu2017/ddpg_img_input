"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import time

import numpy as np
import tensorflow as tf

from gym_torcs.gym_torcs import TorcsEnv

##TODO use tf  FLAGS.define_string, define_ing....
DDPG_CFG = tf.app.flags.FLAGS  # alias
DDPG_CFG.action_fields = ['steer', 'accel', 'brake']

from train_agent import  train

# torcs
DDPG_CFG.torcs_relaunch_freq = 3  # relaunch TORCS every episodes because of the memory leak error
DDPG_CFG.policy_output_idx_steer = 0
DDPG_CFG.policy_output_idx_accel = 1
DDPG_CFG.policy_output_idx_brake = 2
# DDPG_CFG.hist_len =4 #stack 4-frames.

# Action = collections.namedtuple('Action', DDPG_CFG.action_fields)
## - [0]:steer [1]:accel [2]:brake --
DDPG_CFG.actor_output_bound_fns = [None for _ in range(len(DDPG_CFG.action_fields))]
DDPG_CFG.actor_output_bound_fns[DDPG_CFG.policy_output_idx_steer] = tf.nn.tanh  # [-1,1]
DDPG_CFG.actor_output_bound_fns[DDPG_CFG.policy_output_idx_accel] = tf.nn.sigmoid  # [0,1]
DDPG_CFG.actor_output_bound_fns[DDPG_CFG.policy_output_idx_brake] = tf.nn.sigmoid  # [0,1]

# logging and saver
# DDPG_CFG.reward_moving_average_discount = 0.8
# DDPG_CFG.steps_moving_average_discount = 0.8
DDPG_CFG.log_dir = './train/gym_torcs/tf_log'
DDPG_CFG.checkpoint_dir = './train/gym_torcs/chk_pnt'
DDPG_CFG.eval_monitor_dir = './train/gym_torcs/eval_monitor'

# global var for eval
prev_eval_time = 0.0
max_avg_episode_reward = 0.0

tf.logging.set_verbosity(tf.logging.INFO)


class torcs_env_wrapper(TorcsEnv):
    def __init__(self,*args, **kwargs):
        super(torcs_env_wrapper,self).__init__(*args,**kwargs)
        self.reset_count = 0

    def make_obs(self,img):
      """

      :param img: (4096,3)
      :return: (64,64,1) grey scale
      """
      # (64,64,3) array
      # downsample to (64,64,3) and change to greyscale (64,64,1)
      img = np.reshape(img, newshape=(DDPG_CFG.screen_height,DDPG_CFG.screen_width, -1))

      #TODO just test what we got
      # global t
      # t+=1
      # if t < 20:
      #   img = img / 255.
      #   plt.imshow(img)
      #   plt.show()

      # scr = obs[:-1:int(np.floor(h / DDPG_CFG.screen_height)), :-1:int(np.floor(w / DDPG_CFG.screen_width)), :]
      # TODO test greyscale.
      # return np.mean(img[:DDPG_CFG.screen_height, :DDPG_CFG.screen_width, :], axis=2, keepdims=True)  # gray scale

      # greyscale= np.mean(img, axis=2, keepdims=True)  # gray scale
      # # TODO just test what we got
      # if t < 20:
      #   show=np.squeeze(greyscale)
      #   plt.imshow(show)
      #   plt.show()
      return np.mean(img, axis=2, keepdims=True)  # gray scale

    def reset(self):
        # relaunch TORCS every 3 episode because of the memory leak error
        obs = self._reset((self.reset_count % DDPG_CFG.torcs_relaunch_freq) == 0)
        self.reset_count += 1
        return self.make_obs(obs.img)

    def step(self, action):
        obs, reward, term, _ = self._step(action)
        return self.make_obs(obs.img), reward,term,_

if __name__ == "__main__":
  tf.logging.info("@@@  start ddpg training gym_torcs @@@ start time:{}".format(time.ctime()))
  # Generate a Torcs environment
  env = torcs_env_wrapper(vision=True, throttle=True, gear_change=False)
  train(env,env)



