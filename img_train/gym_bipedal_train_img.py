
"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-bipedalwalker-v2.
with tensorflow.

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import tensorflow as tf
import time
import gym
from gym.wrappers import Monitor

import numpy as np

##TODO use tf  FLAGS.define_string, define_ing....
DDPG_CFG = tf.app.flags.FLAGS  # alias

#bipedal with action space shape(4,) and each dim with range [-1,+1]
DDPG_CFG.action_fields = ['hip-1', 'knee-1', 'hip-2','knee-2']
DDPG_CFG.actor_output_bound_fns = [tf.nn.tanh for _ in range(len(DDPG_CFG.action_fields))]
#TODO define bount fn
# DDPG_CFG.actor_output_bound_fns = [None]*len(DDPG_CFG.action_fields)


from train_agent import  train

# logging and saver
# DDPG_CFG.reward_moving_average_discount = 0.8
# DDPG_CFG.steps_moving_average_discount = 0.8
DDPG_CFG.log_dir = 'train/gym_bipedal_walker_v2/tf_log/'
DDPG_CFG.checkpoint_dir = 'train/gym_bipedal_walker_v2/chk_pnt/'
DDPG_CFG.eval_monitor_dir = 'train/gym_bipedal_walker_v2/eval_monitor/'

# global var for eval
prev_eval_time = 0.0
max_avg_episode_reward = 0.0

tf.logging.set_verbosity(tf.logging.INFO)


class bipeldal_env_wrapper(object):
    def __init__(self,gym_env):
        self.gym_env = gym_env

    def make_obs(self):
        """

        :return: (64,64,1) greyscale
        """
        #(400,600,3) array
        obs=self.gym_env.render('rgb_array')
        #downsample to (64,64,3) and change to greyscale (64,64,1)
        h,w,c = obs.shape
        scr=obs[:-1:int(np.floor(h/DDPG_CFG.screen_height)), :-1:int(np.floor(w/DDPG_CFG.screen_width)),:]
        #TODO test greyscale.
        return np.mean(scr[:DDPG_CFG.screen_height,:DDPG_CFG.screen_width,:],axis=2,keepdims=True) # gray scale


    #redefine the interface used in train() to render img.
    def reset(self):
        self.gym_env.reset()
        return self.make_obs()

    def step(self,action):
        state,reward, term, _ = self.gym_env.step(action)
        return self.make_obs(),reward,term,_

    #fall back
    def __getattr__(self, item):
        return self.gym_env.__getattribute__(item)


if __name__ == "__main__":
    tf.logging.info("@@@  start ddpg training gym_bipedal_walker_v2 @@@ start time:{}".format(time.ctime()))
  # Generate a Torcs environment
    train_env = gym.make(id='BipedalWalker-v2')
    train_wrapper = bipeldal_env_wrapper(train_env)

    monitor_env = gym.make(id='BipedalWalker-v2')
    monitor_wrapper = bipeldal_env_wrapper(monitor_env)

    eval_monitor = Monitor(monitor_wrapper, directory=DDPG_CFG.eval_monitor_dir,
                           video_callable=lambda x: True, resume=True)

    train(train_wrapper,eval_monitor)





