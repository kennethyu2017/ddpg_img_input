
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
from low_dim_train.train_agent_low_dim import train
from common.common import env_step
from common.UO_process import UO_Process
from common.replay_buffer import preprocess_low_dim

import numpy as np

##TODO use tf  FLAGS.define_string, define_ing....
DDPG_CFG = tf.app.flags.FLAGS  # alias

#bipedal with action space shape(4,) and each dim with range [-1,+1]
DDPG_CFG.action_fields = ['hip-1', 'knee-1', 'hip-2','knee-2']
DDPG_CFG.actor_output_bound_fns = [tf.nn.tanh for _ in range(len(DDPG_CFG.action_fields))]
#TODO define bount fn
# DDPG_CFG.actor_output_bound_fns = [None]*len(DDPG_CFG.action_fields)


DDPG_CFG.log_dir = 'train/gym_bipedal_walker_v2/tf_log/'
DDPG_CFG.checkpoint_dir = 'train/gym_bipedal_walker_v2/chk_pnt/'
DDPG_CFG.eval_monitor_dir = 'train/gym_bipedal_walker_v2/eval_monitor/'
DDPG_CFG.replay_buffer_file_path = 'train/gym_bipedal_walker_2/replay_buffer/'

DDPG_CFG.learn_start= 10000

# global var for eval
prev_eval_time = 0.0
max_avg_episode_reward = 0.0

tf.logging.set_verbosity(tf.logging.INFO)

def agent_action(step, sess, actor, online_state_inputs,is_training,state,replay_buffer,noise_process, env):
  #make random play at beginning . to fill some states in replay buffer.
    if step < DDPG_CFG.learn_start:
      stochastic_action = [np.random.uniform(low,high) for (low,high) in zip(env.action_space.low, env.action_space.high)]
    else:
      policy_output = sess.run(fetches=[actor.online_action_outputs_tensor],
            feed_dict={online_state_inputs:state,
                       is_training:False})  # must reshape to (1,11)
      policy_output=policy_output[0] #list of tensor

      ##add noise and bound
      stochastic_action=policy_output_to_stochastic_action(policy_output, noise_process, env.action_space)

    ## excute a_t and store Transition.
    (state, reward, terminated) = env_step(env, stochastic_action)

    if step % 2000 == 0:
      tf.logging.info('@@@@@@@@@@ global_step:{} action:{}'
                      '  reward:{} term:{} @@@@@@@@@@'.format(step,stochastic_action,reward,terminated))

    # replace transition with new one.
    transition = preprocess_low_dim(action=stochastic_action,
        reward=reward,
        terminated=terminated,
        state=state)

    ##even if terminated ,we still save next_state cause FF Q network
    # will use it, but will discard Q value in the end.
    replay_buffer.store(transition)
    return transition


def policy_output_to_stochastic_action(output, noise_process, action_space):
  global epsilon
  ## mu_s_t shape(1, action_dim), squeeze to (action_dim,)
  output = np.squeeze(output, axis=0)
  stochastic_action = output + noise_process.sample()
  # bound to torcs scope
  bounded = np.clip(stochastic_action, action_space.low, action_space.high)
  return bounded


if __name__ == "__main__":
    tf.logging.info("@@@  start ddpg training gym_bipedal_walker_v2 @@@ start time:{}".format(time.ctime()))
  # Generate a Torcs environment
    train_env = gym.make(id='BipedalWalker-v2')

    eval_monitor = Monitor(gym.make(id='BipedalWalker-v2'), directory=DDPG_CFG.eval_monitor_dir,
                           video_callable=lambda x: False, resume=True)

    mu = np.array([0.0, 0.0, 0.0,0.0])
    # x0=np.array([0, 0.5, -0.1])
    theta = np.array([0.15, 0.15, 0.15, 0.15])
    sigma = np.array([0.3, 0.3, 0.3, 0.3])
    # x0 = np.array([0.1, 0.3, 0.1])
    # TODO greedy accel in the begining
    x0 = np.array([-0.2, 0.2, 0.2, 0.2,])
    noise_process = UO_Process(mu=mu, x0=x0, theta=theta, sigma=sigma, dt=1e-2)

    train(train_env, eval_monitor,agent_action, noise_process)





