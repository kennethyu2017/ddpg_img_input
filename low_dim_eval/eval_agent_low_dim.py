"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import os
import time
import glob

import numpy as np
import tensorflow as tf
from low_dim_train.actor_low_dim import Actor
from low_dim_train.critic_low_dim import Critic
from tensorflow.contrib.layers import variance_scaling_initializer,batch_norm,l2_regularizer
# from tensorflow.contrib.framework.python.ops.variables import model_variable
from tensorflow.python.ops.variables import model_variables

from common.replay_buffer import preprocess_low_dim
from common.common import env_step

##TODO use tf  FLAGS.define_string, define_ing....
DDPG_CFG = tf.app.flags.FLAGS  # alias
tf.logging.set_verbosity(tf.logging.INFO)

## hyper-P
#DDPG_CFG.learning_rate = 2.5e-4  ##TODO anneal learning_rate

DDPG_CFG.actor_learning_rate = 1e-3  ##TODO anneal learning_rate
DDPG_CFG.critic_learning_rate = 1e-4  ##TODO anneal learning_rate

DDPG_CFG.critic_reg_ratio = 1e-2
DDPG_CFG.actor_reg_ratio = 4e-2
DDPG_CFG.tau = 0.001
DDPG_CFG.gamma = 0.99
DDPG_CFG.num_training_steps = 25*(10**5)  # 2.5M steps total
DDPG_CFG.summary_freq= 1*3*(10*3)  # ~1min
# DDPG_CFG.eval_freq = 30*3*(10**3)  # eval during training per steps .~30min.
DDPG_CFG.eval_freq = 2*3*(10**3)  # eval during training per steps .~30min.
# DDPG_CFG.num_eval_steps = 1000  # eval steps. 5min
DDPG_CFG.num_eval_steps = 50  # eval steps. 5min
# DDPG_CFG.save_model_freq = 10**4  # ~30min
DDPG_CFG.learn_start = 1000   # fill replay_buffer some states then start learn.
DDPG_CFG.batch_size = 64
#TODO decrease buffer size.too big to fit into 8G mem. or experience replay.
# DDPG_CFG.replay_buff_size = 10**6  # 1M
#TODO test buff memory asumption, else it will occupy all the memory and die.
DDPG_CFG.replay_buff_size = 10**6  # 1M


# x is from BN.
def scale_sigmoid(x):
  return tf.nn.sigmoid(x * 3.3)  # when x==1, result is 0.964

# x is from BN.
def scale_tanh(x):
  return tf.nn.tanh(x * 2.0)  # when x==1, result is 0.964


# Action = collections.namedtuple('Action', DDPG_CFG.action_fields)
## - [0]:steer [1]:accel [2]:brake --

DDPG_CFG.action_fields = ['steer', 'accel', 'brake']
DDPG_CFG.policy_output_idx_steer = 0
DDPG_CFG.policy_output_idx_accel = 1
DDPG_CFG.policy_output_idx_brake = 2
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



#TODO add normalization layer,reg definition.
# === actor net arch. shared by online and target ==
is_training=tf.placeholder(tf.bool, shape=(), name='is_training')

DDPG_CFG.online_policy_net_var_scope = 'online_policy'
DDPG_CFG.target_policy_net_var_scope = 'target_policy'
DDPG_CFG.actor_summary_keys = 'actor_summaries'


# -- 1 input norm layers --
DDPG_CFG.actor_input_normalizer = batch_norm
DDPG_CFG.actor_input_norm_params =  {'is_training':is_training,
                                       'data_format':'NHWC',  #represent norm over last dim C.
                                       'updates_collections':None,
                                        'scale':False,  # gamma. let next fc layer to scale.
                                        'center':True,   # beta. act as bias. no reg on this.
                                        # 'param_regularizers':{'beta':l2_regularizer(scale=DDPG_CFG.actor_reg_ratio,
                                        #                                             scope=DDPG_CFG.online_policy_net_var_scope)} #TODO try reg on beta.
                                        }  # inplace update running average
# -- 2 fc layers --
DDPG_CFG.actor_n_fc_units = [400, 300]
# DDPG_CFG.actor_n_fc_units = [300, 200]
DDPG_CFG.actor_fc_activations = [tf.nn.elu] * 2
DDPG_CFG.actor_fc_initializers = [variance_scaling_initializer()] * 2  # [He] init
# DDPG_CFG.actor_fc_regularizers = [l2_regularizer(scale=DDPG_CFG.actor_reg_ratio, scope=DDPG_CFG.online_policy_net_var_scope)] * 2
DDPG_CFG.actor_fc_regularizers = [None] * 2
DDPG_CFG.actor_fc_normalizers = [batch_norm] * 2
DDPG_CFG.actor_fc_norm_params =  [{'is_training':is_training,
                                       'data_format':'NHWC',
                                       'updates_collections':None,  # inplace update running average
                                       'scale':False,
                                       'center':True
                                       # 'param_regularizers': {'beta': l2_regularizer(scale=DDPG_CFG.actor_reg_ratio,
                                       #                                               scope=DDPG_CFG.online_policy_net_var_scope)}
                                         # TODO try reg on beta.
                                         }] *2  #no reg on this.

# -- 1 output layer --
#TODO try actor no BN.use l2 reg on weights only.
DDPG_CFG.actor_output_layer_normalizers = batch_norm
DDPG_CFG.actor_output_layer_norm_params = {'is_training':is_training,
                                       'data_format':'NHWC',
                                       'updates_collections':None,# inplace update running average
                                       'scale':False,   # TODO try adding scale to learn to squash to satisfy tanh.\
                                           #  but add l2 reg on it.
                                       'center':False} # no beta. normaly no bias in output layer.
                                       # 'param_regularizers':{'gamma':l2_regularizer(scale=DDPG_CFG.actor_reg_ratio)} }
DDPG_CFG.actor_output_layer_initializer=tf.random_uniform_initializer(-3e-3,3e-3)
#add l2 reg to avoid tanh saturated.
# DDPG_CFG.actor_output_layer_regularizer = l2_regularizer(scale=DDPG_CFG.actor_reg_ratio,
#                                                          scope=DDPG_CFG.online_policy_net_var_scope)
#TODO heavy reg on output layer to avoid tanh satuation
# DDPG_CFG.actor_output_layer_regularizer = l2_regularizer(scale=)

# DDPG_CFG.actor_output_layer_regularizer = None

# === critic net arch. shared by online and target ==
# -- 3 conv layers --
DDPG_CFG.online_q_net_var_scope = 'online_q'
DDPG_CFG.target_q_net_var_scope = 'target_q'
DDPG_CFG.critic_summary_keys = 'critic_summaries'

#-- 1 input norm layers --
DDPG_CFG.critic_input_normalizer = batch_norm
DDPG_CFG.critic_input_norm_params =  {'is_training':is_training,
                                       'data_format':'NHWC',  #represent last dim is C.
                                       'updates_collections':None,
                                       'scale': False,
                                       'center': True  # no reg
                                      }  # inplace update running average

# -- 3 fc layer --
DDPG_CFG.include_action_fc_layer = 2  # in this layer we include action inputs. conting from fc-1 as 1.
DDPG_CFG.critic_n_fc_units = [400, 300, 300]
# DDPG_CFG.critic_n_fc_units = [300, 200]
DDPG_CFG.critic_fc_activations = [tf.nn.elu] * 3
DDPG_CFG.critic_fc_initializers = [variance_scaling_initializer()] * 3  # [He] init
DDPG_CFG.critic_fc_regularizers = [l2_regularizer(scale=DDPG_CFG.critic_reg_ratio,scope=DDPG_CFG.online_q_net_var_scope)] * 3
###TODO try w/o BN
# DDPG_CFG.critic_fc_normalizers = [batch_norm, None, None]  # 2nd fc including action input and no BN but has bias.
DDPG_CFG.critic_fc_normalizers = [batch_norm, batch_norm,batch_norm]  # 2nd fc including action input and no BN but has bias.
DDPG_CFG.critic_fc_norm_params =  [{'is_training':is_training,
                                       'data_format':'NHWC',
                                       'updates_collections':None,
                                       'scale': False,
                                       'center': True  # with beta.
                                    }] * 3 # inplace update running average
# DDPG_CFG.critic_fc_normalizers = [None] * 2  # 2nd fc including action input and no BN
# DDPG_CFG.critic_fc_norm_params =  [{}] * 2

# -- 1 output layer --
DDPG_CFG.critic_output_layer_initializer = tf.random_uniform_initializer(-3e-3, 3e-3)
# DDPG_CFG.critic_output_layer_regularizer = l2_regularizer(scale=DDPG_CFG.critic_reg_ratio,
#                                                           scope=DDPG_CFG.online_q_net_var_scope)



# global var for eval
prev_eval_time = 0.0

max_avg_episode_reward = -5e10

tf.logging.set_verbosity(tf.logging.INFO)

def policy_output_to_deterministic_action(output,action_space):
  ## mu_s_t shape(1, action_dim), squeeze to (action_dim,)
  output = np.squeeze(output, axis=0)
  # bound to torcs scope
  bounded = np.clip(output, action_space.low, action_space.high)
  return bounded


def evaluate(env):
  '''
    :return:
  '''
  action_space = env.action_space
  obs_space = env.observation_space

  ######### instantiate actor,critic, replay buffer, uo-process#########
  ## feed online with state. feed target with next_state.
  online_state_inputs = tf.placeholder(tf.float32,
                                       shape=(None, obs_space.shape[0]),
                                       name="online_state_inputs")

  # tf.logging.info('@@@@ online_state_inputs shape:{}'.format(online_state_inputs.shape))
  target_state_inputs = tf.placeholder(tf.float32,
                                       shape=online_state_inputs.shape,
                                       name="target_state_inputs")

  ##instantiate actor
  actor = Actor(action_dim=action_space.shape[0],
                online_state_inputs=online_state_inputs,
                target_state_inputs=target_state_inputs,
                input_normalizer=DDPG_CFG.actor_input_normalizer,
                input_norm_params=DDPG_CFG.actor_input_norm_params,
                n_fc_units=DDPG_CFG.actor_n_fc_units,
                fc_activations=DDPG_CFG.actor_fc_activations,
                fc_initializers=DDPG_CFG.actor_fc_initializers,
                fc_normalizers=DDPG_CFG.actor_fc_normalizers,
                fc_norm_params=DDPG_CFG.actor_fc_norm_params,
                fc_regularizers=DDPG_CFG.actor_fc_regularizers,
                output_layer_initializer=DDPG_CFG.actor_output_layer_initializer,
                # output_layer_regularizer=DDPG_CFG.actor_output_layer_regularizer,
                output_layer_regularizer=None,
                # output_normalizers=None,
                # output_norm_params=None,
                output_normalizers=DDPG_CFG.actor_output_layer_normalizers,
                output_norm_params=DDPG_CFG.actor_output_layer_norm_params,
                output_bound_fns=DDPG_CFG.actor_output_bound_fns,
                learning_rate=DDPG_CFG.actor_learning_rate,
                is_training=is_training)


  # ===  finish building ddpg graph before this =================#

  ##create tf default session
  sess = tf.Session(graph=tf.get_default_graph())
  # sess.run(fetches=[tf.global_variables_initializer()])

  # Load a previous checkpoint
  saver = tf.train.Saver()
  saved_files= glob.glob(DDPG_CFG.checkpoint_dir+'/*.meta')
  # latest_checkpoint = tf.train.latest_checkpoint(DDPG_CFG.checkpoint_dir)
  if saved_files:
    for f in saved_files:
      if '90000' not in f:
        continue
      print("@@@@@ === start eval , loading model checkpoint: {}".format(f))
      saver.restore(sess, f.strip('.meta'))
      ####### start eval #########
      evaluate_helper(env=env,
               num_eval_steps=DDPG_CFG.num_eval_steps,
               preprocess_fn=preprocess_low_dim,
               estimate_fn=lambda state: sess.run(fetches=actor.online_action_outputs_tensor,
                                                  feed_dict={online_state_inputs:state,
                                                  is_training:False} ))

  sess.close()
  env.close()

def evaluate_helper(env, num_eval_steps, preprocess_fn, estimate_fn):
  total_reward = 0
  episode_reward = 0
  episode_steps = 0
  max_episode_reward = 0
  n_episodes = 0
  n_rewards = 0
  terminated = False
  global prev_eval_time
  global max_avg_episode_reward

  transition = preprocess_fn(state=env.reset())
  estep=0

  while not terminated:
    estep+=1
    if estep > num_eval_steps:
      break
    policy_out = estimate_fn(transition.next_state[np.newaxis,:])  # must reshape to (1,11)
    ##TODO just give some initial speed

    action = policy_output_to_deterministic_action(policy_out,env.action_space)
    # #TODO just test
    # action[0] = 0.
    # action[1] = 1.
    # action[2] *= 1./3

    (state, reward, terminated) = env_step(env, action)
    if estep % 2 == 0:
      tf.logging.info('@@@@@ eval step:{} action:{}'
                      '  reward:{} term:{}  @@@@@@@@@@'.format(estep, action, reward,terminated))

    # we only need state to generate policy.
    transition = preprocess_fn(state)

    # record every reward
    total_reward += reward
    episode_reward += reward
    episode_steps+=1

    if reward != 0:
      n_rewards += 1  # can represent effective(not still) steps in episode

    if terminated:
      tf.logging.info('@@@@@@ eval episode termed - episode_reward:{} -\
                        episode_steps:{}  n_episode:{}- @@@@@@ '.format(episode_reward,episode_steps,n_episodes))
      episode_steps = 0
      if episode_reward > max_episode_reward:
        max_episode_reward = episode_reward

      episode_reward = 0
      n_episodes += 1
      # relaunch
      # only save state.
      transition = preprocess_fn(env.reset())
      if estep < num_eval_steps:
        terminated = False  # continue

  # -- end for estep ---
  avg_episode_reward = total_reward / max(1, n_episodes)
  avg_episode_steps = n_rewards / max(1, n_episodes)
  now = time.time()
  if prev_eval_time == 0:  # first time eval.
    prev_eval_time = now

  # write_summary(summary_writer, global_step, avg_episode_reward, max_episode_reward, avg_episode_steps, now - prev_eval_time)
  prev_eval_time = now

  tf.logging.info('@@@@@@ ==== end of evaluation, result: -steps:{} - avg_episode_reward:{} -\
                   max_episode_reward:{} - avg_episode_steps:{} - @@@@@@ '.format(estep,
                                                                          avg_episode_reward,
                                                                          max_episode_reward,
                                                                          avg_episode_steps,
                                                                          ))

# def action_step(env, action):
#   (state, reward, terminated, _) =env.step(action)
#   return (state, reward, terminated)


# def write_summary(writer, global_step, avg_episode_reward, max_episode_reward,avg_episode_steps, consuming_seconds):
#   eval_summary = tf.Summary()  # proto buffer
#   eval_summary.value.add(node_name='avg_episode_reward' ,simple_value=avg_episode_reward, tag="train_eval/avg_episode_reward")
#   eval_summary.value.add(node_name='max_episode_reward', simple_value=max_episode_reward, tag="train_eval/max_episode_reward")
#   eval_summary.value.add(node_name='avg_episode_steps', simple_value=avg_episode_steps, tag="train_eval/avg_episode_steps")
#   # change to minutes
#   eval_summary.value.add(node_name='tow_eval_interval_minutes',simple_value=(consuming_seconds/60), tag='train/eval/two_eval_interval_minutes')
#
#   # use epoches as 'global_step' tag
#   writer.add_summary(summary=eval_summary, global_step=global_step)
#   writer.flush()

#
# def save_model(saver, sess,global_step):
#   # save model. will save both online and target networks.
#   return saver.save(sess=sess, save_path=DDPG_CFG.checkpoint_dir, global_step=global_step)
#
#
