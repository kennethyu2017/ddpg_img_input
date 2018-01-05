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
DDPG_CFG.num_eval_steps = 100  # eval steps. 5min
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

def build_ddpg_graph(actor, critic, reward_inputs, terminated_inputs, global_step_tensor):
  # if terminated(True), we don't count Q(s,a) , only count rewards.
  ###y_i = r_i + (1-terminated) * gamma*Q'(s_i+1 , mu'(s_i+1))
  # r_i,gamma, terminated  should be broadcasted.
  y_i = reward_inputs + (1.0 - terminated_inputs) * DDPG_CFG.gamma * critic.target_q_outputs_tensor  # shape (batch_size,)
  #list of reg scalar Tensors
  q_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope=DDPG_CFG.online_q_net_var_scope)

  q_loss = tf.add_n([tf.losses.mean_squared_error(labels=y_i, predictions=critic.online_q_outputs_tensor)] + q_reg_loss,
                    name='q_loss')

  tf.summary.scalar(name='q_value_mean',tensor=tf.reduce_mean(critic.online_q_outputs_tensor),
                    collections=[DDPG_CFG.critic_summary_keys])
  tf.summary.scalar(name='q_loss', tensor=q_loss, collections=[DDPG_CFG.critic_summary_keys])


  ##build policy loss.
  # then d(loss)/d(theta_mu) = d(loss)/d(Q) * d(Q)/d(a) * d(a)/ d(theta_mu) = -1 *policy_gradients.
  # so do SGD to minimize policy_loss is same as do SGA to maximum J(theta_mu)
  # add l2 reg.

  # TODO try to pass through gradient when tanh is saturated.
  #list of reg scalar Tensors.

  # -- optimize actor online network --
  # !!!! when run actor optimize op, we feed mu(s_i) to online_q_net as action_inputs to q.!!!
  actor_g_and_v, actor_compute_grads_op = actor.compute_online_policy_net_gradients(policy_loss=policy_loss)

  #add summary for gradients and var values.
  for (g, v) in actor_g_and_v:
    if g is not None:
      tf.summary.scalar(name='gradients_norm_' + v.name.strip(":0"), tensor=tf.norm(g),collections=[DDPG_CFG.actor_summary_keys])
      tf.summary.scalar(name='weights_norm_' + v.name.strip(':0'), tensor=tf.norm(v.value()), collections=[DDPG_CFG.actor_summary_keys])

  actor_apply_grads_op = actor.apply_online_policy_net_gradients(grads_and_vars=actor_g_and_v)
  train_online_policy_op = actor_apply_grads_op

  # -- optimize critic online network --
  ##!!!!!!! Very important: we must compute policy gradients before update q, cause the policy gradient!!!!
  ##!!!!!!! depends on the q params before this time-step update !!!
  ## on the other hand, update policy net params, will not affect the calculation of q gradients.
  # !!!! when run critic optimize op, we feed a_i(from replay buffer) to online_q_net as action_inputs to q.!!!
  critic_g_and_v, critic_compute_grads_op = critic.compute_online_q_net_gradients(q_loss=q_loss)

  # add summary for gradients and var values.
  for (g, v) in critic_g_and_v:
    if g is not None:
      tf.summary.scalar(name='gradients_norm_'+v.name.strip(':0'), tensor=tf.norm(g),collections=[DDPG_CFG.critic_summary_keys])
      tf.summary.scalar(name='weights_norm_'+v.name.strip(':0'), tensor=tf.norm(v.value()),collections=[DDPG_CFG.critic_summary_keys])

  # with tf.control_dependencies([actor_compute_grads_op]):
  critic_apply_grads_op = critic.apply_online_q_net_gradients(grads_and_vars=critic_g_and_v)
  train_online_q_op = critic_apply_grads_op

  ##!!!!important:soft update actor/critic targets after all online updates finish.!!!
  ##TODO  control compute/assign seperately of soft update.smaller control grain.
  # with tf.control_dependencies([actor_apply_grads_op, critic_apply_grads_op]):
  actor_update_target_op = actor.soft_update_online_to_target()
  critic_update_target_op = critic.soft_update_online_to_target()

  ## run increment global step <- soft update ops
  with tf.control_dependencies([actor_update_target_op, critic_update_target_op]):
    update_target_op = tf.assign_add(global_step_tensor, 1).op  # increment global step

  # after params initial, copy online -> target
  # with tf.control_dependencies([global_var_initializer]):
  actor_init_target_op = actor.copy_online_to_target()
  critic_init_target_op = critic.copy_online_to_target()

  ## call init target can triger the global init op, so we can just run init target op only to complete
  # all the init.
  copy_online_to_target_op = tf.group(actor_init_target_op, critic_init_target_op)

  # model saver
  # params is for checkpoints in memory. then saver.save() will flush all checkpoints from
  # memory to disk files.
  saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=5)

  return (copy_online_to_target_op, train_online_policy_op, train_online_q_op, update_target_op, saver)


def policy_output_to_stochastic_action(output, noise_process, action_space):
  ## mu_s_t shape(1, action_dim), squeeze to (action_dim,)
  output = np.squeeze(output, axis=0)
  stochastic_action = output + noise_process.sample()
  # bound to torcs scope
  bounded = np.clip(stochastic_action, action_space.low, action_space.high)
  return bounded

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

  ## inputs to q_net for training q.
  online_action_inputs_training_q = tf.placeholder(tf.float32,
                                                   shape=(None, action_space.shape[0]),
                                                   name='online_action_batch_inputs'
                                                   )
  # condition bool scalar to switch action inputs to online q.
  # feed True: training q.
  # feed False: training policy.
  cond_training_q = tf.placeholder(tf.bool, shape=[], name='cond_training_q')

  # batch_size vector.
  terminated_inputs = tf.placeholder(tf.float32, shape=(None), name='terminated_inputs')
  reward_inputs = tf.placeholder(tf.float32, shape=(None), name='rewards_inputs')

  ##instantiate actor, critic.
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


  critic = Critic(online_state_inputs=online_state_inputs,
                  target_state_inputs=target_state_inputs,
                  input_normalizer=DDPG_CFG.critic_input_normalizer,
                  input_norm_params=DDPG_CFG.critic_input_norm_params,
                  online_action_inputs_training_q=online_action_inputs_training_q,
                  online_action_inputs_training_policy=actor.online_action_outputs_tensor,
                  cond_training_q=cond_training_q,
                  target_action_inputs=actor.target_action_outputs_tensor,
                  n_fc_units=DDPG_CFG.critic_n_fc_units,
                  fc_activations=DDPG_CFG.critic_fc_activations,
                  fc_initializers=DDPG_CFG.critic_fc_initializers,
                  fc_normalizers=DDPG_CFG.critic_fc_normalizers,
                  fc_norm_params=DDPG_CFG.critic_fc_norm_params,
                  fc_regularizers=DDPG_CFG.critic_fc_regularizers,
                  output_layer_initializer=DDPG_CFG.critic_output_layer_initializer,
                  # output_layer_regularizer=DDPG_CFG.critic_output_layer_regularizer,
                  output_layer_regularizer = None,
                  learning_rate=DDPG_CFG.critic_learning_rate)

  ## track updates.
  global_step_tensor = tf.train.create_global_step()

  ## build whole graph
  copy_online_to_target_op, train_online_policy_op, train_online_q_op, update_target_op, saver \
    = build_ddpg_graph(actor, critic, reward_inputs, terminated_inputs, global_step_tensor)

  # ===  finish building ddpg graph before this =================#

  ##create tf default session
  sess = tf.Session(graph=tf.get_default_graph())
  '''
  # note: will transfer graph to graphdef now. so we must finish all the computation graph
  # before creating summary writer.
  '''
  # summary_writer = tf.summary.FileWriter(logdir=os.path.join(DDPG_CFG.log_dir, "train"),
  #                                        graph=sess.graph)
  # actor_summary_op = tf.summary.merge_all(key=DDPG_CFG.actor_summary_keys)
  # critic_summary_op = tf.summary.merge_all(key=DDPG_CFG.critic_summary_keys)
  ######### initialize computation graph  ############

  sess.run(fetches=[tf.global_variables_initializer()])

  # Load a previous checkpoint if it exists

  latest_checkpoint = tf.train.latest_checkpoint(DDPG_CFG.checkpoint_dir)
  if latest_checkpoint:
    print("@@@@@ === Loading model checkpoint: {}".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)



  # fc_2_bn_moving_mean=tf.get_variable('fully_connected_2/BatchNorm/moving_mean:0',shape=(3,))
  #
  # model_vars= model_variables(scope='online_policy')
  # moving_vas = sess.run(fetches=model_vars,
  #                              feed_dict={})

  #TODO just for debug
  # fetches=[actor.online_action_outputs_tensor]
  # fetches.extend( actor.action_unpacked_tensors)
  # fetches.extend(actor.action_bounded_tensors)

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
    # action, a_unpacked, a_bounded = out[0],out[1],out[2]
    # tf.logging.info('@@@@@ unpacked:{}  bounded:{} @@@@@@@@@@'.format(a_unpacked,a_bounded))

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

  tf.logging.info('@@@@@@ evaluation result: -steps:{} - avg_episode_reward:{} -\
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
