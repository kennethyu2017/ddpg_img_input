"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import os
import time
import numpy as np
import glob as glob
import math as math
import tensorflow as tf
from low_dim_train.actor_low_dim import Actor
from low_dim_train.critic_low_dim import Critic
from tensorflow.contrib.layers import variance_scaling_initializer,batch_norm,l2_regularizer
from tensorflow.python.framework.dtypes import string
from common.replay_buffer import ReplayBuffer, preprocess_low_dim, Transition
from common.common import env_step,policy_output_to_deterministic_action


##TODO use tf  FLAGS.define_string, define_ing....
DDPG_CFG = tf.app.flags.FLAGS  # alias
DDPG_CFG.random_seed = 187
np.random.seed(DDPG_CFG.random_seed)

DDPG_CFG.train_from_replay_buffer_set_only = False
DDPG_CFG.load_replay_buffer_set = False

## hyper-P
#DDPG_CFG.learning_rate = 2.5e-4  ##TODO anneal learning_rate

# DDPG_CFG.actor_learning_rate = 1e-3  ##TODO anneal learning_rate
DDPG_CFG.actor_learning_rate = 1e-3  ##TODO anneal learning_rate
DDPG_CFG.critic_learning_rate = 1e-4  ##TODO anneal learning_rate

DDPG_CFG.critic_reg_ratio = 1e-2
# DDPG_CFG.critic_reg_ratio = 0.

#!!!!!!+++++
# DDPG_CFG.actor_reg_ratio = 5e-2 # to avoid tanh saturation.
# DDPG_CFG.actor_reg_ratio = 1e-2 # to avoid tanh saturation.

DDPG_CFG.tau = 0.001
DDPG_CFG.gamma = 0.99
# DDPG_CFG.num_training_steps = 2*(10**5)  # 1M steps total
DDPG_CFG.num_training_steps = 20*(10**5)  #2M steps total
# DDPG_CFG.summary_freq= 10*3000  # ~10min
DDPG_CFG.summary_freq= 1*10000  # 10~min
# DDPG_CFG.summary_freq= 1000  # ~5min
# DDPG_CFG.summary_freq= 5*3000  # ~5min
DDPG_CFG.summary_transition_freq=1*10000
DDPG_CFG.eval_freq = 3*10000  # eval during training per steps .~30min.
# DDPG_CFG.eval_freq = 3*10000  # eval and save model during training per steps .~30min.
# DDPG_CFG.num_eval_steps = 1000  # eval steps. 20sec
DDPG_CFG.num_eval_steps = 2000  # eval steps.
# DDPG_CFG.learn_start = 50000   # fill replay_buffer some accel states then start learn.
# DDPG_CFG.learn_start = 3000   # fill replay_buffer some states then start learn.
DDPG_CFG.batch_size = 64
#TODO decrease buffer size.too big to fit into 8G mem. or experience replay.
#TODO test buff memory asumption, else it will occupy all the memory and die.
# DDPG_CFG.replay_buff_size = 1*(10**6)  # 1M. bigger, moving average of BN more accurate.
DDPG_CFG.replay_buff_size = 5*(10**5)  # 500k
DDPG_CFG.replay_buff_save_segment_size = 30*3000  # every 30min, 180,000 Transition data.
DDPG_CFG.l_r_decay_freq= 5*(10**5) # 500k
# DDPG_CFG.l_r_decay_freq= 5

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
DDPG_CFG.critic_output_layer_regularizer = l2_regularizer(scale=DDPG_CFG.critic_reg_ratio,
                                                          scope=DDPG_CFG.online_q_net_var_scope)

# global var for eval
prev_eval_time = 0.0
DDPG_CFG.log_summary_keys = 'log_summaries'


# max_avg_episode_reward = -5e10

tf.logging.set_verbosity(tf.logging.INFO)

def build_ddpg_graph(actor, critic, reward_inputs, terminated_inputs, global_step_tensor):
  # if terminated(True), we don't count Q(s,a) , only count rewards.
  ###y_i = r_i + (1-terminated) * gamma*Q'(s_i+1 , mu'(s_i+1))
  # r_i,gamma, terminated  should be broadcasted.
  y_i = reward_inputs + (1.0 - terminated_inputs) * DDPG_CFG.gamma * critic.target_q_outputs_tensor  # shape (batch_size,)
  #list of reg scalar Tensors
  q_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope=DDPG_CFG.online_q_net_var_scope)

  q_value_loss=tf.losses.mean_squared_error(labels=y_i, predictions=critic.online_q_outputs_tensor)

  q_loss = tf.add_n([q_value_loss] + q_reg_loss, name='q_loss')

  tf.summary.scalar(name='q_value_mean',tensor=tf.reduce_mean(critic.online_q_outputs_tensor),
                    collections=[DDPG_CFG.critic_summary_keys])
  with tf.name_scope('loss'):
    tf.summary.scalar(name='q_value_loss', tensor=q_value_loss, collections=[DDPG_CFG.critic_summary_keys])


  ##build policy loss.
  # then d(loss)/d(theta_mu) = d(loss)/d(Q) * d(Q)/d(a) * d(a)/ d(theta_mu) = -1 *policy_gradients.
  # so do SGD to minimize policy_loss is same as do SGA to maximum J(theta_mu)
  # add l2 reg.

  # TODO try to pass through gradient when tanh is saturated.
  #list of reg scalar Tensors.
  policy_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=DDPG_CFG.online_policy_net_var_scope)

  #!!! important -1 * Q as policy loss.
  policy_value_loss = -1.0 * tf.reduce_mean(critic.online_q_outputs_tensor)
  policy_loss =tf.add_n([policy_value_loss ] + policy_reg_loss,
                        name='policy_loss')
  with tf.name_scope('loss'):
    tf.summary.scalar(name='policy_value_loss', tensor=policy_value_loss, collections=[DDPG_CFG.actor_summary_keys])

  # -- optimize actor online network --
  # !!!! when run actor optimize op, we feed mu(s_i) to online_q_net as action_inputs to q.!!!
  actor_g_and_v, actor_compute_grads_op = actor.compute_online_policy_net_gradients(policy_loss=policy_loss)

  ##TODO just see dQ/da
  # dq_da_grads = tf.gradients(ys=tf.reduce_mean(critic.online_q_outputs_tensor),
  #              xs=actor.online_action_outputs_tensor)
  # tf.summary.scalar(name='dQ_wrt_online_action_outputs_norm', tensor=tf.norm(dq_da_grads[0]),collections=[DDPG_CFG.actor_summary_keys])
  #after tanh/sigmoid/sigmoid
  dq_da_grads = tf.gradients(ys=tf.reduce_mean(critic.online_q_outputs_tensor),
                             xs=actor.action_bounded_tensors)
  with tf.name_scope('dq_da_grads'):
    for i in range(len(dq_da_grads)):
      tf.summary.scalar(name='dq_da_grads_'+str(i)+'norm',tensor=tf.norm(dq_da_grads[i]), collections=[DDPG_CFG.actor_summary_keys])

  with tf.name_scope('d_policy_loss_da_grads'):
    d_policy_loss_da_grads=tf.gradients(ys=policy_loss,xs=actor.action_bounded_tensors)
    for i in range(len(dq_da_grads)):
      tf.summary.scalar(name='d_policy_loss_da_grads_'+str(i)+'norm',tensor=tf.norm(d_policy_loss_da_grads[i]),
                        collections=[DDPG_CFG.actor_summary_keys])

  #before tanh/sigmoid
  with tf.name_scope('d_policy_loss_d_final_outputs'):
    d_policy_loss_d_final_outputs = tf.gradients(ys=policy_loss,xs=actor.action_unpacked_tensors)
    for i in range(len(d_policy_loss_d_final_outputs)):
      tf.summary.scalar(name='d_policy_loss_d_final_layer_outputs_'+str(i)+'norm',
                        tensor=tf.norm(d_policy_loss_d_final_outputs[i]),
                        collections=[DDPG_CFG.actor_summary_keys])

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

  return (copy_online_to_target_op, train_online_policy_op,
          train_online_q_op, update_target_op, saver,q_loss)


def train(train_env, monitor_env, agent_action_fn,noise_process):
  '''
    :return:
  '''
  action_space = train_env.action_space
  obs_space = train_env.observation_space

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

  #for l_r decay
  actor_l_r = tf.placeholder(tf.float32, shape=[], name='actor_l_r')
  critic_l_r = tf.placeholder(tf.float32, shape=[], name='critic_l_r')

  #for summary text
  summary_text_tensor=tf.convert_to_tensor(str('summary_text'),preferred_dtype=string)
  tf.summary.text(name='summary_text',tensor=summary_text_tensor, collections=[DDPG_CFG.log_summary_keys])

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
                output_normalizers=DDPG_CFG.actor_output_layer_normalizers,
                output_norm_params=DDPG_CFG.actor_output_layer_norm_params,
                # output_normalizers=None,
                # output_norm_params=None,
                output_bound_fns=DDPG_CFG.actor_output_bound_fns,
                learning_rate=actor_l_r,
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
                  output_layer_regularizer=DDPG_CFG.critic_output_layer_regularizer,
                  # output_layer_regularizer = None,
                  learning_rate=critic_l_r)

  ## track updates.
  global_step_tensor = tf.train.create_global_step()

  ## build whole graph
  copy_online_to_target_op, train_online_policy_op, train_online_q_op, update_target_op, saver,q_loss_tensor \
    = build_ddpg_graph(actor, critic, reward_inputs, terminated_inputs, global_step_tensor)

  #we save the replay buffer data to files.
  replay_buffer = ReplayBuffer(buffer_size=DDPG_CFG.replay_buff_size,
                               save_segment_size= DDPG_CFG.replay_buff_save_segment_size,
                               save_path=DDPG_CFG.replay_buffer_file_path,
                               seed=DDPG_CFG.random_seed
                               )
  ##TODO test load replay buffer from files.
  if DDPG_CFG.load_replay_buffer_set:
    replay_buffer.load(DDPG_CFG.replay_buffer_file_path)

  # ===  finish building ddpg graph before this =================#

  ##create tf default session
  sess = tf.Session(graph=tf.get_default_graph())
  '''
  # note: will transfer graph to graphdef now. so we must finish all the computation graph
  # before creating summary writer.
  '''
  summary_writer = tf.summary.FileWriter(logdir=os.path.join(DDPG_CFG.log_dir, "train"),
                                         graph=sess.graph)
  actor_summary_op = tf.summary.merge_all(key=DDPG_CFG.actor_summary_keys)
  critic_summary_op = tf.summary.merge_all(key=DDPG_CFG.critic_summary_keys)
  log_summary_op = tf.summary.merge_all(key=DDPG_CFG.log_summary_keys)
  ######### initialize computation graph  ############

  '''
  # -------------trace graphdef only
  whole_graph_def = meta_graph.create_meta_graph_def(graph_def=sess.graph.as_graph_def())
  summary_writer.add_meta_graph(whole_graph_def,global_step=1)
  summary_writer.flush()

  run_options = tf.RunOptions(output_partition_graphs=True, trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()

  # including copy target -> online
  sess.run(fetches=[init_op],
           options=run_options,
           run_metadata=run_metadata
           )
  graphdef_part1 = run_metadata.partition_graphs[0]
  meta_graph_part1 = meta_graph.create_meta_graph_def(graph_def=graphdef_part1)
  part1_metagraph_writer = tf.summary.FileWriter(DDPG_CFG.log_dir + '/part1_metagraph')
  part1_metagraph_writer.add_meta_graph(meta_graph_part1)
  part1_metagraph_writer.close()

  graphdef_part2 = run_metadata.partition_graphs[1]
  meta_graph_part2 = meta_graph.create_meta_graph_def(graph_def=graphdef_part2)
  part2_metagraph_writer = tf.summary.FileWriter(DDPG_CFG.log_dir + '/part2_metagraph')
  part2_metagraph_writer.add_meta_graph(meta_graph_part2)
  part2_metagraph_writer.close()
  # --------------- end trace
  '''

  sess.run(fetches=[tf.global_variables_initializer()])

  #copy init params from online to target
  sess.run(fetches=[copy_online_to_target_op])

  # Load a previous checkpoint if it exists
  latest_checkpoint = tf.train.latest_checkpoint(DDPG_CFG.checkpoint_dir)
  if latest_checkpoint:
    print("=== Loading model checkpoint: {}".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)

  ####### start training #########

  if not DDPG_CFG.train_from_replay_buffer_set_only:
    obs = train_env.reset()
    transition = preprocess_low_dim(obs)

  n_episodes = 1
  update_start = 0.0

  for step in range(1, DDPG_CFG.num_training_steps):
    noise_process.reset()

    #replace with new transition
    if not DDPG_CFG.train_from_replay_buffer_set_only:  #no need new samples
      transition=agent_action_fn(step,sess,actor,online_state_inputs, is_training, transition.next_state[np.newaxis,:], replay_buffer,
                   noise_process,train_env)
    if step % DDPG_CFG.summary_transition_freq == 0:
      summary_transition(summary_writer,action_space.shape[0], transition,step)

    # after fill replay_buffer with some states, we start learn.
    if step > DDPG_CFG.learn_start:
      # test update duration at first 10 update
      if step < (DDPG_CFG.learn_start +10):
        update_start = time.time()

      ## ++++ sample mini-batch and train.++++
      state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = \
        replay_buffer.sample_batch(DDPG_CFG.batch_size)

      if step % 2000 == 0 and DDPG_CFG.train_from_replay_buffer_set_only:
        tf.logging.info('@@@@@ train from buffer only -one sample - global_step:{} action:{}'
                        '  reward:{} term:{} @@@@@@@@@@'.format(step, action_batch[0],reward_batch[0], terminated_batch[0]))

      # ---- 1. train policy.-----------
      # no need to feed reward, next_state, terminated which are un-used in policy update.
      # run_options = tf.RunOptions(output_partition_graphs=True, trace_level=tf.RunOptions.FULL_TRACE)
      if 0 == step % DDPG_CFG.summary_freq :
        # run_metadata = tf.RunMetadata()
        _, actor_summary = sess.run(fetches=[train_online_policy_op,actor_summary_op],
                                   feed_dict={online_state_inputs: state_batch,
                                              cond_training_q: False,
                                              online_action_inputs_training_q: action_batch,  # feed but not used.
                                              actor_l_r:l_r_decay(DDPG_CFG.actor_learning_rate,step),
                                              is_training:True})
                                    # options=run_options,
                                    # run_metadata=run_metadata)
        # summary_writer._add_graph_def(run_metadata.partition_graphs[0])

        # the policy online network is updated above and will not affect training q.
        # ---- 2. train q. --------------
        _, critic_summary = sess.run(fetches=[train_online_q_op, critic_summary_op],
                                   feed_dict={
                                     online_state_inputs: state_batch,
                                     cond_training_q: True,
                                     online_action_inputs_training_q: action_batch,
                                     target_state_inputs: next_state_batch,
                                     reward_inputs: reward_batch,
                                     reward_inputs: reward_batch,
                                     terminated_inputs: terminated_batch,
                                     critic_l_r:l_r_decay(DDPG_CFG.critic_learning_rate, step),
                                     is_training:True})

        summary_writer.add_summary(actor_summary)
        summary_writer.add_summary(critic_summary)
        summary_writer.flush()
      else:
        _ = sess.run(fetches=[train_online_policy_op],
                                 feed_dict={online_state_inputs: state_batch,
                                            cond_training_q: False,
                                            online_action_inputs_training_q: action_batch,  # feed but not used.
                                            actor_l_r:l_r_decay(DDPG_CFG.actor_learning_rate, step),
                                            is_training: True
                                            })

        # the policy online network is updated above and will not affect training q.
        # ---- 2. train q. --------------
        _, q_loss_value = sess.run(fetches=[train_online_q_op,q_loss_tensor],
                                  feed_dict={
                                    online_state_inputs: state_batch,
                                    cond_training_q: True,
                                    online_action_inputs_training_q: action_batch,
                                    target_state_inputs: next_state_batch,
                                    reward_inputs: reward_batch,
                                    terminated_inputs: terminated_batch,
                                    critic_l_r:l_r_decay(DDPG_CFG.critic_learning_rate, step),
                                    is_training: True})
        if step % 2000 ==0:
          tf.logging.info('@@ step:{} q_loss:{}'.format(step,q_loss_value))

      # --end of summary --
      # ----- 3. update target ---------
      # including increment global step.
      _ = sess.run(fetches=[update_target_op],
                   feed_dict=None)

      # test update duration at first 10 update
      if step < (DDPG_CFG.learn_start +10):
        tf.logging.info(' @@@@ one batch learn duration @@@@:{}'.format(time.time() - update_start))

      # do evaluation after eval_freq steps:
      if step % DDPG_CFG.eval_freq == 0: ##and step > DDPG_CFG.eval_freq:
        evaluate(env=monitor_env,
                 num_eval_steps=DDPG_CFG.num_eval_steps,
                 preprocess_fn=preprocess_low_dim,
                 estimate_fn=lambda state: sess.run(fetches=[actor.online_action_outputs_tensor],
                                                    feed_dict={online_state_inputs:state,
                                                    is_training:False} ),
                 summary_writer=summary_writer,
                 saver=saver, sess=sess, global_step=step,
                 log_summary_op=log_summary_op,summary_text_tensor=summary_text_tensor)
        # if monitor_env is train_env:
        #   #torcs share. we should reset
        #   transition.terminated = True #fall through
    #-- end of learn

    #TODO temp solution to on vision .use thread instead
    if step % 2000 == 0:
      v_on=os.path.exists('/home/yuheng/Desktop/train_vision_on')
      if train_env.vision_status==False and v_on:
        train_env.vision_on()  #will display next reset
        transition = preprocess_low_dim(train_env.reset(relaunch=True))
        n_episodes += 1
        tf.logging.info('@@ episodes: {} @@'.format(n_episodes))
        continue
      elif train_env.vision_status==True and not v_on:
        train_env.vision_off()
        transition = preprocess_low_dim(train_env.reset(relaunch=True))
        n_episodes += 1
        tf.logging.info('@@ episodes: {} @@'.format(n_episodes))
        continue


      # if os.path.exists('/home/yuheng/Desktop/eval_vision_on'):
      #   monitor_env.vision_on()  # will display next reset
      # else:
      #   monitor_env.vision_off()

    if (not DDPG_CFG.train_from_replay_buffer_set_only) and (transition.terminated):
      # relaunch TORCS every 3 episode because of the memory leak error
      # replace with transition observed after reset.only save state..
      transition = preprocess_low_dim(train_env.reset())
      n_episodes +=1
      tf.logging.info('@@ episodes: {} @@'.format(n_episodes))
      continue  # begin new episode
  # ====end for t.

  sess.close()
  train_env.close()
  monitor_env.close()

  ##### evalutation and plot .ref ddpg paper. use tf.summary#####


def evaluate(env, num_eval_steps, preprocess_fn, estimate_fn,
             summary_writer, saver, sess,global_step,log_summary_op,summary_text_tensor):
  total_reward = 0
  episode_reward = 0
  max_episode_reward = 0
  n_episodes = 1
  n_rewards = 1
  terminated = False
  global prev_eval_time
  # global max_avg_episode_reward

  transition = preprocess_fn(state=env.reset())
  estep=0

  tf.logging.info(' ####### start evaluate @ global step:{}##  '.format(global_step))

  while not terminated:
    estep+=1
    if estep > num_eval_steps:  #avoid too many low progress steps in one episode
      break

    policy_out = estimate_fn(transition.next_state[np.newaxis,:])  # must reshape to (1,11)
    action = policy_output_to_deterministic_action(policy_out,env.action_space)
    (state, reward, terminated) = env_step(env, action)

    # we only need state to generate policy.
    transition = preprocess_fn(state)

    # record every reward
    total_reward += reward
    episode_reward += reward

    if reward != 0:
      n_rewards += 1  # can represent effective(not still) steps in episode

    if terminated:
      n_episodes += 1
      if episode_reward > max_episode_reward:
        max_episode_reward = episode_reward
        episode_reward = 0

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

  #we always save model each evaluation.
  saved_name = save_model(saver, sess, global_step)
  write_summary(summary_writer, global_step, avg_episode_reward, max_episode_reward,
                avg_episode_steps, now - prev_eval_time, saved_name,sess,log_summary_op,
                summary_text_tensor)
  prev_eval_time = now
  # if avg_episode_reward > max_avg_episode_reward:
  #  max_avg_episode_reward = avg_episode_reward
  tf.logging.info('@@@@@@ eval save model : global_step:{} - avg_episode_reward:{} -\
                   max_episode_reward:{} - avg_episode_steps:{} - saved_file: {} @@@@@@ '.format(global_step,
                                                                          avg_episode_reward,
                                                                          max_episode_reward,
                                                                          avg_episode_steps,
                                                                          saved_name))

def write_summary(writer, global_step, avg_episode_reward, max_episode_reward,
                  avg_episode_steps, consuming_seconds,saved_name,sess,log_summary_op,
                  summary_text_tensor):
  eval_summary = tf.Summary()  # proto buffer
  eval_summary.value.add(node_name='avg_episode_reward',simple_value=avg_episode_reward, tag="train_eval/avg_episode_reward")
  eval_summary.value.add(node_name='max_episode_reward', simple_value=max_episode_reward, tag="train_eval/max_episode_reward")
  eval_summary.value.add(node_name='avg_episode_steps', simple_value=avg_episode_steps, tag="train_eval/avg_episode_steps")
  # change to minutes
  eval_summary.value.add(node_name='two_eval_interval_minutes',simple_value=(consuming_seconds/60), tag='train/eval/two_eval_interval_minutes')
  writer.add_summary(summary=eval_summary, global_step=global_step)

  #saved model name
  log_info = 'eval save model : global_step:{}    avg_episode_reward:{} \
              max_episode_reward:{}   avg_episode_steps:{}  \n saved_file: {} '.format(global_step,
                                                                                            avg_episode_reward,
                                                                                            max_episode_reward,
                                                                                            avg_episode_steps,
                                                                                            saved_name)

  log_summary=sess.run(fetches=[log_summary_op],
                       feed_dict={summary_text_tensor:log_info})
  writer.add_summary(summary=log_summary[0], global_step=global_step)

  # TODO let summary_freq to do flush??
  writer.flush()


def save_model(saver, sess,global_step):
  # save model. will save both online and target networks.
  saved_name=saver.save(sess=sess, save_path=DDPG_CFG.checkpoint_dir, global_step=global_step)

  for _ in range(10): #wait total 10s
    if not glob.glob(saved_name + '*'):
      time.sleep(1.0)
    else:
      return saved_name
  raise FileNotFoundError('@@@@@@@@@@@@ save model failed: {}'.format(saved_name))
  return  saved_name

def summary_transition(summary_writer, action_dim,transition,step):
  with tf.name_scope('transistion'):
    transition_summary = tf.Summary()  # proto buffer
    for i in range(action_dim):
      transition_summary.value.add(node_name='action_' + str(i), simple_value=transition.action[i],
                                   tag="transition/action_" + str(i))
    transition_summary.value.add(node_name='reward', simple_value=transition.reward,
                                 tag="transition/reward")
    summary_writer.add_summary(summary=transition_summary, global_step=step)
    #not flush. too frequently.

#TODO should use tf.natural_exp_decay
def l_r_decay(initial_l_r, global_steps):
  return initial_l_r * math.exp(-0.3 * np.floor_divide(global_steps,DDPG_CFG.l_r_decay_freq))
