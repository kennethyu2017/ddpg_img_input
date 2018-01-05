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
import tensorflow as tf
from actor import Actor
from critic import Critic
from tensorflow.contrib.layers import variance_scaling_initializer,batch_norm,l2_regularizer

from common.UO_process import UO_Process
from common.replay_buffer import ReplayBuffer, preprocess_img

##TODO use tf  FLAGS.define_string, define_ing....
DDPG_CFG = tf.app.flags.FLAGS  # alias

## hyper-P
#DDPG_CFG.learning_rate = 2.5e-4  ##TODO anneal learning_rate

DDPG_CFG.actor_learning_rate = 8e-3  ##TODO anneal learning_rate
DDPG_CFG.critic_learning_rate = 8e-4  ##TODO anneal learning_rate

DDPG_CFG.critic_reg_ratio = 1e-2
DDPG_CFG.actor_reg_ratio = 4e-2
DDPG_CFG.tau = 0.001
DDPG_CFG.gamma = 0.99
DDPG_CFG.num_training_steps = 25*(10**5)  # 2.5M steps total
DDPG_CFG.eval_freq = 4*(10**2)  # eval during training per steps .~30min.
DDPG_CFG.num_eval_steps = 50  # eval steps. 5min
# DDPG_CFG.save_model_freq = 10**4  # ~30min
DDPG_CFG.learn_start = 200   # fill replay_buffer some frames then start learn.
DDPG_CFG.batch_size = 16
#TODO decrease buffer size.too big to fit into 8G mem. or experience replay.
# DDPG_CFG.replay_buff_size = 10**6  # 1M
DDPG_CFG.replay_buff_size = 10**6  # 1M
#TODO search good rpt.
# DDPG_CFG.action_rpt = 4  good for bipedal
DDPG_CFG.action_rpt = 3  # long video clip maybe better for torcs.

#TODO test greyscale
DDPG_CFG.screen_height, DDPG_CFG.screen_width, DDPG_CFG.screen_channels = (64, 64, 1)

#TODO add normalization layer,reg definition.
# === actor net arch. shared by online and target ==
is_training=tf.placeholder(tf.bool, shape=(), name='is_training')
# -- 3 conv layers --
DDPG_CFG.online_policy_net_var_scope = 'online_policy'
DDPG_CFG.target_policy_net_var_scope = 'target_policy'
DDPG_CFG.actor_summary_keys = 'actor_summaries'
DDPG_CFG.actor_conv_n_maps = [32, 32, 32]  # number of filters
DDPG_CFG.actor_kernel_sizes = [(7, 7), (4, 4), (3, 3)]
DDPG_CFG.actor_conv_strides = [3, 2, 1]
DDPG_CFG.actor_conv_paddings = ['VALID'] * 3
DDPG_CFG.actor_conv_activations = [tf.nn.elu] * 3
DDPG_CFG.actor_conv_initializers = [variance_scaling_initializer()] * 3  # [He] init.
#TODO try actor no BN.
DDPG_CFG.actor_conv_normalizers = [batch_norm] * 3
DDPG_CFG.actor_conv_normal_params = [{'is_training':is_training,
                                       'data_format':'NHWC',
                                       'updates_collections':None }] *3 # inplace update running average
# DDPG_CFG.actor_conv_normalizers = [None] * 3
# DDPG_CFG.actor_conv_normal_params = [{}]*3

DDPG_CFG.actor_conv_regularizers = [l2_regularizer(scale=DDPG_CFG.actor_reg_ratio)] * 3
# DDPG_CFG.actor_conv_regularizers = [None] * 3

# -- 2 fc layers --
DDPG_CFG.actor_n_fc_in = 7 * 7 * 32  # output of conv-3
DDPG_CFG.actor_n_fc_units = [200, 200]
DDPG_CFG.actor_fc_activations = [tf.nn.elu] * 2
DDPG_CFG.actor_fc_initializers = [variance_scaling_initializer()] * 2  # [He] init
DDPG_CFG.actor_fc_regularizers = [l2_regularizer(scale=DDPG_CFG.actor_reg_ratio)] * 2
# DDPG_CFG.actor_fc_regularizers = [None] * 2
DDPG_CFG.actor_fc_normalizers = [batch_norm] * 2
DDPG_CFG.actor_fc_norm_params =  [{'is_training':is_training,
                                       'data_format':'NHWC',
                                       'updates_collections':None }] *2 # inplace update running average
#TODO try actor no BN.
# DDPG_CFG.actor_fc_normalizers = [None] * 2
# DDPG_CFG.actor_fc_norm_params =  [None] *2

# -- 1 output layer --
# TODO can use tf.variance_scaling better.
# DDPG_CFG.actor_output_layer_initializer = lambda shape=[200, len(DDPG_CFG.action_fields)], dtype=tf.float32, partition_info=None: \
#   tf.random_uniform(shape=shape, minval=-3.e-4, maxval=3.e-4, dtype=dtype)

DDPG_CFG.actor_output_layer_initializer=tf.random_uniform_initializer(-3e-4,3e-4)
DDPG_CFG.actor_output_layer_regularizer = l2_regularizer(scale=DDPG_CFG.actor_reg_ratio)
# DDPG_CFG.actor_output_layer_regularizer = None

# === critic net arch. shared by online and target ==
# -- 3 conv layers --
DDPG_CFG.online_q_net_var_scope = 'online_q'
DDPG_CFG.target_q_net_var_scope = 'target_q'
DDPG_CFG.critic_summary_keys = 'critic_summaries'
DDPG_CFG.critic_conv_n_maps = [32, 32, 32]  # number of filters
DDPG_CFG.critic_kernel_sizes = [(7, 7), (4, 4), (3, 3)]
DDPG_CFG.critic_conv_strides = [3, 2, 1]
DDPG_CFG.critic_conv_paddings = ['VALID'] * 3
DDPG_CFG.critic_conv_activations = [tf.nn.elu] * 3
DDPG_CFG.critic_conv_initializers = [variance_scaling_initializer()] * 3  # [He] init.
###TODO try w/o BN
DDPG_CFG.critic_conv_normalizers = [batch_norm] * 3
DDPG_CFG.critic_conv_normal_params = [{'is_training':is_training,
                                       'data_format':'NHWC',
                                       'updates_collections':None }] *3 # inplace update running average
# DDPG_CFG.critic_conv_normalizers = [None] * 3
# DDPG_CFG.critic_conv_normal_params = [{}] * 3

DDPG_CFG.critic_conv_regularizers = [l2_regularizer(scale=DDPG_CFG.critic_reg_ratio)] * 3

# -- 2 fc layer --
DDPG_CFG.include_action_fc_layer = 2  # in this layer we include action inputs. conting from fc-1 as 1.
DDPG_CFG.critic_n_fc_in = 7 * 7 * 32  # output of conv-3
DDPG_CFG.critic_n_fc_units = [200, 200]
DDPG_CFG.critic_fc_activations = [tf.nn.elu] * 2
DDPG_CFG.critic_fc_initializers = [variance_scaling_initializer()] * 2  # [He] init
DDPG_CFG.critic_fc_regularizers = [l2_regularizer(scale=DDPG_CFG.critic_reg_ratio)] * 2
###TODO try w/o BN
DDPG_CFG.critic_fc_normalizers = [batch_norm, None]  # 2nd fc including action input and no BN
DDPG_CFG.critic_fc_norm_params =  [{'is_training':is_training,
                                       'data_format':'NHWC',
                                       'updates_collections':None }, None] # inplace update running average
# DDPG_CFG.critic_fc_normalizers = [None] * 2  # 2nd fc including action input and no BN
# DDPG_CFG.critic_fc_norm_params =  [{}] * 2


# -- 1 output layer --
# DDPG_CFG.critic_output_layer_initializer = tf.initializers.variance_scaling
# TODO can use tf.variance_scaling better.
DDPG_CFG.critic_output_layer_initializer = tf.random_uniform_initializer(-3e-4, 3e-4)
DDPG_CFG.critic_output_layer_regularizer = l2_regularizer(scale=DDPG_CFG.critic_reg_ratio)


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

  # tf.summary.scalar(name='q_reg_loss_mean', tensor=tf.reduce_mean(q_reg_loss), collections=[DDPG_CFG.critic_summary_keys])
  tf.summary.scalar(name='q_value_mean',tensor=tf.reduce_mean(critic.online_q_outputs_tensor),
                    collections=[DDPG_CFG.critic_summary_keys])
  tf.summary.scalar(name='q_loss', tensor=q_loss, collections=[DDPG_CFG.critic_summary_keys])


  ##build policy loss.
  # then d(loss)/d(theta_mu) = d(loss)/d(Q) * d(Q)/d(a) * d(a)/ d(theta_mu) = -1 *policy_gradients.
  # so do SGD to minimize policy_loss is same as do SGA to maximum J(theta_mu)
  # add l2 reg.

  # TODO try to pass through gradient when tanh is saturated.
  #list of reg scalar Tensors.
  policy_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=DDPG_CFG.online_policy_net_var_scope)

  policy_loss =tf.add_n([ tf.reduce_mean(critic.online_q_outputs_tensor)] + policy_reg_loss,
                        name='policy_loss')
  tf.summary.scalar(name='policy_loss', tensor=policy_loss, collections=[DDPG_CFG.actor_summary_keys])

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




def train(train_env, monitor_env):
  '''
    :return:
  '''
  action_space = train_env.action_space


  ######### instantiate actor,critic, replay buffer, uo-process#########
  ## feed online with state. feed target with next_state.
  online_state_inputs = tf.placeholder(tf.float32,
                                       shape=(None, DDPG_CFG.screen_height, DDPG_CFG.screen_width,
                                              DDPG_CFG.screen_channels * DDPG_CFG.action_rpt),
                                       name="online_state_inputs")

  # tf.logging.info('@@@@ online_state_inputs shape:{}'.format(online_state_inputs.shape))
  target_state_inputs = tf.placeholder(tf.float32,
                                       shape=(None, DDPG_CFG.screen_height, DDPG_CFG.screen_width,
                                              DDPG_CFG.screen_channels * DDPG_CFG.action_rpt),
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

  # target_action_inputs = tf.placeholder(tf.float32,
  #                                       shape=(None, len(DDPG_CFG.torcs_action_fields)),
  #                                       name='target_action_inputs'
  #                                       )

  # batch_size vector.
  terminated_inputs = tf.placeholder(tf.float32, shape=(None), name='terminated_inputs')
  reward_inputs = tf.placeholder(tf.float32, shape=(None), name='rewards_inputs')

  ##instantiate actor, critic.
  actor = Actor(action_dim=action_space.shape[0],
                online_state_inputs=online_state_inputs,
                target_state_inputs=target_state_inputs,
                conv_n_feature_maps=DDPG_CFG.actor_conv_n_maps,
                conv_kernel_sizes=DDPG_CFG.actor_kernel_sizes,
                conv_strides=DDPG_CFG.actor_conv_strides,
                conv_padding=DDPG_CFG.actor_conv_paddings,
                conv_activations=DDPG_CFG.actor_conv_activations,
                conv_initializers=DDPG_CFG.actor_conv_initializers,
                conv_normalizers=DDPG_CFG.actor_conv_normalizers,
                conv_norm_params=DDPG_CFG.actor_conv_normal_params,
                conv_regularizers=DDPG_CFG.actor_conv_regularizers,
                n_fc_in=DDPG_CFG.actor_n_fc_in,
                n_fc_units=DDPG_CFG.actor_n_fc_units,
                fc_activations=DDPG_CFG.actor_fc_activations,
                fc_initializers=DDPG_CFG.actor_fc_initializers,
                fc_normalizers=DDPG_CFG.actor_fc_normalizers,
                fc_norm_params=DDPG_CFG.actor_fc_norm_params,
                fc_regularizers=DDPG_CFG.actor_fc_regularizers,
                output_layer_initializer=DDPG_CFG.actor_output_layer_initializer,
                output_layer_regularizer=DDPG_CFG.actor_output_layer_regularizer,
                # output_layer_regularizer=None,
                output_bound_fns=DDPG_CFG.actor_output_bound_fns,
                learning_rate=DDPG_CFG.actor_learning_rate,
                is_training=is_training)


  critic = Critic(online_state_inputs=online_state_inputs,
                  target_state_inputs=target_state_inputs,
                  online_action_inputs_training_q=online_action_inputs_training_q,
                  online_action_inputs_training_policy=actor.online_action_outputs_tensor,
                  cond_training_q=cond_training_q,
                  target_action_inputs=actor.target_action_outputs_tensor,
                  conv_n_feature_maps=DDPG_CFG.critic_conv_n_maps,
                  conv_kernel_sizes=DDPG_CFG.critic_kernel_sizes,
                  conv_strides=DDPG_CFG.critic_conv_strides,
                  conv_padding=DDPG_CFG.critic_conv_paddings,
                  conv_activations=DDPG_CFG.critic_conv_activations,
                  conv_initializers=DDPG_CFG.critic_conv_initializers,
                  conv_normalizers=DDPG_CFG.critic_conv_normalizers,
                  conv_norm_params=DDPG_CFG.critic_conv_normal_params,
                  conv_regularizers=DDPG_CFG.critic_conv_regularizers,
                  n_fc_in=DDPG_CFG.critic_n_fc_in,
                  n_fc_units=DDPG_CFG.critic_n_fc_units,
                  fc_activations=DDPG_CFG.critic_fc_activations,
                  fc_initializers=DDPG_CFG.critic_fc_initializers,
                  fc_normalizers=DDPG_CFG.critic_fc_normalizers,
                  fc_norm_params=DDPG_CFG.critic_fc_norm_params,
                  fc_regularizers=DDPG_CFG.critic_fc_regularizers,
                  output_layer_initializer=DDPG_CFG.critic_output_layer_initializer,
                  output_layer_regularizer=DDPG_CFG.critic_output_layer_regularizer,
                  learning_rate=DDPG_CFG.critic_learning_rate)

  ## track updates.
  global_step_tensor = tf.train.create_global_step()

  ## build whole graph
  copy_online_to_target_op, train_online_policy_op, train_online_q_op, update_target_op, saver \
    = build_ddpg_graph(actor, critic, reward_inputs, terminated_inputs, global_step_tensor)

  replay_buffer = ReplayBuffer(buffer_size=DDPG_CFG.replay_buff_size)

  # noise shape (3,)
  noise_process = UO_Process(mu=np.zeros(shape=action_space.shape))

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
  ##ddpg algo-1 processing

  # episode_reward_moving_average = 0.0
  # episode_steps_moving_average = 0

  episode = 1
  # track one epoch consuming time
  # epoch_start_time = time.time()  #in seconds
  # epoches = 0

  obs = train_env.reset()
  n_episodes = 1
  # we dont store the 1st frame.just stacked and used as input to policy network to generate action.
  transition = preprocess_img(frames=[obs])

  # while epoches < DDPG_CFG.num_training_epoches:
  update_start = 0.0
  for step in range(1, DDPG_CFG.num_training_steps):
    # episode_reward = 0.0
    # episode_steps = 0
    noise_process.reset()

    # for t in range(1,DDPG_CFG.num_timesteps_per_episode):

    #make random play at beginning . to fill some frames in replay buffer.
    if step < DDPG_CFG.learn_start:
      # stochastic_action = [np.random.uniform(low,high) for (low,high) in zip(action_space.low, action_space.high)]
      #give some speed at beginning.
      stochastic_action = [None]*3
      stochastic_action[DDPG_CFG.policy_output_idx_steer] = np.random.uniform(-0.1,0.1)
      stochastic_action[DDPG_CFG.policy_output_idx_accel] = np.random.uniform(0.5, 1.0)
      stochastic_action[DDPG_CFG.policy_output_idx_brake] = np.random.uniform(0.01, 0.1)
    else:
      ## calc a_t = mu(s_t) + Noise
      ## FF once to fetch the mu(s_t)
      ## --out[0]:steer, out[1]:accel, out[2]:brake--
      # episode_steps = t
      policy_output = sess.run(fetches=[actor.online_action_outputs_tensor],
            feed_dict={online_state_inputs: transition.next_state[np.newaxis,:,:,:],
                       is_training:False})  # must reshape to (1,64,64,9)
      policy_output=policy_output[0]

      #TODO anneal random prob of actions: from high prob of accel to low prob do nothing.
      if step % 7 ==0 or step < (DDPG_CFG.learn_start + 30):
        #tf.logging.info('@@@@@@ policy output:{}  @@@@@@'.format(policy_output))
        # we add some random speed:
        policy_output[0][DDPG_CFG.policy_output_idx_steer] = np.random.uniform(-0.1, 0.1)
        policy_output[0][DDPG_CFG.policy_output_idx_accel] += np.random.uniform(0.8, 1.0)
        policy_output[0][DDPG_CFG.policy_output_idx_brake] += np.random.uniform(-0.9, 0.1)
      ##add noise and bound
      stochastic_action=policy_output_to_stochastic_action(policy_output, noise_process, action_space)



    ## excute a_t and store Transition.
    (frames, reward, terminated) = action_repeat_steps(train_env, stochastic_action)
    # episode_reward += reward

    if step % 50 == 0:
      tf.logging.info('@@@@@@@@@@ global_step:{} action:{}  reward:{} term:{} @@@@@@@@@@'.format(step,stochastic_action,reward,terminated))

    # replace transition with new one.
    transition = preprocess_img(action=stochastic_action,
        reward=reward,
        terminated=terminated,
        frames=frames)

    ##even if terminated ,we still save next_state cause FF Q network
    # will use it, but will discard Q value in the end.
    replay_buffer.store(transition)

    # after fill replay_buffer with some frames, we start learn.
    if step > DDPG_CFG.learn_start:
      # test update duration at first 10 update
      if step < (DDPG_CFG.learn_start +10):
        update_start = time.time()

      ## ++++ sample mini-batch and train.++++
      state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = \
        replay_buffer.sample_batch(DDPG_CFG.batch_size)

      # FP/BP to SGD, SGA update online mu and Q.
      ## training op will update online then soft-update target.
      # tf.logging.info('@@@@ state batch shape:{}'.format(state_batch.shape))


      # ---- 1. train policy.-----------
      # no need to feed reward, next_state, terminated which are un-used in policy update.
      # run_options = tf.RunOptions(output_partition_graphs=True, trace_level=tf.RunOptions.FULL_TRACE)
      if 0 == step % 20 :
        # run_metadata = tf.RunMetadata()
        _, actor_summary = sess.run(fetches=[train_online_policy_op,actor_summary_op],
                                   feed_dict={online_state_inputs: state_batch,
                                              cond_training_q: False,
                                              online_action_inputs_training_q: action_batch,  # feed but not used.
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
                                     terminated_inputs: terminated_batch,
                                     is_training:True})

        summary_writer.add_summary(actor_summary)
        summary_writer.add_summary(critic_summary)
        summary_writer.flush()
      else:
        _ = sess.run(fetches=[train_online_policy_op],
                                 feed_dict={online_state_inputs: state_batch,
                                            cond_training_q: False,
                                            online_action_inputs_training_q: action_batch,  # feed but not used.
                                            is_training: True
                                            })

        # the policy online network is updated above and will not affect training q.
        # ---- 2. train q. --------------
        _ = sess.run(fetches=[train_online_q_op],
                                  feed_dict={
                                    online_state_inputs: state_batch,
                                    cond_training_q: True,
                                    online_action_inputs_training_q: action_batch,
                                    target_state_inputs: next_state_batch,
                                    reward_inputs: reward_batch,
                                    terminated_inputs: terminated_batch,
                                    is_training: True})


      # ----- 3. update target ---------
      # including increment global step.
      _ = sess.run(fetches=[update_target_op],
                   feed_dict=None)

      # test update duration at first 10 update
      if step < (DDPG_CFG.learn_start +10):
        tf.logging.info(' @@@@ one update duration @@@@:{}'.format(time.time() - update_start))

      # do evaluation after eval_freq steps:
      if step % DDPG_CFG.eval_freq == 0: ##and step > DDPG_CFG.eval_freq:
        evaluate(env=monitor_env,
                 num_eval_steps=DDPG_CFG.num_eval_steps,
                 preprocess_fn=preprocess_img,
                 estimate_fn=lambda state: sess.run(fetches=[actor.online_action_outputs_tensor],
                                                    feed_dict={online_state_inputs:state,
                                                    is_training:False} ),
                 summary_writer=summary_writer,
                 saver=saver, sess=sess, global_step=step)
    #-- end of learn

    if (transition.terminated):
      new_obs = train_env.reset()  # relaunch TORCS every 3 episode because of the memory leak error
      # replace with transition observed after reset.only save frames.
      transition = preprocess_img(frames=[new_obs])
      n_episodes +=1
      tf.logging.info('@@ episodes: {} @@'.format(n_episodes))
      continue  # begin new episode
      # ====end for t. end of one episode ====

      # ---end of save---
  # ---end for episode---

  sess.close()
  eval_monitor.close()
  train_env.close()
  monitor_env.close()

  ##### evalutation and plot .ref ddpg paper. use tf.summary#####


def evaluate(env, num_eval_steps, preprocess_fn, estimate_fn, summary_writer, saver, sess,global_step):
  total_reward = 0
  episode_reward = 0
  max_episode_reward = 0
  n_episodes = 0
  n_rewards = 0
  terminated = False
  global prev_eval_time
  global max_avg_episode_reward

  #start_time=time.time()
  obs = env.reset()
  # stack frames and used as input to policy network to generate action.
  transition = preprocess_fn(frames=[obs])
  estep=0

  while not terminated:
    estep+=1
    policy_out = estimate_fn(transition.next_state[np.newaxis,:,:,:])  # must reshape to (1,64,64,3)
    action = policy_output_to_deterministic_action(policy_out,env.action_space)
    # still do action rpt just like DQN2015 code.
    (frames, reward, terminated) = action_repeat_steps(env, action)

    # we only need frames.
    transition = preprocess_fn(frames=frames)

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
      new_obs = env.reset()
      # only save frames.
      transition = preprocess_fn(frames=[new_obs])
      if estep < num_eval_steps:
        terminated = False  # continue

  # -- end for estep ---

  # eval_time = time.time() - start_time
  # start_time = start_time + eval_time
  # agent: compute_validation_statistics()
  # ind =  # reward_history+1
  avg_episode_reward = total_reward / max(1, n_episodes)
  avg_episode_steps = n_rewards / max(1, n_episodes)

  # if agent.v_avg
  # v_history[ind] = agent.v_avg
  # td_history[ind] = agent.tderr_avg
  # qmax_history[ind] = agent.q_max
  # print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

  # reward_history.append(avg_episode_reward)
  # reward_counts.append(n_rewards)
  # episode_counts.append(n_episodes)
  # time_history[ind + 1] = sys.clock() - start_time

  # local  time_dif = time_history[ind + 1] - time_history[ind]

  # local  training_rate = opt.actrep * opt.eval_freq / time_dif

  now = time.time()
  if prev_eval_time == 0:  # first time eval.
    prev_eval_time = now

  write_summary(summary_writer, global_step, avg_episode_reward, max_episode_reward, avg_episode_steps, now - prev_eval_time)
  prev_eval_time = now

  # save best model.
  # TODO save max avg episode steps model.
  if avg_episode_reward > max_avg_episode_reward:
    saved_name = save_model(saver,sess, global_step)
    max_avg_episode_reward = avg_episode_reward
    tf.logging.info('@@@@@@ best model found: global_step:{} - avg_episode_reward:{} -\
                   max_episode_reward:{} - avg_episode_steps:{} - saved_file: {} @@@@@@ '.format(global_step,
                                                                          avg_episode_reward,
                                                                          max_episode_reward,
                                                                          avg_episode_steps,
                                                                          saved_name))


t = 0
def action_repeat_steps(env, action):
  ##stack action rpt frames along original RGB channels.
  # concated_frames=None
  frames = []
  reward_sum = 0
  terminated = False
  curr_steps = 0
  global t
  t+=1

  '''
  gym torcs step --
        inparams:
          'steer': u[0]
          'accel': u[1]
          'brake': u[2]
        return: self.get_obs(), reward, client.R.d['meta'], {}
  '''

  # u = [action[DDPG_CFG.policy_output_idx_steer],
  #      action[DDPG_CFG.policy_output_idx_accel],
  #      action[DDPG_CFG.policy_output_idx_brake]]
  if (t % 10==0) and t < 100:
    prev_t = time.time()

  while curr_steps < (DDPG_CFG.action_rpt) and not terminated:
    curr_steps += 1
    ##action[0]:steer, action[1]:accel, action[2]:brake
    # if anyone step terminated, we stop following steps and return as 'terminated'
    (obs, reward, terminated, _) = env.step(action)
    if (t % 10==0) and t < 100:
      now = time.time()
      tf.logging.info('++++ @@@@ step interval :{}'.format(now - prev_t))
      prev_t =now


    ##obs is after action exec+ution, so its next_state indeed.
    ##the img from gym obs is :64x64=4096 2-D np.array with rgb values grouped together.i.e.shape(4096,3)
    ##we stack and reshape to (64,64,action_rpt*3) to be as state input to policy/q conv net.
    # screen = obs.img.reshape(DDPG_CFG.torcs_screen_height, DDPG_CFG.torcs_screen_width, -1)
    # use preprocess to reshape.
    frames.append(obs)

    # if concated_frames is None:
    #     concated_frames = screen
    # else:
    #     ##concate along RGB channel axis
    #     concated_frames = np.concatenate(a_tuple=(screen, concated_frames), axis=-1)

    # sum reward
    reward_sum += reward

    # handle terminated. repeat the last screen to achieve full feature maps.
    # if terminated:
    ##next_state is not used to calc Q cause its terminated. but we still use
    # it to FF through mu_prime/Q_prime then Q*0, so we must gurantee the
    # data format is accurate,by concatenating the last screen.
    # for _ in xrange(t+1, DDPG_CFG.action_rpt):
    #     concated_frames = np.concatenate(a_tuple=(screen,concated_frames), axis=-1)
    # break
  ## == end while ==#
  # return (concated_frames, reward_sum, terminated)
  return (frames, reward_sum, terminated)



def write_summary(writer, global_step, avg_episode_reward, max_episode_reward,avg_episode_steps, consuming_seconds):
  eval_summary = tf.Summary()  # proto buffer
  eval_summary.value.add(node_name='avg_episode_reward' ,simple_value=avg_episode_reward, tag="train_eval/avg_episode_reward")
  eval_summary.value.add(node_name='max_episode_reward', simple_value=max_episode_reward, tag="train_eval/max_episode_reward")
  eval_summary.value.add(node_name='avg_episode_steps', simple_value=avg_episode_steps, tag="train_eval/avg_episode_steps")
  # change to minutes
  eval_summary.value.add(node_name='tow_eval_interval_minutes',simple_value=(consuming_seconds/60), tag='train/eval/two_eval_interval_minutes')

  # use epoches as 'global_step' tag
  writer.add_summary(summary=eval_summary, global_step=global_step)
  writer.flush()


def save_model(saver, sess,global_step):
  # save model. will save both online and target networks.
  return saver.save(sess=sess, save_path=DDPG_CFG.checkpoint_dir, global_step=global_step)


#
# def torcs_action(output):
#   ## --output[0]:steer, output[1]:accel, output[2]:brake--
#   action = Action(steer=stochastic_action[DDPG_CFG.policy_output_idx_steer],
#                   accel=stochastic_action[DDPG_CFG.policy_output_idx_accel],
#                   brake=stochastic_action[DDPG_CFG.policy_output_idx_brake])
#   return  action




