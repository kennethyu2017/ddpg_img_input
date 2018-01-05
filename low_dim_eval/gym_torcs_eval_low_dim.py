"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import tensorflow as tf

import time
from low_dim_train.gym_torcs_train_low_dim import torcs_env_wrapper
from low_dim_eval.eval_agent_low_dim import evaluate

DDPG_CFG = tf.app.flags.FLAGS  # alias
DDPG_CFG.log_dir = '/home/yuheng/PycharmProjects/rl/kenneth_ddpg/ddpg_add_low_dim/train/gym_torcs_low_dim/tf_log/'
DDPG_CFG.checkpoint_dir = '/home/yuheng/PycharmProjects/rl/kenneth_ddpg/ddpg_add_low_dim/train/gym_torcs_low_dim/chk_pnt/'
DDPG_CFG.eval_monitor_dir = '/home/yuheng/PycharmProjects/rl/kenneth_ddpg/ddpg_add_low_dim/train/gym_torcs_low_dim/eval_monitor/'


tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":
  tf.logging.info("@@@  start ddpg evaluation gym_torcs @@@ start time:{}".format(time.ctime()))
  # Generate a Torcs environment
  env = torcs_env_wrapper(vision=True, throttle=True, gear_change=False)
  evaluate(env)





