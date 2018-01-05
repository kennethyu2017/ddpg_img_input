
import numpy as np
from collections import deque,namedtuple
import copy
import os
import glob
import pickle as pkl
import time
import sys
import common.replay_buffer

if __name__ == '__main__':
  sys.path.append('/home/yuheng/PycharmProjects/rl/kenneth_ddpg/ddpg_add_low_dim/')

  print(sys.path)
  # from common.replay_buffer import Transition
  abs_file_pattern = '/home/yuheng/PycharmProjects/rl/kenneth_ddpg/' \
                     'ddpg_add_low_dim/train/gym_torcs_low_dim/replay_buffer/replay_buffer_*'

  data_l = []
  buffer_files = glob.glob(abs_file_pattern)
  for f_name in buffer_files:
    with open(f_name,'rb') as f:
      data = pkl.load(f) #list of T
      print('@@@@ === load replay buffer data from file -{} '
                      '-- data length {}  data[87]: {}'.format(f_name,len(data)))
      for T in data:
        data_l.append(T)

      data_arr=np.array(data_l)
      if np.any(np.abs(data_arr)==np.inf):
        print ('GOT inf')



