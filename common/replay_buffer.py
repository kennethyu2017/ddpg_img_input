"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import tensorflow as tf
import numpy as np
from collections import deque,namedtuple
import copy
import os
import glob
import pickle as pkl
import time

DDPG_CFG = tf.app.flags.FLAGS  # alias

DDPG_CFG.replay_buffer_file_name = 'replay_buffer'
##transition is (s, a, r, term, next_s)
## == NOTE:next_state is stacked-frames with shape (64,64,9)
## == NOTE: action is np.array of shape(action_dim,)
transition_fields = ['action', 'reward', 'terminated', 'next_state']
Transition = namedtuple('Transition', transition_fields)


def concat_frames(frames):
  '''
    :param frames: frames: list of screen frames(shape 64,64,3) to be stacked.
    :return:
    '''
  ##each element of frames is a np.array representing one screen frame.
  f = frames[0]
  concated = f

  ##TODO if only one frame in frames.
  for i in range(1, DDPG_CFG.action_rpt):
    if i < len(frames):
      f = frames[i]  # else re-use last frame
    concated = np.concatenate((f,concated), axis=-1)

  return concated


def preprocess_img(frames, action=None, reward=None, terminated=None):
  """
    :param frames: list of screen frames representing next_state.
    each frame from gym_torcs is ndarray shape (64*64,3).
     need to be stacked to be shape(64,64,9)
    :param action: None in case  1st frame of new episode
    :param reward: None in case  1st frame of new episode
    :param terminated: None in case  1st frame of new episode
    :return:
    """

  frames = [f.reshape(DDPG_CFG.screen_height, DDPG_CFG.screen_width, -1)
            for f in frames]

  ##concat list of frames to be shape (64,64,6).
  stacked_frames = concat_frames(frames)

  ##TODO need downsample? placed in each env training agent
  ##crop
  stacked_frames = stacked_frames[:DDPG_CFG.screen_height,
                   :DDPG_CFG.screen_width,
                   :]

  # normalize from [0,255] to [0,1]
  stacked_frames=np.divide(stacked_frames, 255.0)

  transition = Transition(action=action,
                          reward=reward,
                          terminated=terminated,
                          next_state=stacked_frames)
  return transition

def preprocess_low_dim(state, action=None, reward=None, terminated=None):
  """
    :param state  shape: obs_space.shape
    :param action: None in case  1st frame of new episode
    :param reward: None in case  1st frame of new episode
    :param terminated: None in case  1st frame of new episode
    :return:
    """
  transition = Transition(action=action,
                          reward=reward,
                          terminated=terminated,
                          next_state=state)
  return transition


class ReplayBuffer(object):
  def __init__(self, buffer_size, seed, save_segment_size=None, save_path=None):
    """
        The right side of the deque contains the most recent experiences 
        """
    self.buffer_size = buffer_size
    ##TODO use tf.RandomShuffleQueue instead.
    self.buffer = deque([], maxlen=buffer_size)
    if seed is not None:
      np.random.seed(seed)
    self.stored_cnt=0  # maybe > buffer_size.
    self.term_cnt=0  # term transition
    self.penalty_cnt=0 # penalty transition

    self.save=False
    if save_segment_size is not None:
      assert save_path is not None
      self.save = True
      self.save_segment_size = save_segment_size
      self.save_path = save_path
      self.save_data_cnt=0
      self.save_segment_cnt=0

  def store(self, transition):
    ##deque can take care of max len.
    T = copy.deepcopy(transition)

    # TODO try to adjust replay buffer distribution.
    #TODO better use individual queue for penalty T.
    # 1 .term: out of track, run backward,low progress term.
    if transition.terminated and transition.reward<=-1:
      # tf.logging.info('+++++ found term, stored_cnt:{} term_cnt:{}'.format(self.stored_cnt, self.term_cnt))
      # if self.term_cnt < (self.stored_cnt * 1./300):  #occupy partion
      if self.term_cnt < (self.stored_cnt * 1./300):  #occupy partion
        self.buffer.extend([T]*5)
        # tf.logging.info('+++++ found term, store *10')
        self.stored_cnt += 5
        self.term_cnt+=5
      else:
        self.buffer.append(T)
        self.stored_cnt+=1
        self.term_cnt+=1
      # tf.logging.info('+++++ found term, but reached occupy partion, append 1')
    # # 2. penalty:reward=-1. collision
    elif transition.reward <= -1 :
      # tf.logging.info('+++++ found penalty, stored_cnt:{} penalty_cnt:{}'.format(self.stored_cnt, self.penalty_cnt))
      if self.penalty_cnt < (self.stored_cnt * 1./500): # occupy partion
        self.buffer.extend([T]*3)
        # tf.logging.info('+++++ found penalty, *5,')
        self.stored_cnt += 3
        self.penalty_cnt +=3
      else:
        self.buffer.append(T)
        self.stored_cnt+=1
        self.penalty_cnt+=1
        # tf.logging.info('+++++ found penalty, but reached occupy partion, append 1 ')
    else:
      self.buffer.append(T)
      self.stored_cnt += 1

    if self.save:
      self.save_data_cnt+=1
      if self.save_data_cnt >= self.save_segment_size:
        self.save_segment()
        self.save_data_cnt=0
    del transition

  def get_item(self,idx):
    return self.buffer[idx]

  @property
  def length(self):
    return len(self.buffer)

  @property
  def size(self):
    return self.buffer.__sizeof__()

  ##TODO use named_tuple to represent transition.
  def sample_batch(self, batch_size):
    ## if length < batch_size, we return the length size data as a batch.
    ##random range end at length-1 so we can handle the buffer tail as next_state.
    ##TODO as DQN[2015], we should uniform sample 50k frames one time,then use it to
    # pop out training frames. and then sample 50k again... avoid sample every update.
    indices = np.random.permutation(self.length - 1)[:batch_size]
    ##TODO use np.array?
    state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = [], [], [], [], []
    # cols = [[],[],[],[],[]]  #state, action, reward, next_state, continue
    for idx in indices:
      ##trans_1 : (a_1, r_1, term_1, s_2)
      trans_1 = self.buffer[idx]
      ##TODO stack hist-frmaes here.
      if trans_1.terminated is not True:
        # the trans_2 : (a_2, r_2, term_2, s_3)
        trans_2 = self.buffer[idx + 1]  # idx < length-1
        # we use the data (s_2, a_2, r_2, term_2, s_3)
        state_batch.append(trans_1.next_state)
        action_batch.append(trans_2.action)
        reward_batch.append(trans_2.reward)
        next_state_batch.append(trans_2.next_state)
        terminated_batch.append(trans_2.terminated)
      else:
        ##term_1 is true, so buffer[idx+1] is beginning of new episode,
        # we traverse back.
        if idx != 0:
          trans_0 = self.buffer[idx - 1]  # a_0, r_0, s_1, term_0 = self.buffer[idx - 1]
          if trans_0.terminated is True:  # give up
            continue
          # we use the data (s_1, a_1, r_1, term_1, s_2)
          # s_2 is not used to calc Q cause its terminated. but we still use
          # it to FF through mu_prime/Q_prime then Q*0. guarantee s_2 is accurate formatted.
          state_batch.append(trans_0.next_state)
          action_batch.append(trans_1.action)
          reward_batch.append(trans_1.reward)
          next_state_batch.append(trans_1.next_state)
          terminated_batch.append(trans_1.terminated)
        else:
          # head of buffer, we dont know the previous state , so give up.
          continue

          ###transfer to np.array.
    return (np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch),np.array(terminated_batch))

  def save_segment(self):
    self.save_segment_cnt+=1

    data = []
    start = self.length - self.save_segment_size  #always save latest data of segment_size
    end = self.length

    for idx in range(start, end):
      data.append(self.buffer[idx])

    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)

    abs_file_name = os.path.abspath(os.path.join(self.save_path,
                            '_'.join([DDPG_CFG.replay_buffer_file_name,str(self.save_segment_cnt),time.ctime()])))

    with open(abs_file_name,'wb') as f:
      pkl.dump(data, f)


  def load(self, path):
    #load from file to buffer
    abs_file_pattern = os.path.abspath(os.path.join(path,
                            '_'.join([DDPG_CFG.replay_buffer_file_name,'*'])))
    buffer_files = glob.glob(abs_file_pattern)
    for f_name in buffer_files:
      with open(f_name,'rb') as f:
        data = pkl.load(f)
        tf.logging.info('@@@@ === load replay buffer data from file -{} '
                        '-- data length {}  data[87]: {}'.format(f_name,len(data),data[87]))
        self.buffer.extend(data)
        tf.logging.info('@@@@ +++ replay buffer length - {}'.format(self.length))

  def clear(self):
    self.buffer.clear()




# from openai baseline
class RingBuffer(object):
  def __init__(self, maxlen, shape, dtype='float32'):
    self.maxlen = maxlen
    self.start = 0
    self.length = 0
    self.data = np.zeros((maxlen,) + shape).astype(dtype)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    if idx < 0 or idx >= self.length:
      raise KeyError()
    return self.data[(self.start + idx) % self.maxlen]

  def get_batch(self, idxs):
    return self.data[(self.start + idxs) % self.maxlen]

  def append(self, v):
    if self.length < self.maxlen:
      # We have space, simply increase the length.
      self.length += 1
    elif self.length == self.maxlen:
      # No space, "remove" the first item.
      self.start = (self.start + 1) % self.maxlen
    else:
      # This should never happen.
      raise RuntimeError()
    self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
  x = np.array(x)
  if x.ndim >= 2:
    return x
  return x.reshape(-1, 1)


#from open ai baseline.
class Memory(object):
  def __init__(self, limit, action_shape, observation_shape):
    self.limit = limit

    self.observations0 = RingBuffer(limit, shape=observation_shape)
    self.actions = RingBuffer(limit, shape=action_shape)
    self.rewards = RingBuffer(limit, shape=(1,))
    self.terminals1 = RingBuffer(limit, shape=(1,))
    self.observations1 = RingBuffer(limit, shape=observation_shape)

  def sample(self, batch_size):
    # Draw such that we always have a proceeding element.
    batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

    obs0_batch = self.observations0.get_batch(batch_idxs)
    obs1_batch = self.observations1.get_batch(batch_idxs)
    action_batch = self.actions.get_batch(batch_idxs)
    reward_batch = self.rewards.get_batch(batch_idxs)
    terminal1_batch = self.terminals1.get_batch(batch_idxs)

    result = {
      'obs0': array_min2d(obs0_batch),
      'obs1': array_min2d(obs1_batch),
      'rewards': array_min2d(reward_batch),
      'actions': array_min2d(action_batch),
      'terminals1': array_min2d(terminal1_batch),
    }
    return result

  def append(self, obs0, action, reward, obs1, terminal1, training=True):
    if not training:
      return

    self.observations0.append(obs0)
    self.actions.append(action)
    self.rewards.append(reward)
    self.observations1.append(obs1)
    self.terminals1.append(terminal1)

  @property
  def nb_entries(self):
    return len(self.observations0)
