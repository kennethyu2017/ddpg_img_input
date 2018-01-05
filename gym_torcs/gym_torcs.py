import collections as col
import copy
import os
import time

import numpy as np
from gym import spaces
from gym.envs.registration import EnvSpec

import gym_torcs.snakeoil3_gym as snakeoil3
import math

pi = 3.1416


def end():
  os.system('pkill torcs')


def obs_vision_to_image_rgb(obs_image_vec):
  image_vec = obs_image_vec
  rgb = []
  temp = []
  # convert size 64x64x3 = 12288 to 64x64=4096 2-D list
  # with rgb values grouped together.
  # Format similar to the observation in openai gym
  for i in range(0, 12286, 3):
    temp.append(image_vec[i])
    temp.append(image_vec[i + 1])
    temp.append(image_vec[i + 2])
    rgb.append(temp)
    temp = []
  return np.array(rgb, dtype=np.uint8)


class TorcsEnv:

  def __init__(self, vision=False, throttle=False, gear_change=False,port=3101):
    self.vision = vision
    self.throttle = throttle
    self.gear_change = gear_change
    self.port=port

    self.initial_run = True
    self.observation = None
    self.client = None
    self.time_step = 0
    self.last_u = None

    # self.default_speed =50  #km/h
    self.default_speed =200  #km/h

    # self.terminal_judge_start = 100  # Speed limit is applied after this step
    self.low_progress_cnt = 0
    # self.low_progress=0.01  # meters
    self.low_progress=0.4 # meters/20ms
    # self.low_progress=1  # meters
    self.low_progress_term_steps=50 # term if no progress along track made after 100 steps.
    # self.low_progress_term_steps=500 # term if no progress along track made after 500 steps.
    # self.termination_limit_progress = 5 /self.default_speed  # normalized .episode terminates if car is running slower than this limit
    # self.low_progress = 3 # km/h along the track.


    self.initial_reset = True

    ##print("launch torcs")
    os.system('pkill torcs')
    time.sleep(0.5)
    if self.vision is True:
      os.system('torcs  -nofuel -nodamage -nolaptime  -vision  &')
    else:
      os.system('torcs  -nofuel -nodamage -nolaptime -T &')
    time.sleep(0.5)
    os.system('sh /home/yuheng/PycharmProjects/rl/kenneth_ddpg/ddpg_add_low_dim/gym_torcs/autostart.sh')
    time.sleep(0.5)

    """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=self.port, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
    if throttle is False:
      self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
    else:
      # kenneth. #
      #   self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
      # steering, accel, brake.
      high = np.array([1., 1., 1.])
      low = np.array([-1., 0., 0.])
      self.action_space = spaces.Box(low=low, high=high)


    # TODO. kenneth. define the accurate obs space for #
    # vision case. ref below. #
    if vision is False:
      # ['speedX', 'speedY', 'speedZ', 'angle',
      #  'rpm',
      #  'track',
      #  'trackPos',
      #  'wheelSpinVel'] we only choose 9 params with 29 dims total:
      high = np.array([np.inf] * 29)
      low = np.array([-np.inf] * 29)
      self.observation_space = spaces.Box(low=low, high=high)
    else:  #kenneth. still puse 29 for low dim eval.
      high = np.array([np.inf] * 29)
      low = np.array([-np.inf] * 29)
      self.observation_space = spaces.Box(low=low, high=high)

    # high = np.array([-255])
    # low = np.array([255])
    # self.reward_range = spaces.Box(low=low, high=high)

    # self.metadata = {'render.modes': ['rgb_array']}
    #malfold
    # self.spec = EnvSpec('Breakout-v0')


  def _step(self, u):
    # print("Step")
    # convert thisAction to the actual torcs actionstr
    client = self.client

    this_action = self.agent_to_torcs(u)

    # Apply Action
    action_torcs = client.R.d

    # Steering
    action_torcs['steer'] = this_action['steer']  # in [-1, 1]

    #  Simple Autnmatic Throttle Control by Snakeoil
    if self.throttle is False:
      target_speed = self.default_speed
      if client.S.d['speedX'] < target_speed - (client.R.d['steer'] * 50):
        client.R.d['accel'] += .01
      else:
        client.R.d['accel'] -= .01

      if client.R.d['accel'] > 0.2:
        client.R.d['accel'] = 0.2

      if client.S.d['speedX'] < 10:
        client.R.d['accel'] += 1 / (client.S.d['speedX'] + .1)

      # Traction Control System
      if ((client.S.d['wheelSpinVel'][2] + client.S.d['wheelSpinVel'][3]) -
            (client.S.d['wheelSpinVel'][0] + client.S.d['wheelSpinVel'][1]) > 5):
        action_torcs['accel'] -= .2
    else:
      action_torcs['accel'] = this_action['accel']  # [0,1] #
      action_torcs['brake'] = this_action['brake']  # [0,1] #

    # Automatic Gear Change by Snakeoil
    if self.gear_change is True:
      action_torcs['gear'] = this_action['gear']
    else:
      #  Automatic Gear Change by Snakeoil is possible
      action_torcs['gear'] = 1
      if self.throttle:
            if client.S.d['speedX'] > 50:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 80:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 110:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 140:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 170:
                action_torcs['gear'] = 6
    # Save the privious full-obs from torcs for the reward calculation
    obs_pre = copy.deepcopy(client.S.d)

    # One-Step Dynamics Update #################################
    # Apply the Agent's action into torcs
    client.respond_to_server()
    # Get the response of TORCS
    # !!! blocking wait for the sensory update from server_bot.!!!
    # race_server update sensory_info/20ms to server_bot -> client.
    # so looped step() executions average interval can not be less than 20ms,
    # i.e. we can not send action to server_bot averaging faster than 50Hz.
    client.get_servers_input()

    # Get the current full-observation from torcs
    obs = client.S.d

    # Make an obsevation from a raw observation vector from TORCS
    ## Note: this state is after the race_server execute the actions.
    ## so it should be s_t+1.
    self.observation = self.make_observaton(obs)

    # Reward setting Here #######################################
    # direction-dependent positive reward
    track = np.array(obs['track'])
    trackPos = np.array(obs['trackPos'])
    sp = np.array(obs['speedX'])  #un-normalized velocity
    damage = np.array(obs['damage'])
    rpm = np.array(obs['rpm'])
    dist=obs['distFromStart']  # dist from start line along the track line.

    # progress = sp*np.cos(obs['angle'])
    # kenneth. modify according to yanpanlau.
    # TODO.maybe dont use sin(angle) according to DDPG paper.
    # tf.logging.info('@@@@@ sp')
    #TODO kenneth. try normalized speed.
    # norm_sp = self.observation.speedX #normalized speed.

    # progress = norm_sp * np.cos(obs['angle'])  #normalized speed.
    # un-normalized speedX.
    # print('++++++++ dist diff:{}'.format(obs['distFromStart'] - obs_pre['distFromStart']))
    # print('++++++++ total speed :{}'.format(obs['distFromStart']/obs['curLapTime']))
    progress = sp * np.cos(obs['angle']) - np.abs(sp * np.sin(obs['angle']))
               # +obs['distFromStart'] / obs['curLapTime']
               # + (obs['distFromStart'] - obs_pre['distFromStart'])

    #TODO use normalized speed, maybe more stable.
    # progress = norm_sp * math.cos(obs['angle']) - abs(norm_sp * math.sin(obs['angle']))

    reward = progress
    episode_terminate = False

    # collision detection. not term.
    if obs['damage'] - obs_pre['damage'] > 0:
      reward = -1
      # print('++++ collision +++++')
      # print(' +++++ if out of track :', obs['trackPos'])
      # #TODO.kenneth .term after collision
      # episode_terminate = True
      # client.R.d['meta'] = True


    # Termination judgement #########################
    # episode_terminate = False
    #TODO should term?? when -T mode, when collision , always out of track, but we dont
    # want to always reset to start line. drive as far as possible. use collision penalty
    # instead.
    # if abs(obs['trackPos']) > 1:  # Episode is terminated if the car is out of track
    #   reward = - 1
    #   print('++++++++++++++++++++++++++ out of track term ++++++++++++++++')
    #   episode_terminate = True
    #   client.R.d['meta'] = True

    #TODO dont term , penalty only ,and we drive as far as possible.
    # term if no progress along the track after limit consecutive steps:
    # print('+++ dist:', obs['distFromStart'])
    if obs['distFromStart'] - obs_pre['distFromStart'] > self.low_progress:
      self.low_progress_cnt = 0 #clear
    else:
      # print('++++ low progress +++++')
      self.low_progress_cnt += 1

    #TODO not term . go as far as possible.? or term to avoid low speed T dominate.
    # allow some time to tolerate no progress and maybe the robot can adjust to find speed up.
    if self.low_progress_cnt == self.low_progress_term_steps:
      reward = - 1
      self.low_progress_cnt = 0 #clear
      print('++++++++++++++++++++++++++++++++ low progress , penalty -1 +++++++++++++++++')
      episode_terminate = True
      client.R.d['meta'] = True

    #TODO should term? or just low reward?
    # if self.terminal_judge_start < self.time_step:  # Episode terminates if the progress of agent is small
    #   if progress < self.termination_limit_progress: #normalized data
    #     # TODO kenneth .add reward penalty -1 . or let it just be equal to progress??#
    #     reward = -1
    #     episode_terminate = True
    #     client.R.d['meta'] = True

    # use no progress judgement to cover.
    if np.cos(obs['angle']) < 0:  # Episode is terminated if the agent runs backward
      # kenneth. no need make reward=-1, cause the progress give the negative reward?
      #TODO progress is tiny, can not change steering. so we use -1.
      # reward=-1
      print('++++++++++++++++++++++++++++ backward , term +++++++++++++++++')
      episode_terminate = True
      client.R.d['meta'] = True

    if client.R.d['meta'] is True:  # Send a reset signal
      self.initial_run = False
      client.respond_to_server()

    self.time_step += 1

    return self.get_obs(), reward, client.R.d['meta'], {}

  def _reset(self, relaunch=False):
    # print("Reset")

    self.time_step = 0

    if self.initial_reset is not True:
      self.client.R.d['meta'] = True
      self.client.respond_to_server()

      ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
      if relaunch is True:
        self.reset_torcs()
        print("### TORCS is RELAUNCHED ###")

    # Modify here if you use multiple tracks in the environment
    self.client = snakeoil3.Client(p=self.port, vision=self.vision)  # Open new UDP in vtorcs
    self.client.MAX_STEPS = np.inf

    client = self.client
    client.get_servers_input()  # Get the initial input from torcs

    obs = client.S.d  # Get the current full-observation from torcs
    self.observation = self.make_observaton(obs)

    self.last_u = None

    self.initial_reset = False
    return self.get_obs()

  def get_obs(self):
    return self.observation

  def reset_torcs(self):
    # print("relaunch torcs")
    os.system('pkill torcs')
    time.sleep(0.5)
    if self.vision is True:
      os.system('torcs -nofuel -nodamage -nolaptime -vision &')
    else:
      os.system('torcs -nofuel -nodamage -nolaptime -T &')
      # os.system('torcs -nofuel -nodamage -nolaptime &')
    time.sleep(0.5)
    # kenneth
    os.system('sh /home/yuheng/PycharmProjects/rl/kenneth_ddpg/ddpg_add_low_dim/gym_torcs/autostart.sh')
    time.sleep(0.5)

  def agent_to_torcs(self, u):
    torcs_action = {'steer': u[0]}

    if self.throttle is True:  # throttle action is enabled
      torcs_action.update({'accel': u[1]})
      torcs_action.update({'brake': u[2]})

    if self.gear_change is True:  # gear change action is enabled
      torcs_action.update({'gear': int(u[3])})

    return torcs_action

  def make_observaton(self, raw_obs):
    if self.vision is False:
      names = ['focus',
               'speedX', 'speedY', 'speedZ', 'angle', 'damage',
               'opponents',
               'rpm',
               'track',
               'trackPos',
               'wheelSpinVel']
      Observation = col.namedtuple('Observaion', names)
      # print('@@@ ++++++++++++++++++++++++ raw spx:{} default_speed :{}'.format(raw_obs['speedX'],self.default_speed))

      #TODO kenneth . normalize to [-1,+1]
      return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
                         speedX=np.array(raw_obs['speedX'], dtype=np.float32) / self.default_speed,
                         speedY=np.array(raw_obs['speedY'], dtype=np.float32) / self.default_speed,
                         speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / self.default_speed,
                         angle=np.array(raw_obs['angle'], dtype=np.float32) / pi,
                         damage=np.array(raw_obs['damage'], dtype=np.float32),
                         opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
                         rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                         track=np.array(raw_obs['track'], dtype=np.float32) / 200.,
                         trackPos=np.array(raw_obs['trackPos'], dtype=np.float32),
                         wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32)/1000. )
    else:
      names = ['focus',
               'speedX', 'speedY', 'speedZ',
               'angle',
               'damage',
               'opponents',
               'rpm',
               'track',
               'trackPos',
               'wheelSpinVel',
               'img']
      Observation = col.namedtuple('Observaion', names)

      # Get RGB from observation
      # kennth. names[11] is img.
      image_rgb = obs_vision_to_image_rgb(raw_obs[names[-1]])

      # kenneth. add angle,  trackPos,damage.
      # note : same normalized magnititude as no vision.
      return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
                         speedX=np.array(raw_obs['speedX'], dtype=np.float32) / self.default_speed,
                         speedY=np.array(raw_obs['speedY'], dtype=np.float32) / self.default_speed,
                         speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / self.default_speed,
                         angle=np.array(raw_obs['angle'], dtype=np.float32) / pi,
                         damage=np.array(raw_obs['damage'], dtype=np.float32),
                         opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
                         rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                         track=np.array(raw_obs['track'], dtype=np.float32) / 200.,
                         trackPos=np.array(raw_obs['trackPos'], dtype=np.float32) / 1.,
                         wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32)/1000,
                         img=image_rgb)

  # kenneth  add for gym.wrapper.
  # def render(self, close=False, mode='rgb_array'):
  #   if mode == 'rgb_array':
  #     # gym.wrapper require (w,h,3) shape.
  #     # cause gym.wrapper call render after each step, so we dont need to receive new obs.
  #     # just use the current obs.
  #     return self.observation.img.reshape(64, 64, -1)
  #   else:
  #     raise ValueError('unsupported render mode:{}'.format(mode))
