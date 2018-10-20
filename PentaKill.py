from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np

import os
import sys
from environment_static_2 import environment

from Model import AC

use_cuda = False  # set to True if training with GPU

maps = ["map_54.npy", "map_61.npy", "map_62.npy"]
games = {"map_54.npy":[[20,200]], "map_61.npy":[[90,300]] ,"map_62.npy":[[70, 300]]}

ENV = environment(maps=maps, map_size=(50, 50), games=games, only_when_success=True, digital=False, reward_type="two", way_back=False, running_reward=True,
                 running_reward_interval=100)

action_dic = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
GAMMA = 0.99
max_time = 100

# actor = Actor(A_DIM=8)
# critic = Critic(GAMMA)
ac = AC(A_DIM=8, gamma=GAMMA)

if __name__ == '__main__':

    # define input data and variable
    episode = 0
    success_counter = 0
    # actor.build_net()  # build net first
    # critic.build_net()
    ac.build_net()

    while True:

        current_state, location_information, mask = ENV.reset(plot=True)

        current_state = np.stack([current_state, location_information[0], location_information[1]], axis=0)

        current_state_record = []
        next_state_record = []
        reward_record = []
        td_error_record = []
        final_reward = 0

        for step in range(max_time):


            action_pros = ac.act(current_state)  # get real action
            action_pros = np.array(action_pros[0]) * mask
            print(action_pros.ravel())
            print(np.sum(action_pros.ravel()))
            action = np.random.choice(range(action_pros.shape[1]), p=action_pros.ravel()/np.sum(action_pros.ravel()))  # generate distribution
            real_action = action_dic[action]  # get real action

            # next_state, reward, done, info, success = ENV.step(real_action)
            if step == max_time - 1:
                next_state, location_information, mask, r, success, f_r, running_mean_reward = ENV.step(real_action, last_step=True)
            else:
                next_state, location_information, mask, r, success, f_r, running_mean_reward = ENV.step(real_action)

            if (step == max_time - 1) or success:
                if running_mean_reward:
                    final_reward = f_r - running_mean_reward
                else:
                    final_reward = f_r

            next_state = np.stack([next_state, location_information[0], location_information[1]], axis=0)

            current_state_record.append(current_state)
            next_state_record.append(next_state)
            reward_record.append(r)

            if success:
                success_counter += 1
                break

            current_state = next_state

        if final_reward:
            reward_record = [(reward_tmp + final_reward) for reward_tmp in reward_record]

            # print("reward is {0}".format(reward))

        for step in range(max_time):

            current_state = current_state_record[step]
            next_state = next_state_record[step]
            reward = reward_record[step]

            td_error = ac.get_td_error(current_state, next_state, reward)
            td_error_record.append(td_error)

        # current_state_record = np.stack(current_state_record)
        # next_state_record = np.stack(next_state_record)
        # reward_record = np.stack(reward_record)
        # td_error_record = np.stack(td_error_record)
        for step in range(max_time):
            ac.train('c', current_state=current_state_record[step], next_state=next_state_record[step], reward=reward_record[step], td_error=td_error_record[step])
            print("critic train!")

            #   Actor Learn
            ac.train('a', current_state=current_state_record[step], next_state=next_state_record[step], reward=reward_record[step], td_error=td_error_record[step])
            print("actor train!")
        print('Episode: {} success times:{}'.format(episode, success_counter))





























