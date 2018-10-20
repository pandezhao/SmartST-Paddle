from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np
import sys

def conv_bn_layer(main_input, ch_out, filter_size, stride, padding, act='relu', name=None):
    conv = fluid.layers.conv2d(
        input=main_input,  # shape = [N,C,H,W]
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        use_cudnn=True,
        act=act,
        name=name
    )
    return conv

class AC(object):
    def __init__(self, A_DIM=8, gamma=0.94):
        self.A_DIM = A_DIM
        self.probs = None
        self.prob_program = None
        self.gamma = gamma
        self.reward = None
        self.td_error = None

        self.a_p = fluid.Program()
        self.c_p = fluid.Program()
        self.startup = fluid.Program()

    def act(self, state):
        with fluid.unique_name.guard():
            with fluid.program_guard(self.a_p, self.startup):
                state = np.array(state, dtype='float32')
                state = np.expand_dims(state, axis=0)
                return self.exe.run(self.prob_program,
                                    feed={'current_state': state},
                                    fetch_list=[self.probs])

    def get_td_error(self, current_state, next_state, reward):
        with fluid.unique_name.guard():
            with fluid.program_guard(self.c_p, self.startup):
                current_state = np.array(current_state, dtype='float32')
                current_state = np.expand_dims(current_state, axis=0)
                next_state = np.array(next_state, dtype='float32')
                next_state = np.expand_dims(next_state, axis=0)
                reward = np.expand_dims(np.array([reward], dtype='float32'), axis=0)
                return self.exe.run(self.td_program,
                                    feed={'current_state': current_state,
                                          'next_state': next_state,
                                          'reward': reward},
                                    fetch_list=[self.c_target])[0]

    def build_a(self, s):
        a_conv1 = conv_bn_layer(main_input=s, ch_out=8, filter_size=3, stride=2, padding=1,name='a_conv1')
        a_conv2 = conv_bn_layer(main_input=a_conv1, ch_out=16, filter_size=3, stride=2, padding=1,name='a_conv2')
        a_conv3 = conv_bn_layer(main_input=a_conv2, ch_out=32, filter_size=3, stride=1, padding=1,name='a_conv3')
        a_conv4 = conv_bn_layer(main_input=a_conv3, ch_out=32, filter_size=3, stride=1, padding=1,name='a_conv4')
        a_fc5 = fluid.layers.fc(input=a_conv4, size=50, act='relu',name='a_fc5')
        a_fc5 = fluid.layers.fc(input=a_conv4, size=50, act='relu', name='a_fc5')
        a_fc5 = fluid.layers.fc(input=a_conv4, size=50, act='relu', name='a_fc5')
        return fluid.layers.fc(input=a_fc5, size=self.A_DIM, act='softmax',name='a_out')

    def build_c(self, s):
        c_conv1 = conv_bn_layer(main_input=s, ch_out=32, filter_size=3, stride=2, padding=0,name='c_conv1')
        c_conv2 = conv_bn_layer(main_input=c_conv1, ch_out=32, filter_size=3, stride=2, padding=0,name='c_conv2')
        c_conv3 = conv_bn_layer(main_input=c_conv2, ch_out=32, filter_size=3, stride=1, padding=0,name='c_conv3')
        c_conv4 = conv_bn_layer(main_input=c_conv3, ch_out=32, filter_size=3, stride=1, padding=0,name='c_conv4')
        c_fc5 = fluid.layers.fc(input=c_conv4, size=256, act='relu',name='c_fc5')
        value = fluid.layers.fc(input=c_fc5, size=1,name='c_out')
        return value

    def build_net(self,):

        #######  Actor Part
        with fluid.unique_name.guard():
            with fluid.program_guard(self.a_p, self.startup):
                s = fluid.layers.data(name='current_state', shape=[3, 50, 50], dtype='float32')
                td_error = fluid.layers.data(name='td_error', shape=[1], dtype='float32')
                self.probs = self.build_a(s)
                # define actor optimizer
                self.prob_program = fluid.default_main_program().clone()  # 1*8
                lprobs = fluid.layers.log(self.probs)  # log operation 1*8
                log_prob = fluid.layers.reduce_max(lprobs, dim=1, keep_dim=True)

                a_target = fluid.layers.reduce_mean(log_prob * td_error * -1.0)

                optimizer_a = fluid.optimizer.Adam(learning_rate=0.001)
                optimizer_a.minimize(a_target)

                self.train_program_actor = fluid.default_main_program()

        #######  Critic Part
        with fluid.unique_name.guard():
            with fluid.program_guard(self.c_p, self.startup):
                s = fluid.layers.data(name='current_state', shape=[3, 50, 50], dtype='float32')
                s_ = fluid.layers.data(name='next_state', shape=[3, 50, 50], dtype='float32')
                reward = fluid.layers.data(name='reward', shape=[1], dtype='float32')
                reward = fluid.layers.clip(reward, min=-10.0, max=10.0)
                v_curr = self.build_c(s)
                v_next = self.build_c(s_)
                self.c_target = fluid.layers.reduce_mean(reward + self.gamma * v_next - v_curr)
                print(self.c_target)
                self.td_program = fluid.default_main_program().clone()  # 1*1
                print(self.td_program)
                # define while optimizer
                optimizer_c = fluid.optimizer.SGD(learning_rate=0.001)
                optimizer_c.minimize(self.c_target)

                self.train_program_critic = fluid.default_main_program()

        place = fluid.CPUPlace()
        self.exe = fluid.Executor(place)

        # fluid exe
        self.exe.run(self.startup)

    def train(self, flag, current_state=None, next_state=None, reward=None, td_error=None):
        current_state = np.array(current_state, dtype='float32')
        current_state = np.expand_dims(current_state, axis=0)
        print(current_state.shape)
        if flag == 'a':
            td_error = np.expand_dims(np.array(td_error, dtype='float32'), axis=0)
            print(td_error.shape)
            self.exe.run(self.a_p,
                         feed={
                             'current_state': current_state,
                             'td_error': td_error}
                         )

        elif flag == 'c':
            next_state = np.array(next_state, dtype='float32')
            next_state = np.expand_dims(next_state, axis=0)
            print(next_state.shape)
            reward = np.expand_dims(np.array([reward], dtype='float32'), axis=0)
            print(reward.shape)

            self.exe.run(self.c_p,
                         feed={
                             'current_state': current_state,
                             'next_state': next_state,
                             'reward': reward}
                         )
