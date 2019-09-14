import numpy as np
import os
import sys
import tensorflow as tf

from forward_model import ForwardModel
from policy import Policy
sys.path.append('./Utils')
import EnvUtils


def re_parametrization(state_e, state_a):
    nu = state_e - state_a
    nu = tf.stop_gradient(nu)
    return state_a + nu, nu

def normalize(x, mean, std):
    return (x - mean)/std

def denormalize(x, mean, std):
    return x * std + mean


class Agent(object):
    def __init__(self, environment, observation_size, action_size, expert_vals_file, core_indices, obs_ind, history_length,
                    which_gpu, snapshot_file, ego_snapshot_file):
        gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(which_gpu))
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.env = environment
        self.sigma = 1.
        self.do_keep_prob = tf.placeholder(tf.float32, shape=(), name='do_keep_prob')
        self.states = tf.placeholder(tf.float32, shape=(None, observation_size), name='states')
        self.gru_states = tf.placeholder(tf.float32, shape=(None, 150), name='gru_states')
        self.core_env_loop = tf.placeholder(tf.float32, shape=(None, len(core_indices)), name='core_env_loop')
        self.forward_model = ForwardModel(state_size=observation_size,
                                          action_size=action_size,
                                          encoding_size=150)
        self.policy = Policy(in_dim=observation_size,
                              out_dim=action_size,
                              size=[200, 100],
                              do_keep_prob=self.do_keep_prob,
                              use_separate_ego=1)
        expert = np.load(expert_vals_file)
        esm = expert[0]
        ess = expert[1]
        eam = expert[2]
        eas = expert[3]
        self.expert_state_mean = esm
        self.expert_state_std = ess
        self.expert_action_mean = eam
        self.expert_action_std = eas
        self.total_trans_err_allowed = 1000.
        self.core_indices = core_indices
        self.obs_ind = obs_ind
        self.history_length = history_length

        def policy_cost_dist(state_):
            state = denormalize(state_, self.expert_state_mean, self.expert_state_std)
            return EnvUtils.tf_ttc(state)

        def policy_stop_condition(state_, gru_state_, t, cost, trans_err, env_term_sig, core_env):
            cond = tf.logical_not(env_term_sig)
            cond = tf.logical_and(cond, t < self.env.nsteps)
            cond = tf.logical_and(cond, trans_err < self.total_trans_err_allowed)
            return cond

        def policy_loop_dummy(state_, gru_state_, t, total_cost, total_trans_err, env_term_sig, core_env):
            return state_, gru_state_, t, total_cost, total_trans_err, env_term_sig, core_env

        def policy_loop(state_, gru_state_, t, total_cost, total_trans_err, _, core_env_):
            mu, logstd = self.policy.forward(state_, reuse=tf.AUTO_REUSE)
            std = self.sigma * tf.exp(logstd)
            eta = tf.multiply(std, tf.random_normal(shape=tf.shape(mu)))
            action = mu + eta

            cost = policy_cost_dist(state_)
            total_cost = tf.minimum(total_cost, cost)
            a_sim = denormalize(action, self.expert_action_mean, self.expert_action_std)

            state_env, _, env_term_sig, = EnvUtils.tf_step(self.env, a_sim)
            core_env = tf.gather(state_env, axis=1, indices=self.core_indices)
            state_env = tf.gather(state_env, axis=1, indices=self.obs_ind)
            state_e = normalize(state_env, self.expert_state_mean, self.expert_state_std)
            state_e = tf.stop_gradient(state_e)

            if self.history_length == 1:
                state_a, _ = self.forward_model.forward([state_, action, initial_gru_state], reuse=tf.AUTO_REUSE)
                gru_state = initial_gru_state
            else:
                state_a, gru_state = self.forward_model.forward([state_, action, gru_state_], reuse=tf.AUTO_REUSE)

            state, nu = re_parametrization(state_e=state_e, state_a=state_a)
            total_trans_err += tf.reduce_mean(abs(nu))
            t += 1
            return state, gru_state, t, total_cost, total_trans_err, env_term_sig, core_env

        def manual_while_loop(states, gru_states, core_env_loop):
            MAXTIME = self.env.nsteps
            states_loop = states
            grustates_loop = gru_states
            t_loop = 0
            totalcost_loop = tf.float32.max
            transerr_loop = 0.
            envtermsig_loop = False
            #core_env_loop = 0.
            core_envs = []
            for _ in xrange(MAXTIME):
                keepgoing = policy_stop_condition(states_loop, grustates_loop, t_loop, totalcost_loop, transerr_loop, envtermsig_loop, core_env_loop)
                states_loop, grustates_loop, t_loop, totalcost_loop, transerr_loop, envtermsig_loop, core_env_loop = tf.cond(keepgoing,
                    lambda: policy_loop(states_loop, grustates_loop, t_loop, totalcost_loop, transerr_loop, envtermsig_loop, core_env_loop),
                    lambda: policy_loop_dummy(states_loop, grustates_loop, t_loop, totalcost_loop, transerr_loop, envtermsig_loop, core_env_loop))
                core_envs.append(core_env_loop)
            return states_loop, grustates_loop, t_loop, totalcost_loop, transerr_loop, envtermsig_loop, core_envs
        
        states_normalized = normalize(self.states, self.expert_state_mean, self.expert_state_std)
        self.loop_outputs = manual_while_loop(states_normalized, self.gru_states, self.core_env_loop)

        if snapshot_file is not None:
            collect = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='forward_model')
            saver = tf.train.Saver(var_list = collect)
            saver.restore(self.sess, snapshot_file)
            collect = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
            saver = tf.train.Saver(var_list = collect)
            saver.restore(self.sess, snapshot_file)
            collect = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ego_policy')
            saver = tf.train.Saver(var_list = collect)
            saver.restore(self.sess, ego_snapshot_file)
        else:
            self.sess.run(tf.global_variables_initializer())





