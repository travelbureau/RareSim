import gym
import numpy as np
import os
import sys
import tensorflow as tf
import zmq

sys.path.append('./simulator')
from Scenario import ScenarioWrapper
sys.path.append('./Utils')
import EnvUtils
import Utils

sys.path.append('./Agent')
from agent import Agent


def build_observation_indices(observation_size, glob_indices):
    l = []
    for i in xrange(observation_size):
        if i not in glob_indices:
            l.append(i)
    return np.asarray(l, dtype=np.int32)

env_dict, args = EnvUtils.makeEnvArgs()
env_id = "Pseudo-v0"
ScenarioWrapper.set_args(env_dict)
gym.envs.register(id=env_id, entry_point='Scenario:ScenarioWrapper')
env = gym.make(env_id)
source_worker_socket, worker_sink_socket, source_worker_direct_socket, context = \
    Utils.openSearchSocket(args.source_worker_port,
                           args.worker_sink_port,
                           args.source_worker_direct_port)
observation_size_agent = 51
observation_size = 55
action_size = 2
glob_indices = [8,9,10,11]
core_indices = [0,1,2,3,4,5,6,7,8,9,10,11]
obs_ind = build_observation_indices(observation_size, glob_indices)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.which_gpu)
agent = Agent(env, observation_size_agent, action_size,
              './Agent/expert_vals.npy', core_indices, obs_ind, 11,
                0,
                './Agent/snapshots/forward_policy', './Agent/snapshots/ego_policy')
feed_dict={ agent.do_keep_prob: 1.}
source_worker_direct_socket.recv()
source_worker_direct_socket.send("Ack")


isfirstrun = True
x0 = lambda: None
while True:
    tf.set_random_seed(12345)
    np.random.seed(12345)

    try:
        s = source_worker_direct_socket.recv(flags=zmq.NOBLOCK)
        source_worker_direct_socket.send("Ack")
        assert(s=="end")
        break
    except zmq.Again as e:
        pass
    try:
        wholething = Utils.recv_array(source_worker_socket, flags=zmq.NOBLOCK)
    except zmq.Again as e:
        continue

    stuffsize = 4
    numcars = len(wholething-2)/(stuffsize+1)
    index = int(wholething[-2])
    run = int(wholething[-1])
    temp = np.reshape(wholething[:-2], [stuffsize+1, numcars])
    x0.v = temp[0]
    x0.w = temp[1]
    x0.s = temp[2]
    x0.t = temp[3]
    x0.lane = temp[4].astype(int)

    assert(numcars==3)
    numcars = 2*numcars
    x0.v = np.append(x0.v, x0.v + np.asarray([0,-2,-1]))
    x0.w = np.append(x0.w, x0.w + np.asarray([0,0,0]))
    x0.s = np.append(x0.s, x0.s + np.asarray([20,27,22]))
    x0.t = np.append(x0.t, x0.t + np.asarray([0,0,0]))
    x0.lane = np.append(x0.lane, x0.lane)
    x0.index = index
    x0.run = run

    x0.s[x0.lane==1]+=0.5
    x0.s[x0.lane==2]+=14.
    x0.s[x0.lane==3]-=0.15
    x0.s[x0.lane==5]-=1.

    if isfirstrun:
        x0.numv = numcars
        x0.lengths = np.array([4.0]*x0.numv)
        x0.widths = np.array([2.0]*x0.numv)
        x0.types = np.array([1]*x0.numv)
        initial_gru_state = np.ones((x0.numv, agent.forward_model.encoding_size), dtype=np.float32)
        feed_dict[agent.gru_states] = initial_gru_state
        isfirstrun = False

    env.prepare_for_reset(x0)
    obs = env.reset()
    core_env0 = (obs[:,core_indices]).astype(np.float32)
    obs0 = (obs[:,obs_ind]).astype(np.float32)
    feed_dict[agent.states] = obs0
    feed_dict[agent.core_env_loop] = core_env0

    out = agent.sess.run(agent.loop_outputs, feed_dict = feed_dict)
    temp = np.asarray([out[3], index])
    env.finish_scene()
    print temp, out[2]
    Utils.send_array(worker_sink_socket, temp)

context.destroy()

