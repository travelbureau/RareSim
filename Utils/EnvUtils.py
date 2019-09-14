import argparse
import numpy as np
import tensorflow as tf

def tf_step(env, action):
    state, reward, done = tf.py_func(lambda action: _step(env, action), inp=[action],
                                     Tout=[tf.float32, tf.float32, tf.bool],
                                     name='env_step_func')
    state.set_shape([None, env.observation_space.shape[0]])
    done.set_shape(())
    return state, reward, done

def _step(env, action):
    action = np.squeeze(action)
    result = env.step(action)
    state, reward, done, info = result
    return np.float32(state), np.float32(reward), done

def tf_ttc(state):
    ranges = state[0,11:31]
    rangerates = state[0,31:]
    ranges = tf.divide(ranges, -1*rangerates)
    ranges = tf.where(ranges>0, ranges, 1./tf.zeros_like(ranges))
    cost = tf.reduce_min(ranges)
    return cost

def makeEnvArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_worker_port', type=int)
    parser.add_argument('--worker_sink_port', type=int)
    parser.add_argument('--source_worker_direct_port', type=int)
    parser.add_argument('--which_gpu', type=int)

    parser.add_argument('--write_frames', type=int)
    parser.add_argument('--write_log', type=int)
    parser.add_argument('--max_traj_len', type=int, default=100)
    parser.add_argument('--use_reactive', type=int, default=1)
    parser.add_argument('--model_all', type=int, default=1)
    parser.add_argument('--reactive_threshold_brake', type=float, default=-10.)
    parser.add_argument('--frame_prefix', default='')
    parser.add_argument('--join_index_and_run_in_log', type=int, default=1)

    args = parser.parse_args()
    args.write_frames = bool(args.write_frames)
    args.write_log = bool(args.write_log)
    args.use_reactive = bool(args.use_reactive)
    args.model_all = bool(args.model_all)
    args.join_index_and_run_in_log = bool(args.join_index_and_run_in_log)
    env_dict = {'write_log': args.write_log,
                'write_frames': args.write_frames,
                'frame_prefix': args.frame_prefix,
                'join_index_and_run_in_log': args.join_index_and_run_in_log,
                'reactive_threshold_brake': args.reactive_threshold_brake,
                'use_reactive': args.use_reactive,
                'model_all': args.model_all,
                'nsteps': args.max_traj_len
    }
    return env_dict, args