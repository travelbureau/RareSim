import numpy as np
from Roadway import Roadway
from Vehicle import Vehicle
from Visualizer import Visualizer
import gym
import os
import gym.spaces
import numba
from matplotlib.animation import FFMpegWriter


class Scenario(gym.Env):

    OBSERVATION_SIZE = 55
    ACTION_SIZE = 2
    ACT_LO = np.asarray([-5.0, -1.0])
    ACT_HI = np.asarray([3.0, 1.0])
    metadata = {}

    def __init__(self, write_log=True, write_frames=False, frame_prefix='', join_index_and_run_in_log=True,
                 reactive_threshold_brake=-10.,
                 use_reactive=False, model_all=True,
                 nsteps=100):
        self.write_log = write_log
        self.write_frames = write_frames
        if self.write_frames and not self.write_log:
            print 'Have to write log to write frames'
            assert(1 == 0)
        self.frame_prefix = frame_prefix
        self.join_index_and_run_in_log = join_index_and_run_in_log

        self.reactive_threshold_brake = reactive_threshold_brake
        self.use_reactive = use_reactive
        self.model_all = model_all
        self.nsteps = nsteps

        self.road = Roadway()
        self.ego_id = None
        self.num_cars = None
        self.index = None
        self.run = None
        self.State_record = []
        self.State = np.asarray([])  # X,Y,Theta,V,S,T,Phi,Lane
        self.reactive_ids = np.asarray([])
        self.reactive_desired_speeds = np.asarray([])
        self.VehicleTypes = np.asarray([])
        self.LengthWidth = np.asarray([])
        self.Radii = np.asarray([])
        self.reset_info = None

    @property
    def action_space(self):
        return gym.spaces.Box(np.array(Scenario.ACT_LO), np.array(Scenario.ACT_HI))

    @property
    def observation_space(self):
        temp = np.full((Scenario.OBSERVATION_SIZE,), np.inf)
        return gym.spaces.Box(-temp, temp)

    def prepare_for_reset(self, x0):
        reset_info = {}
        reset_info['ego_id'] = 0
        reset_info['num_cars'] = len(x0.v)
        reset_info['index'] = x0.index
        reset_info['run'] = x0.run
        reset_info['static_vehicle_data'] = np.stack([x0.types, x0.lengths, x0.widths], axis=1)
        reset_info['vehicle_states'] = self.road.local2global(x0.v, x0.w, x0.s, x0.t, x0.lane)
        self.reset_info = reset_info

    def reset(self):
        if not self.reset_info:
            print 'Must set reset_info'
        self.State_record = []

        self.ego_id = self.reset_info['ego_id']
        self.num_cars = self.reset_info['num_cars']
        self.reactive_ids = np.zeros(self.num_cars, dtype=bool)
        self.index = self.reset_info['index']
        self.run = self.reset_info['run']
        geoms = self.reset_info['static_vehicle_data']
        self.VehicleTypes = np.copy(geoms[:, 0])
        self.LengthWidth = np.copy(geoms[:, 1:])
        self.Radii = np.linalg.norm(self.LengthWidth/2., axis=1, ord=2)
        self.State = np.copy(self.reset_info['vehicle_states'])
        if self.write_log:
            self.State_record.append(np.copy(self.State))
        self.reactive_desired_speeds = np.copy(self.State[:, 3])

        self.VehicleTypes.setflags(write=False)
        self.LengthWidth.setflags(write=False)
        self.Radii.setflags(write=False)
        self.reset_info = None
        Obs = self._observe()
        return Obs

    def finish_scene(self):
        if (not self.write_log) and (not self.write_frames):
            return
        if self.join_index_and_run_in_log:
            name = self.frame_prefix + '%06d' % (self.index+self.run)
        else:
            name = self.frame_prefix + '%03d_%03d' % (self.index, self.run)
        # save log
        if self.write_log:
            filename = name + '.npz'
            with open(filename, 'wb') as file:
                np.savez_compressed(file,
                                    State_record=np.asarray(self.State_record),
                                    ego_id=self.ego_id,
                                    LengthWidth=self.LengthWidth,
                                    num_cars=self.num_cars,
                                    index=self.index,
                                    run=self.run,
                                    name=name)
        # make movie
        if self.write_frames:
            filename = name + '.mp4'
            self.visualizer = Visualizer(self.ego_id,
                                         self.LengthWidth[:, 0], self.LengthWidth[:, 1],
                                         self.num_cars,
                                         self.road)
            metadata = dict(title=name, artist='Pseudo', comment='')
            writer = FFMpegWriter(fps=2*int(1./Vehicle.DEL_T), codec='libx264',
                                  metadata=metadata,
                                  extra_args=['-pix_fmt', 'yuv420p'])
            with writer.saving(self.visualizer.fig,
                               filename,
                               self.visualizer.fig.dpi):
                for state in self.State_record:
                    self.visualizer.update(state[:, :3])
                    writer.grab_frame()
            Scenario.flip_video(filename)

    def write_frames_from_log(self, logfile):
        stuff = np.load(logfile)
        filename = str(stuff['name']) + '.mp4'
        self.visualizer = Visualizer(int(stuff['ego_id']),
                                     stuff['LengthWidth'][:, 0], stuff['LengthWidth'][:, 1],
                                     int(stuff['num_cars']),
                                     self.road)
        metadata = dict(title=str(stuff['name']), artist='Pseudo', comment='')
        writer = FFMpegWriter(fps=2*int(1./Vehicle.DEL_T), codec='libx264',
                              metadata=metadata,
                              extra_args=['-pix_fmt', 'yuv420p'])
        with writer.saving(self.visualizer.fig,
                           filename,
                           self.visualizer.fig.dpi):
            for state in stuff['State_record']:
                self.visualizer.update(state[:, :3])
                writer.grab_frame()
        Scenario.flip_video(filename)

    @staticmethod
    def flip_video(filename):
        os.system('ffmpeg -y -hide_banner -loglevel panic -i ' + filename + ' -vf \"transpose=1\" ' + filename[:-4] + '_rot.mp4')

    def step(self, Action):
        Action = np.clip(Action, Scenario.ACT_LO*10, Scenario.ACT_HI*10)
        if self.model_all and not self.use_reactive:
            Vehicle.do_pointmass_dynamics(self.State, Action)

        elif self.model_all and self.use_reactive:
            # set reactive speeds
            V = self.State[:, 3]
            non_idm = np.ones(self.num_cars, dtype=bool)
            non_idm[self.reactive_ids] = False
            self.reactive_desired_speeds[non_idm] = np.copy(V[non_idm])
            # see if anybody should become reactive if they aren't already
            reactive_actions = Vehicle.get_reactive_actions(self.State,
                                                            self.LengthWidth[:, 0],
                                                            self.reactive_desired_speeds)
            actions, self.reactive_ids = _substep(reactive_actions,
                                                  self.ego_id,
                                                  self.reactive_threshold_brake,
                                                  V,
                                                  self.reactive_ids)
            Action[self.reactive_ids, :] = actions
            Vehicle.do_pointmass_dynamics(self.State, Action)

        else:
            print 'Must set model_all to be true for now'
            assert(1 == 0)

        Obs = self._observe()
        if self.write_log:
            self.State_record.append(np.copy(self.State))
        reward = self._reward()
        # ego collision, or ego offroad or ego reversing
        done = np.sum(Obs[self.ego_id, 12:15]) > 0.1  # safe check for > 0
        # convert the offroad non-ego guys to idm for the next step
        self.reactive_ids = _offroad_idm_conversion(self.reactive_ids, Obs[:, 13], self.ego_id)
        info = {}

        return Obs, reward, done, info

    def _observe(self):
        # needs road, X,Y,Theta,V, LengthWidth, Radii, num_cars set
        # uses old s and old lanes - S0, Lanes0
        Poses = self.State[:, :2]
        Theta = self.State[:, 2]
        V = self.State[:, 3]
        Lengths = self.LengthWidth[:, 0]
        Widths = self.LengthWidth[:, 1]
        S0 = self.State[:, 4]
        Lanes0 = self.State[:, 7]

        road_features = self.road.get_road_features(Poses,
                                                    Theta,
                                                    S0,
                                                    Lanes0,
                                                    Lengths,
                                                    Widths,
                                                    self.Radii)
        BoundingBoxes = Vehicle.get_bounding_boxes(Poses, Theta, Lengths, Widths)

        collision_features = Vehicle.get_collisions(Poses, self.Radii, BoundingBoxes)
        lidar_features = []
        for vehicle_id in xrange(self.num_cars):
            lidar_features.append(Vehicle.get_lidar_features(vehicle_id,
                                                             Poses[:, 0],
                                                             Poses[:, 1],
                                                             Theta,
                                                             V,
                                                             BoundingBoxes))
        lidar_features = np.vstack(lidar_features)

        # set S, T, Phi, Lane in State
        self.State[:, 4] = road_features[:, 7]
        self.State[:, 5] = road_features[:, 0]
        self.State[:, 6] = road_features[:, 1]
        self.State[:, 7] = road_features[:, 5]

        Obs = np.concatenate([road_features[:, :2],
                              V[:, None],
                              self.LengthWidth,
                              road_features[:, 2:6],
                              self.State[:, :3],
                              collision_features[:, None],
                              road_features[:, 6, None],
                              (V[:, None] < 0).astype(np.float),
                              lidar_features], axis=1)
        return Obs

    # stub
    def _reward(self):
        return 0

    # stub
    def render(self, close=False):
        return


@numba.njit
def _substep(reactive_actions, ego_id, brake_threshold, V, reactive_ids):
    idx = reactive_actions[:, 0] < brake_threshold
    idx[ego_id] = False
    reactive_ids = np.logical_or(reactive_ids, idx)
    idx = np.logical_and(V < 0, reactive_ids)
    reactive_actions[idx, 0] = np.maximum(reactive_actions[idx, 0], 0)
    return reactive_actions[reactive_ids, :], reactive_ids


@numba.njit
def _offroad_idm_conversion(reactive_ids, is_offroad, ego_id):
    idx = is_offroad > 0.1  # safe chck for greater than zero
    idx[ego_id] = False
    return np.logical_or(reactive_ids, idx)


# because gym can't make an env have a custom constructor with arguments
class ScenarioWrapper(Scenario):
    _args = {}

    def __init__(self):
        super(ScenarioWrapper, self).__init__(ScenarioWrapper._args['write_log'],
                                              ScenarioWrapper._args['write_frames'],
                                              ScenarioWrapper._args['frame_prefix'],
                                              ScenarioWrapper._args['join_index_and_run_in_log'],
                                              ScenarioWrapper._args['reactive_threshold_brake'],
                                              ScenarioWrapper._args['use_reactive'],
                                              ScenarioWrapper._args['model_all'],
                                              ScenarioWrapper._args['nsteps'])

    @classmethod
    def set_args(cls, args):
        cls._args = args
