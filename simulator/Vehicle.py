from enum import Enum
import numba
import numpy as np

import GJK


class Vehicle(object):

    DEL_T = 0.1

    LIDAR_NUM_BEAMS = 20
    LIDAR_MAXRANGE = 100.
    LIDAR_ANGLE_OFFSET = -np.pi

    LANETRACKER_KP = 0.5  # 2.0
    LANETRACKER_KD = 0.5
    LANETRACKER_SIGMA = 0.01

    IDM_KSPD = 1.0
    IDM_DELTA = 4.0
    IDM_HEADWAY = 0.5
    IDM_SMIN = 1.0
    IDM_AMAX = 3.0
    IDM_DCMF = 2.5
    IDM_DMAX = 9.0
    IDM_SIGMA = 0.1

    class VehicleType(Enum):
        CAR = 1
        TRUCK = 2
        MOTORCYCLE = 3

    def __init__(self):
        return

    @staticmethod
    def do_pointmass_dynamics(State, Action):
        # State is X,Y,Theta,Speed,S,T,Phi,Lane
        # Action is Throttle, Steer
        X = State[:, 0]
        Y = State[:, 1]
        Theta = State[:, 2]
        V = State[:, 3]
        Throttle = Action[:, 0]
        Steer = Action[:, 1]
        X += V*np.cos(Theta)*Vehicle.DEL_T
        Y += V*np.sin(Theta)*Vehicle.DEL_T
        Theta += Steer*Vehicle.DEL_T
        V += Throttle*Vehicle.DEL_T
        # sets new X,Y,Theta,V

    @staticmethod
    def get_reactive_actions(State, Length, V_des):
        out = _get_reactive_actions(State, Length, V_des,
            Vehicle.LANETRACKER_KP, Vehicle.LANETRACKER_KD, Vehicle.LANETRACKER_SIGMA,
            Vehicle.IDM_SMIN, Vehicle.IDM_HEADWAY, Vehicle.IDM_AMAX, Vehicle.IDM_DCMF, Vehicle.IDM_DELTA,
            Vehicle.IDM_KSPD, Vehicle.IDM_DMAX)
        return out

    @staticmethod
    def get_collisions(poses, radii, BoundingBoxes):
        return _get_collisions(poses, radii, BoundingBoxes)

    @staticmethod
    def get_bounding_boxes(poses, Theta, Lengths, Widths):
        x = (Lengths/2.)[:, None]*np.asarray([np.cos(Theta), np.sin(Theta)]).T
        y = (Widths/2.)[:, None]*np.asarray([-np.sin(Theta), np.cos(Theta)]).T
        corners = np.asarray([x-y, x+y, y-x, -x-y])
        return np.transpose(corners + poses[None, :, :], axes=[1, 0, 2])

    @staticmethod
    def get_lidar_features(vehicle_id, X, Y, Theta, V, BoundingBoxes):
        return _get_lidar_features(vehicle_id, X, Y, Theta, V, BoundingBoxes,
            Vehicle.LIDAR_NUM_BEAMS, Vehicle.LIDAR_ANGLE_OFFSET, Vehicle.LIDAR_MAXRANGE)


@numba.njit
def _get_reactive_actions(State, Length, V_des, KP, KD, SIGMA, SMIN, HEADWAY, AMAX, DCMF, DELTA, KSPD, DMAX):
    # State is X,Y,Theta,Speed,S,T,Phi,Lane
    # idm for throttle, pd for lane following
    V = State[:, 3]
    S = State[:, 4]
    T = State[:, 5]
    Phi = State[:, 6]
    Lane = State[:, 7]

    dT = V*np.sin(Phi)
    steer = -T*KP/np.exp(np.abs(T)) - dT*KD/np.exp(np.abs(dT))
    steer = np.random.rand(len(V),)*SIGMA + steer

    # assumes driving forward
    accel = (V_des-V)*KSPD
    for i in xrange(len(State)):
        idx = np.where(np.logical_and(Lane == Lane[i], S > S[i]))[0]
        if len(idx) > 0:
            ind = idx[np.argmin(S[idx])]
            s_front = S[ind]-Length[ind]/2.*np.cos(Phi[ind])
            s_ego = S[i]+Length[i]/2.*np.cos(Phi[i])
            s_gap = s_front-s_ego
            delta_v = V[ind]-V[i]
            s_des = SMIN + V[i]*HEADWAY \
                    - V[i]*delta_v/(2.*np.sqrt(AMAX * DCMF))
            v_ratio = V[i]/V_des[i]
            if (V_des[i] <= 0.0):
                v_ratio = 1.0
            accel[i] = AMAX * (1.0 - np.power(v_ratio, DELTA) - np.power(s_des/s_gap, 2))

    accel = np.maximum(accel, -DMAX)
    accel = np.minimum(accel, AMAX)
    return np.stack((accel, steer), axis=1)


@numba.njit
def _get_collisions(poses, radii, BoundingBoxes):
    num_cars = len(radii)
    collisions = np.zeros((num_cars,))
    for i in xrange(num_cars):
        ri = radii[i]
        for j in xrange(i+1, num_cars):
            rj = radii[j]
            if np.linalg.norm(poses[i, :]-poses[j, :]) < (ri + rj):
                if GJK.collision(BoundingBoxes[i], BoundingBoxes[j]):
                    collisions[i] = 1
                    collisions[j] = 1
    return collisions


@numba.njit
def _get_lidar_features(vehicle_id, X, Y, Theta, V, BoundingBoxes, numbeams, angle_offset, maxrange):
    '''
    X is an array of vehicle x coordinates
    Y is an array of vehicle y coordinates
    V is an array of vehicle speeds (v)
    BoundingBoxes is array of arrays of size 4 x 2
    '''

    lidar_measurement_ranges_and_rates = np.empty((2*numbeams,))
    beams = np.linspace(angle_offset, 2*np.pi+angle_offset, numbeams+1)[1:]
    num_vehicles = len(X)
    poses = np.stack((X, Y, Theta), axis=1)
    for i, angle in enumerate(beams):
        ray_angle = poses[vehicle_id, 2] + angle
        pose = poses[vehicle_id, :2]
        ray_vector = np.array([np.cos(ray_angle), np.sin(ray_angle)])
        range_ = maxrange
        range_rate = 0.0
        for agent_id in xrange(num_vehicles):
            if agent_id != vehicle_id:
                bounding_box = BoundingBoxes[agent_id]

                range_temp = _lidar_observation(pose, ray_angle, bounding_box)
                if range_temp < range_:
                    range_ = range_temp
                    relative_speed = np.array([np.cos(Theta[agent_id])*V[agent_id],
                                               np.sin(Theta[agent_id])*V[agent_id]]) \
                                    - np.array([np.cos(Theta[vehicle_id])*V[vehicle_id],
                                                np.sin(Theta[vehicle_id])*V[vehicle_id]])
                    range_rate = np.dot(relative_speed, ray_vector)
        lidar_measurement_ranges_and_rates[i] = range_
        lidar_measurement_ranges_and_rates[i+numbeams] = range_rate
    return lidar_measurement_ranges_and_rates


@numba.njit
def _lidar_observation(pose, beam_theta, agent_bounding_box):
    ranges = np.empty((len(agent_bounding_box),))
    for i in xrange(len(agent_bounding_box)):
        ranges[i] = _range_observation(pose, beam_theta, agent_bounding_box[i-1, :], agent_bounding_box[i, :])
    return np.amin(ranges)


@numba.njit
def _range_observation(pose, beam_theta, line_segment_a, line_segment_b):
    o = pose
    v1 = o - line_segment_a
    v2 = line_segment_b - line_segment_a
    v3 = np.array([np.cos(beam_theta + np.pi/2.), np.sin(beam_theta + np.pi/2.)])

    denom = np.dot(v2, v3)

    x = np.inf
    if np.abs(denom) > 0.0:
        d1 = cross(v2, v1)/denom  # length of ray (law of sines)
        d2 = np.dot(v1, v3)/denom  # length of seg/v2 (law of sines)
        if d1 >= 0 and d2 >= 0 and d2 <= 1.0:
            x = d1
    elif _are_collinear(pose, line_segment_a, line_segment_b):
        dist_a = np.linalg.norm(line_segment_a - o)
        dist_b = np.linalg.norm(line_segment_b - o)
        x = np.minimum(dist_a, dist_b)
    return x


@numba.njit
def _are_collinear(pt_a, pt_b, pt_c, tol=1e-8):
    return np.abs(cross(pt_b-pt_a, pt_a-pt_c)) < tol


@numba.njit
def cross(vec1, vec2):
    return vec1[0]*vec2[1] - vec1[1]*vec2[0]
