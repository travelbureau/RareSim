import scipy.interpolate
import numpy as np
import os
import numba

os.system('pwd')
ROADWAY_FILE = './simulator/i80.txt'
BOUNDARY_FILE = './simulator/i80boundaries.txt'


class Roadway(object):
    S_TOL = 0.01

    def __init__(self):
        l = np.loadtxt(ROADWAY_FILE)*0.3048
        lanes_ = [l[1537:1980, :],
                  l[695:1140, :],
                  l[0:456, :],
                  l[4111:-170, :],
                  l[3486:3930, :],
                  l[2378:2815, :]]

        lb = np.loadtxt(BOUNDARY_FILE)*0.3048
        boundaries_ = [lb[203:221, :],
                       lb[170:188, :],
                       lb[142:160, :],
                       lb[63:82, :],
                       lb[117:136, :],
                       lb[244:259, :]]

        self.lanes = []
        self.lane_u_spacings = []
        self.lane_lengths = []
        for lane in lanes_:
            tck, u = scipy.interpolate.splprep(lane.T, s=0)
            npts = len(lane)
            npts_fine = 100*3*npts
            u_fine = np.arange(0, 1+1e-12, 1./npts_fine)
            lane_fine = scipy.interpolate.splev(u_fine, tck)
            lane_fine = np.asarray([lane_fine[0], lane_fine[1]]).T
            diffs = np.linalg.norm(lane_fine[1:]-lane_fine[:-1], axis=1)
            s = np.cumsum(diffs)
            s = np.insert(s, 0, 0)
            derivs = scipy.interpolate.spalde(u_fine, tck)
            Dx = np.asarray(derivs[0])
            Dy = np.asarray(derivs[1])
            dx = Dx[:, 1]
            dy = Dy[:, 1]
            ddx = Dx[:, 2]
            ddy = Dy[:, 2]
            curvature_fine = (dx*ddy - dy*ddx)/((dx*dx + dy*dy)**1.5)
            theta_fine = np.arctan2(dy, dx)
            lane_fine = np.asarray([lane_fine[:, 0], lane_fine[:, 1], s, curvature_fine, theta_fine, dx, dy, ddx, ddy]).T
            self.lanes.append(lane_fine)
            self.lane_u_spacings.append(1./npts_fine)
            self.lane_lengths.append(s[-1])
            assert(np.amax(diffs) < Roadway.S_TOL)
        self.lane_lengths = np.asarray(self.lane_lengths)
        self.tolerance = Roadway.S_TOL/np.amax(self.lane_lengths)

        boundary_right, _ = Roadway._make_boundaries_hack(self.lanes[0], 1.5)
        boundaries_.insert(0, boundary_right)
        assert(len(boundaries_) == 7)

        self.boundaries = []
        self.boundary_u_spacings = []
        for boundary in boundaries_:
            tck, u = scipy.interpolate.splprep(boundary.T, s=0)
            npts = len(boundary)
            npts_fine = 2400*3*npts if (npts < 100) else npts
            u_fine = np.arange(0, 1+1e-12, 1./npts_fine)
            boundary_fine = scipy.interpolate.splev(u_fine, tck)
            boundary_fine = np.asarray([boundary_fine[0], boundary_fine[1]]).T
            diffs = np.linalg.norm(boundary_fine[1:]-boundary_fine[:-1], axis=1)
            s = np.cumsum(diffs)
            s = np.insert(s, 0, 0)
            derivs = scipy.interpolate.spalde(u_fine, tck)
            Dx = np.asarray(derivs[0])
            Dy = np.asarray(derivs[1])
            dx = Dx[:, 1]
            dy = Dy[:, 1]
            ddx = Dx[:, 2]
            ddy = Dy[:, 2]
            curvature_fine = (dx*ddy - dy*ddx)/((dx*dx + dy*dy)**1.5)
            theta_fine = np.arctan2(dy, dx)
            boundary_fine = np.asarray([boundary_fine[:, 0], boundary_fine[:, 1], s, curvature_fine, theta_fine, dx, dy, ddx, ddy]).T
            self.boundaries.append(boundary_fine)
            self.boundary_u_spacings.append(1./npts_fine)
            assert(np.amax(diffs) < Roadway.S_TOL)

    @staticmethod
    def _make_boundaries_hack(lane, offset):
        pts = lane[:, :2]
        theta = lane[:, 4]
        dx = offset*np.cos(theta)
        dy = offset*np.sin(theta)

        bndry1 = pts + np.asarray([dy, -dx]).T
        bndry2 = pts - np.asarray([dy, -dx]).T

        return bndry1, bndry2

    def local2global(self, V, W, S, T, Lane):
        # X,Y,Theta,V,S,T,Phi,Lane
        State = np.empty((len(Lane), 8))
        State[:, 3] = V
        State[:, 5] = T
        State[:, 6] = W
        State[:, 7] = Lane
        pt_s = np.empty((len(Lane),))
        pt_theta = np.empty((len(Lane),))
        pt_xy = np.empty((len(Lane), 2))
        for i in xrange(len(Lane)):
            lane = self.lanes[Lane[i]]
            idx = np.searchsorted(lane[:, 2], S[i])
            pt = lane[idx]
            pt_s[i] = pt[2]
            pt_theta[i] = pt[4]
            pt_xy[i, :] = pt[:2]
        assert(np.linalg.norm(pt_s-S, ord=np.inf) < Roadway.S_TOL)
        State[:, 4] = pt_s
        angle = pt_theta + np.pi/2.
        Pose = T[:, None]*np.asarray([np.cos(angle), np.sin(angle)]).T + pt_xy
        State[:, :2] = Pose
        State[:, 2] = pt_theta + W
        return State

    def get_road_features(self, poses, Theta, S0, Lanes0, Lengths, Widths, Radii):
        # poses is current positions, Theta is current rotations, S0 is old s, Lanes0 is old lanes
        # Lengths, Widths are the lengths and widths of the vehicles
        out = _get_road_features(S0, Lanes0.astype(np.int32), poses, Theta, self.lanes, self.lane_lengths,
                                 self.lane_u_spacings, self.tolerance, self.boundaries, self.boundary_u_spacings,
                                 Lengths, Widths, Radii, Roadway.S_TOL)
        return out


@numba.njit
def _get_road_features(s0, lanes0, poses, Theta, lanes, lane_lengths, lane_u_spacings, tolerance,
                       boundaries, boundary_u_spacings, Lengths, Widths, Radii, TOL):
    lanes_indices = _get_lane_s(s0, lanes0, poses, lanes, lane_lengths, lane_u_spacings, tolerance)
    dists_left_right, indices_left_right = _get_boundary_dists(s0, lanes_indices[:, 0], poses,
                                                               lane_lengths, boundaries, boundary_u_spacings, tolerance)
    out = _road_features_helper(poses, Theta, lanes_indices, dists_left_right, indices_left_right,
                                lanes, boundaries, Lengths, Widths, Radii, TOL)
    return out


@numba.njit
def _road_features_helper(poses, Theta, lanes_indices, dists_left_right, indices_left_right, lanes, boundaries,
                          Lengths, Widths, Radii, TOL):
    out = np.empty((len(poses), 8))
    for i in xrange(len(poses)):
        lane = lanes_indices[i, 0]
        ind = lanes_indices[i, 1]
        pose = poses[i]
        theta = Theta[i]
        lane_tuple = lanes[lane][ind, :]
        pt = lane_tuple[:2]
        pt_s = lane_tuple[2]
        pt_curvature = lane_tuple[3]
        pt_theta = lane_tuple[4]
        t = _signed_distance(pose, pt, pt_theta)
        phi = np.arctan2(np.sin(theta-pt_theta), np.cos(theta-pt_theta))

        dist_left = dists_left_right[i, 0]
        index_left = indices_left_right[i, 0]
        dist_right = dists_left_right[i, 1]
        index_right = indices_left_right[i, 1]

        length = Lengths[i]
        width = Widths[i]
        radius = Radii[i]

        is_offroad = 0
        if lane == 0:
            ind = index_right
            bnd = boundaries[0]
            is_offroad = _rectangle_pt_intersection(ind, bnd, radius, TOL, pose, length, width, theta)
        elif lane == len(lanes)-1:
            ind = index_left
            bnd = boundaries[-1]
            is_offroad = _rectangle_pt_intersection(ind, bnd, radius, TOL, pose, length, width, theta)
        out[i] = [t, phi, pt_curvature, dist_left, dist_right, lane, is_offroad, pt_s]
    return out


@numba.njit
def _rectangle_pt_intersection(ind, bnd, radius, TOL, pose, length, width, theta):
    start = np.int32(np.maximum(ind-radius*1./TOL, 0))
    end = np.int32(np.minimum(ind+radius*1./TOL + 50, len(bnd)))
    pts = bnd[start:end:50, :2] - np.expand_dims(pose, axis=0)
    mat = np.array([[2./length*np.cos(-theta), 2./width*np.sin(-theta)],
                    [-2./length*np.sin(-theta), 2./width*np.cos(-theta)]])
    # scaledrotpts = np.dot(pts, mat)
    return np.any(np.sum(np.floor(np.abs(np.dot(pts, mat))), axis=1) < 1)


@numba.njit
def _get_lane_s(s0, lanes0, poses, lanes, lane_lengths, lane_u_spacings, tolerance):
    # s0 is old s, lanes0 is old lanes, poses is new poses
    u0 = s0/lane_lengths[lanes0]
    u0temp = np.expand_dims(u0, axis=0)
    posestemp = np.expand_dims(poses, axis=0)
    u0 = u0temp
    poses = posestemp
    for _ in xrange(len(lanes)-1):
        u0 = np.vstack((u0, u0temp))
        poses = np.vstack((poses, posestemp))

    u = u0
    u_index = np.empty_like(u, dtype=np.int32)
    for _ in xrange(100):
        for i in xrange(u.shape[0]):
            for j in xrange(u.shape[1]):
                u_index[i, j] = np.round_(u[i, j]/lane_u_spacings[i])
        lane = lanes[0]
        vals_all = np.expand_dims(lane[u_index[0, :]], axis=0)
        for i in xrange(1, len(lanes)):
            lane = lanes[i]
            vals2 = np.expand_dims(lane[u_index[i, :]], axis=0)
            vals_all = np.vstack((vals_all, vals2))
        diff = vals_all[:, :, :2]-poses
        d1 = vals_all[:, :, 5:7]
        d2 = vals_all[:, :, 7:]
        d = np.sum(diff*d1, axis=2)
        dd = np.sum(diff*d2, axis=2) + np.sum(d1*d1, axis=2)
        u_new = u - d/dd
        dist = np.amax(np.abs(u-u_new))
        u = u_new
        u = np.minimum(u, 1.0)
        u = np.maximum(u, 0.0)
        if dist < tolerance:
            break
    pts = np.empty_like(poses)
    for i in xrange(u.shape[0]):
        lane = lanes[i]
        for j in xrange(u.shape[1]):
            u_index[i, j] = np.round_(u[i, j]/lane_u_spacings[i])
            pts[i, j, :] = lane[u_index[i, j], :2]
    diff = pts - poses
    out = np.sum(diff*diff, axis=2)
    lanes_and_indices = np.empty((u.shape[1], 2), dtype=np.int32)
    for j in xrange(u.shape[1]):
        lane_ = np.argmin(out[:, j])
        lanes_and_indices[j, 0] = lane_
        lanes_and_indices[j, 1] = u_index[lane_, j]
    return lanes_and_indices


@numba.njit
def _get_boundary_dists(s0, lanes, poses_, lane_lengths, boundaries, boundary_u_spacings, tolerance):
    # s0 is old positions, lanes is new lanes, poses_ is new poses
    dists_left_right = np.empty_like(poses_)
    indices_left_right = np.empty_like(poses_, dtype=np.int32)

    u0 = s0/lane_lengths[lanes]
    for ii in xrange(len(lane_lengths)):
        idx = np.where(lanes == ii)[0]
        if len(idx) == 0:
            continue
        u = np.expand_dims(u0[idx], axis=0)
        poses = np.expand_dims(poses_[idx, :], axis=0)
        u = np.vstack((u, u))
        poses = np.vstack((poses, poses))
        lane = boundaries[ii]
        lane2 = boundaries[ii+1]
        u_index = np.empty_like(u, dtype=np.int32)
        for _ in xrange(100):
            for i in xrange(2):
                for j in xrange(u.shape[1]):
                    u_index[i, j] = np.round_(u[i, j]/boundary_u_spacings[ii+i])
            vals = np.expand_dims(lane[u_index[0, :]], axis=0)
            vals2 = np.expand_dims(lane2[u_index[1, :]], axis=0)
            vals_all = np.vstack((vals, vals2))
            diff = vals_all[:, :, :2]-poses
            d1 = vals_all[:, :, 5:7]
            d2 = vals_all[:, :, 7:]
            d = np.sum(diff*d1, axis=2)
            dd = np.sum(diff*d2, axis=2) + np.sum(d1*d1, axis=2)
            u_new = u - d/dd
            dist = np.amax(np.abs(u-u_new))
            u = u_new
            u = np.minimum(u, 1.0)
            u = np.maximum(u, 0.0)
            if dist < tolerance:
                break
        pts = np.empty_like(poses)
        for i in xrange(2):
            lane = boundaries[ii+i]
            for j in xrange(u.shape[1]):
                u_index[i, j] = np.round_(u[i, j]/boundary_u_spacings[ii+i])
                pts[i, j, :] = lane[u_index[i, j], :2]
        diff = pts - poses
        dist_ = np.sqrt(np.sum(diff*diff, axis=2))

        dists_left_right[idx, 0] = dist_[1, :]
        dists_left_right[idx, 1] = dist_[0, :]

        indices_left_right[idx, 0] = u_index[1, :]
        indices_left_right[idx, 1] = u_index[0, :]
    return dists_left_right, indices_left_right


@numba.njit
def _signed_distance(pose, pt, pt_theta):
    # pose is where car is
    # pt is closest pt from pose to line by nn search
    # pt_theta is slope at pt
    # call the oriented ray from the pt p
    # call the oriented ray (pose-pt) q
    # signed distacne obtains the sign of p cross q (postitive up from page)
    dx = np.cos(pt_theta)
    dy = np.sin(pt_theta)
    t = cross(np.array([dx, dy]), pose-pt)  # /np.linalg.norm(pt_front-pt)
    return t


@numba.njit
def cross(vec1, vec2):
    return vec1[0]*vec2[1] - vec1[1]*vec2[0]
