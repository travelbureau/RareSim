import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np


class Visualizer(object):

    def __init__(self, ego_id, lengths, widths, num_cars, road):
        self.ego_id = ego_id
        self.num_cars = num_cars
        self.lengths = lengths
        self.widths = widths
        p = 0.5

        # make road
        left = road.boundaries[-1][:, :2]
        right = road.boundaries[0][:, :2]
        right = right[::-1, :]
        road_polygon = np.concatenate([left, right], axis=0)
        verts = np.append(road_polygon, (left[0])[None, :], axis=0)
        codes = Path.LINETO*np.ones(len(verts))
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        path = Path(verts, codes)
        self.fig = plt.figure(frameon=False, figsize=(8/2., 12.8/2.))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.road_patch = patches.PathPatch(path, facecolor='gray', lw=0, zorder=1)
        self.ax.add_patch(self.road_patch)
        for i in xrange(len(road.boundaries)):
            if i == 0 or i == len(road.boundaries)-1:
                lane = road.boundaries[i][:, :2]
                self.ax.plot(lane[:, 0], lane[:, 1], lw=1, color='black', zorder=2)
            else:
                lane = road.boundaries[i][:, :2]
                step = 1000
                ministep = step*3/10
                for i in xrange(0, len(lane), step):
                    self.ax.plot(lane[i:i+ministep, 0], lane[i:i+ministep, 1],
                                 lw=0.5, color='white', ms=10, zorder=2)

        # make vehicles
        self.vehicle_patches = []
        self.arrow_patches = []
        for i in xrange(self.num_cars):
            l = self.lengths[i]
            w = self.widths[i]
            color = 'cyan' if i == ego_id else 'pink'
            vehicle_patch = patches.FancyBboxPatch([-(l/2.)+p, -(w/2.)+p], l-(p*2.), w-(p*2.),
                                                   boxstyle=patches.BoxStyle("Round", pad=p), lw=1,
                                                   color=color, zorder=3)
            arrow_patch = patches.FancyArrow(-l/4, 0, l*0.5, 0,
                                             width=w*0.15, edgecolor='gray', facecolor='white',
                                             head_length=0.5*w, head_width=0.5*w,
                                             length_includes_head=True, lw=1, zorder=4)
            self.ax.add_patch(vehicle_patch)
            self.ax.add_patch(arrow_patch)
            self.vehicle_patches.append(vehicle_patch)
            self.arrow_patches.append(arrow_patch)

    def update(self, state):
        ego_id = self.ego_id
        X = state[:, 0]
        Y = state[:, 1]
        Theta = state[:, 2]

        self.ax.set_xlim(X[ego_id] - 33.75, X[ego_id] + 33.75)
        self.ax.set_ylim(Y[ego_id] - 60., Y[ego_id] + 60.)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('off')

        for i in xrange(self.num_cars):
            r = mpl.transforms.Affine2D().rotate_deg_around(0, 0, np.rad2deg(Theta[i]))
            t = mpl.transforms.Affine2D().translate(X[i], Y[i])
            transform = r + t + self.ax.transData
            self.vehicle_patches[i].set_transform(transform)
            self.arrow_patches[i].set_transform(transform)
