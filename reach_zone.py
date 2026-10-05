"""
Zone reachable by the arm tip (point 2') and automatic sweep.

- halo(): draws in the viewer scene a red 3D halo with every point the tip
  can reach, combining the slide along the rails, the turntable rotation
  and the two arm stages (you see the shell of the volume).
- Sweep: moves the cart (rails and rotation) and the two stages by itself so as to
  pass through the whole volume, leaving a green trail behind the tip. It is turned on
  with the "auto_sweep" slider in the Control panel (0 = off, 1 = on).
"""

import numpy as np
import mujoco

import control

TIP = "point2_s2"
CELLS = 30                           # cells per side of the angle grid (sweep)
VOXEL = 0.10                         # m: side of the voxels of the 3D halo volume
PLANE_STEP = 0.03                    # m: cells of the zone in the arm's plane
Y_STEP = 0.05                        # m: steps along the rails to build the volume
HALO_COLOR = (1.0, 0.0, 0.0, 0.07)
TRAIL_COLOR = (0.1, 1.0, 0.2, 0.9)
TRAIL_STEP = 0.03                    # m between one trail dot and the next
STAGE1_LEVELS = 3                    # how many stage 1 positions for each cart pose
SPEEDUP = 4                          # the sweep runs 4 times faster than real time (the PC manages ~4.4)
TOLERANCE = 1.5                      # degrees: waypoint reached


def _range(model, joint):
    g = model.joint(joint)
    return (g.range[0], g.range[1]) if model.jnt_limited[g.id] else (-np.pi / 2, np.pi / 2)


def points(model, n=CELLS + 1):
    """Tip position (n x n x 3) for every pair of angles of the two joints."""
    d = mujoco.MjData(model)
    q1 = np.linspace(*_range(model, "C"), n)
    q2 = np.linspace(*_range(model, "C_s2"), n)
    a1, a2 = model.joint("C").qposadr[0], model.joint("C_s2").qposadr[0]
    site = model.site(TIP).id
    p = np.zeros((n, n, 3))
    for i, x in enumerate(q1):
        for j, y in enumerate(q2):
            d.qpos[a1], d.qpos[a2] = x, y
            mujoco.mj_kinematics(model, d)
            p[i, j] = d.site_xpos[site]
    return p


def _add(scn, kind, size, position, matrix, color):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], kind, np.asarray(size, float),
                        np.asarray(position, float), matrix, np.asarray(color, np.float32))
    scn.ngeom += 1


def _plane(model, step=PLANE_STEP, widen=0.0):
    """Reachable zone in the arm's plane (T centered, turntable straight):
    (x, z) grid of `step` m cells, True where the tip gets or within
    `widen` m of a point where it gets."""
    p = points(model, n=301)
    x, z = p[..., 0].ravel(), p[..., 2].ravel()
    border = 2 * step + widen
    x0, z0 = x.min() - border, z.min() - border
    nx = int((x.max() + border - x0) / step) + 2
    nz = int((z.max() + border - z0) / step) + 2
    cell = np.zeros((nx, nz), bool)
    cell[((x - x0) / step).astype(int), ((z - z0) / step).astype(int)] = True
    # fill the gaps between samples: a cell is inside if a neighbor is
    full = cell.copy()
    full[1:] |= cell[:-1]
    full[:-1] |= cell[1:]
    full[:, 1:] |= cell[:, :-1]
    full[:, :-1] |= cell[:, 1:]
    # widen by `widen` m in every direction (disc of cells)
    k = int(np.ceil(widen / step))
    if k > 0:
        base = full.copy()
        for dx in range(-k, k + 1):
            for dz in range(-k, k + 1):
                if dx * dx + dz * dz > k * k:
                    continue
                full[max(dx, 0):nx + min(dx, 0), max(dz, 0):nz + min(dz, 0)] |=                     base[max(-dx, 0):nx + min(-dx, 0), max(-dz, 0):nz + min(-dz, 0)]
    return full, x0, z0, step


def volume(model):
    """3D volume reachable by the tip, in VOXEL m cells.
    Built forwards: every point of the zone in the arm's plane (PLANE_STEP m
    cells) is rotated with the turntable at every degree of its range and moved
    along the rails in Y_STEP m steps, as far as the travel allows for that
    rotation; the voxel it falls in is full.
    Returns (boolean grid of full voxels, (x, y, z) of the centers)."""
    plane, x0, z0, pp = _plane(model)
    ix, iz = np.nonzero(plane)
    r = x0 + (ix + 0.5) * pp                      # distance from the axis in the arm's plane
    z = z0 + (iz + 0.5) * pp
    g = model.joint("R")
    if model.jnt_limited[g.id]:
        psi = np.radians(np.arange(np.degrees(g.range[0]), np.degrees(g.range[1]) + 0.5, 1.0))
    else:
        psi = np.zeros(1)
    ymax = control.max_y(model, psi)
    radius = np.abs(r).max()
    xs = np.arange(-radius - VOXEL, radius + 2 * VOXEL, VOXEL)
    wide = ymax.max() + radius + VOXEL
    ys = np.arange(-wide, wide + VOXEL, VOXEL)
    zs = np.arange(z.min() - VOXEL, z.max() + 2 * VOXEL, VOXEL)
    inside = np.zeros((len(xs), len(ys), len(zs)), bool)
    kz = np.rint((z - zs[0]) / VOXEL).astype(int)
    for ps, ym in zip(psi, ymax):
        x = r * np.cos(ps)
        dy = r * np.sin(ps)
        kx = np.rint((x - xs[0]) / VOXEL).astype(int)
        travel = np.linspace(-ym, ym, max(2, int(np.ceil(2 * ym / Y_STEP)) + 1))
        for Y in travel:
            ky = np.rint((Y + dy - ys[0]) / VOXEL).astype(int)
            inside[kx, ky, kz] = True
    return inside, (xs, ys, zs)


def halo(scn, model):
    """Empties the scene and draws the red 3D halo in it: the shell of the volume the tip
    can reach by moving slide, turntable and the two stages. It is fixed in the world.
    Returns how many geoms it used (the sweep trail starts from there)."""
    scn.ngeom = 0
    inside, (xs, ys, zs) = volume(model)
    # shell: full voxels with at least one empty neighbor (inner ones wouldn't be visible)
    empty = np.pad(~inside, 1, constant_values=True)
    shell = inside & (empty[2:, 1:-1, 1:-1] | empty[:-2, 1:-1, 1:-1] | empty[1:-1, 2:, 1:-1]
                      | empty[1:-1, :-2, 1:-1] | empty[1:-1, 1:-1, 2:] | empty[1:-1, 1:-1, :-2])
    for i, j, k in zip(*np.nonzero(shell)):
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.6 * VOXEL, 0, 0], [xs[i], ys[j], zs[k]],
             np.eye(3).flatten(), HALO_COLOR)
    return scn.ngeom


class Sweep:
    """Automatic 3D sweep. The cart goes to the cart poses (center and the four
    corners of its range: rail ends with the max rotations) and at each one
    the arm sweeps: stage 1 over STAGE1_LEVELS positions, stage 2 from one
    end of its range to the other (alternating back and forth)."""

    def __init__(self, model):
        g1 = np.degrees(_range(model, "C"))
        g2 = np.degrees(_range(model, "C_s2"))
        self.cart = True
        try:
            gr = np.degrees(_range(model, "R"))
            model.actuator("position_y"), model.actuator("rotation")
        except KeyError:
            self.cart, gr = False, (0.0, 0.0)
        poses = [(0.0, 0.0)]
        if self.cart:
            for rot in (gr[0], gr[1]):
                ym = float(control.max_y(model, np.radians(rot))) * 0.98
                poses += [(-ym, rot), (ym, rot)]
        self.waypoints = []      # (y, rotation, stage 1, stage 2)
        direction = 0
        for y, rot in poses:
            for a1 in np.linspace(g1[0], g1[1], STAGE1_LEVELS):
                start, end = (g2[0], g2[1]) if direction % 2 == 0 else (g2[1], g2[0])
                self.waypoints += [(y, rot, a1, start), (y, rot, a1, end)]
                direction += 1
        self.waypoints.append((0.0, 0.0, 0.0, 0.0))      # at the end go back to rest
        self.active = False
        self.waypoint = 0
        self.trail = []
        self.first_trail_geom = 0

    def start(self, first_trail_geom):
        self.active, self.waypoint, self.trail = True, 0, []
        self.first_trail_geom = first_trail_geom

    def step(self, model, data, scn):
        """Call at every step: moves the sliders towards the waypoint and updates the trail.
        Returns False when the sweep is over."""
        if not self.active:
            return False
        y, rot, a1, a2 = self.waypoints[self.waypoint]
        data.ctrl[model.actuator("winding").id] = control.slider_for_angle("", a1)
        data.ctrl[model.actuator("winding_s2").id] = control.slider_for_angle("_s2", a2)
        groups = ["", "_s2"]
        if self.cart:
            data.ctrl[model.actuator("position_y").id] = y
            data.ctrl[model.actuator("rotation").id] = rot
            groups.append("cart")
        arrived = all(control._state[s]["brake"] for s in groups)
        actual = -np.degrees(data.qpos[[model.joint("C").qposadr[0], model.joint("C_s2").qposadr[0]]])
        close = abs(actual[0] - a1) < TOLERANCE and abs(actual[1] - a2) < TOLERANCE
        if self.cart:
            close = close and abs(data.qpos[model.joint("Y").qposadr[0]] - y) < 0.01 and \
                abs(np.degrees(data.qpos[model.joint("R").qposadr[0]]) - rot) < TOLERANCE
        if arrived and close:
            self.waypoint += 1
            if self.waypoint >= len(self.waypoints):
                self.active = False
                return False
        # trail: one dot every TRAIL_STEP meters
        p = data.site_xpos[model.site(TIP).id].copy()
        if not self.trail or np.linalg.norm(p - self.trail[-1]) > TRAIL_STEP:
            self.trail.append(p)
            if scn.ngeom < self.first_trail_geom:
                scn.ngeom = self.first_trail_geom
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0, 0], p, np.eye(3).flatten(), TRAIL_COLOR)
        return True

    def text(self):
        if not self.active:
            return None
        y, rot, a1, a2 = self.waypoints[self.waypoint]
        return "auto sweep", "waypoint %d of %d: y %+.2f m, rot %+.0f, arms %+.0f / %+.0f deg (x%d)" % (
            self.waypoint + 1, len(self.waypoints), y, rot, a1, a2, SPEEDUP)
