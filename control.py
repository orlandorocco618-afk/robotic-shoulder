"""
Cables and winders of shoulder.mjcf.xml, controlled by cable length.
There are two identical stages: the first (names without suffix) moves arm 1-2,
the second (names with "_s2") moves arm 1'-2' and hangs at the bottom of the first.

For each stage:
- The "winding" slider (or "winding_s2") says how many cm to wind,
  relative to the start: + on A (arm towards A), - on A'. It corresponds to an angle.
- The two cables are synchronized: they move together along the same reference
  angle, each one wound or unwound exactly as much as that angle needs
  (from the geometry), not by the same amount. This way there is never a
  slack cable: it pulls what is needed, whatever the direction of the load.
  With the brake both stop: the arm is locked in both directions.
- The motors are dynamic: rotor with inertia, reduced by the gears
  (small on the motor, big on the spool, with a chain). The torque follows the
  curve of a DC motor: maximum at standstill (stall), zero at
  NO_LOAD_RPM. A speed controller brings the cable to the slider; once there,
  the brake locks the spool: locked, the cable doesn't unwind by itself.
- The cable is a steel wire rope: tension = EA/L x elongation when taut,
  zero when slack. It can only pull.
"""

import numpy as np
import mujoco

# Small motor: the electric motor of a windshield wiper (fast and not very powerful).
# DC motor curve: speed drops linearly with torque,
# from NO_LOAD_RPM to 0 at STALL_TORQUE.
# Max power = STALL_TORQUE x no-load speed / 4 ~ 100 W (wiper ~50 W).
NO_LOAD_RPM = 3000.0     # rpm without load
STALL_TORQUE = 1.28      # N*m with the motor stalled (twice the wiper: power x2)
ROTOR_INERTIA = 6e-5     # kg*m2, rotor of a wiper motor (~0.3 kg, r ~2 cm)

# Worm gear inside the wiper motor, between rotor and pinion.
# It is self-locking: at standstill it holds the load by itself (the "brake").
# WORM_RATIO = 1 means bare motor, no worm gear.
WORM_RATIO = 50.0
WORM_EFFICIENCY = 0.6

# External reduction: pinion on the motor, big gear on the spool, chain.
# One ratio per stage (big teeth / small teeth).
RATIOS = {"": 6.0, "_s2": 3.0}
EFFICIENCY = 0.95        # roller chain: ~5% is lost
SPOOL_RADIUS = 0.03      # m
SPOOL_INERTIA = 1e-4     # kg*m2, only the spool (the big gear is separate)

# ISO 04B roller chain: 6 mm pitch. Small sprocket with SMALL_TEETH teeth,
# big one with SMALL_TEETH x ratio. The big one is a solid steel disc.
CHAIN_PITCH = 0.006      # m
SMALL_TEETH = 10
GEAR_THICKNESS = 0.003   # m
STEEL_DENSITY = 7850.0

# Speed controller, like real motor drivers: the cable follows a speed that
# drops with deceleration DECEL when arriving; the motor voltage immediately
# compensates the cable load and chases that speed.
# Soft start: the chasing part changes at most from 0 to full in RAMP
# seconds, so the cable doesn't jerk.
DECEL = 0.5              # m/s2 of cable
GAIN = 5.0               # how much the speed error is corrected
RAMP = 0.3               # s
BRAKE_TOLERANCE = 0.0005  # m: within this of the slider, and almost still, it brakes

# 7x19 steel wire rope per stage: diameter in mm. Stage 1 carries all of
# stage 2 and the load: with 50 kg it reaches ~4 kN, too much for the 5 mm (working 2.9 kN).
CABLE_DIAMETER = {"": 8.0, "_s2": 5.0}


def cable_ea(s):
    """N: axial stiffness EA of the rope (metallic section ~48%, modulus ~100 GPa)."""
    return 0.48 * np.pi * (CABLE_DIAMETER[s] / 2) ** 2 * 1e-6 * 100e9


def cable_breaking_load(s):
    """N: minimum breaking load of the 7x19 rope, 1770 N/mm2 steel."""
    return 0.33 * CABLE_DIAMETER[s] ** 2 * 1770
CABLE_DAMPING = 1500.0   # N*s/m, damps the vibrations of the taut cable


def sprocket_radius(teeth):
    """Pitch radius of a chain sprocket (m)."""
    return CHAIN_PITCH / (2 * np.sin(np.pi / teeth))


def big_gear_inertia(R):
    """kg*m2 of the big gear (solid disc) for ratio R."""
    r = sprocket_radius(round(SMALL_TEETH * R))
    mass = STEEL_DENSITY * np.pi * r ** 2 * GEAR_THICKNESS
    return 0.5 * mass * r ** 2


def cable_speed(R):
    """m/s of cable wound with no load."""
    return NO_LOAD_RPM * 2 * np.pi / 60 / (WORM_RATIO * R) * SPOOL_RADIUS


def max_pull(R):
    """N on the cable when a motor stalls."""
    return STALL_TORQUE * WORM_RATIO * WORM_EFFICIENCY * R * EFFICIENCY / SPOOL_RADIUS


def motor_mass(R):
    """Inertia of a motor with its spool as seen from the cable, in kg."""
    J = ROTOR_INERTIA * (WORM_RATIO * R) ** 2 + SPOOL_INERTIA + big_gear_inertia(R)
    return J / SPOOL_RADIUS ** 2


# state of the model in use, per stage: reset when live.py loads a new model
_state = {"model": None}


def _table(model, s, L0):
    """For stage s: arm angles (rad, + towards A) and cable wound on A and A'
    (m) needed exactly for each angle. Depends only on the geometry."""
    joint = model.joint("C" + s)
    lim = joint.range[1] if model.jnt_limited[joint.id] else np.radians(90)
    angles = np.linspace(-lim, lim, 881)
    cables = [model.tendon("cable1" + s).id, model.tendon("cable2" + s).id]
    d = mujoco.MjData(model)
    wound = np.zeros((len(angles), 2))
    for i, q in enumerate(angles):
        d.qpos[joint.qposadr[0]] = -q
        mujoco.mj_fwdPosition(model, d)
        wound[i] = L0 - d.ten_length[cables]
    slope = np.gradient(wound, angles, axis=0)
    return angles, wound, slope


def _target(st, slider):
    """Angle (rad) matching the slider: m wound on A if +, on A' if -."""
    angles, wound = st["angles"], st["tab"]
    if slider >= 0:
        side = angles >= 0
        return float(np.interp(slider, wound[side, 0], angles[side]))
    side = angles <= 0
    return float(np.interp(-slider, wound[side, 1][::-1], angles[side][::-1]))


def _stage(model, data, s, st):
    """Computes cables and motors of one stage (suffix s). Adds the cable
    forces to data.qfrc_applied and returns the text lines for the viewer."""
    R = RATIOS[s]
    cables = [model.tendon("cable1" + s).id, model.tendon("cable2" + s).id]
    command = model.actuator("winding" + s).id
    joint = model.joint("C" + s)

    J = np.zeros((2, model.nv))
    for r, c in enumerate(cables):
        start = model.ten_J_rowadr[c]
        for k in range(model.ten_J_rownnz[c]):
            J[r, model.ten_J_colind[start + k]] = data.ten_J[start + k]
    length = data.ten_length[cables]
    velocity = J @ data.qvel

    # The two cables are synchronized on a reference angle q: each one is
    # wound exactly as much as that angle needs (geometric table).
    # q moves towards the slider's target; the motors follow it.
    q, qv = st["q"], st["qv"]
    wound = np.array([np.interp(q, st["angles"], st["tab"][:, k]) for k in (0, 1)])
    w = np.array([np.interp(q, st["angles"], st["slope"][:, k]) for k in (0, 1)])  # m of cable per radian

    # tension: steel spring only when the cable is longer than its free length
    free = st["length"] - wound
    elongation = length - free
    tension = cable_ea(s) / free * elongation + CABLE_DAMPING * (velocity + w * qv)
    tension = np.where(elongation > 0, np.maximum(tension, 0.0), 0.0)
    st["tension"] = tension
    st["wound"] = wound

    # the cable pulls: force on the joints = -tension x (how much the cable lengthens per radian)
    data.qfrc_applied[:] -= J.T @ tension

    target = _target(st, data.ctrl[command] / 100.0)
    error = target - q
    v0 = cable_speed(R)
    wmax = max(abs(w).max(), 1e-3)
    if st["brake"] and abs(error) * wmax > 2 * BRAKE_TOLERANCE:
        st["brake"] = False
    u = np.zeros(2)
    if st["brake"]:
        qv = 0.0
        st["u"][:] = 0.0
    else:
        # desired angular speed: the max the motors can handle, dropping when arriving
        qv_ref = np.sign(error) * min(0.9 * v0 / wmax, np.sqrt(2 * DECEL / wmax * abs(error)))
        # each motor chases its own cable speed (w x qv_ref) and compensates its cable
        compensate = tension / max_pull(R)
        chase = w * qv_ref / v0 + GAIN * w * (qv_ref - qv) / v0
        jump = model.opt.timestep / RAMP
        chase = np.clip(chase, st["u"] - jump, st["u"] + jump)
        st["u"] = chase
        u = np.clip(compensate + chase, -1.0, 1.0)
        # motor forces on the cables (DC motor curve), mapped onto the angle
        force = max_pull(R) * (u - w * qv / v0)
        mass = (w ** 2).sum() * motor_mass(R)
        qv += ((force - tension) * w).sum() / mass * model.opt.timestep
        q += qv * model.opt.timestep
        # arrived (or just overshot): the motors stop and the self-locking worms hold
        overshot = np.sign(target - q) != np.sign(error)
        if overshot or (abs(target - q) * wmax < BRAKE_TOLERANCE and abs(qv) * wmax < 0.01 * v0):
            q, qv = target, 0.0
            st["brake"] = True
    st["q"], st["qv"] = q, qv
    stall = np.abs(u).max() == 1.0 and abs(qv) * wmax < 0.02 * v0
    rpm = w * qv / SPOOL_RADIUS * R * WORM_RATIO * 60 / (2 * np.pi)

    angle = -np.degrees(data.qpos[joint.qposadr[0]])  # + towards A
    name = "stage 1" if s == "" else "stage 2"
    return ([name + " (cable %g mm)" % CABLE_DIAMETER[s], "  angle / target", "  wound A / A'",
             "  tension A / A'", "  motors A / A'"],
            ["ratio 1:%g" % R,
             "%+.1f / %+.1f deg" % (angle, np.degrees(target)),
             "%+.1f / %+.1f cm" % (wound[0] * 100, wound[1] * 100),
             "%.0f / %.0f N (%.1f%% of breaking)" % (tension[0], tension[1], 100 * tension.max() / cable_breaking_load(s)),
             "braked" if st["brake"] else ("STALL" if stall else "%+.0f / %+.0f rpm" % tuple(rpm))])


# Cart: 4 motors at the ends of the rails, same small motor with its worm gear,
# spool directly on the output (no chain: little force is needed).
CART_RATIO = 1.0
CABLE_DIAMETER["cart"] = 5.0
CART_CABLES = ["cable_r2_front", "cable_r2_back", "cable_r_front", "cable_r_back"]
# the pulleys on the turntable must stay behind the anchor points (at 1.40 m): margin
PULLEY_Y_LIMIT = 1.32           # m: max y of the edge of the cart pulleys


def _cart_lengths(model, st, q):
    """Lengths of the 4 cart cables with the T at y = q[0] and the turntable rotated by q[1] (rad)."""
    d = st["probe"]
    d.qpos[:] = model.qpos0
    d.qpos[model.joint("Y").qposadr[0]] = q[0]
    d.qpos[model.joint("R").qposadr[0]] = q[1]
    mujoco.mj_kinematics(model, d)
    mujoco.mj_comPos(model, d)
    mujoco.mj_tendon(model, d)
    return d.ten_length[st["cables"]].copy()


def _cart_target(model, data):
    """Position and rotation wanted by the sliders, reduced to what the cables allow."""
    y = data.ctrl[model.actuator("position_y").id]
    psi = np.radians(data.ctrl[model.actuator("rotation").id])
    limit = max_y(model, psi)
    return np.array([np.clip(y, -limit, limit), psi])


def max_y(model, psi):
    """How far the T can go along the rails (m, both ways) with the turntable
    rotated by psi (rad): the pulleys must stay behind the anchor points, and the
    slide can't go past the stops."""
    cables = PULLEY_Y_LIMIT - 0.5 * np.abs(np.sin(psi)) - 0.16 * np.cos(psi) - SPOOL_RADIUS
    return np.minimum(cables, model.jnt_range[model.joint("Y").id][1])


def _cart(model, data, st):
    """The 4 motors at the ends of the rails move the T along y and rotate the turntable.
    Like the stages: a common reference q = (y, rotation) moves towards the sliders,
    and each cable is wound exactly as much as that reference needs."""
    R = CART_RATIO
    cables = st["cables"]
    J = np.zeros((4, model.nv))
    for r, c in enumerate(cables):
        start = model.ten_J_rowadr[c]
        for k in range(model.ten_J_rownnz[c]):
            J[r, model.ten_J_colind[start + k]] = data.ten_J[start + k]
    length = data.ten_length[cables]
    velocity = J @ data.qvel

    q, qv = st["q"], st["qv"]
    L = _cart_lengths(model, st, q)
    h = np.array([1e-4, 1e-4])
    G = np.column_stack([-(_cart_lengths(model, st, q + h * e) - L) / h[k]
                         for k, e in enumerate(np.eye(2))])          # m of cable wound per m and per rad
    wound = st["length"] - L

    free = st["length"] - wound
    elongation = length - free
    tension = cable_ea("cart") / free * elongation + CABLE_DAMPING * (velocity + G @ qv)
    tension = np.where(elongation > 0, np.maximum(tension, 0.0), 0.0)
    st["tension"], st["wound"] = tension, wound
    data.qfrc_applied[:] -= J.T @ tension

    target = _cart_target(model, data)
    error = target - q
    scale = np.array([1.0, 0.5])            # rotation counts as the arc at 0.5 m (point A)
    distance = np.linalg.norm(error * scale)
    v0 = cable_speed(R)
    if st["brake"] and distance > 2 * BRAKE_TOLERANCE:
        st["brake"] = False
    u = np.zeros(4)
    if st["brake"]:
        qv = np.zeros(2)
        st["u"][:] = 0.0
    else:
        direction = error / max(distance, 1e-9)
        # max speed along the direction: no motor above 90% of its own
        fastest = max(np.abs(G @ direction).max(), 1e-6)
        ref = min(0.9 * v0 / fastest, np.sqrt(2 * DECEL / fastest * distance))
        qv_ref = direction * ref
        compensate = tension / max_pull(R)
        chase = G @ qv_ref / v0 + GAIN * G @ (qv_ref - qv) / v0
        jump = model.opt.timestep / RAMP
        chase = np.clip(chase, st["u"] - jump, st["u"] + jump)
        st["u"] = chase
        u = np.clip(compensate + chase, -1.0, 1.0)
        force = max_pull(R) * (u - G @ qv / v0)
        mass = G.T @ G * motor_mass(R) + np.eye(2) * 1e-6
        qv = qv + np.linalg.solve(mass, G.T @ (force - tension)) * model.opt.timestep
        q = q + qv * model.opt.timestep
        after = target - q
        if np.dot(after, error) < 0 or (np.linalg.norm(after * scale) < BRAKE_TOLERANCE
                                        and np.abs(G @ qv).max() < 0.01 * v0):
            q, qv = target.copy(), np.zeros(2)
            st["brake"] = True
    st["q"], st["qv"] = q, qv

    y = data.qpos[model.joint("Y").qposadr[0]]
    psi = np.degrees(data.qpos[model.joint("R").qposadr[0]])
    return (["cart (4 motors)", "  y / target", "  rotation / target", "  tensions r'f r'b r f r b"],
            ["ratio 1:%g, %s" % (R, "braked" if st["brake"] else "moving"),
             "%+.3f / %+.3f m" % (y, target[0]),
             "%+.1f / %+.1f deg" % (psi, np.degrees(target[1])),
             "%.0f %.0f %.0f %.0f N" % tuple(tension)])


def step(model, data):
    """Computes the cable forces of all stages and puts them in data.qfrc_applied.
    Call it before every mj_step. Returns the text for the viewer,
    or None if the model has no cables and motors."""
    stages = []
    for s in RATIOS:
        try:
            model.tendon("cable1" + s), model.tendon("cable2" + s)
            model.actuator("winding" + s), model.joint("C" + s)
        except KeyError:
            continue
        stages.append(s)
    try:
        cart_cables = [model.tendon(n).id for n in CART_CABLES]
        model.actuator("position_y"), model.actuator("rotation"), model.joint("Y"), model.joint("R")
    except KeyError:
        cart_cables = None
    if not stages and cart_cables is None:
        return None

    if _state["model"] is not model:
        # cable length at start = with the arms vertical (initial position)
        initial = mujoco.MjData(model)
        mujoco.mj_fwdPosition(model, initial)
        _state.clear()
        _state["model"] = model
        for s in stages:
            cables = [model.tendon("cable1" + s).id, model.tendon("cable2" + s).id]
            L0 = initial.ten_length[cables].copy()
            angles, tab, slope = _table(model, s, L0)
            _state[s] = dict(length=L0, angles=angles, tab=tab, slope=slope, q=0.0, qv=0.0,
                             brake=True, u=np.zeros(2), wound=np.zeros(2), tension=np.zeros(2))
        if cart_cables is not None:
            _state["cart"] = dict(cables=cart_cables, length=initial.ten_length[cart_cables].copy(),
                                  probe=mujoco.MjData(model), q=np.zeros(2), qv=np.zeros(2), brake=True,
                                  u=np.zeros(4), wound=np.zeros(4), tension=np.zeros(4))

    # length and speed of the cables in the current position
    mujoco.mj_fwdPosition(model, data)
    data.qfrc_applied[:] = 0.0
    left, right = [], []
    for s in stages:
        a, b = _stage(model, data, s, _state[s])
        left += a
        right += b
    if cart_cables is not None:
        a, b = _cart(model, data, _state["cart"])
        left += a
        right += b
    return "\n".join(left), "\n".join(right)


def tensions():
    """Current cable tensions per stage: {suffix: [A, A']} (for tests)."""
    return {s: v["tension"].copy() for s, v in _state.items() if s != "model"}


def align(model, data):
    """Sets the cables of all stages exactly to the length the current arm
    position needs (neither slack nor pulling), with the brake on.
    Useful to start a test from an angle other than vertical."""
    step(model, data)                      # initializes the state if needed
    for s, st in _state.items():
        if s == "model":
            continue
        if s == "cart":
            st.update(q=np.array([data.qpos[model.joint("Y").qposadr[0]], data.qpos[model.joint("R").qposadr[0]]]),
                      qv=np.zeros(2), brake=True)
        else:
            st.update(q=-float(data.qpos[model.joint("C" + s).qposadr[0]]), qv=0.0, brake=True)
        st["u"][:] = 0.0


def slider_for_angle(s, degrees):
    """Slider value (cm) that brings stage s to the given angle:
    cm wound on A if the angle is positive, minus the cm wound on A' if negative."""
    st = _state[s]
    q = np.radians(degrees)
    if q >= 0:
        return float(np.interp(q, st["angles"], st["tab"][:, 0])) * 100
    return -float(np.interp(q, st["angles"], st["tab"][:, 1])) * 100
