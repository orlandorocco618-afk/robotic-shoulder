"""
Live reload for MuJoCo models.
Keeps the viewer open and reloads the model every time you save the XML,
in the same window (the camera keeps its position).

    python live.py
    (on macOS:  mjpython live.py)
"""

import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import control      # drives the motors: see control.py
import reach_zone   # halo of the reachable zone and automatic sweep

PATH = "shoulder.mjcf.xml"   # <-- your file
INTERVAL = 0.3               # how often to check the file, in seconds

# Box with angles and tensions. The viewer only writes text at the size
# of its panels (Option > Font), so I draw it myself as an image.
# POINTS = font size: 13 is about 1.5 times smaller than 150%.
POINTS = 13
try:
    FONT = ImageFont.truetype("consola.ttf", POINTS)
except OSError:
    FONT = ImageFont.load_default()


def text_box(left, right):
    """RGB image with the two columns of text, on a dark background."""
    lines = list(zip(left.split("\n"), right.split("\n")))
    height = POINTS + 4
    col = max(FONT.getlength(a) for a, _ in lines) + 12
    width = int(col + max(FONT.getlength(b) for _, b in lines) + 8)
    img = Image.new("RGB", (width, height * len(lines) + 8), (25, 25, 25))
    draw = ImageDraw.Draw(img)
    for i, (a, b) in enumerate(lines):
        draw.text((5, 4 + i * height), a, font=FONT, fill=(170, 170, 170))
        draw.text((col, 4 + i * height), b, font=FONT, fill=(255, 255, 255))
    return np.asarray(img)


def load():
    model = mujoco.MjModel.from_xml_path(PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# the first load must succeed, otherwise there is nothing to show
while True:
    try:
        model, data = load()
        break
    except Exception as error:
        print("\n[MODEL ERROR]\n", error, "\n")
        time.sleep(1)

timestamp = os.path.getmtime(PATH)
last_check = time.time()
last_text = 0
print("Model loaded.")

with mujoco.viewer.launch_passive(model, data) as v:
    # show the world axes: red = x, green = y, blue = z
    with v.lock():
        v.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        # red halo: every point reachable by the tip (reach_zone.py)
        halo_geoms = reach_zone.halo(v.user_scn, model)
    sweep = reach_zone.Sweep(model)

    while v.is_running():
        t0 = time.time()

        # automatic sweep: turned on and off with the "auto_sweep" slider
        try:
            switch = model.actuator("auto_sweep").id
        except KeyError:
            switch = None
        if switch is not None:
            if data.ctrl[switch] >= 0.5 and not sweep.active:
                sweep.start(halo_geoms)
            elif data.ctrl[switch] < 0.5:
                sweep.active = False

        # during the sweep, more steps per cycle: it runs SPEEDUP times faster
        for _ in range(reach_zone.SPEEDUP if sweep.active else 1):
            text = control.step(model, data)
            if sweep.active:
                with v.lock():
                    if not sweep.step(model, data, v.user_scn) and switch is not None:
                        data.ctrl[switch] = 0.0   # done: switch back to 0
            mujoco.mj_step(model, data)
        v.sync()
        if text and sweep.text():
            text = (text[0] + "\n" + sweep.text()[0], text[1] + "\n" + sweep.text()[1])

        # box with angle and tensions, updated 10 times per second
        view = v.viewport
        if text and view is not None and time.time() - last_text > 0.1:
            last_text = time.time()
            img = text_box(text[0], text[1])
            height, width = img.shape[:2]
            # top left of the scene (the viewer counts pixels from the bottom)
            spot = mujoco.MjrRect(view.left + 5, view.bottom + view.height - height - 5, width, height)
            v.set_images((spot, img))

        # check the file every now and then, not at every step
        if time.time() - last_check > INTERVAL:
            last_check = time.time()
            new_timestamp = os.path.getmtime(PATH)
            if new_timestamp != timestamp:
                timestamp = new_timestamp
                try:
                    new_model, new_data = load()
                except Exception as error:
                    # broken XML: say so and keep the old model
                    print("\n[MODEL ERROR]\n", error, "\n")
                    continue

                # when loading, the viewer resets the camera:
                # save it and put it back where it was
                c = v.cam
                cam = (c.type, c.fixedcamid, c.trackbodyid, c.lookat.copy(),
                       c.distance, c.azimuth, c.elevation)

                v._get_sim().load(new_model, new_data, PATH)
                model, data = new_model, new_data

                with v.lock():
                    (c.type, c.fixedcamid, c.trackbodyid, c.lookat[:],
                     c.distance, c.azimuth, c.elevation) = cam
                    # if the camera or the tracked body no longer exist,
                    # go back to the free camera
                    if c.fixedcamid >= model.ncam or c.trackbodyid >= model.nbody:
                        c.type = mujoco.mjtCamera.mjCAMERA_FREE
                    # the new model may have another zone: redo halo and sweep
                    halo_geoms = reach_zone.halo(v.user_scn, model)
                sweep = reach_zone.Sweep(model)
                print("File changed, reloaded.")

        # keep the simulation at real-time speed
        time.sleep(max(0, model.opt.timestep - (time.time() - t0)))

print("Closed.")
