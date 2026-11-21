"""Interactive fluid: the native SPH simulator next to learned simulators, driven by the same controls.

    run.bat interactive.py                                  # native vs the first trained run found
    run.bat interactive.py --runs les_M2 gnn_M2             # native + several learned models, side by side
    run.bat interactive.py --record out/demo.gif --seconds 3 --runs les_M2   # headless, writes a GIF

Controls (window):
    left mouse drag    push the fluid away          right mouse drag   pull it in
    arrow keys         rotate gravity               g                  gravity back to straight down
    1 2 3 4            scene presets                r                  restart the scene
    space              pause                        +/-                finger strength
    q / Esc            quit

Every panel gets the same gravity and finger.  Native advances M*STRIDE substeps per frame so all panels
cover the same simulated time; the title of each panel shows its own milliseconds per frame, which is
the honest speed comparison.  Learned panels start from the native state and drift away from it: the
number on the right is their mean error against native, in particle spacings.
"""
import argparse
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import evaluate
import gen_data
import learned
import sph

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = {
    "1": ("dam break", [(0.0, 0.0, 0.35, 0.6)]),
    "2": ("two blocks", [(0.05, 0.3, 0.3, 0.4), (0.62, 0.0, 0.3, 0.45)]),
    "3": ("pool + drop", [(0.0, 0.0, 1.0, 0.18), (0.4, 0.5, 0.2, 0.25)]),
    "4": ("tall column", [(0.35, 0.0, 0.25, 0.7)]),
    "5": ("calm pool (should stay calm)", [(0.0, 0.0, 1.0, 0.16)]),
}


def scene_state(key):
    blocks = SCENES[key][1]
    pos = torch.cat([sph.lattice_block(*b) for b in blocks])
    return pos, torch.zeros_like(pos)


class Twin:
    """Native sim plus any number of learned models on the same controls."""
    def __init__(self, run_names, scene="1", use_native=True):
        self.use_native = use_native
        self.nframe, self.timing = 0, True
        self.pool = ThreadPoolExecutor(1)
        self.runs = []
        for n in run_names:
            m, a = evaluate.load_model(n)
            self.runs.append(dict(name=n, model=m, M=a["M"], ms=0.0, err=0.0))
        self.M = self.runs[0]["M"] if self.runs else 2
        self.scene = scene
        self.reset()
        self.g_angle = 0.0
        self.finger = [0.5, 0.5, 0.0, 0.14, 0.0]
        self.native_ms = 0.0

    def reset(self, scene=None):
        self.scene = scene or self.scene
        pos, vel = scene_state(self.scene)
        self.native = sph.Sim(pos, vel)
        dev = learned.DEV
        for r in self.runs:
            r["pos"] = pos.to(dev)[None].clone()
            r["vel"] = vel.to(dev)[None].clone()
            r["prev"] = r["vel"].clone()
        self.mask = torch.ones(1, len(pos), dtype=torch.bool, device=dev)

    def controls(self):
        g = sph.GRAVITY * np.array([math.sin(self.g_angle), -math.cos(self.g_angle)], np.float32)
        return g, np.array(self.finger, np.float32)

    def _advance_native(self, g, fin):
        t0 = time.time()
        n_sub = self.M * gen_data.STRIDE
        chunk = min(32, n_sub)                            # refresh the controls every 32 solver steps (4 ms), not every 8: fewer Python calls
        for _ in range(max(1, n_sub // chunk)):
            self.native.ext = gen_data.ext_for(self.native.pos, g, fin)
            self.native.step(chunk)
        torch.cuda.synchronize()
        if self.timing:
            self.native_ms = 0.7 * self.native_ms + 0.3 * (time.time() - t0) * 1000

    def _advance_learned(self, g, fin):
        frame_dt = gen_data.STRIDE * sph.DT
        dev = learned.DEV
        gt = torch.tensor(g, device=dev)[None]
        ft = torch.tensor(fin, device=dev)[None]
        for r in self.runs:
            t0 = time.time()
            with torch.no_grad():
                ext = learned.ext_batch(r["pos"], gt, ft)
                dt = r["M"] * frame_dt
                r["prev"] = r["vel"]
                r["pos"], r["vel"] = r["model"].step(r["pos"], r["vel"], r["prev"], ext, self.mask, dt)
            torch.cuda.synchronize()
            if self.timing:
                r["ms"] = 0.7 * r["ms"] + 0.3 * (time.time() - t0) * 1000

    def advance(self, parallel=True):
        """One frame.  The native solver and the learned models are independent, so they run in two threads.
        synchronize() waits for the whole GPU, so a frame run in parallel cannot be used to time one panel (the
        first learned model would be charged for the native kernel running beside it): every 20th frame is
        run one panel at a time and only those frames update the ms/frame numbers."""
        g, fin = self.controls()
        self.nframe += 1
        self.timing = (self.nframe % 20 == 0) or not self.use_native
        if self.timing and self.use_native:
            parallel = False
        if self.use_native and parallel:
            f = self.pool.submit(self._advance_native, g, fin)
            self._advance_learned(g, fin)
            f.result()
        else:
            if self.use_native:
                self._advance_native(g, fin)
            self._advance_learned(g, fin)
        if self.use_native:
            for r in self.runs:
                r["err"] = ((r["pos"][0] - self.native.pos).norm(dim=-1).mean() / sph.SPACING).item()

    def panels(self):
        out = []
        if self.use_native:
            out.append(("native SPH", self.native.pos.cpu().numpy(), self.native.vel.cpu().numpy(), f"{self.native_ms:.1f} ms/frame"))
        for r in self.runs:
            info = f"{r['ms']:.1f} ms/frame" + (f"  err {r['err']:.1f} sp" if self.use_native else "")
            out.append((r["name"], r["pos"][0].cpu().numpy(), r["vel"][0].cpu().numpy(), info))
        return out


def color(speed):
    t = np.clip(speed / 5.0, 0, 1)
    r = (40 + 215 * t).astype(int); g = (110 + 90 * (1 - np.abs(2 * t - 1))).astype(int); b = (255 - 200 * t).astype(int)
    return r, g, b


def render_gif(twin, path, seconds, size=360, controls=True):
    from PIL import Image, ImageDraw
    frames = []
    nframes = int(seconds / (twin.M * gen_data.STRIDE * sph.DT))
    for f in range(nframes):
        # scripted demo controls: gravity swings, finger pushes mid-clip
        if controls:
            twin.g_angle = 0.6 * math.sin(f / nframes * 2 * math.pi) if f > nframes // 3 else 0.0
            twin.finger = [0.3 + 0.4 * f / nframes, 0.25, 40.0 if nframes // 4 < f < nframes // 2 else 0.0, 0.14,
                           1.0 if nframes // 4 < f < nframes // 2 else 0.0]
        else:                                          # no controls at all: straight gravity (the calm-pool demo)
            twin.g_angle = 0.0
            twin.finger = [0.5, 0.5, 0.0, 0.14, 0.0]
        twin.advance()
        pans = twin.panels()
        img = Image.new("RGB", (size * len(pans), size + 22), (16, 18, 24))
        d = ImageDraw.Draw(img)
        for i, (title, pos, vel, info) in enumerate(pans):
            ox = i * size
            d.rectangle([ox, 22, ox + size - 1, 22 + size - 1], outline=(70, 75, 90))
            r, g, b = color(np.linalg.norm(vel, axis=1))
            for (x, y), rr, gg, bb in zip(pos, r, g, b):
                cx, cy = ox + x * (size - 2) + 1, 22 + (1 - y) * (size - 2) + 1
                d.ellipse([cx - 2.2, cy - 2.2, cx + 2.2, cy + 2.2], fill=(int(rr), int(gg), int(bb)))
            d.text((ox + 6, 5), f"{title}   {info}", fill=(220, 225, 235))
        frames.append(img)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 * twin.M * gen_data.STRIDE * sph.DT * 4),
                   loop=0)
    print("wrote", path, len(frames), "frames", flush=True)


DISC = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if dx * dx + dy * dy <= 5]
DX = np.array([d[0] for d in DISC], np.int32)
DY = np.array([d[1] for d in DISC], np.int32)
BG = np.array([16, 18, 24], np.uint8)


def rasterise(pans, size):
    """All panels into one RGB image with numpy (no per-particle Python calls)."""
    n = len(pans)
    img = np.empty((size, size * n, 3), np.uint8)
    img[:] = BG
    for i, (_, pos, vel, _) in enumerate(pans):
        ox = i * size
        px = (pos[:, 0] * (size - 2) + 1).astype(np.int32) + ox
        py = ((1.0 - pos[:, 1]) * (size - 2) + 1).astype(np.int32)
        r, g, b = color(np.linalg.norm(vel, axis=1))
        rgb = np.stack([r, g, b], 1).astype(np.uint8)
        x = np.clip(px[:, None] + DX[None], ox, ox + size - 1)          # (N, disc) pixel coordinates, one scatter
        y = np.clip(py[:, None] + DY[None], 0, size - 1)
        img[y, x] = rgb[:, None, :]
        img[0, ox:ox + size] = img[-1, ox:ox + size] = (70, 75, 90)
        img[:, ox] = img[:, ox + size - 1] = (70, 75, 90)
    return b"P6 %d %d 255\n" % (size * n, size) + img.tobytes()


def run_window(twin, self_test=0, realtime=True, size=300):
    """Physics runs in its own thread and publishes snapshots; the window only draws the latest one, so a slow
    physics step never blocks drawing or input (and vice versa)."""
    import tkinter as tk
    npan = len(twin.panels())
    root = tk.Tk()
    if self_test:                                    # invisible: run N frames, then exit (for automated checks)
        root.withdraw()
    root.title("LESNet fluid: native SPH vs learned")
    cv = tk.Canvas(root, width=size * npan, height=size + 24, bg="#10121a", highlightthickness=0)
    cv.pack()
    photo = tk.PhotoImage(width=size * npan, height=size)
    cv.create_image(0, 24, image=photo, anchor="nw")
    state = dict(paused=False, drag=0, mx=0.5, my=0.5, strength=40.0, quit=False, reset=None, fast=not realtime)
    shared = dict(snap=None, sim_fps=0.0, speed=0.0)
    lock = threading.Lock()
    frame_dt = twin.M * gen_data.STRIDE * sph.DT

    def sim_loop():
        n, t_win, t_next = 0, time.time(), time.time()
        while not state["quit"]:
            if state["reset"] is not None:
                twin.reset(None if state["reset"] == "same" else state["reset"]); state["reset"] = None
            if state["paused"]:
                time.sleep(0.01); t_next = time.time(); continue
            twin.finger = [state["mx"], state["my"], state["drag"] * state["strength"], 0.14, 1.0 if state["drag"] else 0.0]
            twin.advance()
            pans = twin.panels()
            n += 1
            now = time.time()
            if now - t_win >= 0.5:
                with lock:
                    shared["sim_fps"] = n / (now - t_win); shared["speed"] = shared["sim_fps"] * frame_dt
                n, t_win = 0, now
            with lock:
                shared["snap"] = pans
            if not state["fast"]:                    # keep simulated time at or below wall-clock time
                t_next += frame_dt
                d = t_next - time.time()
                if d > 0:
                    time.sleep(d)
                else:
                    t_next = time.time()

    def to_world(ev):
        px = ev.x % size / size
        return min(max(px, 0), 1), 1 - min(max((ev.y - 24) / size, 0), 1)

    def down(ev, sign):
        state["drag"] = sign
        state["mx"], state["my"] = to_world(ev)

    def move(ev):
        state["mx"], state["my"] = to_world(ev)

    def up(_):
        state["drag"] = 0

    cv.bind("<ButtonPress-1>", lambda e: down(e, 1)); cv.bind("<B1-Motion>", move); cv.bind("<ButtonRelease-1>", up)
    cv.bind("<ButtonPress-3>", lambda e: down(e, -1)); cv.bind("<B3-Motion>", move); cv.bind("<ButtonRelease-3>", up)

    def shutdown():
        state["quit"] = True
        root.destroy()

    def key(ev):
        k = ev.keysym
        if k in ("q", "Escape"):
            shutdown()
        elif k == "space":
            state["paused"] = not state["paused"]
        elif k == "Left":
            twin.g_angle -= 0.15
        elif k == "Right":
            twin.g_angle += 0.15
        elif k == "g":
            twin.g_angle = 0.0
        elif k == "r":
            state["reset"] = "same"
        elif k in SCENES:
            state["reset"] = k
        elif k in ("plus", "equal"):
            state["strength"] *= 1.25
        elif k == "minus":
            state["strength"] /= 1.25
        elif k == "f":
            state["fast"] = not state["fast"]

    root.bind("<Key>", key)
    root.protocol("WM_DELETE_WINDOW", shutdown)
    th = threading.Thread(target=sim_loop, daemon=True)
    th.start()
    ticks = {"n": 0, "seen": None, "t": time.time(), "draws": 0, "fps": 0.0}
    labels = [cv.create_text(i * size + 6, 12, text="", fill="#dce1eb", anchor="w", font=("Consolas", 9)) for i in range(npan)]

    def tick():
        ticks["n"] += 1
        if self_test and ticks["n"] > self_test:
            shutdown()
            return
        if self_test and ticks["n"] == 5:
            state["drag"] = 1; state["mx"], state["my"] = 0.3, 0.2      # simulate a finger push
        with lock:
            snap, sfps, speed = shared["snap"], shared["sim_fps"], shared["speed"]
        if snap is not None and snap is not ticks["seen"]:               # draw only when there is a new frame
            ticks["seen"] = snap
            photo.configure(data=rasterise(snap, size))
            for lab, (title, _, _, info) in zip(labels, snap):
                cv.itemconfigure(lab, text=f"{title}   {info}")
            ticks["draws"] += 1
        now = time.time()
        if now - ticks["t"] >= 0.5:
            ticks["fps"] = ticks["draws"] / (now - ticks["t"]); ticks["draws"] = 0; ticks["t"] = now
            root.title(f"LESNet fluid | {SCENES[twin.scene][0]} | draw {ticks['fps']:.0f} fps | sim {sfps:.0f} steps/s "
                       f"({speed:.2f}x real time{'' if not state['fast'] else ', uncapped'}) | gravity {math.degrees(twin.g_angle):.0f} deg")
        root.after(4, tick)

    tick()
    root.mainloop()
    state["quit"] = True
    th.join(timeout=2)
    if self_test:
        print(f"self-test: draw {ticks['fps']:.0f} fps, sim {shared['sim_fps']:.0f} steps/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--scene", default="1")
    ap.add_argument("--self-test", type=int, default=0, help="open an invisible window for N frames and exit")
    ap.add_argument("--size", type=int, default=300, help="panel size in pixels (smaller = faster drawing)")
    ap.add_argument("--no-native", action="store_true", help="learned models only (no native panel, no error numbers)")
    ap.add_argument("--fast", action="store_true", help="do not cap the simulation at real time ('f' toggles it in the window)")
    a = ap.parse_args()
    runs = a.runs
    if runs is None:
        d = os.path.join(HERE, "runs")
        runs = [n for n in sorted(os.listdir(d)) if os.path.exists(os.path.join(d, n, "ckpt.pt")) and n != "smoke"][:2] if os.path.isdir(d) else []
    twin = Twin(runs, a.scene, use_native=not a.no_native)
    if a.record:
        render_gif(twin, a.record, a.seconds, controls=(a.scene != "5"))
    else:
        run_window(twin, a.self_test, realtime=not a.fast, size=a.size)


if __name__ == "__main__":
    main()
