"""crawl_bench.py - why static fine detail crawls under the TAA resolve, measured on a synthetic livery.

A livery-like scene is drawn at 8x supersampling: dark text strokes 0.6-2 px wide on a light fuselage,
0.3-0.5 px panel lines, sub-pixel bright rivet specks. Each frame is rasterised the way the GPU does it
(one point sample per pixel, displaced by the frame's Halton jitter), then run through a numpy copy of
taa.comp's resolve for a static pixel:

  currA = Catmull-Rom fetch of the raster at the un-jittered position         (taa.cr_unjitter=1)
  box   = 3x3 bilinear taps at the same offset, tm -> YCoCg, mu +- clipK*sigma within [nmin, nmax]
          clipK = varclip * mix(0.55, 1, smoothstep(0.02, 0.15, luma(tm(currA))))   ("tighter in the dark")
  moved = max(|clamp(h) - h| - moved_dead * floorS, 0) / max(|sigma|, floorS)
  a     = clamp(alpha + gain * moved * alpha, alpha, 1);  out = mix(history, currA, a)   (history NOT clamped)

Optionally the camera wobbles by a sub-pixel amount each frame (the chase camera's spring) with exact or
mis-scaled vectors, history then fetched with Catmull-Rom at the reprojected position.

Reported over the steady state (last 16 frames of 240):
  crawl  - mean |(out_t - out_t-1) - (truth_t - truth_t-1)| in tonemapped luma over detail pixels, x1000:
           frame-to-frame change the scene itself did not make (what the eye sees crawl)
  error  - mean |out_t - truth| over detail pixels, x1000 (truth = the 8x8 box-filtered scene)
Lower is better for both.

  python tools/nn/crawl_bench.py
SPDX-License-Identifier: GPL-3.0-or-later
"""
import argparse
import numpy as np

SS = 8
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b; r += f * (i % b); i //= b
    return r


def scene(n, seed=3):
    """n x n pixels at SS x supersampling, RGB linear."""
    rng = np.random.default_rng(seed)
    N = n * SS
    y, x = np.mgrid[0:N, 0:N].astype(np.float32) / SS          # in output pixels
    img = np.empty((N, N, 3), np.float32)
    img[:] = np.array([0.62, 0.64, 0.68], np.float32) * (0.9 + 0.1 * (x / n))[..., None]
    # text strokes: rotated bars, 0.6-2 px wide, dark blue livery ink
    for _ in range(70):
        cx, cy = rng.uniform(4, n - 4, 2); L = rng.uniform(3, 12); w = rng.uniform(0.6, 2.0)
        th = rng.uniform(0, np.pi)
        dx, dy = x - cx, y - cy
        u = dx * np.cos(th) + dy * np.sin(th); v = -dx * np.sin(th) + dy * np.cos(th)
        m = (np.abs(u) < L / 2) & (np.abs(v) < w / 2)
        img[m] = np.array([0.03, 0.05, 0.12], np.float32)
    # panel lines: long, 0.3-0.5 px, slightly darker
    for _ in range(14):
        th = rng.uniform(0, np.pi); c = rng.uniform(0, n); w = rng.uniform(0.3, 0.5)
        d = np.abs(x * np.sin(th) - y * np.cos(th) - c * np.sin(th) + (n / 2) * np.cos(th) * 0)
        img[(d % n) < w] *= 0.6
    # rivets: sub-pixel bright specks (specular highlights)
    for _ in range(260):
        cx, cy = rng.uniform(1, n - 1, 2); r = rng.uniform(0.15, 0.35)
        img[(x - cx) ** 2 + (y - cy) ** 2 < r * r] = 1.6
    return img


def raster(hi, n, ox, oy):
    """GPU-style point sampling: pixel p takes the scene at p + 0.5 + (ox, oy)."""
    N = n * SS
    yy = ((np.arange(n) + 0.5 + oy) * SS).astype(np.int64).clip(0, N - 1)
    xx = ((np.arange(n) + 0.5 + ox) * SS).astype(np.int64).clip(0, N - 1)
    return hi[yy][:, xx]


def shift(img, dx, dy, kind):
    """sample img at (x + dx, y + dy), separable, edge-clamped. kind: 'catmull' or 'bilinear'."""
    def axis(a, d, ax):
        i0 = int(np.floor(d)); t = d - i0
        if kind == "bilinear":
            taps, w = [0, 1], [1 - t, t]
        elif kind in ("lanczos3", "lanczos2"):
            R = 3 if kind == "lanczos3" else 2
            taps = list(range(-R + 1, R + 1))
            xs = np.array([t - k for k in taps], np.float64)
            w = np.where(np.abs(xs) < 1e-6, 1.0, R * np.sin(np.pi * xs) * np.sin(np.pi * xs / R) / (np.pi * np.pi * xs * xs + 1e-12))
            w = list(w / w.sum())
        elif kind.startswith("cubicC"):   # Mitchell-Netravali B=0, C=<value>: C 0.5 = Catmull-Rom, larger = sharper
            C = float(kind[6:]); taps = [-1, 0, 1, 2]
            def k(x):
                x = abs(x)
                if x < 1: return (2 - C) * x ** 3 + (C - 3) * x ** 2 + 1
                if x < 2: return -C * x ** 3 + 5 * C * x ** 2 - 8 * C * x + 4 * C
                return 0.0
            w = [k(t - tk) for tk in taps]
        else:
            t2, t3 = t * t, t * t * t
            taps = [-1, 0, 1, 2]
            w = [-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1, -1.5 * t3 + 2 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2]
        n = a.shape[ax]; idx = np.arange(n)
        out = 0
        for k, wk in zip(taps, w):
            out = out + wk * np.take(a, np.clip(idx + i0 + k, 0, n - 1), axis=ax)
        return out
    return axis(axis(img, dx, 1), dy, 0)


def shift_dering(img, dx, dy, kind):
    """shift() clamped to the 2x2 source texels around each sample - FSR-style de-ringing"""
    out = shift(img, dx, dy, kind)
    h, w = img.shape[:2]
    ix, iy = int(np.floor(dx)), int(np.floor(dy))
    xs = [np.clip(np.arange(w) + ix + k, 0, w - 1) for k in (0, 1)]
    ys = [np.clip(np.arange(h) + iy + k, 0, h - 1) for k in (0, 1)]
    q = [img[y][:, x] for y in ys for x in xs]
    return np.clip(out, np.minimum.reduce(q), np.maximum.reduce(q))


def tm(c): return c / (1.0 + (c @ LUMA))[..., None]
def ycocg(c):
    return np.stack([0.25 * c[..., 0] + 0.5 * c[..., 1] + 0.25 * c[..., 2], 0.5 * c[..., 0] - 0.5 * c[..., 2],
                     -0.25 * c[..., 0] + 0.5 * c[..., 1] - 0.25 * c[..., 2]], -1)
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)


def neigh(a):
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge"); h, w = a.shape[:2]
    return np.stack([p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])


def run(hi, n, cfg, frames=240, wobble=0.0, vel_err=0.0, seed=5):
    rng = np.random.default_rng(seed)
    alpha, gain, varclip, md = cfg.get("alpha", 0.05), cfg.get("gain", 4.0), cfg.get("varclip", 1.5), cfg.get("moved_dead", 0.0)
    dark = cfg.get("dark", True); mode = cfg.get("mode", "gain")
    cam = np.zeros(2); hist = None; outs = []; truths = []
    for f in range(frames):
        jx, jy = halton(f % 8 + 1, 2) - 0.5, halton(f % 8 + 1, 3) - 0.5
        # the chase camera's spring: a bounded, mean-reverting sub-pixel sway (stationary std = wobble)
        new = 0.7 * cam + rng.normal(0, wobble * np.sqrt(1 - 0.49), 2) if wobble > 0 else cam
        step = new - cam; cam = new
        r = raster(hi, n, cam[0] + jx, cam[1] + jy)                 # scene seen through a camera at `cam`
        cf = cfg.get("cfilt", "catmull")                           # taa.cr_unjitter: catmull (1) / bilinear (0)
        currA = np.maximum(shift_dering(r, -jx, -jy, cf) if cfg.get("cdering") else shift(r, -jx, -jy, cf), 0)   # un-jittered current
        T = ycocg(tm(neigh(shift(r, -jx, -jy, "bilinear"))))       # 9 taps, tm-YCoCg
        mu, sg = T.mean(0), T.std(0); nmin, nmax = T.min(0), T.max(0)
        lc = tm(currA) @ LUMA
        k = varclip * (0.55 + 0.45 * smoothstep(0.02, 0.15, lc))[..., None] if dark else varclip
        lo, hiB = np.maximum(mu - k * sg, nmin), np.minimum(mu + k * sg, nmax)
        if hist is None:
            hist = currA.copy()
        else:
            # reproject: the scene moved by -step on screen; history was rendered with the old camera
            v = step * (1.0 + vel_err)
            hf = cfg.get("hfilt", "catmull")
            hist = np.maximum(shift_dering(hist, v[0], v[1], hf) if cfg.get("dering") else shift(hist, v[0], v[1], hf), 0)
        hRaw = ycocg(tm(hist)); hy = np.clip(hRaw, lo, hiB)
        floorS = 0.01 * lc + 1e-4
        if mode == "gain":
            moved = np.maximum(np.linalg.norm(hy - hRaw, axis=-1) - md * floorS, 0) / np.maximum(np.linalg.norm(sg, axis=-1), floorS)
            a = np.clip(alpha + gain * moved * alpha, alpha, 1.0)[..., None]
            out = hist + (currA - hist) * a
        elif mode == "clamp":   # classic: clamp history in tm-YCoCg (scaled back by the history's own tm factor), fixed alpha
            Y = hy; rgb_t = np.stack([Y[..., 0] + Y[..., 1] - Y[..., 2], Y[..., 0] + Y[..., 2], Y[..., 0] - Y[..., 1] - Y[..., 2]], -1)
            hc = np.maximum(rgb_t * (1.0 + hist @ LUMA)[..., None], 0)
            out = hc + (currA - hc) * alpha
        hist = np.maximum(out, 0)
        if f >= frames - 16:
            outs.append(tm(hist) @ LUMA)
            truths.append(tm(raster_box(hi, n, cam)) @ LUMA)
    outs = np.stack(outs); truths = np.stack(truths)
    return outs, truths


_box_cache = {}
def raster_box(hi, n, cam):
    """the ideal anti-aliased frame for camera `cam`: the 8x8 box average around p + 0.5 + cam"""
    key = (round(cam[0] * SS), round(cam[1] * SS))
    if key not in _box_cache:
        N = n * SS
        sh = np.roll(hi, (-key[1], -key[0]), axis=(0, 1))
        _box_cache[key] = sh.reshape(n, SS, n, SS, 3).mean((1, 3))
        if len(_box_cache) > 64: _box_cache.pop(next(iter(_box_cache)))
    return _box_cache[key]


def score(outs, truths, detail):
    crawl = np.abs(np.diff(outs, axis=0) - np.diff(truths, axis=0)).mean(0)   # change the truth did not make
    err = np.abs(outs - truths).mean(0)
    m = detail
    return crawl[m].mean() * 1000, err[m].mean() * 1000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()
    hi = scene(a.n)
    t0 = tm(raster_box(hi, a.n, np.zeros(2))) @ LUMA
    g = np.abs(np.gradient(t0)[0]) + np.abs(np.gradient(t0)[1])
    p = np.pad(g, 1, mode="edge"); gm = np.max([p[1 + dy:1 + dy + a.n, 1 + dx:1 + dx + a.n] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 0)
    detail = gm > 0.02
    detail[:3] = detail[-3:] = False; detail[:, :3] = detail[:, -3:] = False
    cfgs = [
        ("shipped (gain 4, dead 0, varclip 1.5, dark)", {}),
        ("history Lanczos-3", {"hfilt": "lanczos3"}),
        ("history Lanczos-3 + de-ring", {"hfilt": "lanczos3", "dering": True}),
        ("history Lanczos-2", {"hfilt": "lanczos2"}),
        ("history Catmull + de-ring", {"dering": True}),
        ("history cubic C=0.75", {"hfilt": "cubicC0.75"}),
        ("history cubic C=1.0", {"hfilt": "cubicC1.0"}),
        ("history bilinear", {"hfilt": "bilinear"}),
        ("alpha 0.1", {"alpha": 0.1}),
        ("alpha 0.2", {"alpha": 0.2}),
        ("moved_dead 1", {"moved_dead": 1.0}),
        ("moved_dead 3", {"moved_dead": 3.0}),
        ("no dark tightening", {"dark": False}),
        ("varclip 2.5", {"varclip": 2.5}),
        ("gain 1", {"gain": 1.0}),
        ("gain 0 (never reject: reprojection check)", {"gain": 0.0}),
        ("classic clamp, fixed alpha", {"mode": "clamp"}),
        ("alpha 1.0 (no TAA)", {"alpha": 1.0}),
    ]
    for label, extra in (("static", {}), ("wobble 0.3 px, exact vectors", {"wobble": 0.3}),
                         ("wobble 0.3 px, vectors 20% off", {"wobble": 0.3, "vel_err": 0.2})):
        print(f"\n== {label} ==   {'crawl':>7s} {'error':>7s}")
        for name, cfg in cfgs:
            outs, truths = run(hi, a.n, cfg, **extra)
            c, e = score(outs, truths, detail)
            print(f"{name:46s} {c:7.2f} {e:7.2f}", flush=True)


if __name__ == "__main__":
    main()
