"""sharpen_eval.py - score post-resolve sharpen variants against the burst-accumulated truth.

The capture's `resolved` plane is history[hw_]: the resolve's output BEFORE the sharpen and FSCAA's CMAA2
(taa.h, rp.resolved). pack_aa.accumulate builds an independent multi-sample truth for the same frame. So
any post-resolve filter can be applied offline and scored against the truth, split by region:

  edge     - the top 5% of truth gradient: geometric edges, where aliasing lives
  texture  - the middle of the gradient range: surface detail a sharpen exists to restore
  all      - every pixel with a reliable truth (>= --min-count samples)

Error is mean |x - truth| in tonemapped space (c / (1 + luma)), x1000. Lower is better. detail/truth is the
mean |Laplacian| over the texture region relative to the truth's: below 1 is softer, above 1 crisper.

  python tools/nn/sharpen_eval.py --root E:/NNCap --session 2026-09-11_200844_fresh --bursts 20
SPDX-License-Identifier: GPL-3.0-or-later
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack_aa as pa
import train_captures as tc
from quick_train import tm, luma


def taps(x):
    """x (C,H,W) -> the 3x3 neighbourhood, each (C,H,W): a b c / d e f / g h i (edge-clamped)."""
    p = F.pad(x[None], (1, 1, 1, 1), mode="replicate")[0]
    H, W = x.shape[-2:]
    return [p[:, dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)]


def sharpen_lap(e_lin, conf, amount):
    """The shipped MODE_SHARPEN: 4-neighbour Laplacian, confidence-gated, clamped to the 5-tap range."""
    a, b, c, d, e, f, g, h, i = taps(e_lin)
    lo = torch.minimum(e, torch.minimum(torch.minimum(b, h), torch.minimum(d, f)))
    hi = torch.maximum(e, torch.maximum(torch.maximum(b, h), torch.maximum(d, f)))
    s = e + amount * conf * (4 * e - b - h - d - f)
    return torch.clamp(torch.minimum(torch.maximum(s, lo), hi), min=0.0)


def sharpen_cas(e_lin, conf, sharpness):
    """AMD FidelityFX CAS (no-scale path). The adaptive weight is computed on tonemapped values (the scene
    can exceed 1); the filter is applied to the linear colour. Confidence-gated like the shipped pass."""
    T = taps(tm(e_lin[None])[0]); L = taps(e_lin)
    a, b, c, d, e, f, g, h, i = T
    mn = torch.minimum(torch.minimum(torch.minimum(d, e), torch.minimum(f, b)), h)
    mn2 = torch.minimum(torch.minimum(mn, a), torch.minimum(torch.minimum(c, g), i))
    mx = torch.maximum(torch.maximum(torch.maximum(d, e), torch.maximum(f, b)), h)
    mx2 = torch.maximum(torch.maximum(mx, a), torch.maximum(torch.maximum(c, g), i))
    mn = mn + mn2; mx = mx + mx2
    amp = torch.sqrt(torch.clamp(torch.minimum(mn, 2.0 - mx) / mx.clamp_min(1e-6), 0.0, 1.0))
    w = amp * (-1.0 / (8.0 + (5.0 - 8.0) * sharpness)) * conf
    la, lb, lc, ld, le, lf, lg, lh, li = L
    out = (lb * w + ld * w + lf * w + lh * w + le) / (1.0 + 4.0 * w)
    lo = torch.minimum(le, torch.minimum(torch.minimum(lb, lh), torch.minimum(ld, lf)))
    hi = torch.maximum(le, torch.maximum(torch.maximum(lb, lh), torch.maximum(ld, lf)))
    return torch.clamp(torch.minimum(torch.maximum(out, lo), hi), min=0.0)


# Not in build/ itself: build/vulkan-1.dll is the layer's shim and would load first.
# Build: g++ -O1 -std=c++17 -static tools/cmaa2_file.cpp -o build/evaltools/cmaa2_file.exe
#        -I"$VULKAN_SDK/Include" -I src/vklayer -L"$VULKAN_SDK/Lib" -lvulkan-1
CMAA2_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build", "evaltools", "cmaa2_file.exe")


def cmaa2(x, preset):
    """the shipped CMAA2 kernels (tools/cmaa2_file.cpp, in place, as after the resolve) on x (3,H,W)"""
    H, W = x.shape[-2:]
    a = np.ones((H, W, 4), np.float16); a[..., :3] = x.permute(1, 2, 0).cpu().numpy().astype(np.float16)
    d = tempfile.mkdtemp(); fi, fo = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
    a.tofile(fi)
    r = subprocess.run([CMAA2_EXE, fi, str(W), str(H), fo, str(preset)], capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError("cmaa2_file failed: " + r.stdout + r.stderr)
    o = np.fromfile(fo, np.float16).reshape(H, W, 4)[..., :3].astype(np.float32)
    os.remove(fi); os.remove(fo); os.rmdir(d)
    return torch.from_numpy(o.transpose(2, 0, 1).copy()).to(x.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="E:/NNCap")
    ap.add_argument("--session", default="2026-09-11_200844_fresh", help="only bursts from this session folder")
    ap.add_argument("--bursts", type=int, default=20)
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--min-count", type=float, default=4.0)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--cmaa", action="store_true", help="score whole pipelines: sharpen, then CMAA2 (shipped kernels)")
    a = ap.parse_args()
    # (name, sharpen kind, strength, CMAA2 preset or None)
    variants = [("none", None, 0.0, None), ("lap 0.25", "lap", 0.25, None), ("lap 0.5 (old shipped)", "lap", 0.5, None)] + \
               [("cas %.1f" % s, "cas", s, None) for s in (0.0, 0.3, 0.5, 0.7, 1.0)]
    if a.cmaa:
        variants = [("none", None, 0.0, None), ("none + cmaa2 p2", None, 0.0, 2),
                    ("lap 0.5", "lap", 0.5, None), ("lap 0.5 + cmaa2 p2 (old)", "lap", 0.5, 2),
                    ("cas 1.0", "cas", 1.0, None)] + \
                   [("cas 1.0 + cmaa2 p%d" % q, "cas", 1.0, q) for q in (0, 1, 2, 3)]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tc.calibrate_warp(tc.list_frames([os.path.join(a.root, a.session)]))
    bl = [b for b in pa.bursts(a.root, a.min_len) if b[0] == a.session]
    step = max(1, len(bl) // a.bursts); bl = bl[::step][:a.bursts]
    tot = {v[0]: np.zeros(4) for v in variants}; n = 0
    for sess, bid, fl in bl:
        frames = [pa.Frame(f, m, dev) for f, m in fl]
        t = len(fl) - 1
        truth, cnt, _, _ = pa.accumulate(frames, t, a.tol, dev)
        f, m = fl[t]
        r = tc.load_plane(f, m, "resolved").astype(np.float32)
        r = np.clip(np.nan_to_num(r, nan=0.0, posinf=64.0, neginf=0.0), 0.0, 64.0)
        res = torch.from_numpy(r[..., :3].transpose(2, 0, 1).copy()).to(dev)
        conf = (1.0 - torch.from_numpy(r[..., 3].copy()).to(dev).clamp(0.0, 1.0))[None]
        ttm = tm(truth[None])[0]
        gl = luma(ttm[None])[0] if luma(ttm[None]).dim() == 3 else luma(ttm[None])[0, 0]
        gx = torch.zeros_like(gl); gy = torch.zeros_like(gl)
        gx[:, 1:-1] = (gl[:, 2:] - gl[:, :-2]) * 0.5; gy[1:-1, :] = (gl[2:, :] - gl[:-2, :]) * 0.5
        gm = torch.sqrt(gx * gx + gy * gy)
        ok = (cnt >= a.min_count)
        ok[:4, :] = False; ok[-4:, :] = False; ok[:, :4] = False; ok[:, -4:] = False
        if ok.sum() < 10000: continue
        q95 = torch.quantile(gm[ok][::7].float(), 0.95); q40 = torch.quantile(gm[ok][::7].float(), 0.40)
        q80 = torch.quantile(gm[ok][::7].float(), 0.80)
        masks = [ok & (gm >= q95), ok & (gm >= q40) & (gm < q80), ok]
        def detail(img):   # mean |Laplacian| of tonemapped luma over the texture region
            l = luma(tm(img[None]))[0]; l = l[0] if l.dim() == 3 else l
            lp = torch.zeros_like(l); lp[1:-1, 1:-1] = 4 * l[1:-1, 1:-1] - l[:-2, 1:-1] - l[2:, 1:-1] - l[1:-1, :-2] - l[1:-1, 2:]
            return float(lp.abs()[masks[1]].mean())
        dT = detail(truth)
        for name, kind, s, q in variants:
            x = res if kind is None else (sharpen_lap(res, conf, s) if kind == "lap" else sharpen_cas(res, conf, s))
            if q is not None: x = cmaa2(x, q)
            err = (tm(x[None])[0] - ttm).abs().mean(0)
            tot[name] += np.array([float(err[mk].mean()) for mk in masks] + [detail(x) / dT])
        n += 1
        print(f"  {sess}/{bid}: scored", flush=True)
    print(f"\n{n} bursts, error x1000 vs truth (lower is better)")
    print(f"{'variant':28s} {'edge':>8s} {'texture':>8s} {'all':>8s} {'detail/truth':>13s}")
    for name, _, _, _ in variants:
        e = tot[name] / max(n, 1)
        print(f"{name:28s} {e[0]*1000:8.3f} {e[1]*1000:8.3f} {e[2]*1000:8.3f} {e[3]:13.3f}")


if __name__ == "__main__":
    main()
