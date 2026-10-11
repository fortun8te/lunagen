#!/usr/bin/env python3
"""lunagen look: make renders match a reference photo's look, measured, not guessed.

The image model always renders crisp, clean, HD output, even when the prompt asks
for a blurry low-res film still. Prompting cannot fix that, so this does it after:
it measures the reference (effective resolution, tone curve, colour cast,
saturation, grain) and pushes each render to the same numbers.

  lunagen look analyze REF.jpg [IMG ...]          print the measured look
  lunagen look match REF.jpg -d outdir [--strength 0.85]
        -> outdir/look/<name>.jpg for every <name>.png in outdir, plus _look-sheet.jpg
  lunagen look match REF.jpg a.png b.png -o folder

Needs numpy and Pillow only.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------- colour maths (sRGB <-> CIELAB, D65), float arrays in 0..1 ----------
_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_MI = np.linalg.inv(_M)
_WP = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WP
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * _WP
    lin = np.clip(xyz @ _MI.T, 0, 1)
    return np.clip(np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055), 0, 1)


def load(p) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def to_img(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def _jpeg(im: Image.Image, q: int) -> Image.Image:
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q, subsampling=2)
    b.seek(0)
    return Image.open(b).convert("RGB")


# ---------- measurements ----------
def detail(rgb: np.ndarray) -> float:
    """How much fine detail the image holds: the spatial frequency (as a fraction of
    the image width) below which 99% of its non-DC spectral energy sits. A sharp
    1024px render scores ~0.33, a 200px thumbnail blown up to 1024 ~0.19. Compare
    images of similar content only; use it relatively, as match() does."""
    g = rgb.mean(-1)
    H, W = g.shape
    F = np.abs(np.fft.fft2((g - g.mean()) * np.outer(np.hanning(H), np.hanning(W)))) ** 2
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    m = (r > 0.01) & (r <= 0.5)
    o = np.argsort(r[m])
    c = np.cumsum(F[m][o])
    return float(2 * r[m][o][np.searchsorted(c, 0.99 * c[-1])])


def effective_width(rgb: np.ndarray) -> int:
    """Rough real width, from detail() calibrated on blown-up thumbnails of a
    1024px render (200px -> 0.19, 300 -> 0.23, 450 -> 0.27, 600 -> 0.30, sharp 0.33)."""
    d = detail(rgb)
    pts = ([0.12, 0.183, 0.228, 0.271, 0.297, 0.332], [64, 200, 300, 450, 600, 1024])
    return int(np.interp(d, *pts) * rgb.shape[1] / 1024)


def grain(rgb: np.ndarray, width: int) -> float:
    """Noise left in the midtones once the image is shrunk to its effective width."""
    im = to_img(rgb)
    W, H = im.size
    s = im.resize((width, max(1, round(width * H / W))), Image.BOX)
    a = np.asarray(s, np.float32).mean(-1)
    r = a - np.asarray(s.filter(ImageFilter.GaussianBlur(1.2)), np.float32).mean(-1)
    m = (a > 40) & (a < 200)
    return float(r[m].std()) if m.any() else 0.0


def analyze(p) -> dict:
    rgb = load(p)
    lab = rgb2lab(rgb)
    L = lab[..., 0]
    mid = (L > 15) & (L < 85)
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = np.where(mx > 0.08, (mx - mn) / np.maximum(mx, 1e-6), 0)
    ew = effective_width(rgb)
    return {
        "size": f"{rgb.shape[1]}x{rgb.shape[0]}",
        "effective_width": ew,
        "detail": round(detail(rgb), 3),
        "L_p1_p50_p99": [round(float(np.percentile(L, q)), 1) for q in (1, 50, 99)],
        "black_share": round(float((L < 8).mean()), 2),
        "mid_a": round(float(lab[..., 1][mid].mean()), 1) if mid.any() else 0.0,
        "mid_b": round(float(lab[..., 2][mid].mean()), 1) if mid.any() else 0.0,
        "shadow_b": round(float(lab[..., 2][L < 15].mean()), 1) if (L < 15).any() else 0.0,
        "saturation": round(float(sat[mx > 0.08].mean()), 3) if (mx > 0.08).any() else 0.0,
        "grain": round(grain(rgb, ew), 2),
    }


def describe(s: dict) -> str:
    """The measured look in words, for writing the next prompt."""
    res = ("a blown-up thumbnail" if s["detail"] < 0.2 else
           "soft, low-res" if s["detail"] < 0.28 else "sharp, full-res")
    a, b = s["mid_a"], s["mid_b"]
    cast = (("magenta" if a > 4 else "green" if a < 1 else "neutral") + "/" +
            ("warm" if b > 14 else "cool" if b < 8 else "neutral"))
    return (f"detail {s['detail']} (~{s['effective_width']}px real): {res}; "
            f"{int(s['black_share'] * 100)}% near-black (blacks b {s['shadow_b']}, "
            f"{'blue' if s['shadow_b'] < -2 else 'warm' if s['shadow_b'] > 2 else 'neutral'}); "
            f"midtone cast {cast} (a {a}, b {b}); "
            f"saturation {s['saturation']}; grain {s['grain']}")


# ---------- matching ----------
def _quantile_map(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    qs = np.linspace(0, 100, 257)
    return np.interp(src, np.percentile(src, qs), np.percentile(ref, qs))


def match(render_path, ref_path, out_path, strength: float = 0.85, seed: int = 1) -> dict:
    ref = load(ref_path)
    src = load(render_path)
    rl, sl = rgb2lab(ref), rgb2lab(src)

    # 1. tone: map the render's lightness distribution onto the reference's
    L = sl[..., 0]
    out = sl.copy()
    out[..., 0] = L + strength * (_quantile_map(L, rl[..., 0]) - L)

    # 2. colour, per tonal band: shadows, mids and highlights each move to the
    #    reference's own a/b, so a tint in the render's blacks (a navy void) is
    #    pulled to the reference's black instead of inheriting the midtone shift
    bands = [(0, 15), (15, 85), (85, 101)]
    centres = [7.5, 50.0, 92.5]
    # chroma first: scale a/b so midtone colourfulness matches (a saturated render
    # stays saturated under mean/spread shifts alone)
    rm, sm = (rl[..., 0] > 15) & (rl[..., 0] < 85), (L > 15) & (L < 85)
    if rm.sum() > 500 and sm.sum() > 500:
        cr = np.hypot(rl[..., 1], rl[..., 2])[rm].mean() / (np.hypot(sl[..., 1], sl[..., 2])[sm].mean() + 1e-6)
        sl = sl.copy()
        sl[..., 1:] *= 1 + strength * (cr - 1)
        out[..., 1:] = sl[..., 1:]
    for c in (1, 2):
        shifts, scale = [], 1.0
        for lo_, hi_ in bands:
            r_m = (rl[..., 0] >= lo_) & (rl[..., 0] < hi_)
            s_m = (L >= lo_) & (L < hi_)
            ok = r_m.sum() > 500 and s_m.sum() > 500
            shifts.append(rl[..., c][r_m].mean() - sl[..., c][s_m].mean() if ok else 0.0)
            if (lo_, hi_) == (15, 85) and ok:
                scale = (rl[..., c][r_m].std() + 1e-6) / (sl[..., c][s_m].std() + 1e-6)
        mu = sl[..., c][(L >= 15) & (L < 85)].mean()
        mid_w = np.clip(1 - np.abs(L - 50) / 35, 0, 1)  # spread scaling only in mids
        t = mu + (sl[..., c] - mu) * (1 + mid_w * (scale - 1)) + np.interp(L, centres, shifts)
        out[..., c] = sl[..., c] + strength * (t - sl[..., c])
    graded = lab2rgb(out)

    # 3. resolution: find the shrink width whose result holds the same fine detail
    #    as the reference (bisection on detail()), so "how blurry" is measured, not guessed
    W, H = src.shape[1], src.shape[0]
    g_img = to_img(graded)
    target = detail(ref)
    ref_w = max(64, effective_width(ref))
    extra = max(0.0, grain(ref, ref_w) ** 2 - grain(graded, ref_w) ** 2) ** 0.5
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (H, W)).astype(np.float32)

    def degrade(w: int) -> Image.Image:
        h = max(1, round(w * H / W))
        small = g_img.resize((w, h), Image.LANCZOS)
        if extra > 0.3:  # 4. grain: only what the reference has beyond the render's own
            a = np.asarray(small, np.float32)
            a += extra * np.asarray(Image.fromarray(noise).resize((w, h)), np.float32)[..., None]
            small = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        small = _jpeg(small, 45)
        return _jpeg(small.resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(0.6)), 75)

    lo, hi = 64, W
    for _ in range(9):
        mid = (lo + hi) // 2
        if detail(np.asarray(degrade(mid), np.float32) / 255.0) > target:
            hi = mid
        else:
            lo = mid
    big = degrade(lo)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    big.save(out_path, "JPEG", quality=90)
    return {"width_used": lo, "grain_added": round(extra, 2)}


def sheet(ref_path, paths: list[Path], out: Path, h: int = 520) -> None:
    items = [("REFERENCE", Path(ref_path))] + [(p.stem, p) for p in paths]
    tiles = []
    for name, p in items:
        im = Image.open(p).convert("RGB")
        im = im.resize((round(im.width * h / im.height), h))
        tiles.append((name, im))
    cols = min(5, len(tiles))
    tw = max(t.width for _, t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    s = Image.new("RGB", (cols * (tw + 8), rows * (h + 26)), (24, 24, 24))
    d = ImageDraw.Draw(s)
    for i, (name, im) in enumerate(tiles):
        x, y = (i % cols) * (tw + 8), (i // cols) * (h + 26)
        s.paste(im, (x, y + 22))
        d.text((x + 3, y + 5), name, fill=(255, 220, 0) if i else (120, 220, 255))
    s.save(out, "JPEG", quality=85)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lunagen look", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("images", nargs="+")
    m = sub.add_parser("match")
    m.add_argument("ref")
    m.add_argument("images", nargs="*", help="renders; default: every PNG in -d")
    m.add_argument("-d", "--dir", help="batch folder: reads <dir>/*.png, writes <dir>/look/")
    m.add_argument("-o", "--out", help="output folder (default <dir>/look)")
    m.add_argument("--strength", type=float, default=0.85, help="tone and colour pull, 0..1")
    args = p.parse_args(argv)

    if args.cmd == "analyze":
        for f in args.images:
            s = analyze(f)
            print(f"{Path(f).name}: {describe(s)}")
        return 0

    if not args.images and not args.dir:
        p.error("give renders or -d folder")
    imgs = [Path(x) for x in args.images] or sorted(
        x for x in Path(args.dir).expanduser().glob("*.png") if ".prev" not in x.name)
    out = Path(args.out or (Path(args.dir) / "look")).expanduser()
    done = []
    for f in imgs:
        dst = out / f"{f.stem}.jpg"
        r = match(f, args.ref, dst, args.strength)
        done.append(dst)
        print(f"look {f.stem}: {r['width_used']}px detail, grain +{r['grain_added']}")
    sheet(args.ref, done, out / "_look-sheet.jpg")
    print(f"ref: {describe(analyze(args.ref))}")
    print(f"out: {describe(analyze(done[0]))}  (first render, after)")
    print(f"sheet: {out / '_look-sheet.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
