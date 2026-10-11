#!/usr/bin/env python3
"""Offline checks for lunagen (no rendering, no network). Run: python3 -I tools/test_lunagen.py
Run it after every edit to lunagen.py or ocr/ocr.swift."""
import importlib.util
import json
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lg", HERE / "lunagen.py")
lg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lg)
fails = []


def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


def jobs_file(d, rows):
    p = d / "jobs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


d = Path(tempfile.mkdtemp())

# copy matching ignores case, spacing and punctuation
check("compact", lg._compact("No Relief!") == lg._compact("no  relief"))

# versions: three redos keep three old copies
for v in ("v1", "v2", "v3"):
    (d / "A.png").write_text(v)
    lg.archive_png(d, "A")
got = {x.name: x.read_text() for x in d.glob("A.*")}
check("history", got == {"A.prev.png": "v3", "A.prev2.png": "v2", "A.prev3.png": "v1"})

# n variants and base name
jobs = lg.load_jobs(jobs_file(d, [{"name": "M", "prompt": "p", "n": 3}, {"name": "S", "prompt": "q"},
                                  {"name": "O", "prompt": "r", "n": 1}]), "auto")
check("variants: job n beats the default of 2",
      [j["name"] for j in jobs] == ["M", "M-2", "M-3", "S", "S-2", "O"] and jobs[1]["base"] == "M")
check("--n 1 gives one each", len(lg.load_jobs(jobs_file(d, [{"name": "S", "prompt": "q"}]), "auto", 1)) == 1)

# duplicate names abort
try:
    lg.load_jobs(jobs_file(d, [{"name": "A", "prompt": "x"}, {"name": "A", "prompt": "y"}]), "auto", 1)
    check("duplicates abort", False)
except SystemExit:
    check("duplicates abort", True)

# dry run: flags a bad size and a missing ref, spots a stale prompt, renders nothing
out = d / "out"
out.mkdir()
(out / "S.png").write_bytes(b"x" * 2000)
(out / "S.txt").write_text("old prompt")
jobs = lg.load_jobs(jobs_file(d, [{"name": "S", "prompt": "new prompt"},
                                  {"name": "B", "prompt": "x", "size": "tall", "refs": ["/nope.png"]}]), "auto", 1)
check("dry run finds problems", lg.dry_run(out, jobs, set(), 4) == 1)

# quality log: last render per name wins
(out / "_results.jsonl").write_text(
    json.dumps({"name": "S", "ok": True, "meta": {"quality": "low"}}) + "\n"
    + json.dumps({"name": "S", "ok": True, "meta": {"quality": "medium"}}) + "\n")
check("latest quality", lg.latest_quality(out) == {"S": "medium"})

# look: Lab round trip, detail tracks real resolution, match hits the reference's detail
import numpy as np
from PIL import Image, ImageFilter

lk = lg._look_module()
rgb = np.random.default_rng(0).random((40, 40, 3)).astype(np.float32)
check("look: lab round trip", float(np.abs(lk.lab2rgb(lk.rgb2lab(rgb)) - rgb).max()) < 2e-3)

rng = np.random.default_rng(1)
sharp = Image.fromarray((rng.random((96, 64)) * 255).astype(np.uint8)).resize((512, 768), Image.NEAREST)
sharp = sharp.filter(ImageFilter.GaussianBlur(1)).convert("RGB")
blown = lambda w: sharp.resize((w, w * 3 // 2), Image.BOX).resize(sharp.size, Image.BICUBIC)
arr = lambda im: np.asarray(im, np.float32) / 255
ds = [lk.detail(arr(blown(w))) for w in (100, 200, 350)] + [lk.detail(arr(sharp))]
check("look: detail rises with real resolution", ds == sorted(ds) and ds[0] < ds[-1] * 0.8)

ref_p, src_p, dst_p = d / "ref.png", d / "src.png", d / "look" / "src.jpg"
blown(110).save(ref_p)
sharp.save(src_p)
lk.match(src_p, ref_p, dst_p)
got, want = lk.detail(arr(Image.open(dst_p))), lk.detail(arr(Image.open(ref_p)))
check("look: match lands on the reference's detail", Image.open(dst_p).size == sharp.size
      and abs(got - want) < 0.25 * want)

jl = lg.load_jobs(jobs_file(d, [{"name": "L", "prompt": "p", "look": str(ref_p)}]), "auto", 1)
check("look: job keeps its look reference", jl[0]["look"] == str(ref_p))
lo = d / "lookrun"
lo.mkdir()
sharp.save(lo / "L.png")
check("look: batch writes look/<name>.jpg", lg.apply_looks(lo, jl) == 1 and (lo / "look" / "L.jpg").exists())
check("look: dry run flags a missing look ref", lg.dry_run(
    lo, lg.load_jobs(jobs_file(d, [{"name": "M", "prompt": "p", "look": "/nope.jpg"}]), "auto", 1), set(), 4) == 1)

print("\nall passed" if not fails else f"\n{len(fails)} failed: {fails}")
raise SystemExit(1 if fails else 0)
