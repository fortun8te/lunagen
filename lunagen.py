#!/usr/bin/env python3
"""lunagen: lean ChatGPT image generation. One direct call per image, no agent.

Each image is a single POST to the Codex backend (your ChatGPT login in
~/.codex/auth.json) with a one-line instruction, your prompt and your refs.
No Codex agent session, no memory/skills/hooks preamble, no image bytes in any
model's context. Output stays one short line per image so the calling agent
spends almost nothing reading it.

  lunagen "prompt" -o out.png [-i ref.png ...] [--size 1024x1536] [--copy "exact text" ...]
  lunagen batch jobs.jsonl -d outdir [-j 4] [--n 2] [--redo A,B] [--redo-low] [--dry-run]
  lunagen check jobs.jsonl -d outdir      (checks + sheets only, no generation)
  lunagen look analyze REF.jpg | lunagen look match REF.jpg -d outdir   (see look.py)

jobs.jsonl, one job per line:
  {"name": "BLOOM-V1", "prompt": "...", "refs": ["/abs/ref.png"], "size": "1024x1536",
   "copy": ["Comfort", "should be your baseline.", "25% off with code EARLYBF25"]}
  optional "look": "/abs/reference.jpg" writes look/<name>.jpg matched to that photo's
  resolution, tone, colour and grain after the batch (the model always renders crisp HD)

batch skips names that already have a PNG (so it resumes), stops cleanly on the
usage limit, retries a failed job once, writes <name>.png + <name>.txt (the exact
prompt) and _results.jsonl. Then it auto-checks every image in the set:
  size  the output's aspect ratio matches the job's size
  copy  every "copy" string is in the image (local macOS OCR, free; letters,
        digits and %$ must match, case, spaces and other punctuation are ignored)
Results go to _checks.json; flagged images are listed and framed in red on the
_sheet-N.jpg contact sheets (6 per sheet), so QA costs one image read per six.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(os.path.realpath(__file__)).parent
ENGINE_PATH = Path(os.environ.get("LUNAGEN_ENGINE") or HERE / "chatgpt-imagegen.py")
DEFAULT_MODEL = os.environ.get("LUNAGEN_MODEL", "gpt-6-luna")
_PREV_RE = re.compile(r"\.prev\d*\.png$")
SHEET_PER = 6
SHEET_COLS = 3
SHEET_CELL_W = 500

# Homebrew python's OpenSSL points at a stale cert path; certifi fixes it.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", os.environ["SSL_CERT_FILE"])
except ImportError:
    pass


def _load_engine():
    spec = importlib.util.spec_from_file_location("chatgpt_imagegen", ENGINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load_engine()
_usage = threading.local()
_orig_stream = E._stream


def _stream_with_usage(*a, **k):
    for evt in _orig_stream(*a, **k):
        if evt.get("type") == "response.completed" and isinstance(evt.get("response"), dict):
            _usage.value = evt["response"].get("usage") or {}
        yield evt


E._stream = _stream_with_usage


def _payload(prompt: str, size: str, model: str, refs: list[tuple[str, str]]) -> dict:
    # quality is not settable: the subscription backend always serves "medium".
    tool = {"type": "image_generation", "output_format": "png"}
    if size and size != "auto":
        tool["size"] = size
    lead = "Call the image_generation tool exactly once. "
    if refs:
        lead += ("The attached images are references, numbered in attach order; use each "
                 "one only as the request describes. ")
    if size and size != "auto":
        # The backend ignores the tool's size field; the model only follows text.
        w, _, h = size.partition("x")
        shape = "portrait" if int(h) > int(w) else "landscape" if int(w) > int(h) else "square"
        lead += f"Set the tool's size to {size} ({shape}); the output must be {shape}. "
    text = lead + "Pass the request to the tool as written, without shortening it.\n\nRequest:\n" + prompt
    content = [{"type": "input_image", "image_url": f"data:{m};base64,{b}"} for b, m in refs]
    content.append({"type": "input_text", "text": text})
    return {
        "model": model,
        "stream": True,
        # The backend only accepts allowlisted instructions; this one is proven.
        "instructions": "You are an image generation assistant.",
        "input": [{"type": "message", "role": "user", "content": content}],
        "tools": [tool],
        "tool_choice": "required" if refs else "auto",
        "parallel_tool_calls": False,
        "store": False,
        "reasoning": {"effort": "low", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "text": {"verbosity": "low"},
    }


def generate(prompt: str, out: Path, refs: list[str], size: str, model: str,
             timeout: int = 300) -> dict:
    """One image. Never raises; returns a result dict."""
    t0 = time.monotonic()
    _usage.value = {}
    res = {"name": out.stem, "out": str(out), "ok": False}
    try:
        auth = E._load_auth()
        access, account_id, refresh = E._extract_access_token(auth)
        if not access:
            res["error"] = "no ChatGPT login in ~/.codex/auth.json (run: codex login)"
            return res
        loaded = [E._load_reference(r) for r in refs]
        payload = _payload(prompt, size, model, loaded)
        args = argparse.Namespace(timeout=timeout)
        limit = E._backend_limit("codex", E.CODEX_CONCURRENCY_DEFAULT)
        with E._concurrency_slot("codex", limit, False, t0):
            img, meta = E._run_codex_locked(args, False, float(E.DEFAULT_STALL_TIMEOUT), t0,
                                            payload, account_id, access, refresh,
                                            E._detect_codex_version(), auth)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(img)
        res.update(ok=True, bytes=len(img), revised_prompt=meta.get("revised_prompt", ""),
                   meta={k: v for k, v in meta.items()
                         if k in ("quality", "size", "background", "output_format", "model")})
        try:
            from PIL import Image

            with Image.open(out) as im:
                res["px"] = f"{im.width}x{im.height}"
        except Exception:  # noqa: BLE001
            pass
    except SystemExit as e:  # the engine exits with a message on failure
        res["error"] = str(e.code if e.code is not None else "failed")
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    res["secs"] = round(time.monotonic() - t0, 1)
    u = getattr(_usage, "value", {}) or {}
    res["tokens"] = {
        "input": u.get("input_tokens", 0),
        "cached": (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
        "output": u.get("output_tokens", 0),
    }
    err = res.get("error", "")
    if re.search(r"usage_limit_reached|HTTP 429", err):
        res["rate_limited"] = True
        m = re.search(r"resets_in_seconds\"?\s*:?\s*(\d+)", err)
        res["resets_in_s"] = int(m.group(1)) if m else None
        res["error"] = "usage limit reached"
    return res


def _tok(r: dict) -> int:
    t = r.get("tokens") or {}
    return t.get("input", 0) + t.get("output", 0)


def _line(r: dict) -> str:
    if r["ok"]:
        return f"OK   {r['name']}  {r.get('px', '?')}  {r['secs']}s  {_tok(r)} tok"
    return f"FAIL {r['name']}: {str(r.get('error'))[:200]}"


# ---------- auto-checks ----------

OCR_DIR = HERE / "ocr"
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def _compact(s: str) -> str:
    """What copy matching compares: lowercase letters, digits and %$£€&+."""
    return re.sub(r"[^\w%$£€&+]|_", "", s.translate(_QUOTES).lower())


def run_ocr(paths: list[Path]) -> dict[str, list[str]]:
    """Text lines per image via macOS Vision. Builds the helper on first use."""
    binary = OCR_DIR / "ocr"
    if not binary.exists():
        import subprocess
        subprocess.run(["swiftc", "-O", str(OCR_DIR / "ocr.swift"), "-o", str(binary)],
                       check=True, capture_output=True)
    import subprocess
    out = subprocess.run([str(binary), *map(str, paths)], capture_output=True, text=True,
                         timeout=600).stdout
    res = {}
    for raw in out.splitlines():
        d = json.loads(raw)
        res[d["path"]] = d["lines"]
    return res


def _texts(lines: list[dict]) -> list[str]:
    return [x["t"] for x in lines]


def second_look(png: Path, lines: list[dict], idxs: list[int]) -> list[str]:
    """Re-read the region of the given OCR lines, cropped and enlarged.

    A small line inside a busy image is often misread ("relief" as "reliet");
    the same words cropped on their own read cleanly.
    """
    import subprocess
    import tempfile
    from PIL import Image

    boxes = [lines[i]["b"] for i in idxs if lines[i].get("b")]
    if not boxes:
        return []
    with Image.open(png) as im:
        im = im.convert("RGB")
        W, H = im.size
        x0 = min(b[0] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y0 = min(b[1] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        pad = 0.012
        box = (max(0, int((x0 - pad) * W)), max(0, int((1 - y1 - pad) * H)),
               min(W, int((x1 + pad) * W)), min(H, int((1 - y0 + pad) * H)))
        crop = im.crop(box)
        scale = 3 if crop.height < 200 else 2
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "crop.png"
        crop.save(tmp)
        got = run_ocr([tmp]).get(str(tmp), [])
    return _texts(got)


def _closest(want: str, lines: list[str]) -> tuple[str, float, list[int]]:
    from difflib import SequenceMatcher

    w = _compact(want)
    best, score, where = "", 0.0, []
    for size in (1, 2, 3):
        for i in range(len(lines) - size + 1):
            cand = " / ".join(lines[i:i + size])
            c = _compact(cand)
            # score the best same-length window, so a long line can still match a short want
            for k in range(max(1, len(c) - len(w) + 1)):
                r = SequenceMatcher(None, w, c[k:k + len(w)]).ratio()
                if r > score:
                    best, score, where = cand, r, list(range(i, i + size))
    return best, score, where


def check_image(job: dict, png: Path, lines: list[dict]) -> dict:
    from PIL import Image

    with Image.open(png) as im:
        w, h = im.size
    c = {"px": f"{w}x{h}", "flags": []}
    size = job.get("size") or "auto"
    if size != "auto":
        rw, _, rh = size.partition("x")
        if abs((w / h) / (int(rw) / int(rh)) - 1) > 0.03:
            c["flags"].append(f"size {w}x{h}, asked {size}")
    want = job.get("copy") or []
    texts = _texts(lines)
    if want:
        full = _compact(" ".join(texts))
        missing, rechecked = [], []
        for s in want:
            if _compact(s) not in full:
                read, score, where = _closest(s, texts)
                if where and score >= 0.6 and _compact(s) in _compact(" ".join(second_look(png, lines, where))):
                    rechecked.append(s)  # a misread on the first pass, clean on a close look
                    continue
                missing.append({"want": s, "closest_read": read, "match": round(score, 2)})
        c["copy"] = f"{len(want) - len(missing)}/{len(want)}"
        if rechecked:
            c["second_look_ok"] = rechecked
        for m in missing:
            c["flags"].append(f"copy missing \"{m['want']}\" (closest read: \"{m['closest_read']}\", {m['match']:.0%})")
        c["missing"] = missing
    c["ocr"] = texts
    return c


def run_checks(d: Path, jobs: list[dict]) -> dict[str, dict]:
    have = [(j, d / f"{j['name']}.png") for j in jobs if (d / f"{j['name']}.png").exists()]
    need_ocr = [p for j, p in have if j.get("copy")]
    try:
        ocr = run_ocr(need_ocr) if need_ocr else {}
    except Exception as e:  # noqa: BLE001
        print(f"warning: OCR unavailable, copy checks skipped ({type(e).__name__}: {e})")
        ocr = {}
    checks = {}
    quality = latest_quality(d)
    for j, p in have:
        jj = j if str(p) in ocr or not j.get("copy") else {**j, "copy": []}
        checks[j["name"]] = check_image(jj, p, ocr.get(str(p), []))
        if quality.get(j["name"]):
            checks[j["name"]]["quality"] = quality[j["name"]]
    (d / "_checks.json").write_text(json.dumps(checks, indent=1, ensure_ascii=False))
    flagged = {n: c for n, c in checks.items() if c["flags"]}
    with_copy = sum(1 for c in checks.values() if "copy" in c)
    print(f"checks: {len(checks)} images, {len(checks) - len(flagged)} clean, {len(flagged)} flagged "
          f"(copy checked on {with_copy})")
    for n, c in flagged.items():
        for f in c["flags"]:
            print(f"FLAG {n}: {f}")
    saved = [n for n, c in checks.items() if c.get("second_look_ok")]
    if saved:
        print(f"note: {len(saved)} first-pass text misreads cleared by a close second look")
    low = [n for n, c in checks.items() if c.get("quality") == "low"]
    if low:
        print(f"note: {len(low)} of {len(checks)} rendered at backend quality \"low\" (busy hours; "
              f"rerun them later with --redo-low): {','.join(low)}")
    return checks


def make_sheets(folder: Path, names: list[str] | None = None,
                checks: dict[str, dict] | None = None,
                bases: dict[str, str] | None = None) -> list[Path]:
    """Contact sheets. Variants of one concept sit side by side in one row, so judging a
    pair is one glance; single images flow three to a row."""
    from PIL import Image, ImageDraw, ImageFont

    checks = checks or {}
    bases = bases or {}
    if names is None:
        names = sorted(p.stem for p in folder.glob("*.png") if not p.name.startswith("_")
                       and not _PREV_RE.search(p.name))
    names = [n for n in names if (folder / f"{n}.png").exists()]
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(bases.get(n, n), []).append(n)
    m = min(4, max((len(g) for g in groups.values()), default=1))
    if m == 1:
        flat = [g[0] for g in groups.values()]
        rows = [flat[i:i + SHEET_COLS] for i in range(0, len(flat), SHEET_COLS)]
        per_sheet = 2
    else:
        rows = [g[i:i + m] for g in groups.values() for i in range(0, len(g), m)]
        per_sheet = max(1, SHEET_PER // m)
    for old in folder.glob("_sheet-*.jpg"):
        old.unlink()
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    except OSError:
        font = ImageFont.load_default()
    sheets = []
    label_h, gap, border = 40, 12, 6
    cols = SHEET_COLS if m == 1 else m
    for page, i in enumerate(range(0, len(rows), per_sheet), 1):
        chunk = rows[i:i + per_sheet]
        thumbs = []
        for row in chunk:
            r = []
            for n in row:
                im = Image.open(folder / f"{n}.png").convert("RGB")
                h = round(im.height * SHEET_CELL_W / im.width)
                r.append((n, im.resize((SHEET_CELL_W, h), Image.LANCZOS)))
            thumbs.append(r)
        row_h = [max(t.height for _, t in r) + label_h for r in thumbs]
        sheet = Image.new("RGB", (cols * (SHEET_CELL_W + gap) + gap, sum(row_h) + gap * (len(thumbs) + 1)), "white")
        d = ImageDraw.Draw(sheet)
        y = gap
        for r, rh in zip(thumbs, row_h):
            for k, (name, t) in enumerate(r):
                x = gap + k * (SHEET_CELL_W + gap)
                c = checks.get(name, {})
                if c.get("flags"):
                    tag, color = "FLAG " + ", ".join(sorted({f.split()[0] for f in c["flags"]})), (200, 0, 0)
                    d.rectangle([x - border, y - border, x + SHEET_CELL_W + border - 1,
                                 y + label_h + t.height + border - 1], fill=color)
                    d.rectangle([x, y, x + SHEET_CELL_W - 1, y + label_h - 1], fill="white")
                elif "copy" in c:
                    tag, color = "copy ok", (0, 130, 0)
                else:
                    tag, color = "", (0, 0, 0)
                if c.get("quality") == "low":
                    tag, color = (tag + " " if tag else "") + "low q", (200, 110, 0)
                label = name[:26] + (f"  {tag}" if tag else "")
                d.text((x + 4, y + 6), label, fill=color if tag else "black", font=font)
                sheet.paste(t, (x, y + label_h))
            y += rh + gap
        out = folder / f"_sheet-{page}.jpg"
        sheet.save(out, "JPEG", quality=82)
        sheets.append(out)
    return sheets


# ---------- commands ----------

def archive_png(d: Path, name: str) -> None:
    """Move <name>.png aside as <name>.prev.png, shifting older ones to .prev2, .prev3, ...
    so a second redo never overwrites the first version."""
    cur = d / f"{name}.png"
    if not cur.exists():
        return
    old = []
    n = 1
    while (d / (f"{name}.prev.png" if n == 1 else f"{name}.prev{n}.png")).exists():
        old.append(n)
        n += 1
    for k in reversed(old):
        src = d / (f"{name}.prev.png" if k == 1 else f"{name}.prev{k}.png")
        src.rename(d / f"{name}.prev{k + 1}.png")
    cur.rename(d / f"{name}.prev.png")


def latest_quality(d: Path) -> dict[str, str]:
    """Quality the backend reported for each image's most recent render."""
    q: dict[str, str] = {}
    f = d / "_results.jsonl"
    if f.exists():
        for raw in f.read_text().splitlines():
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            if r.get("ok"):
                q[r["name"]] = (r.get("meta") or {}).get("quality") or ""
    return q


def load_jobs(path: str, default_size: str, default_n: int = 2) -> list[dict]:
    """Read jobs.jsonl. Each job renders n times with the same prompt (independent renders): the
    first keeps the plain NAME, the rest are NAME-2, NAME-3. n comes from the job's "n" field,
    else the --n default (2)."""
    jobs = []
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        if not raw.strip():
            continue
        j = json.loads(raw)
        base = j.get("name") or j.get("_name") or f"job{n:03d}"
        reps = max(1, int(j.get("n") or default_n))
        for k in range(1, reps + 1):
            jobs.append({"name": base if k == 1 else f"{base}-{k}", "base": base,
                         "prompt": j["prompt"],
                         "refs": [str(Path(r).expanduser()) for r in (j.get("refs") or j.get("references") or [])],
                         "size": j.get("size", default_size),
                         "copy": j.get("copy") or [],
                         "look": str(Path(j["look"]).expanduser()) if j.get("look") else ""})
    seen: dict[str, int] = {}
    for j in jobs:
        seen[j["name"]] = seen.get(j["name"], 0) + 1
    dup = sorted(k for k, v in seen.items() if v > 1)
    if dup:
        sys.exit(f"duplicate job names in {path} (they would overwrite each other): {', '.join(dup)}")
    return jobs


def dry_run(d: Path, jobs: list[dict], redo: set[str], parallel: int) -> int:
    """Say what a batch would do and roughly what it would cost. Renders nothing."""
    problems, todo, stale = [], [], []
    for j in jobs:
        p, t = d / f"{j['name']}.png", d / f"{j['name']}.txt"
        if not re.fullmatch(r"auto|\d+x\d+", j["size"] or "auto"):
            problems.append(f"{j['name']}: size \"{j['size']}\" is not auto or WxH")
        for r in j["refs"]:
            if not Path(r).exists():
                problems.append(f"{j['name']}: reference not found: {r}")
        if j.get("look") and not Path(j["look"]).exists():
            problems.append(f"{j['name']}: look reference not found: {j['look']}")
        if j["name"] in redo or j["base"] in redo:
            todo.append(j)
        elif p.exists() and p.stat().st_size > 1000:
            if t.exists() and t.read_text() != j["prompt"]:
                stale.append(j["name"])
                todo.append(j)
        else:
            todo.append(j)
    with_ref = sum(1 for j in todo if j["refs"])
    tokens = (len(todo) - with_ref) * 2300 + with_ref * 7300  # measured 6 Oct 2026
    minutes = -(-len(todo) // max(1, parallel)) * 37 / 60  # median 37s per image over 412 renders
    print(f"dry run: {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} would render "
          f"({with_ref} with references)")
    if stale:
        print(f"  stale (prompt changed, would re-render): {','.join(stale)}")
    print(f"  estimate: about {tokens / 1000:.0f}k tokens, about {minutes:.0f} min at {parallel} at a time")
    for q in problems:
        print(f"  PROBLEM {q}")
    print("  nothing was rendered")
    return 1 if problems else 0


def concept_summary(jobs: list[dict], checks: dict[str, dict]) -> dict[str, dict]:
    """Per concept (job base name): which variants passed the auto-checks."""
    out: dict[str, dict] = {}
    for j in jobs:
        c = checks.get(j["name"])
        if c is None:
            continue
        e = out.setdefault(j["base"], {"variants": [], "clean": [], "flagged": {}, "low_quality": []})
        e["variants"].append(j["name"])
        if c["flags"]:
            e["flagged"][j["name"]] = c["flags"]
        else:
            e["clean"].append(j["name"])
        if c.get("quality") == "low":
            e["low_quality"].append(j["name"])
    return out


def check_and_sheet(d: Path, jobs: list[dict], sheet: bool = True) -> int:
    """Run the checks, print the per-concept verdict, write sheets. Returns how many concepts
    have no clean variant (those need a redo)."""
    checks = run_checks(d, jobs)
    summ = concept_summary(jobs, checks)
    (d / "_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    redo = [b for b, e in summ.items() if not e["clean"]]
    mixed = {b: e for b, e in summ.items() if e["clean"] and e["flagged"]}
    print(f"concepts: {len(summ)}, {len(summ) - len(redo)} have a clean variant to judge, "
          f"{len(redo)} need a redo")
    for b in redo:
        print(f"REDO {b}: every variant flagged ({'; '.join(f for fl in summ[b]['flagged'].values() for f in fl)[:160]})")
    for b, e in mixed.items():
        print(f"ONE-OF {b}: judge only {','.join(e['clean'])}; skip {','.join(e['flagged'])}")
    bases = {j["name"]: j["base"] for j in jobs}
    for sh in (make_sheets(d, [j["name"] for j in jobs], checks, bases) if sheet else []):
        print(f"sheet: {sh}")
    return len(redo)


def cmd_one(a) -> int:
    out = Path(a.out).expanduser().resolve()
    r = generate(a.prompt, out, a.ref or [], a.size, a.model)
    print(_line(r) + (f"  -> {out}" if r["ok"] else ""))
    if not r["ok"]:
        return 2 if r.get("rate_limited") else 1
    if a.copy or a.size != "auto":
        job = {"name": out.stem, "size": a.size, "copy": a.copy or []}
        lines = run_ocr([out]).get(str(out), []) if a.copy else []
        c = check_image(job, out, lines)
        print("check: clean" if not c["flags"] else "\n".join(f"FLAG {out.stem}: {f}" for f in c["flags"]))
        return 3 if c["flags"] else 0
    return 0


def cmd_batch(a) -> int:
    d = Path(a.dir).expanduser().resolve()
    jobs = load_jobs(a.jobs, a.size, a.n)
    if not a.dry_run:
        d.mkdir(parents=True, exist_ok=True)
    redo = set(filter(None, (a.redo or "").split(",")))
    if a.dry_run:
        return dry_run(d, jobs, redo, a.j)
    redo |= {j["name"] for j in jobs if j["base"] in redo}
    if a.redo_low:
        redo |= {n for n, q in latest_quality(d).items() if q == "low"}
    for j in jobs:
        p, t = d / f"{j['name']}.png", d / f"{j['name']}.txt"
        if p.exists() and j["name"] not in redo and t.exists() and t.read_text() != j["prompt"]:
            print(f"stale: {j['name']} prompt changed since it was rendered, rendering again")
            redo.add(j["name"])
        if j["name"] in redo:
            archive_png(d, j["name"])
    todo = [j for j in jobs if not ((d / f"{j['name']}.png").exists()
                                    and (d / f"{j['name']}.png").stat().st_size > 1000)]
    print(f"batch: {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to make, {a.j} at a time")
    os.environ["CHATGPT_IMAGEGEN_CODEX_CONCURRENCY"] = str(a.j)
    stop = threading.Event()
    lock = threading.Lock()
    results = []

    def run(j):
        if stop.is_set():
            return
        out = d / f"{j['name']}.png"
        (d / f"{j['name']}.txt").write_text(j["prompt"])
        r = generate(j["prompt"], out, j["refs"], j["size"], a.model)
        if not r["ok"] and not r.get("rate_limited") and not stop.is_set():
            r = generate(j["prompt"], out, j["refs"], j["size"], a.model)
            r["retried"] = True
        if r.get("rate_limited"):
            stop.set()
        with lock:
            results.append(r)
            print(_line(r), flush=True)
            with open(d / "_results.jsonl", "a") as f:
                f.write(json.dumps({**r, "at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")

    with ThreadPoolExecutor(max_workers=max(1, a.j)) as ex:
        list(ex.map(run, todo))

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    tail = ""
    if stop.is_set():
        rl = next((r for r in results if r.get("rate_limited")), {})
        hrs = f"~{rl['resets_in_s'] / 3600:.1f}h" if rl.get("resets_in_s") else "unknown"
        tail = f"; STOPPED at usage limit, resets in {hrs}, rerun the same command to resume"
    print(f"done: {len(ok)}/{len(todo)} made, {len(bad)} failed, {sum(_tok(r) for r in results)} tokens{tail}")
    if bad:
        print("failed: " + ",".join(r["name"] for r in bad))
    flagged = check_and_sheet(d, jobs, sheet=not a.no_sheet)
    apply_looks(d, jobs)
    if bad:
        return 2 if stop.is_set() else 1
    return 3 if flagged else 0


def _look_module():
    spec = importlib.util.spec_from_file_location("lunagen_look", HERE / "look.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_looks(d: Path, jobs: list[dict]) -> int:
    """For jobs with a "look" reference: write look/<name>.jpg matched to it (resolution,
    tone, colour, grain; see look.py) whenever the PNG is newer than its look copy."""
    todo = [j for j in jobs if j.get("look") and (d / f"{j['name']}.png").exists()]
    if not todo:
        return 0
    lk = _look_module()
    out = d / "look"
    made = []
    for j in todo:
        src, dst = d / f"{j['name']}.png", out / f"{j['name']}.jpg"
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            lk.match(src, j["look"], dst)
        made.append(dst)
    by_ref: dict[str, list[Path]] = {}
    for j, dst in zip(todo, made):
        by_ref.setdefault(j["look"], []).append(dst)
    for i, (ref, paths) in enumerate(by_ref.items(), 1):
        lk.sheet(ref, paths, out / f"_look-sheet-{i}.jpg")
    print(f"look: {len(made)} matched to {len(by_ref)} reference(s) -> {out}/_look-sheet-*.jpg")
    return len(made)


def cmd_check(a) -> int:
    d = Path(a.dir).expanduser().resolve()
    return 3 if check_and_sheet(d, load_jobs(a.jobs, a.size, a.n)) else 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "look":
        return _look_module().main(argv[1:])
    if argv and argv[0] in ("batch", "check"):
        p = argparse.ArgumentParser(prog=f"lunagen {argv[0]}")
        p.add_argument("jobs", help="jobs.jsonl")
        p.add_argument("-d", "--dir", required=True, help="output folder")
        p.add_argument("--size", default="auto", help="default size for jobs without one")
        p.add_argument("--n", type=int, default=2,
                       help="renders per job unless the job sets its own \"n\" (default 2; use --n 1 for one)")
        if argv[0] == "check":
            return cmd_check(p.parse_args(argv[1:]))
        p.add_argument("-j", type=int, default=4, help="parallel images (default 4)")
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--redo", help="comma-separated names to regenerate (old kept as .prev.png, .prev2.png, ...)")
        p.add_argument("--dry-run", action="store_true",
                       help="show what would render and a rough cost, then stop; also checks refs and sizes")
        p.add_argument("--redo-low", action="store_true",
                       help="also regenerate every image the backend served at quality \"low\"")
        p.add_argument("--no-sheet", action="store_true")
        return cmd_batch(p.parse_args(argv[1:]))
    p = argparse.ArgumentParser(prog="lunagen", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("-i", "--ref", action="append", help="reference image (repeatable, order matters)")
    p.add_argument("--size", default="auto", help="auto | 1024x1024 | 1024x1536 | 1536x1024")
    p.add_argument("--copy", action="append", help="exact text that must appear (repeatable)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    return cmd_one(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
