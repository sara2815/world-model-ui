"""Live audit gallery served from the workstation filesystem.

Instead of copying thousands of full-res screenshots next to an HTML file
(which breaks under Cursor Live Preview / SSH anyway), this starts a small
HTTP server on the workstation that:

  • streams original PNGs from /bigdata/... on demand  (/full?p=...)
  • generates JPEG thumbs on the fly with a disk cache   (/thumb?p=...)
  • serves a polished Kept + Dropped gallery UI

Usage (on the workstation):
    python data/audit_server.py                  # http://127.0.0.1:8765
    python data/audit_server.py --port 8765

Then from your laptop, port-forward and open in a normal browser:
    ssh -L 8765:127.0.0.1:8765 <user>@<workstation>
    open http://127.0.0.1:8765/

Cursor Live Preview of a local .html file will NOT see these images —
use a real browser tab pointed at the tunneled URL.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import mimetypes
import os
import sys
import urllib.parse
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (  # noqa: E402
    _DEFAULT_DATA_ROOT,
    _DEFAULT_DROP_ACTION_PATTERNS,
    _DEFAULT_MAX_CHANGE_METRIC,
    _DEFAULT_MIN_CHANGE_METRIC,
    _RESULT_DIRS,
    _TRIPLETS_FALLBACK,
    _TRIPLETS_FILENAME,
    _swap_annotated,
    clean_step_actions,
    compile_action_patterns,
    entry_has_blocked_action,
    extract_action_kinds,
    metric_bucket,
)

try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # older Pillow
    _RESAMPLE = Image.LANCZOS

REASON_ORDER = [
    "kept",
    "metric_below_min",
    "metric_above_max",
    "missing_metric",
    "missing_images",
    "blocked_action",
    "sleep_only",
]
REASON_LABELS = {
    "kept": "Kept",
    "metric_below_min": "Dropped – below min metric",
    "metric_above_max": "Dropped – above max metric",
    "missing_metric": "Dropped – missing DINO metric",
    "missing_images": "Dropped – missing images",
    "blocked_action": "Dropped – blocked action (e.g. ctrl+a)",
    "sleep_only": "Dropped – sleep-only step",
}


_METRIC_REASON = {
    "missing": "missing_metric",
    "low": "metric_below_min",
    "high": "metric_above_max",
}


def evaluate_entry(entry, episode_dir, data_root, use_annotated_image, min_m, max_m,
                   using_fallback, blocked_patterns=()):
    """Classify one triplet the same way WorldModelDataset would."""
    source = str(episode_dir.relative_to(data_root))
    metric = entry.get("change_metric_value")
    if using_fallback and (min_m is not None or max_m is not None):
        metric = None

    before = entry.get("screenshot_before", "")
    after = entry.get("screenshot_after", "")
    if use_annotated_image:
        before, after = _swap_annotated(before), _swap_annotated(after)
    before_path = episode_dir / before
    after_path = episode_dir / after
    images_exist = before_path.exists() and after_path.exists()

    ar_raw = entry.get("actions_resolved", [])
    af_raw = entry.get("action_full", "")
    cleaned, action_full, action_text, had_sleep = clean_step_actions(
        ar_raw, af_raw, strip_sleep=True, use_annotated=use_annotated_image,
    )

    bucket = metric_bucket(metric, min_m, max_m)
    if bucket != "ok":
        reason = _METRIC_REASON[bucket]
    elif not images_exist:
        reason = "missing_images"
    elif entry_has_blocked_action(ar_raw, af_raw, blocked_patterns):
        reason = "blocked_action"
    elif not cleaned:
        reason = "sleep_only"
    else:
        reason = "kept"

    return {
        "reason": reason,
        "source": source,
        "step_number": entry.get("step_number"),
        "change_metric_value": metric,
        "action_full": action_full,
        "action_text": action_text,
        "action_kinds": extract_action_kinds(cleaned) if cleaned else [],
        "had_sleep": had_sleep,
        "before_path": str(before_path),
        "after_path": str(after_path),
        "images_exist": images_exist,
    }


def collect(data_root, use_annotated_image, min_m, max_m, blocked_patterns=()):
    records_by_reason = defaultdict(list)
    counts = Counter()
    kinds_kept = Counter()
    kinds_drop = Counter()
    metrics = {"all": [], "kept": [], "low": [], "high": []}

    root = Path(data_root)
    for result_dir in sorted(root.iterdir()):
        if not result_dir.is_dir() or result_dir.name not in _RESULT_DIRS:
            continue
        for episode_dir in sorted(
            result_dir.iterdir(),
            key=lambda p: (p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name),
        ):
            path = episode_dir / _TRIPLETS_FILENAME
            using_fallback = False
            if not path.exists():
                path = episode_dir / _TRIPLETS_FALLBACK
                using_fallback = True
            if not path.exists():
                continue
            try:
                triplets = json.load(open(path))
            except Exception:
                counts["json_error"] += 1
                continue
            for entry in triplets:
                rec = evaluate_entry(
                    entry, episode_dir, root, use_annotated_image, min_m, max_m,
                    using_fallback, blocked_patterns,
                )
                counts[rec["reason"]] += 1
                records_by_reason[rec["reason"]].append(rec)
                m = rec["change_metric_value"]
                if isinstance(m, (int, float)):
                    metrics["all"].append(m)
                if rec["reason"] == "kept":
                    for k in rec["action_kinds"]:
                        kinds_kept[k] += 1
                    if isinstance(m, (int, float)):
                        metrics["kept"].append(m)
                else:
                    for k in rec["action_kinds"]:
                        kinds_drop[k] += 1
                    if rec["reason"] == "metric_below_min":
                        metrics["low"].append(m)
                    elif rec["reason"] == "metric_above_max":
                        metrics["high"].append(m)

    return records_by_reason, counts, kinds_kept, kinds_drop, metrics


def stats(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)

    def p(q):
        return round(s[min(n - 1, int(q / 100 * (n - 1)))], 2)

    return {
        "n": n, "min": round(s[0], 2), "p25": p(25), "median": p(50),
        "p75": p(75), "p95": p(95), "max": round(s[-1], 2), "mean": round(sum(s) / n, 2),
    }


def print_report(counts, kinds_kept, kinds_drop, metrics, min_m, max_m):
    total = sum(v for k, v in counts.items() if k != "json_error") or 1
    print("=" * 64)
    print(f"Data cleaning audit  (min={min_m}, max={max_m})")
    print("-" * 64)
    for reason in REASON_ORDER:
        if reason in counts:
            c = counts[reason]
            print(f"  {reason:20s} {c:7d}  ({100 * c / total:5.1f}%)")
    print("-" * 64)
    print("  Metric stats:")
    for name in ("all", "kept", "low", "high"):
        st = stats(metrics[name])
        if st:
            print(
                f"    {name:5s} n={st['n']:6d} median={st['median']:>12} "
                f"mean={st['mean']:>12} max={st['max']:>12}"
            )
    print("-" * 64)
    print("  Action kinds (kept):  ", dict(kinds_kept.most_common()))
    print("  Action kinds (dropped):", dict(kinds_drop.most_common()))
    print("=" * 64)


# ---------------------------------------------------------------------------
# Shared state filled at startup
# ---------------------------------------------------------------------------
STATE: dict = {}


def _safe_resolve(raw: str) -> Path | None:
    """Resolve `raw` only if it stays under the configured data_root."""
    if not raw:
        return None
    root: Path = STATE["data_root"]
    try:
        p = Path(raw).resolve()
        root_r = root.resolve()
        if root_r in p.parents or p == root_r:
            return p if p.is_file() else None
    except Exception:
        return None
    return None


def _thumb_cache_path(src: Path, max_side: int) -> Path:
    cache_dir: Path = STATE["thumb_cache"]
    key = hashlib.sha1(f"{src}|{src.stat().st_mtime_ns}|{max_side}".encode()).hexdigest()
    return cache_dir / f"{key}.jpg"


def _make_thumb_bytes(src: Path, max_side: int = 720) -> bytes:
    cache = _thumb_cache_path(src, max_side)
    if cache.exists():
        return cache.read_bytes()
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), _RESAMPLE)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=78, optimize=True)
        data = buf.getvalue()
    try:
        cache.write_bytes(data)
    except Exception:
        pass
    return data


def _img_url(kind: str, path: str) -> str:
    return f"/{kind}?p={urllib.parse.quote(path, safe='')}"


def _serialize_record(rec: dict) -> dict:
    before = rec.get("before_path") or ""
    after = rec.get("after_path") or ""
    exist = bool(rec.get("images_exist"))
    return {
        "reason": rec["reason"],
        "source": rec["source"],
        "step": rec["step_number"],
        "metric": rec["change_metric_value"] if isinstance(rec["change_metric_value"], (int, float)) else None,
        "action": rec.get("action_text") or rec.get("action_full") or "",
        "kinds": rec.get("action_kinds") or [],
        "had_sleep": bool(rec.get("had_sleep")),
        "thumb_before": _img_url("thumb", before) if exist else None,
        "thumb_after": _img_url("thumb", after) if exist else None,
        "full_before": _img_url("full", before) if exist else None,
        "full_after": _img_url("full", after) if exist else None,
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>World-model data audit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root, [data-theme="dark"] {
  color-scheme: dark;
  --bg: #0c0e12;
  --bg-elev: #141821;
  --bg-card: #171b24;
  --bg-hover: #1c2230;
  --line: #2a3142;
  --line-soft: #222836;
  --text: #e8eaef;
  --muted: #8b93a7;
  --faint: #5c657a;
  --accent: #6ea8fe;
  --accent-soft: rgba(110,168,254,.14);
  --accent-ring: rgba(110,168,254,.25);
  --good: #3dd68c;
  --good-soft: rgba(61,214,140,.12);
  --bad: #f07178;
  --bad-soft: rgba(240,113,120,.12);
  --warn: #e6c07b;
  --warn-soft: rgba(230,192,123,.12);
  --radius: 12px;
  --shadow: 0 1px 0 rgba(255,255,255,.04) inset, 0 8px 24px rgba(0,0,0,.35);
  --top-bg: rgba(12,14,18,.88);
  --lb-bg: rgba(4,6,10,.92);
  --shot-bg: #0a0c10;
  --cap-bg: rgba(0,0,0,.65);
  --code-bg: rgba(0,0,0,.35);
}
[data-theme="light"] {
  color-scheme: light;
  --bg: #f4f6fa;
  --bg-elev: #ffffff;
  --bg-card: #ffffff;
  --bg-hover: #eef1f7;
  --line: #d5dbe8;
  --line-soft: #e3e8f2;
  --text: #1a2030;
  --muted: #5c657a;
  --faint: #8b93a7;
  --accent: #2f6fed;
  --accent-soft: rgba(47,111,237,.10);
  --accent-ring: rgba(47,111,237,.22);
  --good: #0d9f6e;
  --good-soft: rgba(13,159,110,.10);
  --bad: #d64550;
  --bad-soft: rgba(214,69,80,.10);
  --warn: #b8860b;
  --warn-soft: rgba(184,134,11,.12);
  --shadow: 0 1px 2px rgba(16,24,40,.06), 0 8px 24px rgba(16,24,40,.06);
  --top-bg: rgba(244,246,250,.92);
  --lb-bg: rgba(20,24,35,.78);
  --shot-bg: #e8ecf4;
  --cap-bg: rgba(255,255,255,.85);
  --code-bg: rgba(16,24,40,.06);
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
  font-family: "DM Sans", system-ui, sans-serif; font-optical-sizing: auto;
  -webkit-font-smoothing: antialiased; }
button, input, select { font: inherit; color: inherit; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

.app { min-height: 100vh; display: flex; flex-direction: column; }

/* header */
.top {
  position: sticky; top: 0; z-index: 20;
  background: var(--top-bg); backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--line-soft);
}
.top-inner { max-width: 1680px; margin: 0 auto; padding: 16px 24px 0; }
.brand-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
.brand { display: flex; align-items: baseline; gap: 12px; min-width: 0; }
.brand h1 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: -0.02em; }
.brand .sub { color: var(--muted); font-size: 13px; }
.theme-toggle {
  appearance: none; border: 1px solid var(--line); background: var(--bg-card);
  color: var(--muted); border-radius: 999px; padding: 6px 12px; cursor: pointer;
  font-size: 12px; font-weight: 500; white-space: nowrap;
}
.theme-toggle:hover { border-color: var(--accent); color: var(--text); }
.tabs { display: flex; gap: 4px; }
.tab {
  appearance: none; border: 0; background: transparent; color: var(--muted);
  padding: 10px 14px; border-radius: 8px 8px 0 0; cursor: pointer;
  font-weight: 500; font-size: 13px; border-bottom: 2px solid transparent;
}
.tab:hover { color: var(--text); background: var(--bg-hover); }
.tab.active { color: var(--text); border-bottom-color: var(--accent); }

/* body */
.main { max-width: 1680px; margin: 0 auto; padding: 20px 24px 64px; width: 100%; }

/* overview */
.stats {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px;
}
@media (max-width: 900px) { .stats { grid-template-columns: repeat(2, 1fr); } }
.stat {
  background: var(--bg-card); border: 1px solid var(--line-soft);
  border-radius: var(--radius); padding: 16px 18px; box-shadow: var(--shadow);
}
.stat .label { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
.stat .value { font-size: 28px; font-weight: 600; letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }
.stat.good .value { color: var(--good); }
.stat.bad .value { color: var(--bad); }
.stat.info .value { color: var(--accent); }

.panels { display: grid; grid-template-columns: 1.1fr 1fr 1fr; gap: 14px; }
@media (max-width: 1100px) { .panels { grid-template-columns: 1fr; } }
.panel {
  background: var(--bg-card); border: 1px solid var(--line-soft);
  border-radius: var(--radius); padding: 16px; box-shadow: var(--shadow);
}
.panel h2 { margin: 0 0 12px; font-size: 13px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: .06em; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 7px 8px; text-align: left; border-bottom: 1px solid var(--line-soft); }
th { color: var(--faint); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; font-family: "JetBrains Mono", monospace; font-size: 12px; }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }
.dot.good { background: var(--good); }
.dot.bad { background: var(--bad); }
.dot.warn { background: var(--warn); }
.dot.muted { background: var(--faint); }

/* gallery toolbar */
.toolbar {
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  margin-bottom: 12px; padding: 12px; background: var(--bg-elev);
  border: 1px solid var(--line-soft); border-radius: var(--radius);
}
.toolbar input[type=search], .toolbar input[type=number], .toolbar select {
  background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
  padding: 8px 12px; font-size: 13px; outline: none;
}
.toolbar input[type=search] { flex: 1; min-width: 200px; }
.toolbar input[type=search]:focus, .toolbar input[type=number]:focus, .toolbar select:focus {
  border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft);
}
.toolbar label { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.kinds { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
.chip {
  appearance: none; border: 1px solid var(--line); background: var(--bg-card);
  color: var(--muted); border-radius: 999px; padding: 5px 12px; font-size: 12px;
  cursor: pointer; transition: .12s ease;
}
.chip:hover { border-color: var(--accent); color: var(--text); }
.chip.active { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); font-weight: 500; }
.chip .n { opacity: .7; margin-left: 4px; font-variant-numeric: tabular-nums; }

.pager {
  display: flex; align-items: center; gap: 10px; margin-bottom: 14px;
  font-size: 13px; color: var(--muted);
}
.pager button {
  appearance: none; background: var(--bg-card); border: 1px solid var(--line);
  border-radius: 8px; padding: 7px 14px; cursor: pointer; color: var(--text);
}
.pager button:hover:not(:disabled) { border-color: var(--accent); }
.pager button:disabled { opacity: .35; cursor: default; }
#status { font-variant-numeric: tabular-nums; font-family: "JetBrains Mono", monospace; font-size: 12px; }

.grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
}
@media (max-width: 900px) {
  .grid { grid-template-columns: 1fr; }
}
.card {
  background: var(--bg-card); border: 1px solid var(--line-soft);
  border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow);
  display: flex; flex-direction: column;
}
.card-body { padding: 12px 14px 10px; }
.meta { font-size: 12px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 6px; }
.meta b { color: var(--text); font-weight: 500; }
.meta .metric {
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  background: var(--bg); border: 1px solid var(--line-soft); border-radius: 6px;
  padding: 2px 7px; color: var(--accent);
}
.badge {
  display: inline-block; font-size: 10px; font-weight: 500; letter-spacing: .02em;
  padding: 2px 7px; border-radius: 999px; background: var(--accent-soft); color: var(--accent);
  border: 1px solid transparent;
}
.badge.sleep { background: var(--warn-soft); color: var(--warn); }
.badge.reason { background: var(--bad-soft); color: var(--bad); }
.badge.kept-side { background: var(--good-soft); color: var(--good); }
.badge.dist {
  font-family: "JetBrains Mono", monospace; font-size: 10px;
  background: var(--bg); color: var(--muted); border: 1px solid var(--line-soft);
}
.action {
  font-size: 12.5px; color: var(--muted); line-height: 1.4;
  max-height: 2.8em; overflow: hidden; margin-bottom: 2px;
}
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--line-soft); }
.shot {
  position: relative; background: var(--shot-bg); cursor: zoom-in; aspect-ratio: 16/9;
  overflow: hidden;
}
.shot img {
  width: 100%; height: 100%; object-fit: cover; display: block;
  transition: transform .2s ease;
}
.shot:hover img { transform: scale(1.03); }
.shot .cap {
  position: absolute; left: 8px; bottom: 8px;
  font-size: 10px; letter-spacing: .04em; text-transform: uppercase;
  background: var(--cap-bg); color: var(--text); padding: 3px 7px; border-radius: 4px;
  pointer-events: none;
}
.empty { color: var(--faint); padding: 40px; text-align: center; font-size: 14px; }

/* lightbox — before | after, fills the viewport */
#lightbox {
  display: none; position: fixed; inset: 0; z-index: 100;
  background: var(--lb-bg); backdrop-filter: blur(6px);
  padding: 0; cursor: zoom-out;
}
#lightbox.open { display: block; }
#lightbox .lb-pair {
  position: absolute; inset: 0 0 44px 0;
  display: grid; grid-template-columns: 1fr 1fr; gap: 6px; padding: 6px;
}
@media (max-width: 900px) {
  #lightbox .lb-pair { grid-template-columns: 1fr; overflow: auto; }
}
#lightbox .lb-pane {
  position: relative; min-width: 0; min-height: 0; height: 100%;
  display: flex; align-items: center; justify-content: center;
  background: rgba(0,0,0,.25); border-radius: 6px; overflow: hidden;
}
#lightbox .lb-pane .cap {
  position: absolute; left: 10px; top: 10px; z-index: 2;
  font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
  color: #fff; font-weight: 600;
  background: rgba(0,0,0,.55); padding: 4px 8px; border-radius: 4px;
  pointer-events: none;
}
#lightbox .lb-pane img {
  width: 100%; height: 100%; object-fit: contain;
  background: #000; cursor: default;
}
#lightbox .lb-meta { color: var(--muted); font-size: 12px; font-family: "JetBrains Mono", monospace; }
#lightbox .lb-row {
  position: absolute; left: 0; right: 0; bottom: 0; height: 44px;
  display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: center;
  padding: 0 56px 0 12px; background: rgba(0,0,0,.35);
}
#lightbox button, #lightbox a.btn {
  appearance: none; background: var(--bg-card); color: var(--text);
  border: 1px solid var(--line); border-radius: 8px; padding: 5px 12px;
  cursor: pointer; font-size: 12px; text-decoration: none;
}
#lightbox .close { position: absolute; top: 10px; right: 12px; z-index: 3; cursor: pointer; }
.hidden { display: none !important; }
.callout {
  background: var(--accent-soft); border: 1px solid var(--accent-ring);
  border-radius: var(--radius); padding: 12px 14px; font-size: 13px; color: var(--text);
  margin-bottom: 16px; line-height: 1.45;
}
.callout code {
  font-family: "JetBrains Mono", monospace; font-size: 12px;
  background: var(--code-bg); padding: 1px 6px; border-radius: 4px;
}
</style>
</head>
<body>
<script>
(function () {
  const pref = localStorage.getItem("audit-theme");
  const theme = pref || (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  document.documentElement.setAttribute("data-theme", theme);
})();
</script>
<div class="app">
  <header class="top">
    <div class="top-inner">
      <div class="brand-row">
        <div class="brand">
          <h1>World-model data audit</h1>
          <span class="sub" id="brand-sub"></span>
        </div>
        <button type="button" class="theme-toggle" id="theme-toggle" aria-label="Toggle color theme">Dark</button>
      </div>
      <nav class="tabs">
        <button class="tab active" data-view="overview">Overview</button>
        <button class="tab" data-view="kept">Kept</button>
        <button class="tab" data-view="dropped">Dropped</button>
        <button class="tab" data-view="borderline">Borderline</button>
      </nav>
    </div>
  </header>

  <main class="main">
    <section id="view-overview">
      <div class="callout">
        Full-resolution screenshots are streamed live from this workstation — nothing is copied.
        From your laptop: <code>ssh -L 8765:127.0.0.1:8765 &lt;host&gt;</code> then open
        <code>http://127.0.0.1:8765/</code> in a browser (not Cursor Live Preview).
        Click any card image to compare <b>before | after</b> full-res side by side.
        Use the <b>Borderline</b> tab to inspect samples near the min/max metric cutoffs.
      </div>
      <div class="stats" id="stats"></div>
      <div class="panels">
        <div class="panel"><h2>Filter funnel</h2><table id="funnel"></table></div>
        <div class="panel"><h2>Action kinds</h2><table id="kinds-tbl"></table></div>
        <div class="panel"><h2>Metric stats</h2><table id="metric-tbl"></table></div>
      </div>
    </section>

    <section id="view-kept" class="hidden">
      <div class="toolbar" id="kept-toolbar"></div>
      <div class="kinds" id="kept-kinds"></div>
      <div class="pager" id="kept-pager"></div>
      <div class="grid" id="kept-grid"></div>
    </section>

    <section id="view-dropped" class="hidden">
      <div class="toolbar" id="drop-toolbar"></div>
      <div class="kinds" id="drop-kinds"></div>
      <div class="pager" id="drop-pager"></div>
      <div class="grid" id="drop-grid"></div>
    </section>

    <section id="view-borderline" class="hidden">
      <div class="callout">
        Samples within <b>margin</b> of the metric thresholds
        (min=<span id="bl-min"></span>, max=<span id="bl-max"></span>).
        Compare <span class="badge">just kept</span> vs <span class="badge reason">just dropped</span>
        to judge whether the cutoffs feel right.
      </div>
      <div class="toolbar" id="bl-toolbar"></div>
      <div class="kinds" id="bl-kinds"></div>
      <div class="pager" id="bl-pager"></div>
      <div class="grid" id="bl-grid"></div>
    </section>
  </main>
</div>

<div id="lightbox">
  <button class="close" id="lb-close" type="button">Close · Esc</button>
  <div class="lb-pair">
    <div class="lb-pane">
      <div class="cap">Before</div>
      <img id="lb-before" alt="before full resolution">
    </div>
    <div class="lb-pane">
      <div class="cap">After</div>
      <img id="lb-after" alt="after full resolution">
    </div>
  </div>
  <div class="lb-row">
    <div class="lb-meta" id="lb-meta"></div>
    <a class="btn" id="lb-open-before" target="_blank" rel="noopener">Open before</a>
    <a class="btn" id="lb-open-after" target="_blank" rel="noopener">Open after</a>
  </div>
</div>

<script>
const META = __META_JSON__;
const REASON_LABELS = __REASON_LABELS__;

(function themeInit() {
  const btn = document.getElementById("theme-toggle");
  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("audit-theme", theme);
    btn.textContent = theme === "dark" ? "Light mode" : "Dark mode";
  }
  apply(document.documentElement.getAttribute("data-theme") || "dark");
  btn.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    apply(next);
  });
})();

function fmt(n) {
  if (n == null || Number.isNaN(n)) return "—";
  return Math.round(n).toLocaleString();
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

document.getElementById("brand-sub").textContent =
  `${META.total.toLocaleString()} considered · min=${fmt(META.thresholds.min)} · max=${fmt(META.thresholds.max)}`;

/* ---------- overview ---------- */
(function buildOverview() {
  const c = META.counts;
  const kept = c.kept || 0;
  const dropped = META.total - kept;
  document.getElementById("stats").innerHTML = `
    <div class="stat"><div class="label">Considered</div><div class="value">${META.total.toLocaleString()}</div></div>
    <div class="stat good"><div class="label">Kept</div><div class="value">${kept.toLocaleString()}</div></div>
    <div class="stat bad"><div class="label">Dropped</div><div class="value">${dropped.toLocaleString()}</div></div>
    <div class="stat info"><div class="label">Keep rate</div><div class="value">${(100*kept/META.total).toFixed(1)}%</div></div>`;

  const funnelRows = META.reason_order.filter(r => c[r]).map(r => {
    const n = c[r];
    const tone = r === "kept" ? "good" : (r.includes("metric") || r === "blocked_action" || r === "sleep_only" ? "bad" : "muted");
    return `<tr><td><span class="dot ${tone}"></span>${esc(REASON_LABELS[r]||r)}</td>
      <td class="num">${n.toLocaleString()}</td>
      <td class="num">${(100*n/META.total).toFixed(1)}%</td></tr>`;
  }).join("");
  document.getElementById("funnel").innerHTML =
    `<tr><th>Bucket</th><th class="num">Count</th><th class="num">%</th></tr>${funnelRows}`;

  const kinds = Object.keys({...META.kinds_kept, ...META.kinds_drop})
    .sort((a,b) => ((META.kinds_kept[b]||0)+(META.kinds_drop[b]||0)) - ((META.kinds_kept[a]||0)+(META.kinds_drop[a]||0)));
  document.getElementById("kinds-tbl").innerHTML =
    `<tr><th>Kind</th><th class="num">Kept</th><th class="num">Dropped</th></tr>` +
    kinds.map(k => `<tr><td>${esc(k)}</td><td class="num">${(META.kinds_kept[k]||0).toLocaleString()}</td>
      <td class="num">${(META.kinds_drop[k]||0).toLocaleString()}</td></tr>`).join("");

  const ms = META.metric_stats;
  document.getElementById("metric-tbl").innerHTML =
    `<tr><th>Set</th><th class="num">N</th><th class="num">Median</th><th class="num">Mean</th><th class="num">Max</th></tr>` +
    ["all","kept","low","high"].filter(k => ms[k]).map(k => {
      const s = ms[k];
      return `<tr><td>${k}</td><td class="num">${s.n.toLocaleString()}</td>
        <td class="num">${fmt(s.median)}</td><td class="num">${fmt(s.mean)}</td><td class="num">${fmt(s.max)}</td></tr>`;
    }).join("");
})();


/* ---------- shared card renderer ---------- */
function renderCard(r, { showReason = false } = {}) {
  const metric = r.metric == null ? "—" : fmt(r.metric);
  const badges = (r.kinds || []).map(k => `<span class="badge">${esc(k)}</span>`).join("");
  const sleep = r.had_sleep ? `<span class="badge sleep">sleep stripped</span>` : "";
  const reasonBadge = showReason
    ? `<span class="badge reason">${esc(REASON_LABELS[r.reason] || r.reason)}</span>` : "";
  let borderBadges = "";
  if (r.border_band) {
    const sideCls = r.border_side === "kept" ? "kept-side" : "reason";
    const sideLabel = r.border_side === "kept" ? "just kept" : "just dropped";
    borderBadges =
      `<span class="badge ${sideCls}">${sideLabel}</span>` +
      `<span class="badge">near ${esc(r.border_band)}</span>` +
      `<span class="badge dist">Δ ${fmt(r.dist)}</span>`;
  }
  const imgs = (r.thumb_before && r.thumb_after)
    ? `<div class="pair" data-before="${esc(r.full_before)}" data-after="${esc(r.full_after)}" data-source="${esc(r.source)}" data-step="${esc(r.step ?? "")}">
        <div class="shot"><img loading="lazy" src="${esc(r.thumb_before)}" alt="before"><span class="cap">before</span></div>
        <div class="shot"><img loading="lazy" src="${esc(r.thumb_after)}" alt="after"><span class="cap">after</span></div>
      </div>`
    : `<div class="empty">images missing</div>`;
  return `<article class="card">
    <div class="card-body">
      <div class="meta"><b>${esc(r.source)}</b> · step ${r.step ?? "—"}
        <span class="metric">${metric}</span>${borderBadges}${badges}${sleep}${reasonBadge}</div>
      <div class="action">${esc(r.action || "—")}</div>
    </div>${imgs}</article>`;
}

function bindPairClicks(gridEl) {
  gridEl.querySelectorAll(".pair").forEach(pair => {
    pair.addEventListener("click", () => {
      openLb(pair.dataset.before, pair.dataset.after, pair.dataset.source, pair.dataset.step);
    });
  });
}

/* ---------- gallery factory ---------- */
function makeGallery({ records, toolbarEl, kindsEl, pagerEl, gridEl, showReason }) {
  let kind = "";
  let reason = "";
  let page = 0;
  let pageSize = 40;
  let q = "";
  let minM = null, maxM = null;

  const kindCounts = {};
  const reasonCounts = {};
  for (const r of records) {
    reasonCounts[r.reason] = (reasonCounts[r.reason] || 0) + 1;
    for (const k of (r.kinds || [])) kindCounts[k] = (kindCounts[k] || 0) + 1;
  }

  toolbarEl.innerHTML = `
    <input type="search" placeholder="Search episode / action…">
    ${showReason ? `<select id="reason-sel"><option value="">all drop reasons</option>
      ${Object.keys(reasonCounts).sort().map(r =>
        `<option value="${esc(r)}">${esc(REASON_LABELS[r]||r)} (${reasonCounts[r]})</option>`).join("")}
    </select>` : ""}
    <label>metric ≥ <input type="number" class="minm" style="width:88px" placeholder="any"></label>
    <label>metric ≤ <input type="number" class="maxm" style="width:88px" placeholder="any"></label>
    <label>page <input type="number" class="ps" value="40" min="10" max="120" style="width:64px"></label>`;

  kindsEl.innerHTML = `<button class="chip active" data-kind="">all</button>` +
    Object.entries(kindCounts).sort((a,b)=>b[1]-a[1]).map(([k,n]) =>
      `<button class="chip" data-kind="${esc(k)}">${esc(k)}<span class="n">${n}</span></button>`).join("");

  pagerEl.innerHTML = `<button class="prev" type="button">← prev</button>
    <span class="status"></span><button class="next" type="button">next →</button>`;

  const search = toolbarEl.querySelector('input[type=search]');
  const minEl = toolbarEl.querySelector('.minm');
  const maxEl = toolbarEl.querySelector('.maxm');
  const psEl = toolbarEl.querySelector('.ps');
  const reasonSel = toolbarEl.querySelector('#reason-sel');
  const status = pagerEl.querySelector('.status');
  const prev = pagerEl.querySelector('.prev');
  const next = pagerEl.querySelector('.next');

  function matches(r) {
    if (kind && !(r.kinds || []).includes(kind)) return false;
    if (reason && r.reason !== reason) return false;
    if (q) {
      const hay = ((r.source||"") + " " + (r.action||"")).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    if (minM != null && !(r.metric >= minM)) return false;
    if (maxM != null && !(r.metric <= maxM)) return false;
    return true;
  }

  function render() {
    const matched = records.filter(matches);
    const pages = Math.max(1, Math.ceil(matched.length / pageSize));
    if (page >= pages) page = pages - 1;
    if (page < 0) page = 0;
    const start = page * pageSize;
    const slice = matched.slice(start, start + pageSize);
    gridEl.innerHTML = slice.length
      ? slice.map(r => renderCard(r, { showReason })).join("")
      : `<div class="empty">No samples match these filters.</div>`;
    status.textContent = matched.length
      ? `showing ${start+1}–${Math.min(start+pageSize, matched.length)} of ${matched.length.toLocaleString()}  ·  page ${page+1}/${pages}`
      : "no matches";
    prev.disabled = page <= 0;
    next.disabled = page >= pages - 1;
    bindPairClicks(gridEl);
  }

  kindsEl.querySelectorAll(".chip").forEach(btn => {
    btn.addEventListener("click", () => {
      kindsEl.querySelectorAll(".chip").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      kind = btn.dataset.kind || "";
      page = 0; render();
    });
  });
  search.addEventListener("input", () => { q = search.value.trim().toLowerCase(); page = 0; render(); });
  minEl.addEventListener("input", () => { minM = minEl.value === "" ? null : parseFloat(minEl.value); page = 0; render(); });
  maxEl.addEventListener("input", () => { maxM = maxEl.value === "" ? null : parseFloat(maxEl.value); page = 0; render(); });
  psEl.addEventListener("input", () => { pageSize = Math.max(10, Math.min(120, parseInt(psEl.value,10)||40)); page = 0; render(); });
  if (reasonSel) reasonSel.addEventListener("change", () => { reason = reasonSel.value; page = 0; render(); });
  prev.addEventListener("click", () => { page--; render(); });
  next.addEventListener("click", () => { page++; render(); });

  render();
}

/* ---------- borderline gallery ---------- */
function collectBorderline(kept, dropped, margin) {
  const tMin = META.thresholds.min;
  const tMax = META.thresholds.max;
  const out = [];
  function push(r, band, side, threshold) {
    if (r.metric == null) return;
    const dist = Math.abs(r.metric - threshold);
    if (dist > margin) return;
    out.push({ ...r, border_band: band, border_side: side, dist });
  }
  for (const r of kept) {
    push(r, "min", "kept", tMin);
    push(r, "max", "kept", tMax);
  }
  for (const r of dropped) {
    if (r.reason === "metric_below_min") push(r, "min", "dropped", tMin);
    else if (r.reason === "metric_above_max") push(r, "max", "dropped", tMax);
  }
  const seen = new Set();
  const deduped = [];
  for (const r of out.sort((a, b) => a.dist - b.dist)) {
    const key = `${r.border_band}|${r.border_side}|${r.source}|${r.step}|${r.metric}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(r);
  }
  return deduped;
}

function makeBorderlineGallery(kept, dropped) {
  const toolbarEl = document.getElementById("bl-toolbar");
  const kindsEl = document.getElementById("bl-kinds");
  const pagerEl = document.getElementById("bl-pager");
  const gridEl = document.getElementById("bl-grid");
  document.getElementById("bl-min").textContent = fmt(META.thresholds.min);
  document.getElementById("bl-max").textContent = fmt(META.thresholds.max);

  let band = "both";   // min | max | both
  let side = "both";   // kept | dropped | both
  let margin = 5000;
  let page = 0;
  let pageSize = 40;
  let q = "";

  toolbarEl.innerHTML = `
    <input type="search" placeholder="Search episode / action…">
    <label>margin ± <input type="number" class="margin" value="5000" min="100" step="500" style="width:90px"></label>
    <select class="band">
      <option value="both">near min &amp; max</option>
      <option value="min">near min only (${fmt(META.thresholds.min)})</option>
      <option value="max">near max only (${fmt(META.thresholds.max)})</option>
    </select>
    <select class="side">
      <option value="both">just kept + just dropped</option>
      <option value="kept">just kept only</option>
      <option value="dropped">just dropped only</option>
    </select>
    <label>page <input type="number" class="ps" value="40" min="10" max="120" style="width:64px"></label>`;

  kindsEl.innerHTML = "";
  pagerEl.innerHTML = `<button class="prev" type="button">← prev</button>
    <span class="status"></span><button class="next" type="button">next →</button>`;

  const search = toolbarEl.querySelector('input[type=search]');
  const marginEl = toolbarEl.querySelector('.margin');
  const bandEl = toolbarEl.querySelector('.band');
  const sideEl = toolbarEl.querySelector('.side');
  const psEl = toolbarEl.querySelector('.ps');
  const status = pagerEl.querySelector('.status');
  const prev = pagerEl.querySelector('.prev');
  const next = pagerEl.querySelector('.next');

  function render() {
    margin = Math.max(100, parseFloat(marginEl.value) || 5000);
    let recs = collectBorderline(kept, dropped, margin);
    if (band !== "both") recs = recs.filter(r => r.border_band === band);
    if (side !== "both") recs = recs.filter(r => r.border_side === side);
    if (q) {
      const qq = q.toLowerCase();
      recs = recs.filter(r => ((r.source||"") + " " + (r.action||"")).toLowerCase().includes(qq));
    }

    const nMinK = recs.filter(r => r.border_band==="min" && r.border_side==="kept").length;
    const nMinD = recs.filter(r => r.border_band==="min" && r.border_side==="dropped").length;
    const nMaxK = recs.filter(r => r.border_band==="max" && r.border_side==="kept").length;
    const nMaxD = recs.filter(r => r.border_band==="max" && r.border_side==="dropped").length;
    kindsEl.innerHTML = `
      <span class="chip" style="cursor:default">min: <b>${nMinK}</b> kept / <b>${nMinD}</b> dropped</span>
      <span class="chip" style="cursor:default">max: <b>${nMaxK}</b> kept / <b>${nMaxD}</b> dropped</span>
      <span class="chip" style="cursor:default">sorted by |metric − threshold|</span>`;

    const pages = Math.max(1, Math.ceil(recs.length / pageSize));
    if (page >= pages) page = pages - 1;
    if (page < 0) page = 0;
    const start = page * pageSize;
    const slice = recs.slice(start, start + pageSize);
    gridEl.innerHTML = slice.length
      ? slice.map(r => renderCard(r, { showReason: r.border_side === "dropped" })).join("")
      : `<div class="empty">No borderline samples in this margin — try increasing margin.</div>`;
    status.textContent = recs.length
      ? `showing ${start+1}–${Math.min(start+pageSize, recs.length)} of ${recs.length.toLocaleString()}  ·  page ${page+1}/${pages}`
      : "no matches";
    prev.disabled = page <= 0;
    next.disabled = page >= pages - 1;
    bindPairClicks(gridEl);
  }

  search.addEventListener("input", () => { q = search.value.trim(); page = 0; render(); });
  marginEl.addEventListener("input", () => { page = 0; render(); });
  bandEl.addEventListener("change", () => { band = bandEl.value; page = 0; render(); });
  sideEl.addEventListener("change", () => { side = sideEl.value; page = 0; render(); });
  psEl.addEventListener("input", () => { pageSize = Math.max(10, Math.min(120, parseInt(psEl.value,10)||40)); page = 0; render(); });
  prev.addEventListener("click", () => { page--; render(); });
  next.addEventListener("click", () => { page++; render(); });
  render();
}

/* ---------- lightbox ---------- */
const lb = document.getElementById("lightbox");
const lbBefore = document.getElementById("lb-before");
const lbAfter = document.getElementById("lb-after");
const lbMeta = document.getElementById("lb-meta");
const lbOpenBefore = document.getElementById("lb-open-before");
const lbOpenAfter = document.getElementById("lb-open-after");
function openLb(beforeSrc, afterSrc, source, step) {
  lbBefore.src = beforeSrc;
  lbAfter.src = afterSrc;
  lbOpenBefore.href = beforeSrc;
  lbOpenAfter.href = afterSrc;
  lbMeta.textContent = `${source || ""} · step ${step ?? "—"} · before | after`;
  lb.classList.add("open");
}
function closeLb() {
  lb.classList.remove("open");
  lbBefore.removeAttribute("src");
  lbAfter.removeAttribute("src");
}
document.getElementById("lb-close").addEventListener("click", closeLb);
lb.addEventListener("click", e => {
  // Dismiss on empty space / pane chrome; keep clicks on the images & controls.
  if (e.target.closest("img, a.btn, button.close")) return;
  closeLb();
});
document.addEventListener("keydown", e => { if (e.key === "Escape") closeLb(); });

/* ---------- tabs + lazy-load record payloads ---------- */
const views = {
  overview: document.getElementById("view-overview"),
  kept: document.getElementById("view-kept"),
  dropped: document.getElementById("view-dropped"),
  borderline: document.getElementById("view-borderline"),
};
let keptReady = false, dropReady = false, borderlineReady = false;
let keptCache = null, dropCache = null;

async function loadJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function ensureKeptDropped() {
  if (!keptCache) keptCache = (await loadJson("/api/kept")).records;
  if (!dropCache) dropCache = (await loadJson("/api/dropped")).records;
}

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", async () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    const view = tab.dataset.view;
    Object.entries(views).forEach(([k, el]) => el.classList.toggle("hidden", k !== view));
    if (view === "kept" && !keptReady) {
      await ensureKeptDropped();
      makeGallery({
        records: keptCache,
        toolbarEl: document.getElementById("kept-toolbar"),
        kindsEl: document.getElementById("kept-kinds"),
        pagerEl: document.getElementById("kept-pager"),
        gridEl: document.getElementById("kept-grid"),
        showReason: false,
      });
      keptReady = true;
    }
    if (view === "dropped" && !dropReady) {
      await ensureKeptDropped();
      makeGallery({
        records: dropCache,
        toolbarEl: document.getElementById("drop-toolbar"),
        kindsEl: document.getElementById("drop-kinds"),
        pagerEl: document.getElementById("drop-pager"),
        gridEl: document.getElementById("drop-grid"),
        showReason: true,
      });
      dropReady = true;
    }
    if (view === "borderline" && !borderlineReady) {
      await ensureKeptDropped();
      makeBorderlineGallery(keptCache, dropCache);
      borderlineReady = true;
    }
  });
});
</script>
</body>
</html>
"""


def build_page(meta: dict) -> str:
    return (
        APP_HTML
        .replace("__META_JSON__", json.dumps(meta))
        .replace("__REASON_LABELS__", json.dumps(REASON_LABELS))
    )


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # Quieter: skip successful static image spam
        msg = fmt % args
        if " /thumb?" in msg or " /full?" in msg:
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), msg))

    def _send(self, code: int, body: bytes, content_type: str, cache: str | None = None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if cache:
            self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code=200):
        data = json.dumps(obj, default=str).encode("utf-8")
        self._send(code, data, "application/json; charset=utf-8", cache="no-store")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            body = STATE["page_html"].encode("utf-8")
            return self._send(200, body, "text/html; charset=utf-8", cache="no-store")

        if path == "/api/meta":
            return self._send_json(STATE["meta"])

        if path == "/api/kept":
            return self._send_json({"records": STATE["kept"]})

        if path == "/api/dropped":
            return self._send_json({"records": STATE["dropped"]})

        if path in ("/thumb", "/full"):
            raw = (qs.get("p") or [None])[0]
            if not raw:
                return self._send(400, b"missing p", "text/plain")
            # quote was used; unquote
            raw = urllib.parse.unquote(raw)
            src = _safe_resolve(raw)
            if src is None:
                return self._send(404, b"not found / not allowed", "text/plain")
            try:
                if path == "/thumb":
                    data = _make_thumb_bytes(src, max_side=STATE["thumb_side"])
                    return self._send(200, data, "image/jpeg", cache="public, max-age=86400")
                data = src.read_bytes()
                ctype = mimetypes.guess_type(str(src))[0] or "application/octet-stream"
                return self._send(200, data, ctype, cache="public, max-age=86400")
            except Exception as e:
                return self._send(500, str(e).encode(), "text/plain")

        self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=_DEFAULT_DATA_ROOT)
    ap.add_argument("--min", type=float, default=_DEFAULT_MIN_CHANGE_METRIC, dest="min_m")
    ap.add_argument("--max", type=float, default=_DEFAULT_MAX_CHANGE_METRIC, dest="max_m")
    ap.add_argument("--not-annotated", action="store_true")
    ap.add_argument("--no-block", action="store_true")
    ap.add_argument("--drop-action-pattern", action="append", default=None)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (keep 127.0.0.1 and SSH-tunnel)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--thumb-side", type=int, default=720)
    ap.add_argument("--cache-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".thumb_cache"))
    args = ap.parse_args()

    if args.no_block:
        blocked_src = ()
    else:
        blocked_src = tuple(args.drop_action_pattern) if args.drop_action_pattern else _DEFAULT_DROP_ACTION_PATTERNS
    blocked = compile_action_patterns(blocked_src)

    use_annotated = not args.not_annotated
    print("Indexing triplets…", flush=True)
    records_by_reason, counts, kinds_kept, kinds_drop, metrics = collect(
        args.data_root, use_annotated, args.min_m, args.max_m, blocked
    )
    print_report(counts, kinds_kept, kinds_drop, metrics, args.min_m, args.max_m)

    print("Serializing gallery payloads…", flush=True)

    kept_recs = list(records_by_reason.get("kept", []))
    kept_recs.sort(key=lambda r: (r["source"], r["step_number"] if r["step_number"] is not None else -1))

    dropped_recs = []
    for reason in REASON_ORDER:
        if reason == "kept":
            continue
        dropped_recs.extend(records_by_reason.get(reason, []))
    dropped_recs.sort(key=lambda r: (r["reason"], r["source"], r["step_number"] if r["step_number"] is not None else -1))

    total = sum(v for k, v in counts.items() if k != "json_error")
    meta = {
        "thresholds": {"min": args.min_m, "max": args.max_m},
        "total": total,
        "counts": dict(counts),
        "kinds_kept": dict(kinds_kept),
        "kinds_drop": dict(kinds_drop),
        "metric_stats": {k: stats(v) for k, v in metrics.items()},
        "reason_order": REASON_ORDER,
        "result_dirs": list(_RESULT_DIRS),
        "n_kept": len(kept_recs),
        "n_dropped": len(dropped_recs),
    }

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    STATE.update({
        "data_root": Path(args.data_root),
        "thumb_cache": cache_dir,
        "thumb_side": args.thumb_side,
        "meta": meta,
        "kept": [_serialize_record(r) for r in kept_recs],
        "dropped": [_serialize_record(r) for r in dropped_recs],
        "page_html": build_page(meta),
    })

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print()
    print("=" * 64)
    print(f"Audit server listening on {url}")
    print(f"  kept={len(kept_recs):,}  dropped={len(dropped_recs):,}")
    print(f"  thumb cache: {cache_dir}")
    print()
    print("On your laptop, port-forward then open a real browser:")
    print(f"  ssh -L {args.port}:127.0.0.1:{args.port} <user>@<this-host>")
    print(f"  open http://127.0.0.1:{args.port}/")
    print("=" * 64)
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
