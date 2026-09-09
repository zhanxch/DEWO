#!/usr/bin/env python3
"""Write a self-contained HTML of Pass@M vs prefix for each failure rollout.

No plotly required. Reads shard prefix_results.jsonl + event_pairs.

Example::

    python scripts/fold_glasses/plot_recoverability_prefix_curves.py \\
      --scan-root prepare_results/dexjoco/fold_glasses/<stamp>/step_055000/recoverability_pairs
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_episodes(scan_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(scan_root.glob("shard*/prefix_results.jsonl")):
        rows.extend(load_jsonl(path))
    merged = scan_root / "prefix_results.jsonl"
    if merged.is_file() and not rows:
        rows = load_jsonl(merged)

    by_ep: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_ep[int(row["source_failure_episode_index"])].append(row)

    pairs: dict[int, dict] = {}
    for path in sorted(scan_root.glob("shard*/event_pairs/*/pair.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        frontier = payload.get("frontier") or {}
        ep = int(payload["source_failure_episode_index"])
        pairs[ep] = {
            "t": int(frontier.get("t_frame") or frontier.get("last_recoverable_frame") or 0),
            "M": int(frontier.get("first_zero_frame") or frontier.get("failure_frame") or 0),
            "k_at_t": int(frontier.get("last_recoverable_success_count") or 0),
            "status": payload.get("status"),
        }

    episodes: list[dict] = []
    for ep in sorted(by_ep):
        pts = sorted(by_ep[ep], key=lambda row: int(row["prefix_frame"]))
        first = pts[0]
        pair = pairs.get(ep, {})
        last_zero = int(pts[-1]["prefix_frame"]) if int(pts[-1]["success_count"]) == 0 else None
        episodes.append(
            {
                "ep": ep,
                "seed": int(first["seed"]),
                "cls": first.get("seed_classification"),
                "pass_m": int(first.get("pass_m") or 10),
                "curve": [
                    {"f": int(row["prefix_frame"]), "k": int(row["success_count"])}
                    for row in pts
                ],
                "t": pair.get("t"),
                "M": pair.get("M") or last_zero,
                "kAtT": pair.get("k_at_t"),
                "hasPair": pair.get("status") == "complete",
            }
        )
    return episodes


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>Recoverability Pass@M vs prefix</title>
<style>
  :root { color-scheme: light; --bg:#f7f6f3; --ink:#1b1b1b; --muted:#5c5c5c; --line:#d9d6cf; --card:#fff; }
  body { font: 13px/1.45 ui-sans-serif, system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--ink); }
  header { padding: 20px 24px 8px; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 6px; }
  .sub { color: var(--muted); }
  .stats { display: flex; gap: 24px; padding: 8px 24px 16px; }
  .stat b { display: block; font-size: 22px; font-weight: 600; }
  .stat span { color: var(--muted); }
  section { background: var(--card); margin: 0 24px 20px; padding: 16px; border: 1px solid var(--line); }
  section h2 { font-size: 14px; font-weight: 600; margin: 0 0 12px; }
  .heat { overflow: auto; }
  table.heat { border-collapse: collapse; font-variant-numeric: tabular-nums; }
  table.heat th, table.heat td { min-width: 28px; height: 22px; text-align: center; padding: 0 2px; border: 1px solid #eee; }
  table.heat th { font-weight: 500; color: var(--muted); font-size: 11px; }
  table.heat td.lab { text-align: left; min-width: 120px; padding: 0 8px; white-space: nowrap; background: #fff; position: sticky; left: 0; }
  table.heat tr.sel td.lab { font-weight: 600; }
  .legend { display: flex; gap: 8px; align-items: center; margin-top: 10px; color: var(--muted); font-size: 12px; }
  .sw { width: 12px; height: 12px; display: inline-block; }
  select { font: inherit; padding: 4px 8px; }
  canvas { width: 100%; height: 280px; background: #fff; }
  caption-note { display: block; margin-top: 8px; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>fold_glasses recoverability along each failure rollout</h1>
  <div class="sub">SOURCE · Pass@__PASS_M__ · scan stops at the first 0/M cliff · blank = not scanned</div>
</header>
<div class="stats" id="stats"></div>
<section>
  <h2>Pass@__PASS_M__ successes at each prefix frame</h2>
  <div class="heat" id="heat"></div>
  <div class="legend" id="legend"></div>
</section>
<section>
  <h2>
    Selected rollout
    <select id="pick"></select>
  </h2>
  <canvas id="curve" width="1100" height="280"></canvas>
  <div class="sub" id="curveCap"></div>
</section>
<script>
const EPS = __EPISODES__;
const PASS_M = __PASS_M__;
const frames = [...new Set(EPS.flatMap(e => e.curve.map(p => p.f)))].sort((a,b)=>a-b);
function viridis(t) {
  const stops = [
    [68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]
  ];
  t = Math.max(0, Math.min(1, t));
  const x = t * (stops.length-1);
  const i = Math.min(stops.length-2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i+1];
  const r = Math.round(a[0]+(b[0]-a[0])*f);
  const g = Math.round(a[1]+(b[1]-a[1])*f);
  const bl = Math.round(a[2]+(b[2]-a[2])*f);
  return `rgb(${r},${g},${bl})`;
}
function cellColor(k) {
  if (k == null) return "#f2f1ee";
  return viridis(k / PASS_M);
}
const ts = EPS.map(e => e.t).filter(v => v != null);
const Ms = EPS.map(e => e.M).filter(v => v != null);
const med = arr => { const s=[...arr].sort((a,b)=>a-b); return s[Math.floor(s.length/2)]; };
document.getElementById("stats").innerHTML = [
  ["Completed failure rollouts", EPS.length],
  ["Pass@M", PASS_M],
  ["t* median (last recoverable prefix)", med(ts)],
  ["M median (first 0/M cliff)", med(Ms)],
  ["t* range", Math.min(...ts) + " – " + Math.max(...ts)],
].map(([k,v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");

const heat = document.getElementById("heat");
let html = "<table class='heat'><thead><tr><th></th>" + frames.map(f=>`<th>${f}</th>`).join("") + "</tr></thead><tbody>";
EPS.forEach((e, idx) => {
  const map = Object.fromEntries(e.curve.map(p => [p.f, p.k]));
  html += `<tr data-i="${idx}"><td class="lab">ep${String(e.ep).padStart(3,"0")} seed ${e.seed}</td>`;
  frames.forEach(f => {
    const k = map[f];
    const label = k == null ? "" : k;
    html += `<td style="background:${cellColor(k)}" title="ep${e.ep} prefix ${f}: ${k == null ? "not scanned" : k+"/"+PASS_M}">${label}</td>`;
  });
  html += "</tr>";
});
html += "</tbody></table>";
heat.innerHTML = html;
document.getElementById("legend").innerHTML =
  "0 " + Array.from({length: PASS_M+1}, (_,k) => `<span class="sw" style="background:${cellColor(k)}"></span>`).join("") + ` ${PASS_M} successes &nbsp; gray = not scanned`;

const pick = document.getElementById("pick");
EPS.forEach((e,i) => {
  const o = document.createElement("option");
  o.value = i;
  o.textContent = `ep${e.ep}  seed ${e.seed}  t*=${e.t}  M=${e.M}  k(t*)=${e.kAtT}`;
  pick.appendChild(o);
});

function draw(i) {
  const e = EPS[i];
  const cv = document.getElementById("curve");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0,0,W,H);
  const pad = {l:48, r:16, t:16, b:36};
  const xs = e.curve.map(p => p.f);
  const xmin = Math.min(...frames), xmax = Math.max(...frames);
  const x = f => pad.l + (f-xmin)/(xmax-xmin) * (W-pad.l-pad.r);
  const y = k => pad.t + (1 - k/PASS_M) * (H-pad.t-pad.b);
  ctx.strokeStyle = "#d9d6cf";
  ctx.fillStyle = "#5c5c5c";
  ctx.font = "11px ui-sans-serif";
  for (let k=0;k<=PASS_M;k+=2) {
    ctx.beginPath(); ctx.moveTo(pad.l,y(k)); ctx.lineTo(W-pad.r,y(k)); ctx.stroke();
    ctx.fillText(String(k), 8, y(k)+4);
  }
  ctx.beginPath();
  e.curve.forEach((p,j) => {
    const px=x(p.f), py=y(p.k);
    if (j===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  });
  ctx.strokeStyle = "#1b1b1b"; ctx.lineWidth = 2; ctx.stroke();
  e.curve.forEach(p => {
    ctx.beginPath(); ctx.arc(x(p.f), y(p.k), 3.5, 0, Math.PI*2);
    ctx.fillStyle = cellColor(p.k); ctx.fill();
    ctx.strokeStyle = "#1b1b1b"; ctx.lineWidth=1; ctx.stroke();
  });
  if (e.t != null) {
    ctx.setLineDash([4,4]); ctx.strokeStyle="#888";
    ctx.beginPath(); ctx.moveTo(x(e.t), pad.t); ctx.lineTo(x(e.t), H-pad.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x(e.M), pad.t); ctx.lineTo(x(e.M), H-pad.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle="#1b1b1b";
    ctx.fillText("t*", x(e.t)-6, H-12);
    ctx.fillText("M", x(e.M)-4, H-12);
  }
  document.getElementById("curveCap").textContent =
    `ep${e.ep} seed ${e.seed}: last recoverable prefix t*=${e.t} with ${e.kAtT}/${PASS_M} successes; first 0/${PASS_M} cliff M=${e.M}. Event window is [M-33, M).`;
  document.querySelectorAll("table.heat tr").forEach(tr => tr.classList.toggle("sel", tr.dataset.i === String(i)));
}
pick.onchange = () => draw(+pick.value);
heat.addEventListener("click", ev => {
  const tr = ev.target.closest("tr[data-i]");
  if (!tr) return;
  pick.value = tr.dataset.i;
  draw(+tr.dataset.i);
});
draw(0);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    scan_root = args.scan_root.expanduser().resolve()
    episodes = load_episodes(scan_root)
    if not episodes:
        raise SystemExit(f"No prefix_results under {scan_root}")
    pass_m = int(episodes[0]["pass_m"])
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else scan_root / "recoverability_prefix_curves.html"
    )
    html = (
        HTML.replace("__EPISODES__", json.dumps(episodes, separators=(",", ":")))
        .replace("__PASS_M__", str(pass_m))
        .replace("SOURCE", str(scan_root))
    )
    output.write_text(html, encoding="utf-8")
    print(f"wrote {output}  episodes={len(episodes)} pass_m={pass_m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
