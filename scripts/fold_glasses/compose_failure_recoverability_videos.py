#!/usr/bin/env python3
"""Compose original-failure mp4s with a synced Pass@M continuation curve.

Layout: factual failure video on top, V(s)=k/10 step plot below. When the
playhead enters a v9.1 CFG event (Δk≥3 and k≥1 over one replan), a badge is
drawn in the top-left.

Example::

    python scripts/fold_glasses/compose_failure_recoverability_videos.py \\
      --scan-root prepare_results/dexjoco/fold_glasses/<stamp>/step_055000/recoverability_pairs \\
      --raw-dataset collect_results/dexjoco/fold_glasses/<stamp>/step_055000/rollout_raw \\
      --output-dir prepare_results/dexjoco/fold_glasses/<stamp>/result
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPLAN_STEPS = 24
MIN_DELTA_K = 3
EVENT_SPAN = 33
PLOT_HEIGHT = 200
DEFAULT_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


@dataclass(frozen=True)
class CfgEvent:
    t: int
    k: int
    k_prev: int
    delta_k: int
    pass_m: int

    @property
    def v(self) -> float:
        return self.k / float(self.pass_m)

    @property
    def overlay_end(self) -> int:
        return self.t + EVENT_SPAN


@dataclass
class EpisodeScan:
    ep: int
    seed: int
    cls: str | None
    pass_m: int
    nodes: list[tuple[int, int]]
    cliff_m: int | None
    dataset: Path | None


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def collect_video_path(raw: Path, ep: int, camera: str = "front") -> Path:
    chunk = int(ep) // 1000
    return (
        raw
        / f"videos/chunk-{chunk:03d}/observation.images.{camera}/episode_{int(ep):06d}.mp4"
    )


def default_result_dir(scan_root: Path) -> Path:
    if scan_root.name == "recoverability_pairs":
        return scan_root.parents[1] / "result"
    return scan_root / "result"


def load_scan_episodes(scan_root: Path) -> list[EpisodeScan]:
    rows: list[dict] = []
    for path in sorted(scan_root.glob("shard*/prefix_results.jsonl")):
        rows.extend(load_jsonl(path))
    merged = scan_root / "prefix_results.jsonl"
    if merged.is_file() and not rows:
        rows = load_jsonl(merged)

    by_ep: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_ep[int(row["source_failure_episode_index"])].append(row)

    episodes: list[EpisodeScan] = []
    for ep in sorted(by_ep):
        pts = sorted(by_ep[ep], key=lambda row: int(row["prefix_frame"]))
        first = pts[0]
        nodes = [(int(row["prefix_frame"]), int(row["success_count"])) for row in pts]
        cliff = next((frame for frame, k in nodes if k == 0), None)
        dataset = first.get("run_signature", {}).get("dataset")
        episodes.append(
            EpisodeScan(
                ep=ep,
                seed=int(first["seed"]),
                cls=first.get("seed_classification"),
                pass_m=int(first.get("pass_m") or 10),
                nodes=nodes,
                cliff_m=cliff,
                dataset=None if dataset is None else Path(dataset),
            )
        )
    return episodes


def cfg_events(
    nodes: list[tuple[int, int]],
    *,
    replan_steps: int = REPLAN_STEPS,
    min_delta_k: int = MIN_DELTA_K,
    pass_m: int = 10,
) -> list[CfgEvent]:
    """Offline v9.1 CFG events: Δk = k_{t-replan} - k_t ≥ 3 and k_t ≥ 1.

    ``t=48`` has no previous scan node, so it is never an event.
    A drop to 0/10 is the cliff, not a CFG event.
    """

    by_frame = {int(frame): int(k) for frame, k in nodes}
    events: list[CfgEvent] = []
    for frame, k in nodes:
        prev = int(frame) - int(replan_steps)
        if prev not in by_frame:
            continue
        k_prev = by_frame[prev]
        delta = k_prev - int(k)
        if delta >= int(min_delta_k) and int(k) >= 1:
            events.append(
                CfgEvent(
                    t=int(frame),
                    k=int(k),
                    k_prev=int(k_prev),
                    delta_k=int(delta),
                    pass_m=int(pass_m),
                )
            )
    return events


def held_v_series(
    nodes: list[tuple[int, int]],
    n_frames: int,
    pass_m: int,
) -> np.ndarray:
    """Hold k/10 from each scan node until the next; NaN before the first node.

    No interpolation between nodes.
    """

    series = np.full(int(n_frames), np.nan, dtype=np.float32)
    if not nodes or n_frames <= 0:
        return series
    ordered = sorted((int(f), int(k) / float(pass_m)) for f, k in nodes)
    starts = [frame for frame, _ in ordered]
    values = [value for _, value in ordered]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else n_frames
        lo = max(0, start)
        hi = min(n_frames, end)
        if hi > lo:
            series[lo:hi] = values[i]
    last_start = starts[-1]
    if last_start < n_frames:
        series[last_start:] = values[-1]
    return series


def active_cfg_event(
    events: list[CfgEvent],
    frame: int,
    *,
    span: int = EVENT_SPAN,
) -> CfgEvent | None:
    matched = [event for event in events if event.t <= frame < event.t + span]
    if not matched:
        return None
    return max(matched, key=lambda event: event.t)


def _v_color_rgb(v: float) -> tuple[int, int, int]:
    t = float(np.clip(v, 0.0, 1.0))
    stops = np.array(
        [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
        dtype=np.float32,
    )
    x = t * (len(stops) - 1)
    i = min(len(stops) - 2, int(np.floor(x)))
    f = x - i
    rgb = stops[i] + (stops[i + 1] - stops[i]) * f
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if DEFAULT_FONT.is_file():
        return ImageFont.truetype(str(DEFAULT_FONT), size=size)
    return ImageFont.load_default()


def render_plot_base(
    *,
    n_frames: int,
    v_series: np.ndarray,
    nodes: list[tuple[int, int]],
    events: list[CfgEvent],
    pass_m: int,
    width: int,
    height: int = PLOT_HEIGHT,
) -> np.ndarray:
    img = np.full((height, width, 3), 247, dtype=np.uint8)
    pad_l, pad_r, pad_t, pad_b = 52, 14, 30, 38
    plot_w = max(1, width - pad_l - pad_r)
    plot_h = max(1, height - pad_t - pad_b)
    denom = max(n_frames - 1, 1)

    def x_of(frame: int) -> int:
        return pad_l + int(round(frame / denom * plot_w))

    def y_of(value: float) -> int:
        return pad_t + int(round((1.0 - float(value)) * plot_h))

    for k in range(0, pass_m + 1, max(1, pass_m // 5)):
        y = y_of(k / pass_m)
        cv2.line(img, (pad_l, y), (width - pad_r, y), (217, 214, 207), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{k}/{pass_m}",
            (4, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (92, 92, 92),
            1,
            cv2.LINE_AA,
        )

    event_ts = {event.t for event in events}
    for event in events:
        x0, x1 = x_of(event.t), x_of(min(n_frames - 1, event.overlay_end - 1))
        overlay = img[pad_t : height - pad_b, x0:x1]
        if overlay.size:
            tint = overlay.astype(np.int16)
            tint[..., 0] = np.minimum(255, tint[..., 0] + 18)
            tint[..., 1] = np.minimum(255, tint[..., 1] + 8)
            overlay[:] = np.clip(tint, 0, 255).astype(np.uint8)

    xs = np.flatnonzero(~np.isnan(v_series))
    if xs.size >= 2:
        pts = np.stack(
            [np.array([x_of(int(f)) for f in xs]), np.array([y_of(float(v_series[f])) for f in xs])],
            axis=1,
        ).astype(np.int32)
        cv2.polylines(img, [pts], False, (27, 27, 27), 2, cv2.LINE_AA)

    for frame, k in nodes:
        center = (x_of(frame), y_of(k / pass_m))
        cv2.circle(img, center, 5, _v_color_rgb(k / pass_m), -1, cv2.LINE_AA)
        cv2.circle(img, center, 5, (27, 27, 27), 1, cv2.LINE_AA)
        if frame in event_ts:
            cv2.circle(img, center, 9, (230, 126, 34), 2, cv2.LINE_AA)

    cv2.putText(
        img,
        "V(s) = Pass@M continuation success   held between scan nodes, not interpolated",
        (pad_l, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (92, 92, 92),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "env step",
        (width // 2 - 28, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (92, 92, 92),
        1,
        cv2.LINE_AA,
    )
    return img


def draw_playhead(plot: np.ndarray, *, frame: int, n_frames: int) -> np.ndarray:
    out = plot.copy()
    pad_l, pad_r, pad_t, pad_b = 52, 14, 30, 38
    denom = max(n_frames - 1, 1)
    x = pad_l + int(round(frame / denom * (plot.shape[1] - pad_l - pad_r)))
    cv2.line(out, (x, pad_t), (x, plot.shape[0] - pad_b), (192, 57, 43), 2, cv2.LINE_AA)
    return out


def draw_cfg_badge(rgb: np.ndarray, event: CfgEvent) -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = _font(22)
    small = _font(16)
    title = "CFG event"
    detail = (
        f"t={event.t}  {event.k_prev}/{event.pass_m} -> {event.k}/{event.pass_m}"
        f"  drop={event.delta_k}"
    )
    pad = 10
    title_box = draw.textbbox((0, 0), title, font=font)
    detail_box = draw.textbbox((0, 0), detail, font=small)
    width = max(title_box[2] - title_box[0], detail_box[2] - detail_box[0]) + 2 * pad
    height = (title_box[3] - title_box[1]) + (detail_box[3] - detail_box[1]) + 3 * pad
    x0, y0 = 12, 12
    draw.rounded_rectangle(
        (x0, y0, x0 + width, y0 + height),
        radius=8,
        fill=(230, 126, 34),
        outline=(146, 79, 21),
        width=2,
    )
    draw.text((x0 + pad, y0 + pad - 2), title, font=font, fill=(255, 255, 255))
    draw.text(
        (x0 + pad, y0 + pad + (title_box[3] - title_box[1]) + 4),
        detail,
        font=small,
        fill=(255, 255, 255),
    )
    return np.asarray(image, dtype=np.uint8)


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


class Mp4Writer:
    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        self.tmp = path.with_name(path.stem + ".tmp" + path.suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(self.tmp), mode="w")
        self.stream = self.container.add_stream("libx264", rate=int(round(fps)) or 30)
        self.stream.width = int(width)
        self.stream.height = int(height)
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"preset": "veryfast", "crf": "20"}

    def write(self, rgb: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.tmp.replace(self.path)


def compose_episode(
    *,
    video_path: Path,
    episode: EpisodeScan,
    output_path: Path,
    events: list[CfgEvent],
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open failure video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 640
    n_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_frames = max(n_hint, episode.nodes[-1][0] + 1 if episode.nodes else 1)
    v_series = held_v_series(episode.nodes, n_frames, episode.pass_m)
    plot_w = _even(src_w)
    plot_h = _even(PLOT_HEIGHT)
    base_plot = render_plot_base(
        n_frames=n_frames,
        v_series=v_series,
        nodes=episode.nodes,
        events=events,
        pass_m=episode.pass_m,
        width=plot_w,
        height=plot_h,
    )
    out_w = _even(src_w)
    out_h = _even(src_h + plot_h)
    writer = Mp4Writer(output_path, out_w, out_h, fps)
    written = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[1] != src_w or rgb.shape[0] != src_h:
                rgb = np.asarray(
                    Image.fromarray(rgb).resize((src_w, src_h), Image.Resampling.BILINEAR)
                )
            event = active_cfg_event(events, written)
            if event is not None:
                rgb = draw_cfg_badge(rgb, event)
            plot = draw_playhead(base_plot, frame=min(written, n_frames - 1), n_frames=n_frames)
            if rgb.shape[1] != plot.shape[1]:
                plot = np.asarray(
                    Image.fromarray(plot).resize((rgb.shape[1], plot.shape[0]), Image.Resampling.BILINEAR)
                )
            stacked = np.concatenate([rgb, plot], axis=0)
            if stacked.shape[1] != out_w or stacked.shape[0] != out_h:
                canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                canvas[: stacked.shape[0], : stacked.shape[1]] = stacked[
                    :out_h, :out_w
                ]
                stacked = canvas
            writer.write(stacked)
            written += 1
    finally:
        cap.release()
        writer.close()
    return {
        "episode": episode.ep,
        "seed": episode.seed,
        "classification": episode.cls,
        "n_frames": written,
        "cliff_m": episode.cliff_m,
        "cfg_events": [
            {
                "t": event.t,
                "k_prev": event.k_prev,
                "k": event.k,
                "delta_k": event.delta_k,
            }
            for event in events
        ],
        "nodes": [{"t": frame, "k": k, "v": k / episode.pass_m} for frame, k in episode.nodes],
        "video": str(output_path),
    }


def write_index(output_dir: Path, rows: list[dict], *, scan_root: Path, raw: Path) -> None:
    payload = {
        "format": "FoldGlassesFailureRecoverabilityResultVideos",
        "scan_root": str(scan_root),
        "raw_dataset": str(raw),
        "cfg_event": "k_{t-24}-k_t >= 3 and k_t >= 1; overlay on [t, t+33)",
        "num_videos": len(rows),
        "episodes": rows,
    }
    (output_dir / "index.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cards = []
    for row in rows:
        name = Path(row["video"]).name
        events = row.get("cfg_events") or []
        event_txt = (
            ", ".join(f"t={e['t']} ({e['k_prev']}→{e['k']})" for e in events)
            if events
            else "none"
        )
        cards.append(
            "<article>"
            f"<h2>ep{row['episode']:06d} · seed {row['seed']}</h2>"
            f"<p>M={row.get('cliff_m')} · CFG events: {event_txt}</p>"
            f'<video controls src="{name}"></video>'
            "</article>"
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>fold_glasses recoverability result videos</title>
<style>
body {{ font: 14px/1.4 ui-sans-serif, system-ui, sans-serif; background:#f7f6f3; color:#1b1b1b; margin:0; }}
header {{ padding:20px 24px 8px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); gap:16px; padding:16px 24px 32px; }}
article {{ background:#fff; border:1px solid #d9d6cf; padding:12px; }}
h2 {{ font-size:14px; margin:0 0 6px; }}
p {{ color:#5c5c5c; margin:0 0 8px; font-size:12px; }}
video {{ width:100%; background:#000; }}
</style></head><body>
<header>
<h1>原始失败轨迹 + 扫描点续行成功率</h1>
<p>Top: original collect failure. Bottom: V=k/10 held between scan nodes (no interpolation). Orange badge = v9.1 CFG event (drop k&gt;=3 and k&gt;=1), shown on [t, t+33).</p>
</header>
<div class="grid">{"".join(cards)}</div>
</body></html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _episode_index_row(episode: EpisodeScan, events: list[CfgEvent], output_path: Path) -> dict:
    return {
        "episode": episode.ep,
        "seed": episode.seed,
        "classification": episode.cls,
        "cliff_m": episode.cliff_m,
        "cfg_events": [
            {
                "t": event.t,
                "k_prev": event.k_prev,
                "k": event.k,
                "delta_k": event.delta_k,
            }
            for event in events
        ],
        "nodes": [
            {"t": frame, "k": k, "v": k / episode.pass_m} for frame, k in episode.nodes
        ],
        "video": str(output_path),
    }


def _compose_job(payload: dict) -> dict:
    episode = EpisodeScan(**payload["episode"])
    events = [CfgEvent(**event) for event in payload["events"]]
    print(
        f"[result-video] ep{episode.ep:06d} seed={episode.seed} "
        f"nodes={len(episode.nodes)} cfg={len(events)} -> {payload['output']}",
        flush=True,
    )
    return compose_episode(
        video_path=Path(payload["video"]),
        episode=episode,
        output_path=Path(payload["output"]),
        events=events,
    )


def compose_scan(
    *,
    scan_root: Path,
    raw_dataset: Path,
    output_dir: Path,
    camera: str = "front",
    skip_existing: bool = True,
    require_cliff: bool = True,
    jobs: int = 1,
) -> list[dict]:
    episodes = load_scan_episodes(scan_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    pending: list[dict] = []
    for episode in episodes:
        if require_cliff and episode.cliff_m is None:
            print(f"[result-video] skip ep{episode.ep:06d}: no 0/10 cliff yet", flush=True)
            continue
        raw = episode.dataset if episode.dataset is not None else raw_dataset
        video_path = collect_video_path(raw, episode.ep, camera=camera)
        if not video_path.is_file():
            print(f"[result-video] skip ep{episode.ep:06d}: missing {video_path}", flush=True)
            continue
        out = output_dir / f"ep{episode.ep:06d}_seed{episode.seed}_{camera}.mp4"
        events = cfg_events(episode.nodes, pass_m=episode.pass_m)
        if skip_existing and out.is_file():
            print(f"[result-video] exists ep{episode.ep:06d} -> {out}", flush=True)
            rows.append(_episode_index_row(episode, events, out))
            continue
        pending.append(
            {
                "episode": {
                    "ep": episode.ep,
                    "seed": episode.seed,
                    "cls": episode.cls,
                    "pass_m": episode.pass_m,
                    "nodes": episode.nodes,
                    "cliff_m": episode.cliff_m,
                    "dataset": None,
                },
                "events": [
                    {
                        "t": event.t,
                        "k": event.k,
                        "k_prev": event.k_prev,
                        "delta_k": event.delta_k,
                        "pass_m": event.pass_m,
                    }
                    for event in events
                ],
                "video": str(video_path),
                "output": str(out),
            }
        )
    if pending:
        workers = max(1, int(jobs))
        if workers == 1:
            for payload in pending:
                rows.append(_compose_job(payload))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_compose_job, payload) for payload in pending]
                for future in as_completed(futures):
                    rows.append(future.result())
    rows.sort(key=lambda row: int(row["episode"]))
    write_index(output_dir, rows, scan_root=scan_root, raw=raw_dataset)
    print(f"[result-video] wrote {len(rows)} videos under {output_dir}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Also render episodes that have not yet hit a 0/10 cliff.",
    )
    args = parser.parse_args()
    scan_root = args.scan_root.expanduser().resolve()
    raw = args.raw_dataset.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_result_dir(scan_root)
    )
    compose_scan(
        scan_root=scan_root,
        raw_dataset=raw,
        output_dir=output,
        camera=str(args.camera),
        skip_existing=not args.overwrite,
        require_cliff=not args.include_partial,
        jobs=int(args.jobs),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
