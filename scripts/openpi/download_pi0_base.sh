#!/usr/bin/env bash
# Resume-download official JAX π0_base from the public OpenPI S3 mirror.
# Source of truth: gs://openpi-assets/checkpoints/pi0_base (~11.19 GiB).
set -euo pipefail

DEST="${1:-/gaozt-test1/zhanxch/FITWAM-dewov9-20260828/checkpoints/openpi}"
MANIFEST="${2:-/tmp/pi0_base_manifest.json}"
INPUT=/tmp/pi0_base_aria2_resume.txt

python3 - "$DEST" "$MANIFEST" "$INPUT" <<'PY'
import json, pathlib, sys
dest, manifest_path, input_path = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
lines = []
todo_bytes = 0
for item in sorted(manifest, key=lambda x: -x["size"]):
    out = dest / item["name"][len("checkpoints/"):]
    out.parent.mkdir(parents=True, exist_ok=True)
    have = out.stat().st_size if out.exists() else 0
    if have == item["size"]:
        continue
    todo_bytes += item["size"] - have
    lines.append("https://openpi-assets.s3.amazonaws.com/" + item["name"])
    lines.append(f"  dir={out.parent}")
    lines.append(f"  out={out.name}")
    lines.append("  continue=true")
print(f"remaining {len(lines)//4} files, {todo_bytes/1024/1024:.1f} MiB")
input_path.write_text("\n".join(lines) + ("\n" if lines else ""))
if not lines:
    raise SystemExit(0)
PY

if [[ ! -s "$INPUT" ]]; then
  echo "pi0_base already complete under ${DEST}/pi0_base"
  exit 0
fi

aria2c -c -x 16 -s 16 -j 4 --file-allocation=none --timeout=60 --retry-wait=3 \
  --max-tries=0 --summary-interval=20 -i "$INPUT"
echo "pi0_base download finished"
