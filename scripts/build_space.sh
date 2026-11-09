#!/bin/sh
# Stage the Hugging Face Space bundle: the server, the shared drowsim package,
# the dashboard, the BiLSTM weights and the recorded demo telemetry.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-"$ROOT/build/space"}

for f in models/bilstm_attention.pt server/demo_telemetry.jsonl; do
  [ -f "$ROOT/$f" ] || { echo "missing $f" >&2; exit 1; }
done

rm -rf "$OUT"
mkdir -p "$OUT/models" "$OUT/server" "$OUT/drowsim" "$OUT/dashboard"

cp "$ROOT"/deploy/huggingface/Dockerfile \
   "$ROOT"/deploy/huggingface/README.md \
   "$ROOT"/deploy/huggingface/requirements.txt "$OUT/"
cp "$ROOT"/server/__init__.py "$ROOT"/server/server.py "$ROOT"/server/demo_telemetry.jsonl "$OUT/server/"
cp "$ROOT"/drowsim/*.py "$OUT/drowsim/"
cp "$ROOT"/dashboard/index.html "$OUT/dashboard/"
cp "$ROOT"/models/bilstm_attention.pt "$OUT/models/"

echo "staged $OUT"
find "$OUT" -type f | sed "s|$OUT/||" | sort
