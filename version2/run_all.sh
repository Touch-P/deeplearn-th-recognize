#!/usr/bin/env bash
# run_all.sh - รัน pipeline PyTorch ทั้งสายต่อเนื่อง (ขั้นที่ 2 ถึง 6)
#
# ขั้นที่ 1 (สำรวจ + แบ่ง train/val/test) รันแยกครั้งเดียวพอ เพราะ split ถูก
# ล็อกด้วย seed แล้วเขียนไว้ที่ outputs/splits/splits.csv
#
# ใช้ --num-workers 4 ทุกขั้นเพื่อคุมการใช้ RAM: บน Windows แต่ละ worker เป็น
# process ใหม่ที่ import torch ใหม่ ถ้าตั้งสูงกว่านี้แล้วมีโปรแกรมอื่นกิน RAM
# อยู่ด้วยจะเจอ "OSError [WinError 1455] The paging file is too small"
#
# รัน:  bash run_all.sh
# log ของแต่ละขั้นอยู่ที่ outputs/logs/stepN_stdout.txt

set -euo pipefail
cd "$(dirname "$0")"

# หา Python interpreter: ตัวแปร PYTHON > venv ใน version2/ > venv ที่ repo root
#   > python ใน PATH (เช่นเมื่อ activate environment ไว้แล้ว หรือรันผ่าน uv run)
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
elif [ -x "../.venv/Scripts/python.exe" ]; then
  PY="../.venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif [ -x "../.venv/bin/python" ]; then
  PY="../.venv/bin/python"
else
  PY="python"
fi
echo "ใช้ Python: $PY"
export PYTHONIOENCODING=utf-8
LOGS="outputs/logs"
mkdir -p "$LOGS"

run_step() {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $name ==="
  "$PY" "$@" > "$LOGS/${name}_stdout.txt" 2>&1
  echo "=== [$(date +%H:%M:%S)] $name เสร็จ (exit $?) ==="
}

run_step step2 run_step2_baseline.py --epochs 3 --num-workers 4
run_step step3 run_step3_augment_plan.py
run_step step4 run_step4_train.py --epochs 10 --num-workers 4
run_step step5 run_step5_evaluate.py
run_step step6 run_step6_test_inference.py

echo "=== [$(date +%H:%M:%S)] ครบทุกขั้น ==="
