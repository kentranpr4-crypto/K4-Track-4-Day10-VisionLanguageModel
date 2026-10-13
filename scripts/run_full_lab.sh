#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(dirname "$(realpath "$0")")")"
mkdir -p work report/evidence

echo '=== E0: zero-shot ==='
if [[ ! -s work/e0.jsonl ]] || [[ "$(wc -l < work/e0.jsonl)" -lt 311 ]]; then
  python run_lab.py --experiment e0 --data data --output work/e0.jsonl
else
  echo 'E0 predictions already complete; skipping inference.'
fi
python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json

for e in e1 e2 e3; do
  echo "=== ${e^^}: training and scoring ==="
  if [[ ! -s "work/$e.jsonl" ]] || [[ ! -s "work/$e/run.json" ]]; then
    python run_lab.py --experiment "$e" --data data \
      --output "work/$e.jsonl" --run-dir "work/$e"
  else
    echo "$e output already exists; skipping training."
  fi
  read -r trainable vram minutes < <(python - "$e" <<'PY'
import json, sys
d = json.load(open(f"work/{sys.argv[1]}/run.json"))
print(d["trainable_params"], d["peak_vram_gb"], d["train_minutes"])
PY
  )
  python scripts/score.py --pred "work/$e.jsonl" --name "${e^^}" \
    --out "report/evidence/$e.json" \
    --meta "trainable_params=$trainable" \
    --meta "peak_vram_gb=$vram" \
    --meta "train_minutes=$minutes"
done

for rank in 4 64; do
  echo "=== E4: LoRA rank $rank ==="
  if [[ ! -s "work/e4_rank_${rank}.jsonl" ]] || [[ ! -s "work/e4_rank_${rank}/run.json" ]]; then
    python run_lab.py --experiment e4 --rank "$rank" --data data \
      --output "work/e4_rank_${rank}.jsonl" --run-dir "work/e4_rank_${rank}"
  else
    echo "E4 rank $rank output already exists; skipping training."
  fi
  python scripts/score.py --pred "work/e4_rank_${rank}.jsonl" \
    --name "E4-r${rank}" --out "report/evidence/e4_rank_${rank}.json"
done

echo '=== submission checks ==='
python scripts/check_submission.py
