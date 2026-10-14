"""Create six E1 prediction examples for REPORT.md from the held-out test split."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, parse_output


def draw_box(draw, box, width, height, color, text):
    x1, y1, x2, y2 = box
    xy = (round(x1 * width / 1000), round(y1 * height / 1000),
          round(x2 * width / 1000), round(y2 * height / 1000))
    draw.rectangle(xy, outline=color, width=3)
    draw.text((xy[0] + 2, max(0, xy[1] - 12)), text, fill=color,
              stroke_width=2, stroke_fill="black")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--pred", default="work/e1.jsonl")
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(args.pred)}
    rows = read_jsonl(data / "splits/test.jsonl")
    chosen = {}
    for row in rows:
        pred = parse_output(preds[row["image_id"]])
        if pred is None:
            continue
        gt = _gt_boxes(row)
        per, _ = _match(gt, pred)
        for label in ("pothole", "crack", "manhole"):
            for kind, index in (("hit", 0), ("miss", 2)):
                key = f"{label}_{kind}"
                if key not in chosen and per[label][index] > 0:
                    chosen[key] = (row, gt, pred)
        if len(chosen) == 6:
            break
    out.mkdir(parents=True, exist_ok=True)
    for key, (row, gt, pred) in chosen.items():
        with Image.open(data / row["image_path"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in gt:
            draw_box(draw, box["bbox"], image.width, image.height,
                     "#37ff75", f"GT {box['label']}")
        for box in pred:
            draw_box(draw, box["bbox"], image.width, image.height,
                     "#ff4343", f"Pred {box['label']}")
        image.save(out / f"{key}.jpg", quality=90)
        print(key, row["image_id"])
    if len(chosen) < 6:
        raise RuntimeError(f"only found {len(chosen)}/6 example types")


if __name__ == "__main__":
    main()
