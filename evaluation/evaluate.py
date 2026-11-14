"""Evaluate B1 checkpoint against a Dev-A or Dev-B CSV manifest.

Example:
python -m evaluation.evaluate \
  --checkpoint path/to/b1_xlsr_best.pt \
  --manifest path/to/dev_b_manifest.csv \
  --audio-dir /path/to/ASVspoof2019_LA_dev/flac \
  --output-dir evaluation/results/dev_b

Manifest must contain label_id and either audio_path or file_id.
When --audio-dir is supplied, file_id.flac is used (portable path override).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor
from tqdm import tqdm

from models.b1_xlsr import load_b1
from evaluation.metrics import compute_metrics


def resolve_audio_path(row, audio_dir):
    if audio_dir:
        file_id = row.get("file_id")
        if pd.isna(file_id) or not str(file_id).strip():
            # Some manifests only have audio_path; use its basename.
            file_id = Path(str(row["audio_path"])).stem
        return Path(audio_dir) / f"{file_id}.flac"
    if "audio_path" not in row or pd.isna(row["audio_path"]):
        raise ValueError("Manifest needs audio_path or --audio-dir with file_id.")
    return Path(str(row["audio_path"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument("--model-name", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Keep at 1 initially to match notebook validation.")
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("This first evaluator intentionally supports batch-size 1.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model, checkpoint = load_b1(args.checkpoint, args.model_name, device)

    df = pd.read_csv(args.manifest)
    required = {"label_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if not ({"audio_path", "file_id"} & set(df.columns)):
        raise ValueError("Manifest needs audio_path or file_id.")

    labels, scores, rows = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        path = resolve_audio_path(row, args.audio_dir)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"{path}: expected 16 kHz, got {sr}")
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError(f"{path}: empty or non-finite audio")

        batch = extractor(
            audio, sampling_rate=16000, return_attention_mask=True,
            return_tensors="pt"
        )
        input_values = batch.input_values.to(device)
        attention_mask = batch.attention_mask.to(device)

        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_values, attention_mask)
            else:
                logits = model(input_values, attention_mask)
            spoof_score = torch.softmax(logits.float(), dim=1)[0, 1].item()

        label = int(row["label_id"])
        if label not in (0, 1):
            raise ValueError(f"{path}: label_id must be 0 or 1, got {label}")
        labels.append(label)
        scores.append(spoof_score)
        rows.append({
            "file_id": str(row.get("file_id", path.stem)),
            "audio_path": str(path),
            "label_id": label,
            "spoof_score": spoof_score,
            "bonafide_score": 1.0 - spoof_score
        })

    metrics = compute_metrics(labels, scores)
    metrics["checkpoint_epoch"] = checkpoint.get("epoch")
    metrics["checkpoint_global_step"] = checkpoint.get("global_step")
    metrics["checkpoint_best_dev_a_eer"] = checkpoint.get("best_dev_a_eer")
    metrics["evaluated_manifest"] = str(Path(args.manifest).resolve())
    metrics["score_definition"] = "softmax(logits)[:, 1] = spoof probability"

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "roc"}, indent=2))
    print(f"\nSaved: {out / 'scores.csv'}")
    print(f"Saved: {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
