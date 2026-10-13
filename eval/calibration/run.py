"""CLI: прогон калибровочных метрик, сохранение reliability diagram."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.calibration.metrics import brier_score, calibration_curve, ece


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration evaluation")
    parser.add_argument("--probs", type=Path, required=True, help="npy file with probabilities")
    parser.add_argument("--outcomes", type=Path, required=True, help="npy file with binary outcomes")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/calibration/out"))
    args = parser.parse_args()

    probs = np.load(args.probs)
    outcomes = np.load(args.outcomes)
    assert probs.shape == outcomes.shape

    metrics = {
        "ece": ece(probs, outcomes, n_bins=args.bins),
        "brier": brier_score(probs, outcomes),
        "n_bins": args.bins,
        "n_samples": len(probs),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fig = calibration_curve(probs, outcomes, n_bins=args.bins)
    fig.savefig(args.out_dir / "reliability.png", dpi=150)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()