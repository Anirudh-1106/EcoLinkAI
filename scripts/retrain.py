"""
Scheduled retraining job.

Trains the MC-GNN on whatever exchange history the database holds now, and
promotes the result only if it beats the model in production on a held-out
split. The platform's data grows as companies trade, so a model trained months
ago steadily falls behind what the graph actually looks like.

Intended to be run unattended, so it:
  - writes a dated log under logs/retrain/ that can be read the next morning
  - exits non-zero when training fails, which is what a scheduler notices
  - reports whether a new model was promoted or the old one was kept

A promoted checkpoint is picked up by a running API server on its next graph
cache refresh, so no restart or deploy is needed.

Usage:
    python -m scripts.retrain
    python -m scripts.retrain --epochs 500

Scheduling (Windows Task Scheduler, weekly at 02:00):
    schtasks /create /tn "EcoLinkAI Retrain" /sc weekly /d SUN /st 02:00 ^
      /tr "cmd /c cd /d E:\\PROJECTS\\EcoLinkAI && .venv\\Scripts\\python.exe -m scripts.retrain"

Scheduling (cron, weekly at 02:00 on Sunday):
    0 2 * * 0 cd /path/to/EcoLinkAI && .venv/bin/python -m scripts.retrain
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
for path in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

LOG_DIR = PROJECT_ROOT / "logs" / "retrain"

logger = logging.getLogger("retrain")


def _configure_logging() -> Path:
    """Log to both the console and a dated file. Returns the log file path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Attach to the root logger so the training module's own output is captured
    # too -- the detail of why a model was or wasn't promoted lives there.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain and conditionally promote the MC-GNN.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Maximum training epochs; early stopping usually ends the run sooner.",
    )
    args = parser.parse_args()

    log_path = _configure_logging()
    logger.info("Retraining run started. Log: %s", log_path)

    try:
        from ai.training.train import train_mc_gnn

        result = train_mc_gnn(epochs=args.epochs)
    except Exception:
        logger.exception("Retraining failed.")
        return 1

    if not result:
        logger.error("Training returned no result; treating as a failure.")
        return 1

    held_out = result["held_out"]
    baseline = result["baseline"]

    logger.info("─" * 60)
    if result["promoted"]:
        logger.info("PROMOTED %s -- it beat the model in production.", result["checkpoint"])
        logger.info("A running server picks this up on its next cache refresh.")
    else:
        logger.info(
            "KEPT the existing production model; %s did not beat it by enough.",
            result["checkpoint"],
        )
    logger.info(
        "Held-out AUC %.4f (baseline %.4f) | NDCG@5 %.4f (baseline %.4f) | best epoch %d",
        held_out["auc"],
        baseline["auc"],
        held_out["ndcg@5"],
        baseline["ndcg@5"],
        result["best_epoch"],
    )
    logger.info("─" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
