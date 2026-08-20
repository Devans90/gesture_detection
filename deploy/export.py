"""
deploy/export.py
----------------
CLI helper: export PyTorch checkpoint → numpy .npz (run on your laptop).

Usage::

    python -m deploy.export --checkpoint runs/<run_id>/best.pt \\
                            --output deploy/weights.npz
"""

import argparse
from deploy.inference import export_weights


def main() -> None:
    p = argparse.ArgumentParser(description="Export PyTorch weights to numpy .npz")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt")
    p.add_argument("--output", default="deploy/weights.npz", help="Output .npz path")
    args = p.parse_args()
    export_weights(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
