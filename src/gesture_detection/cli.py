from __future__ import annotations

import argparse
from pathlib import Path

from gesture_detection.app import GestureDetectionApp
from gesture_detection.capture import CaptureSession
from gesture_detection.collector import DataCollector
from gesture_detection.config import AppConfig
from gesture_detection.factories import build_sensor
from gesture_detection.training.dataset import build_dataset_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture detection project scaffold")
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="Path to the JSON config file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Run the capture/display loop.")

    collect_parser = subparsers.add_parser("collect", help="Capture labelled gesture samples.")
    collect_parser.add_argument("label", help="Gesture label to collect.")
    collect_parser.add_argument("--samples", type=int, default=5, help="Number of samples to capture.")

    subparsers.add_parser("inspect-dataset", help="List labels, sessions, and samples already captured.")

    init_parser = subparsers.add_parser("init-config", help="Write a starter config file.")
    init_parser.add_argument(
        "--output",
        default="configs/default.json",
        help="Where to write the generated config.",
    )

    return parser.parse_args()


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        config = AppConfig()
        config.write(config_path)
        return config
    return AppConfig.from_path(config_path)


def run_command(config: AppConfig) -> None:
    GestureDetectionApp(config).run_forever()


def collect_command(config: AppConfig, label: str, samples: int) -> None:
    sensor = build_sensor(config)
    capture_session = CaptureSession(
        sensor=sensor,
        sample_hz=config.capture.sample_hz,
        max_distance_cm=config.capture.max_distance_cm,
    )
    collector = DataCollector(
        capture_session=capture_session,
        output_dir=config.collection.output_dir,
        settle_seconds=config.collection.settle_seconds,
    )
    try:
        collector.collect(label=label, samples=samples, gesture_seconds=config.collection.gesture_seconds)
    finally:
        sensor.close()


def inspect_dataset_command(config: AppConfig) -> None:
    dataset = build_dataset_index(config.collection.output_dir)
    print(f"root={dataset.root_dir}")
    print(f"labels={dataset.labels}")
    print(f"sessions={dataset.session_ids}")
    print(f"samples={len(dataset.samples)}")


def init_config_command(output: str) -> None:
    AppConfig().write(output)
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.command == "run":
        run_command(config)
    elif args.command == "collect":
        collect_command(config, label=args.label, samples=args.samples)
    elif args.command == "inspect-dataset":
        inspect_dataset_command(config)
    elif args.command == "init-config":
        init_config_command(args.output)
    else:  # pragma: no cover - argparse enforces the command list.
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
