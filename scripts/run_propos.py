from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.config import load_experiment_config, validate_propos_config
from src.runners import run_propos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ProPos from a YAML config.")
    parser.add_argument("config_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config_path)
    validate_propos_config(config)
    result = run_propos(config)
    print(f"Wrote ProPos outputs to {result.output_dir}", flush=True)


if __name__ == "__main__":
    main()
