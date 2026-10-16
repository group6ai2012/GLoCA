"""Hash local folder datasets without downloading or changing their images."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.reproducibility import atomic_json, dataset_records, digest
from src.data import get_dataset_spec, FolderImageDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["plantseg", "plantvillage", "plantwild"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = get_dataset_spec({"name": args.dataset, "root": args.root})
    records = dataset_records(spec)
    atomic_json(args.output, {"version": 1, "dataset": spec.name,
                             "class_names": FolderImageDataset(spec).class_names,
                             "fingerprint": digest(records), "records": records})
    print(f"Hashed {len(records)} images: {args.output}")


if __name__ == "__main__":
    main()
