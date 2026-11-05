"""Run every UAV frame through undistortion and flat-ground orthorectification.

The output folder contains only north-up top-down thermal PNGs. The processing
order is: read/undistort the source frame, intersect rays with the flat ground,
then resample the image onto the ground grid in ``geo_intersection``.

Run from the project directory::

    python3 -m pipeline.create_uav_topdown_folder --clean
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.geo_intersection import OUTPUT_DIR, process  # noqa: E402
from undistort import IMAGES_DIR, METADATA_CSV, undistort  # noqa: E402


def create_topdown_folder(images_dir: Path = IMAGES_DIR,
                          metadata_csv: Path = METADATA_CSV,
                          output_dir: Path | None = None,
                          clean_output: bool = False) -> tuple[int, int]:
    """Process all metadata images and return ``(successes, failures)``."""
    images_dir, metadata_csv = Path(images_dir), Path(metadata_csv)
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "uav_topdown"
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_csv)
    if "filename" not in metadata.columns:
        raise ValueError(f"Metadata CSV has no 'filename' column: {metadata_csv}")

    succeeded = failed = 0
    for filename in metadata["filename"].dropna().astype(str):
        source = images_dir / filename
        if not source.is_file():
            failed += 1
            print(f"[ERROR] Source image missing: {source}", file=sys.stderr)
            continue
        try:
            # Explicit sequence: undistort.py -> geo_intersection.process().
            undistorted = undistort(filename, images_dir, metadata_csv)
            result = process(filename, images_dir, metadata_csv, output_dir,
                             save_diagnostics=False, undistorted=undistorted)
            path = output_dir / f"{Path(filename).stem}_topdown.png"
            if not path.is_file():
                raise FileNotFoundError(f"geo_intersection did not create {path}")
            succeeded += 1
            print(f"[OK] {filename} -> {path} (shape={result['topdown'].shape})")
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {filename}: {exc}", file=sys.stderr)

    print(f"Finished: {succeeded} top-down images saved to {output_dir}; {failed} failed.")
    return succeeded, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Undistort and orthorectify every UAV frame into one folder."
    )
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "uav_topdown")
    parser.add_argument("--clean", action="store_true",
                        help="Replace the output folder contents before processing.")
    args = parser.parse_args()
    _, failed = create_topdown_folder(
        args.images_dir, args.metadata_csv, args.out_dir, clean_output=args.clean
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
