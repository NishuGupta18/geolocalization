#!/usr/bin/env python3
"""run_undistort.py
Simple wrapper to run the `undistort` function on a single raw thermal PNG.

Usage
-----
```bash
python3 -m pipeline.run_undistort \
    --image path/to/raw_image.png \
    --out-dir path/to/output_dir   # optional, defaults to ./undistort_outputs
```
The script:
1. Loads the image metadata from the CSV that ships with the repo (`undistort.IMAGES_DIR/metadata.csv`).
2. Calls ``undistort`` (defined in ``undistort.py``) which returns the undistorted image and the original/updated camera matrices.
3. Saves the undistorted image as ``<stem>_undistorted.png`` in the output directory.
4. Also writes a tiny JSON side‑car containing the 4 matrices for later debugging.

The wrapper works on any platform – it does **not** require the rest of the pipeline.
"""

import argparse
import json
from pathlib import Path
import cv2
from undistort import undistort, IMAGES_DIR, METADATA_CSV

def main():
    parser = argparse.ArgumentParser(description="Run the undistort function on a single thermal image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the raw PNG image (must be present in the image folder or give a full path).")
    parser.add_argument("--out-dir", type=str, default="undistort_outputs", help="Directory where the undistorted PNG and metadata JSON will be saved.")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The original undistort function expects just the filename (basename) and will look for the image in IMAGES_DIR.
    # If the supplied path is already inside IMAGES_DIR we can pass the basename directly; otherwise we copy the file there temporarily.
    if img_path.parent.resolve() == IMAGES_DIR.resolve():
        filename = img_path.name
    else:
        # Copy to the expected folder so the metadata CSV line can be found (metadata rows are keyed by filename).
        tmp_path = IMAGES_DIR / img_path.name
        if not tmp_path.exists():
            # Use OpenCV to read and write to preserve 16‑bit depth.
            tmp_img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(tmp_path), tmp_img)
        filename = img_path.name

    # Run the core undistortion.
    img_ud, K, newK, D = undistort(filename)

    # Save the undistorted image (preserve original dtype – may be 16‑bit).
    out_img_path = out_dir / f"{Path(filename).stem}_undistorted.png"
    cv2.imwrite(str(out_img_path), img_ud)

    # Write matrices for reproducibility.
    meta = {
        "original_K": K.tolist(),
        "new_K": newK.tolist(),
        "distortion_D": D.tolist()
    }
    out_json_path = out_dir / f"{Path(filename).stem}_undistort_meta.json"
    out_json_path.write_text(json.dumps(meta, indent=2))

    print(f"✅ Undistorted image written to: {out_img_path}")
    print(f"📄 Metadata JSON written to: {out_json_path}")

if __name__ == "__main__":
    main()

