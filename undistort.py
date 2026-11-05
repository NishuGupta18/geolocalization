"""undistort.py – Undistort one thermal image using per-row camera intrinsics.

Defaults
--------
IMAGES_DIR   : directory that contains the raw PNG images.
METADATA_CSV : CSV file with one row per image; columns include cam_fx, cam_fy,
               cam_cx, cam_cy, cam_k1, cam_k2, cam_k3, cam_p1, cam_p2.

Both defaults can be overridden at the call site.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── Default paths (edit once here, never scatter magic strings elsewhere) ──────
IMAGES_DIR   = Path(__file__).parent / "images"
METADATA_CSV = IMAGES_DIR / "metadata.csv"


def undistort(filename, images_dir=None, metadata_csv=None):
    """Undistort one drone image.

    Parameters
    ----------
    filename     : str  – basename of the image file (e.g. "abc123.png").
    images_dir   : Path or str – directory containing the image.
                   Defaults to IMAGES_DIR.
    metadata_csv : Path or str – CSV with camera calibration columns.
                   Defaults to METADATA_CSV.

    Returns
    -------
    img_ud : np.ndarray  – undistorted image (same size and dtype as input).
    K      : (3,3) float64 – original camera matrix.
    newK   : (3,3) float64 – updated camera matrix after undistortion.
    D      : (5,)  float64 – distortion coefficients [k1, k2, p1, p2, k3]
                             (OpenCV order).
    """
    images_dir   = Path(images_dir)   if images_dir   else IMAGES_DIR
    metadata_csv = Path(metadata_csv) if metadata_csv else METADATA_CSV

    meta = pd.read_csv(metadata_csv).set_index("filename")
    r    = meta.loc[filename]

    img = cv2.imread(str(images_dir / filename), cv2.IMREAD_UNCHANGED)  # keeps 16-bit
    if img is None:
        raise FileNotFoundError(images_dir / filename)
    h, w = img.shape[:2]

    K = np.array([[r.cam_fx, 0,       r.cam_cx],
                  [0,        r.cam_fy, r.cam_cy],
                  [0,        0,        1       ]], dtype=np.float64)

    # OpenCV distortion order: [k1, k2, p1, p2, k3]
    D = np.array([r.cam_k1, r.cam_k2, r.cam_p1, r.cam_p2, r.cam_k3],
                 dtype=np.float64)

    # Zoom in just enough to remove black edges; keep the same scale in x and y
    # so there is no stretching artefact.
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
    s = max(newK[0, 0] / K[0, 0], newK[1, 1] / K[1, 1])
    newK[0, 0] = K[0, 0] * s
    newK[1, 1] = K[1, 1] * s

    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_32FC1)
    img_ud = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    return img_ud, K, newK, D
