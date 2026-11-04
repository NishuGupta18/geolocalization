"""pipeline/geo_intersection.py
=================================
Stage 3 – Ray/terrain intersection and ground-footprint visualisation.

Pipeline
--------
1. Load metadata row for the requested image.
2. Undistort the image (calls ``../undistort.py``).
3. Build the camera matrix ``K`` (updated after undistortion).
4. Build the world-from-camera rotation:
       R_world_cam = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ R_body_cam
   where the body-camera alignment R_body_cam is:
       image +X  → body -Y  (image right  → body left)
       image +Y  → body +X  (image down   → body front)
       optical Z → body -Z  (optical axis → body down, i.e. nadir camera)
   All angles come from the CSV in *degrees*, stored in [0, 360).  Values > 180
   are folded to the equivalent negative angle before converting to radians.
5. Cast a ray through every pixel; intersect with a flat ground plane at
       Z_ground = takeoff_elevation + 0  (we treat the flat plane as sea-level
                   corrected to takeoff DEM; see NOTE below)
   Altitude above ground = alt_agl  = alt (CSV) + alt_takeoff_pt_dem (DEM at
   takeoff) - terrain_elevation.  For the FLAT model terrain_elevation is
   assumed equal to alt_takeoff_pt_dem everywhere.
6. Convert each intersection point to (lat, lon) using a local tangent-plane
   approximation around (lat_sc, lon_sc).
7. Save diagnostic outputs to ``pipeline/outputs/<stem>/``.

NOTE on altitude convention
---------------------------
``alt`` in the CSV has been verified to be in the range 100 – 350 m with a
constant ``alt_takeoff_pt_dem = 4 m`` for all 500 rows.  We assume ``alt`` is
**height above the take-off point** (AGL-like), so the camera is at:
    Z_cam  = alt + alt_takeoff_pt_dem   [m above MSL / WGS84 ellipsoid approx]
For flat-ground: ground elevation = alt_takeoff_pt_dem everywhere.
Height above ground = alt  (the DEM terms cancel for the flat case).

ASSUMPTIONS / OPEN QUESTIONS (document, do not fabricate)
----------------------------------------------------------
A1. ``alt`` is height above take-off point (not above sea level, not AGL from
    arbitrary datum).  ← Assumed from range 100-350 m with constant DEM=4 m.
A2. Angles are NED-frame: yaw=0 → North, yaw increases clockwise (standard
    aviation / DJI convention).  We fold values in (180, 360] to (−180, 0].
A3. R_body_cam convention:  image +X → body −Y, image +Y → body +X.
    This maps a nadir-pointing camera (optical axis = body −Z) so the image is
    north-up when yaw=0, roll=0, pitch=0.
A4. ``lat_sc`` / ``lon_sc`` is the GPS position of the UAV (not a ground point).
A5. The flat-ground plane elevation equals ``alt_takeoff_pt_dem`` (= 4 m MSL).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Project root on sys.path so we can import undistort ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from undistort import undistort, IMAGES_DIR, METADATA_CSV

OUTPUT_DIR = Path(__file__).parent / "outputs"


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def Rx(deg: float) -> np.ndarray:
    """Rotation about X axis (right-hand rule, radians or degrees?)."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)

def Ry(deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)

def Rz(deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


def wrap180(deg: float) -> float:
    """Fold a 0–360 angle into (−180, 180]."""
    return ((float(deg) + 180.0) % 360.0) - 180.0


# ─────────────────────────────────────────────────────────────────────────────
# Camera-to-body alignment  (R_body_cam)
# ─────────────────────────────────────────────────────────────────────────────
# Columns of R_body_cam are the body-frame coordinates of the camera axes:
#   camera X  (image right)   → body  -Y  (body left)
#   camera Y  (image down)    → body  +X  (body forward)
#   camera Z  (optical, fwd)  → body  -Z  (body down, i.e. nadir)
#
# In matrix form  p_body = R_body_cam @ p_cam :
#   [ 0  1  0 ]
#   [-1  0  0 ]
#   [ 0  0 -1 ]
#
# Verification:
#   cam_X → (0,-1,0) body  ✓ (body -Y = left)
#   cam_Y → (1, 0,0) body  ✓ (body +X = forward)
#   cam_Z → (0, 0,-1) body ✗ Wait — we need optical→body-down.
#
# Body-down is body +Z (NED: Z points down).  So:
#   cam_Z → body +Z  (down)
#   corrected matrix:
#   [ 0  1  0 ]
#   [-1  0  0 ]
#   [ 0  0  1 ]
#
# Double-check with stated convention in the task:
#   "image-right → body-left"  → cam_X → body -Y  ✓
#   "image-down  → body-front" → cam_Y → body +X  ✓
#   "optical axis → body-down" → cam_Z → body +Z  ✓ (NED down = +Z)

R_BODY_CAM = np.array([[ 0,  1,  0],
                        [-1,  0,  0],
                        [ 0,  0,  1]], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Camera pose
# ─────────────────────────────────────────────────────────────────────────────

def build_R_world_cam(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Return R_world_cam (3×3) that rotates a camera-frame vector to NED world.

    NED world: X=North, Y=East, Z=Down.
    Euler convention: ZYX extrinsic (yaw → pitch → roll applied to body, then
    R_body_cam appended on the right).

    Parameters are stored in [0, 360) in the CSV; we fold to (−180, 180] first.
    """
    yaw   = wrap180(yaw_deg)
    pitch = wrap180(pitch_deg)
    roll  = wrap180(roll_deg)

    # R_ned_body: rotates body → NED
    # ZYX Euler for NED: Rz(yaw) applied first (heading), then Ry(pitch), Rx(roll)
    R_ned_body = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    return R_ned_body @ R_BODY_CAM


# ─────────────────────────────────────────────────────────────────────────────
# Local tangent plane  (NED ↔ lat/lon)
# ─────────────────────────────────────────────────────────────────────────────
_EARTH_R = 6_371_000.0   # metres

def ned_to_latlon(dn: float, de: float, lat0_deg: float, lon0_deg: float):
    """Convert a NED offset (North, East in metres) to (lat, lon) in degrees."""
    lat0 = np.radians(lat0_deg)
    dlat = np.degrees(dn / _EARTH_R)
    dlon = np.degrees(de / (_EARTH_R * np.cos(lat0)))
    return lat0_deg + dlat, lon0_deg + dlon


# ─────────────────────────────────────────────────────────────────────────────
# Flat-ground ray intersection
# ─────────────────────────────────────────────────────────────────────────────

def flat_ground_intersection(pixels_uv: np.ndarray,
                             K: np.ndarray,
                             R_world_cam: np.ndarray,
                             cam_height_m: float) -> np.ndarray:
    """Intersect camera rays with the flat ground plane (world Z = 0).

    NED convention: Z=Down, so the camera is at Z = -cam_height_m and the
    ground plane is at Z_ned = 0 (the local horizontal).

    Parameters
    ----------
    pixels_uv   : (N, 2) float array of (u, v) pixel coords [column, row].
    K           : (3, 3) camera matrix (after undistortion).
    R_world_cam : (3, 3) rotation, camera → NED world.
    cam_height_m: scalar, height of the camera above ground in metres.

    Returns
    -------
    ground_ned  : (N, 3) array of NED intersection points; the Z component
                  is 0 for flat ground; in-view rays have positive t.
    valid       : (N,) bool mask – True when the ray actually hits the ground
                  (i.e. the ray has a downward component in NED).
    """
    N = len(pixels_uv)

    # Unproject pixels to normalised camera rays  [direction in camera frame]
    fx, fy   = K[0, 0], K[1, 1]
    cx, cy   = K[0, 2], K[1, 2]
    u        = pixels_uv[:, 0]
    v        = pixels_uv[:, 1]
    ray_cam  = np.stack([(u - cx) / fx,
                         (v - cy) / fy,
                         np.ones(N)], axis=1)   # (N, 3)

    # Rotate to NED world frame
    ray_ned = (R_world_cam @ ray_cam.T).T        # (N, 3)

    # Camera position in NED:  (0, 0, -cam_height_m)  [up = NED-Z negative]
    cam_pos_ned = np.array([0.0, 0.0, -cam_height_m])

    # Intersect with ground plane Z_ned = 0
    #   cam_pos_ned[2] + t * ray_ned[:,2] = 0
    #   t = -cam_pos_ned[2] / ray_ned[:,2]  = cam_height_m / ray_ned[:,2]
    dz      = ray_ned[:, 2]                      # (N,)
    valid   = dz > 1e-6                          # ray hits ground (going down)
    t       = np.where(valid, cam_height_m / dz, np.nan)

    ground_ned = cam_pos_ned + t[:, None] * ray_ned  # (N, 3)
    return ground_ned, valid


# ─────────────────────────────────────────────────────────────────────────────
# Camera-centre ground projection (Step 7 output)
# ─────────────────────────────────────────────────────────────────────────────

def camera_nadir_latlon(lat0_deg: float, lon0_deg: float):
    """Return the ground point directly below the camera centre.

    This is trivially (lat0, lon0) for a pinhole model where GPS reports the
    UAV position – no ray casting needed.  Kept explicit for pipeline clarity.
    """
    return lat0_deg, lon0_deg


# ─────────────────────────────────────────────────────────────────────────────
# Corner-footprint helper
# ─────────────────────────────────────────────────────────────────────────────

def image_footprint(h: int, w: int,
                    K: np.ndarray,
                    R_world_cam: np.ndarray,
                    cam_height_m: float,
                    lat0_deg: float,
                    lon0_deg: float):
    """Return (lats, lons) for the four image corners + centre pixel.

    Returns a dict with keys 'TL','TR','BR','BL','C', each a (lat, lon) tuple.
    """
    corners_uv = np.array([
        [0,     0    ],   # TL
        [w - 1, 0    ],   # TR
        [w - 1, h - 1],   # BR
        [0,     h - 1],   # BL
        [(w-1)/2, (h-1)/2],  # C
    ], dtype=np.float64)

    ground_ned, valid = flat_ground_intersection(corners_uv, K, R_world_cam, cam_height_m)

    labels = ['TL', 'TR', 'BR', 'BL', 'C']
    result = {}
    for i, lbl in enumerate(labels):
        if valid[i]:
            lat, lon = ned_to_latlon(ground_ned[i, 0], ground_ned[i, 1],
                                     lat0_deg, lon0_deg)
            result[lbl] = (lat, lon)
        else:
            result[lbl] = None
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Dense ground map (full-image rays)
# ─────────────────────────────────────────────────────────────────────────────

def dense_ground_map(h: int, w: int,
                     K: np.ndarray,
                     R_world_cam: np.ndarray,
                     cam_height_m: float,
                     lat0_deg: float,
                     lon0_deg: float):
    """Return (lat_map, lon_map) arrays of shape (h, w) for all pixels."""
    vv, uu = np.mgrid[0:h, 0:w]                 # row, col grids
    pixels_uv = np.stack([uu.ravel().astype(float),
                          vv.ravel().astype(float)], axis=1)  # (h*w, 2)

    ground_ned, valid = flat_ground_intersection(pixels_uv, K, R_world_cam, cam_height_m)

    lats = np.full(h * w, np.nan)
    lons = np.full(h * w, np.nan)
    if valid.any():
        gn = ground_ned[valid]
        lat_v, lon_v = ned_to_latlon(gn[:, 0], gn[:, 1], lat0_deg, lon0_deg)
        lats[valid] = lat_v
        lons[valid] = lon_v

    return lats.reshape(h, w), lons.reshape(h, w)


def orthorectify_image(img_ud: np.ndarray, K: np.ndarray,
                       R_world_cam: np.ndarray, cam_height_m: float) -> np.ndarray:
    """Resample an undistorted frame onto a north-up, flat-ground pixel grid.

    Output pixels retain the source dtype (including uint16 thermal values).
    The grid uses the approximate nadir ground-sampling distance.
    """
    h, w = img_ud.shape[:2]
    corner_uv = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                         dtype=np.float64)
    corner_ned, valid = flat_ground_intersection(corner_uv, K, R_world_cam, cam_height_m)
    if not np.all(valid):
        raise ValueError("At least one image corner ray does not intersect the flat ground")

    north_min, north_max = float(corner_ned[:, 0].min()), float(corner_ned[:, 0].max())
    east_min, east_max = float(corner_ned[:, 1].min()), float(corner_ned[:, 1].max())
    gsd = cam_height_m / float(np.sqrt(K[0, 0] * K[1, 1]))
    out_w = max(1, int(np.ceil((east_max - east_min) / gsd)) + 1)
    out_h = max(1, int(np.ceil((north_max - north_min) / gsd)) + 1)

    east = east_min + np.arange(out_w, dtype=np.float64) * gsd
    north = north_max - np.arange(out_h, dtype=np.float64) * gsd
    east_grid, north_grid = np.meshgrid(east, north)
    # Camera-relative NED coordinates; camera is at (0, 0, -height).
    ground_to_camera = np.stack((north_grid, east_grid,
                                 np.full_like(north_grid, cam_height_m)), axis=-1) @ R_world_cam
    z = ground_to_camera[..., 2]
    safe_z = np.where(z > 1e-8, z, np.nan)
    map_x = (K[0, 0] * ground_to_camera[..., 0] / safe_z + K[0, 2]).astype(np.float32)
    map_y = (K[1, 1] * ground_to_camera[..., 1] / safe_z + K[1, 2]).astype(np.float32)
    invalid_map = (
        ~np.isfinite(map_x) | ~np.isfinite(map_y) | (z <= 1e-8)
        | (map_x < 0) | (map_x > w - 1)
        | (map_y < 0) | (map_y > h - 1)
    )
    map_x[invalid_map] = -1
    map_y[invalid_map] = -1
    topdown = cv2.remap(img_ud, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Preserve 16-bit precision while spreading the valid thermal footprint
    # over the image display range. Zero remains the no-data value. This is a
    # monotonic tone mapping, not a conversion to 8-bit.
    if topdown.dtype == np.uint16:
        valid_pixels = topdown > 0
        if np.any(valid_pixels):
            values = topdown[valid_pixels].astype(np.float64)
            low, high = float(values.min()), float(values.max())
            if high > low:
                stretched = np.rint(1 + (values - low) * (65534.0 / (high - low)))
                topdown[valid_pixels] = np.clip(stretched, 1, 65535).astype(np.uint16)
    return topdown


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise(filename: str,
              img_ud: np.ndarray,
              lat_map: np.ndarray,
              lon_map: np.ndarray,
              footprint: dict,
              meta_row,
              out_dir: Path):
    """Save a 3-panel diagnostic figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem

    # Normalise 16-bit image for display
    img_8 = cv2.normalize(img_ud, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img_8.ndim == 2:
        img_8 = cv2.cvtColor(img_8, cv2.COLOR_GRAY2RGB)

    h, w = img_ud.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Flat-ground footprint  |  {filename}\n"
                 f"alt={float(meta_row.alt):.1f} m  "
                 f"yaw={float(meta_row.yaw):.1f}°  "
                 f"pitch={wrap180(float(meta_row.pitch)):.1f}°  "
                 f"roll={wrap180(float(meta_row.roll)):.1f}°",
                 fontsize=11)

    # ── Panel 1: undistorted image ──────────────────────────────────────────
    axes[0].imshow(img_8)
    axes[0].set_title("Undistorted thermal image")
    axes[0].axis("off")

    # ── Panel 2: pixel-to-ground lat offsets (NED North) ───────────────────
    lat0 = float(meta_row.lat_sc)
    dN = (lat_map - lat0) * _EARTH_R * np.pi / 180   # metres North
    dE = (lon_map - float(meta_row.lon_sc)) * _EARTH_R * np.pi / 180 * np.cos(np.radians(lat0))

    im = axes[1].imshow(dN, cmap="RdYlGn", origin="upper")
    axes[1].set_title("Ground offset – North (m)\n[each pixel → ground location]")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # ── Panel 3: footprint polygon on a lat/lon plot ────────────────────────
    ax3 = axes[2]
    corners_order = ['TL', 'TR', 'BR', 'BL', 'TL']  # close the ring
    lats_fp = [footprint[k][1] for k in corners_order if footprint[k] is not None]
    lons_fp = [footprint[k][0] for k in corners_order if footprint[k] is not None]
    # Note: footprint stores (lat, lon), so lat=footprint[k][0], lon=[1]
    lats_fp = [footprint[k][0] for k in corners_order if footprint[k] is not None]
    lons_fp = [footprint[k][1] for k in corners_order if footprint[k] is not None]

    ax3.plot(lons_fp, lats_fp, 'b-o', markersize=4, linewidth=1.5, label="Footprint")

    # Camera nadir
    ax3.plot(float(meta_row.lon_sc), float(meta_row.lat_sc),
             'r*', markersize=14, label="UAV nadir (GPS)")

    # Centre-pixel ground intersection
    if footprint['C'] is not None:
        ax3.plot(footprint['C'][1], footprint['C'][0],
                 'g^', markersize=10, label="Image-centre ray")

    for lbl, pt in footprint.items():
        if pt is not None and lbl != 'C':
            ax3.annotate(lbl, (pt[1], pt[0]), fontsize=8, color='blue',
                         xytext=(3, 3), textcoords='offset points')

    ax3.set_xlabel("Longitude (°)")
    ax3.set_ylabel("Latitude (°)")
    ax3.set_title("Ground footprint (flat terrain)")
    ax3.legend(fontsize=8)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.4)

    plt.tight_layout()
    out_path = out_dir / f"{stem}_footprint.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Ray-direction sanity check
# ─────────────────────────────────────────────────────────────────────────────

def check_ray_directions(K: np.ndarray, R_world_cam: np.ndarray,
                         h: int, w: int, out_dir: Path, stem: str):
    """Plot a quiver of ray directions (NED North/East components) for a 16×16 grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gy, gx = np.mgrid[0:h:16, 0:w:16]
    pixels_uv = np.stack([gx.ravel().astype(float),
                          gy.ravel().astype(float)], axis=1)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = pixels_uv[:, 0], pixels_uv[:, 1]
    ray_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones(len(u))], axis=1)
    ray_ned = (R_world_cam @ ray_cam.T).T   # (N, 3)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(gx.ravel(), gy.ravel(),
              ray_ned[:, 1],   # East component → image X-like
              -ray_ned[:, 0],  # −North → image Y-like (screen coords)
              ray_ned[:, 2],   # colour by Z (down = positive → nadir)
              cmap='RdYlGn_r', alpha=0.8)
    ax.invert_yaxis()
    ax.set_title("Ray directions (NED) — colour = Z-down component\n"
                 "(should all point mostly downward for a nadir camera)")
    ax.set_xlabel("pixel col"); ax.set_ylabel("pixel row")
    out_path = out_dir / f"{stem}_ray_directions.png"
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def process(filename: str,
            images_dir=None,
            metadata_csv=None,
            out_base_dir=None,
            save_diagnostics=True,
            undistorted=None):
    """Run the full flat-ground intersection pipeline for one image.

    Returns a dict with:
        img_ud, K, newK, R_world_cam, footprint, lat_map, lon_map,
        cam_height_m, nadir_lat, nadir_lon
    """
    images_dir   = Path(images_dir)   if images_dir   else IMAGES_DIR
    metadata_csv = Path(metadata_csv) if metadata_csv else METADATA_CSV
    out_base_dir = Path(out_base_dir) if out_base_dir else OUTPUT_DIR

    # ── 1. Metadata ──────────────────────────────────────────────────────────
    meta = pd.read_csv(metadata_csv).set_index("filename")
    r    = meta.loc[filename]

    # ── 2. Use the undistorted frame (or undistort it here for standalone use)
    if undistorted is None:
        img_ud, K_orig, newK, D = undistort(filename, images_dir, metadata_csv)
    else:
        img_ud, K_orig, newK, D = undistorted
    h, w = img_ud.shape[:2]
    print(f"[undistort] {filename}  shape={img_ud.shape}  dtype={img_ud.dtype}")
    print(f"  K_orig:\n{K_orig}")
    print(f"  newK:\n{newK}")
    print(f"  D (OpenCV k1,k2,p1,p2,k3): {D}")

    # ── 3. Camera height above ground ────────────────────────────────────────
    # alt is height above take-off point; alt_takeoff_pt_dem is DEM at takeoff.
    # Flat ground elevation = alt_takeoff_pt_dem (everywhere).
    # cam_height = alt + alt_takeoff_pt_dem - ground_elev
    #            = alt + alt_takeoff_pt_dem - alt_takeoff_pt_dem  = alt
    cam_height_m = float(r.alt)   # metres above flat ground
    print(f"  cam_height_m = {cam_height_m:.2f} m  (= alt, flat-ground assumption)")

    # ── 4. Camera pose ───────────────────────────────────────────────────────
    yaw_raw   = float(r.yaw)
    pitch_raw = float(r.pitch)
    roll_raw  = float(r.roll)
    print(f"  Raw angles (0-360): yaw={yaw_raw:.3f}  pitch={pitch_raw:.3f}  roll={roll_raw:.3f}")
    print(f"  Wrapped (-180..180): yaw={wrap180(yaw_raw):.3f}  "
          f"pitch={wrap180(pitch_raw):.3f}  roll={wrap180(roll_raw):.3f}")

    R_world_cam = build_R_world_cam(yaw_raw, pitch_raw, roll_raw)
    print(f"  R_world_cam (camera→NED):\n{np.round(R_world_cam, 4)}")

    # Sanity: optical axis (cam Z) should map to a downward NED direction
    optical_ned = R_world_cam @ np.array([0, 0, 1])
    print(f"  Optical axis in NED: {np.round(optical_ned, 4)}  "
          f"(Z-down should be >> 0 for near-nadir)")

    # ── 5. Dense ground map ──────────────────────────────────────────────────
    lat_map, lon_map = dense_ground_map(
        h, w, newK, R_world_cam, cam_height_m,
        float(r.lat_sc), float(r.lon_sc))

    # ── 6. Corner footprint ──────────────────────────────────────────────────
    footprint = image_footprint(
        h, w, newK, R_world_cam, cam_height_m,
        float(r.lat_sc), float(r.lon_sc))

    print("\n  Corner footprint (lat, lon):")
    for lbl, pt in footprint.items():
        if pt:
            print(f"    {lbl}: lat={pt[0]:.6f}  lon={pt[1]:.6f}")

    # Footprint width/height sanity check
    if footprint['TL'] and footprint['TR']:
        dE = (footprint['TR'][1] - footprint['TL'][1]) * _EARTH_R * np.pi / 180
        dE *= np.cos(np.radians(float(r.lat_sc)))
        print(f"  Approx footprint width (E): {abs(dE):.1f} m")
    if footprint['TL'] and footprint['BL']:
        dN = (footprint['BL'][0] - footprint['TL'][0]) * _EARTH_R * np.pi / 180
        print(f"  Approx footprint height (N): {abs(dN):.1f} m")

    # ── 7. Nadir (ground point below camera) ────────────────────────────────
    nadir_lat, nadir_lon = camera_nadir_latlon(float(r.lat_sc), float(r.lon_sc))
    print(f"\n  Camera nadir (GPS): lat={nadir_lat:.6f}  lon={nadir_lon:.6f}")

    # ── 8. Orthorectify and save the thermal image ───────────────────────────
    stem    = Path(filename).stem
    topdown = orthorectify_image(img_ud, newK, R_world_cam, cam_height_m)
    topdown_path = out_base_dir / f"{stem}_topdown.png"
    out_base_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(topdown_path), topdown):
        raise OSError(f"Could not write orthorectified image: {topdown_path}")

    # Diagnostic plots are optional so the batch folder can contain only images.
    if save_diagnostics:
        out_dir = out_base_dir / stem
        check_ray_directions(newK, R_world_cam, h, w, out_dir, stem)
        visualise(filename, img_ud, lat_map, lon_map, footprint, r, out_dir)

    return dict(img_ud=img_ud, topdown=topdown, topdown_path=topdown_path,
                K=K_orig, newK=newK, D=D,
                R_world_cam=R_world_cam,
                footprint=footprint,
                lat_map=lat_map, lon_map=lon_map,
                cam_height_m=cam_height_m,
                nadir_lat=nadir_lat, nadir_lon=nadir_lon)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 3 – Flat-ground ray intersection + footprint visualisation")
    parser.add_argument("filename", nargs="?",
                        default="9053fd836a714cdc880ff0267d0607ed.png",
                        help="Image basename (default: first row in metadata)")
    parser.add_argument("--images-dir",   default=None)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--out-dir",      default=None)
    parser.add_argument("--all", action="store_true",
                        help="Process all 500 images (batch mode)")
    parser.add_argument("--no-diagnostics", action="store_true",
                        help="Save only the orthorectified image")
    args = parser.parse_args()

    if args.all:
        meta = pd.read_csv(args.metadata_csv or METADATA_CSV)
        for fn in meta["filename"]:
            try:
                process(fn, args.images_dir, args.metadata_csv, args.out_dir,
                        save_diagnostics=not args.no_diagnostics)
            except Exception as e:
                print(f"  ERROR {fn}: {e}")
    else:
        process(args.filename, args.images_dir, args.metadata_csv, args.out_dir,
                save_diagnostics=not args.no_diagnostics)
