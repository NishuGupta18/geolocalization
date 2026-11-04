"""Evaluate DINOv3 satellite-tile retrieval against UAV GPS metadata.

Example:
    HF_HUB_OFFLINE=1 python pipeline/validate_dinov3.py \
      --model vit_base_patch16_dinov3.lvd1689m

The evaluation considers a query localized when a retrieved tile center is
within the requested radius of the UAV GPS coordinate. It also reports the
distance from the best-ranked tile center, since tile-center error includes
the tile's spatial quantization error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dinov3_retrieval import (  # noqa: E402
    DEFAULT_MODEL,
    DINOv3Embedder,
    TileIndex,
    collect_tile_paths,
)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres."""
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def load_tile_coordinates(db_path: Path, tile_dir: Path):
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT filename, center_lat, center_lon, valid_pct FROM tiles"
        ).fetchall()
    finally:
        con.close()
    by_path = {}
    for filename, lat, lon, valid_pct in rows:
        path = tile_dir / filename
        if path.exists() and lat is not None and lon is not None:
            by_path[path.resolve()] = {
                "lat": float(lat), "lon": float(lon),
                "valid_pct": float(valid_pct or 0),
            }
    return by_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "images")
    parser.add_argument("--metadata", type=Path, default=ROOT / "images/metadata.csv")
    parser.add_argument("--tiles", type=Path, default=ROOT / "satellite_tile/tiles-20261103T143243Z-1-001/tiles")
    parser.add_argument("--tile-db", type=Path, default=ROOT / "satellite_tile/satellite_tiles.db")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--radii-m", type=float, nargs="+", default=[50, 100, 250, 500])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs/dinov3_validation")
    parser.add_argument("--index-cache", type=Path, default=None)
    parser.add_argument("--include-invalid-tiles", action="store_true",
                        help="Include tiles whose DB valid_pct is zero (default excludes them).")
    args = parser.parse_args()

    tile_coords = load_tile_coordinates(args.tile_db, args.tiles)
    paths = collect_tile_paths(args.tiles)
    paths = [p for p in paths if p.resolve() in tile_coords and
             (args.include_invalid_tiles or tile_coords[p.resolve()]["valid_pct"] > 0)]
    if not paths:
        raise SystemExit("No coordinate-indexed satellite tiles found.")

    embedder = DINOv3Embedder(model_name=args.model, device=args.device,
                              batch_size=args.batch)
    if args.index_cache and args.index_cache.exists():
        print(f"Loading tile embeddings: {args.index_cache}")
        index = TileIndex.load(args.index_cache)
        indexed = {Path(p).resolve() for p in index.paths}
        if indexed != {p.resolve() for p in paths}:
            raise SystemExit("Index cache tile set does not match current coordinate-filtered tiles; remove cache and retry.")
    else:
        print(f"Embedding {len(paths)} valid satellite tiles")
        embeddings = embedder.embed_paths(paths)
        index = TileIndex(embeddings, [str(p) for p in paths])
        if args.index_cache:
            args.index_cache.parent.mkdir(parents=True, exist_ok=True)
            index.save(args.index_cache)

    with args.metadata.open(newline="") as f:
        rows = list(csv.DictReader(f))
    records = []
    missing = 0
    valid_rows = []
    nearest_errors = []
    valid_tile_points = [(v["lat"], v["lon"]) for v in tile_coords.values()
                         if args.include_invalid_tiles or v["valid_pct"] > 0]
    for row in rows:
        query_path = args.images / row["filename"]
        if not query_path.exists():
            missing += 1
            continue
        valid_rows.append(row)

    # Batch query inference to avoid 500 separate model calls.
    query_paths = [args.images / row["filename"] for row in valid_rows]
    query_embeddings = embedder.embed_paths(query_paths)
    for row, query_emb in zip(valid_rows, query_embeddings):
        truth_lat, truth_lon = float(row["lat_gtruth"]), float(row["lon_gtruth"])
        nearest_errors.append(min(
            haversine_m(truth_lat, truth_lon, tile_lat, tile_lon)
            for tile_lat, tile_lon in valid_tile_points
        ))
        scores, inds = index.search(query_emb, args.topk)
        ranked = []
        for rank, (score, idx) in enumerate(zip(scores, inds), start=1):
            path = Path(index.paths[int(idx)]).resolve()
            coords = tile_coords[path]
            dist = haversine_m(truth_lat, truth_lon, coords["lat"], coords["lon"])
            ranked.append({
                "rank": rank, "tile": path.name, "score": float(score),
                "tile_lat": coords["lat"], "tile_lon": coords["lon"],
                "distance_m": dist,
            })
        records.append({
            "query": row["filename"], "truth_lat": truth_lat, "truth_lon": truth_lon,
            "top1": ranked[0], "topk": ranked,
        })
        if len(records) % 25 == 0:
            print(f"Evaluated {len(records)}/{len(rows)} queries")

    radii = sorted(set(args.radii_m))
    errors = np.array([r["top1"]["distance_m"] for r in records], dtype=float)
    metrics = {
        "model": args.model,
        "queries": len(records),
        "tiles": len(paths),
        "missing_query_images": missing,
        "valid_tile_filter": not args.include_invalid_tiles,
        "top1_tile_center_error_m": {
            "mean": float(errors.mean()) if len(errors) else None,
            "median": float(np.median(errors)) if len(errors) else None,
            "p90": float(np.percentile(errors, 90)) if len(errors) else None,
        },
        "nearest_tile_center_baseline_error_m": {
            "mean": float(np.mean(nearest_errors)) if nearest_errors else None,
            "median": float(np.median(nearest_errors)) if nearest_errors else None,
            "p90": float(np.percentile(nearest_errors, 90)) if nearest_errors else None,
        },
        "recall_within_m": {},
        "recall_at_k_within_m": {},
    }
    for radius in radii:
        key = str(radius)
        metrics["recall_within_m"][key] = float(np.mean(errors <= radius)) if len(errors) else None
        metrics["recall_at_k_within_m"][key] = float(np.mean([
            any(item["distance_m"] <= radius for item in rec["topk"])
            for rec in records
        ])) if records else None

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out / "predictions.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics and predictions to {args.out}")


if __name__ == "__main__":
    main()
