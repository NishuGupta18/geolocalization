#!/usr/bin/env python3
"""coarse_dino3_jetson.py

Minimal DINOv3‑based coarse retrieval pipeline for UAV‑to‑satellite matching.

Features
--------
* Index satellite tiles (pre‑process, embed with a DINOv3 model from ``timm``).
* Save/load index to ``.npz`` (embeddings + tile paths).
* Compute query embedding for a single UAV orthorectified PNG.
* Cosine similarity (L2‑normed vectors) using pure NumPy – no FAISS, works on ARM.
* GPU‑aware: uses CUDA if ``torch.cuda.is_available()`` else falls back to CPU.
* Optional FP16 inference (``--half``) – useful on Jetson Orin to cut memory.
* Small batch support (``--batch``) to avoid OOM when many tiles.
* Configurable via CLI arguments.

Usage example
-------------
```bash
python3 -m pipeline.coarse_dino3_jetson \
    --tiles   /home/nishu/Desktop/geolocalization/satellite_tile/tiles-20261103T143243Z-1-001/tiles \
    --query   /home/nishu/Desktop/geolocalization/pipeline/outputs/9053fd836a714cdc880ff0267d0607ed/9053fd836a714cdc880ff0267d0607ed_footprint.png \
    --index   /home/nishu/Desktop/geolocalization/pipeline/outputs/dinov3_tile_index.npz \
    --topk    20 \
    --out     /home/nishu/Desktop/geolocalization/pipeline/outputs/retrieval_top20.json \
    --half    # optional, enables fp16 on GPU
```
If ``--index`` already exists the script will load it; otherwise it will create it.

The script is deliberately lightweight so it runs on a Jetson Orin without extra heavy
dependencies (no FAISS, no scikit‑learn, only ``torch``, ``timm``, ``numpy`` and ``Pillow``).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import cv2
from PIL import Image
import torch
import timm

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_device(use_half: bool) -> torch.device:
    """Return the best device (CUDA if available, otherwise CPU).
    If ``use_half`` is requested on a CPU‐only device we silently ignore it.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_model(model_name: str, device: torch.device, use_half: bool):
    """Load a DINOv3 model from ``timm`` and move it to ``device``.
    ``use_half`` forces FP16 when the device supports it (CUDA).
    """
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    if use_half and device.type == "cuda":
        model.half()
    return model

def preprocess_image(img_path: Path, img_size: int = 224) -> torch.Tensor:
    """Load an image, resize to ``img_size`` while keeping aspect ratio, and return a normalized tensor.
    The DINOv3 pre‑training uses ImageNet‑style mean/std.
    """
    raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    if raw.dtype == np.uint16:
        valid = raw > 0
        values = raw[valid] if np.any(valid) else raw.reshape(-1)
        low, high = np.percentile(values, (1, 99))
        if high <= low:
            high = low + 1
        raw = np.clip((raw.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    elif raw.shape[2] == 4:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
    else:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(raw)
    # Simple resize – models are tolerant to slight distortion.
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    # Normalize (ImageNet)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    tensor = torch.from_numpy(arr)
    return tensor

def embed_batch(model, batch: torch.Tensor, device: torch.device, use_half: bool):
    """Run a forward pass for a batch and return L2‑normalised embeddings as ``np.ndarray``.
    ``batch`` is a ``torch.Tensor`` of shape ``(B, C, H, W)``.
    """
    batch = batch.to(device)
    if use_half and device.type == "cuda":
        batch = batch.half()
    with torch.no_grad():
        feats = model(batch)
    # Global average pool is already applied for most timm DINO models (num_classes=0)
    feats = feats.float()  # ensure FP32 for the L2 norm step
    feats = torch.nn.functional.normalize(feats, dim=1)
    return feats.cpu().numpy()

def build_index(
    tile_dir: Path,
    model_name: str,
    index_path: Path,
    batch_size: int = 8,
    use_half: bool = False,
) -> None:
    """Compute embeddings for all ``.png`` tiles under ``tile_dir`` and save ``.npz``.
    The index contains two arrays:
    * ``embeddings`` – shape (N, D)
    * ``paths`` – list of absolute strings, length N
    """
    device = get_device(use_half)
    model = load_model(model_name, device, use_half)
    img_size = getattr(model, "default_cfg", {}).get("input_size", (3, 224, 224))[1]

    tile_paths: List[Path] = sorted([p for p in tile_dir.rglob("*.png")])
    if not tile_paths:
        raise FileNotFoundError(f"No PNG tiles found in {tile_dir}")

    embeddings = []
    # Process in batches to keep memory low on Jetson.
    for i in range(0, len(tile_paths), batch_size):
        batch_paths = tile_paths[i : i + batch_size]
        tensors = [preprocess_image(p, img_size) for p in batch_paths]
        batch_tensor = torch.stack(tensors)  # (B, C, H, W)
        batch_emb = embed_batch(model, batch_tensor, device, use_half)
        embeddings.append(batch_emb)
        print(f"Indexed {i + len(batch_paths)}/{len(tile_paths)} tiles", flush=True)

    embeddings_arr = np.vstack(embeddings)
    # Save as .npz (compressed) – lightweight for Jetson.
    np.savez_compressed(index_path, embeddings=embeddings_arr, paths=[str(p) for p in tile_paths])
    print(f"Index saved to {index_path}")

def load_index(index_path: Path):
    data = np.load(index_path, allow_pickle=True)
    return data["embeddings"], data["paths"].tolist()

def retrieve_topk(
    query_path: Path,
    index_path: Path,
    model_name: str,
    topk: int = 30,
    batch_size: int = 8,
    use_half: bool = False,
) -> List[dict]:
    """Return the top‑k most similar satellite tiles for ``query_path``.
    The return format matches the requirement: list of dicts with ``rank``, ``score`` and ``path``.
    """
    device = get_device(use_half)
    model = load_model(model_name, device, use_half)
    img_size = getattr(model, "default_cfg", {}).get("input_size", (3, 224, 224))[1]

    # Load or build index lazily – caller should ensure the index exists.
    embeddings, paths = load_index(index_path)

    # Embed the query image.
    query_tensor = preprocess_image(query_path, img_size).unsqueeze(0)  # (1, C, H, W)
    query_emb = embed_batch(model, query_tensor, device, use_half)[0]  # (D,)

    # Cosine similarity is just dot product because vectors are L2‑normed.
    sims = embeddings @ query_emb  # (N,)
    top_indices = np.argpartition(-sims, range(topk))[:topk]
    # Sort the top‑k for nice output.
    top_sorted = top_indices[np.argsort(-sims[top_indices])]

    results = []
    for rank, idx in enumerate(top_sorted, start=1):
        results.append({
            "rank": rank,
            "score": float(sims[idx]),
            "path": paths[idx],
        })
    return results

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Coarse DINOv3 retrieval (Jetson‑friendly)")
    parser.add_argument("--tiles", type=str, required=False,
                        help="Directory containing satellite PNG tiles (used only when building index)")
    parser.add_argument("--query", type=str, required=True,
                        help="Path to the UAV orthorectified PNG to query")
    parser.add_argument("--index", type=str, required=True,
                        help="Path to .npz file that stores tile embeddings and paths")
    parser.add_argument("--topk", type=int, default=20,
                        help="Number of top candidates to return")
    parser.add_argument("--out", type=str, default="",
                        help="JSON file to write results; if omitted, prints to stdout")
    parser.add_argument("--model", type=str, default="vit_base_patch16_dinov3.lvd1689m",
                        help="timm model name for DINOv3 (default lightweight ViT‑B)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size for embedding (default 8)")
    parser.add_argument("--half", action="store_true",
                        help="Use FP16 inference on CUDA devices (saves memory)")
    parser.add_argument("--build", action="store_true",
                        help="Force rebuilding of the index even if it exists")
    return parser.parse_args()


def main():
    args = parse_args()
    tile_dir = Path(args.tiles) if args.tiles else None
    query_path = Path(args.query)
    index_path = Path(args.index)

    if not index_path.is_file() or args.build:
        if not tile_dir:
            print("Error: --tiles must be provided when building a new index", file=sys.stderr)
            sys.exit(1)
        print(f"Building index from tiles in {tile_dir} ...")
        build_index(
            tile_dir=tile_dir,
            model_name=args.model,
            index_path=index_path,
            batch_size=args.batch,
            use_half=args.half,
        )
    else:
        print(f"Loading existing index from {index_path}")

    if not query_path.is_file():
        print(f"Error: query image {query_path} not found", file=sys.stderr)
        sys.exit(1)

    results = retrieve_topk(
        query_path=query_path,
        index_path=index_path,
        model_name=args.model,
        topk=args.topk,
        batch_size=args.batch,
        use_half=args.half,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Results written to {out_path}")
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

