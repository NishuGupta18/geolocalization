"""pipeline/dinov3_retrieval.py
=================================
Stage 4 – DINOv3-based satellite tile retrieval.

Given a UAV top-down PNG and a directory of satellite tile PNGs, this module:

  1. Loads a pre-trained **DINOv3** model via ``timm``.
  2. Embeds the UAV query image and all satellite tiles.
  3. Ranks tiles by cosine similarity of the pooled [CLS] embedding.
  4. Returns the **top-K** matches (default K = 20).
  5. Saves a visual montage and a JSON results file.

Model options (pass via ``--model`` or ``MODEL_ID``)
-----------------------------------------------------
| timm model name                             | Notes                              |
|---------------------------------------------|------------------------------------|
| vit_base_patch16_dinov3.lvd1689m            | ViT-B, fast, general purpose       |
| vit_large_patch16_dinov3.lvd1689m           | ViT-L, better quality              |
| vit_large_patch16_dinov3.sat493m  ← DEFAULT | ViT-L pretrained on 493M sat imgs  |
| vit_large_patch16_dinov3_qkvb.sat493m       | ViT-L + qkv-bias, sat-pretrained   |
| convnext_base.dinov3_lvd1689m               | ConvNeXt-B, fast on CPU            |

The satellite-pretrained model (``sat493m``) is the best choice for matching
UAV ortho images against satellite tiles.

Embedding strategy
------------------
* CLS token    : global image descriptor (primary ranking signal)
* Patch mean   : mean of all patch tokens (appended when USE_PATCH_MEAN=True)
* Both vectors are L2-normalised before concatenation → cosine similarity
  reduces to inner product (FAISS IndexFlatIP).

Usage (CLI)
-----------
    python pipeline/dinov3_retrieval.py \\
        --query  pipeline/outputs/<stem>/<stem>_ortho.png \\
        --tiles  /path/to/satellite_tiles/ \\
        --topk   20 \\
        --out    pipeline/outputs/<stem>/retrieval/

    # Re-run with cached tile index (fast):
    python pipeline/dinov3_retrieval.py \\
        --query  query.png \\
        --tiles  /tiles/ \\
        --index-cache /tiles/dinov3_index.npz

    # Use satellite-pretrained ViT-L (recommended):
    python pipeline/dinov3_retrieval.py \\
        --query  query.png  --tiles /tiles/ \\
        --model  vit_large_patch16_dinov3.sat493m

Assumptions
-----------
A1. ``alt`` in the metadata is height above take-off point (verified ~100–350 m).
A2. Satellite tiles are PNG files in a flat directory (use --recursive for nested).
A3. Tiles carry no ground-truth metadata; ranking is by embedding similarity only
    (no label leakage).
A4. GPU is used when available; CPU fallback is automatic but much slower.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Torch & timm ──────────────────────────────────────────────────────────────
try:
    import torch
    import torchvision.transforms as T
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[WARN] PyTorch not found. Install with: pip install torch torchvision")

try:
    import timm
    TIMM_OK = True
except ImportError:
    TIMM_OK = False
    print("[WARN] timm not found. Install with: pip install timm")

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False
    print("[WARN] faiss not found – using numpy fallback. "
          "Install with: pip install faiss-cpu (or faiss-gpu)")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Default: satellite-pretrained ViT-L/16 (best for aerial/satellite matching)
DEFAULT_MODEL   = "vit_large_patch16_dinov3.sat493m"
# Fast alternative for CPU or quick experiments:
FAST_MODEL      = "vit_base_patch16_dinov3.lvd1689m"

# Append mean-patch token to CLS for richer spatial signal
USE_PATCH_MEAN  = True

# DINOv3 standard input resolution
MODEL_INPUT_SIZE = 224

# ImageNet normalisation (DINOv3 uses ImageNet stats)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image_rgb(path: str | Path) -> Image.Image:
    """Load any PNG (8-bit or 16-bit, grey or colour) as 8-bit PIL RGB.

    16-bit thermal images are linearly mapped to [0, 255].
    """
    img_cv = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img_cv is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img_cv.dtype == np.uint16:
        # Orthorectified thermal images use zero for pixels outside the ground
        # footprint. Exclude those pixels when mapping sensor intensity to RGB.
        valid = img_cv > 0 if img_cv.ndim == 2 else np.any(img_cv > 0, axis=2)
        values = img_cv[valid]
        if values.size:
            low, high = np.percentile(values, (1, 99))
            if high <= low:
                high = low + 1
            scaled = np.clip((img_cv.astype(np.float32) - low) * (255.0 / (high - low)),
                             0, 255)
            img_cv = scaled.astype(np.uint8)
        else:
            img_cv = np.zeros(img_cv.shape, dtype=np.uint8)

    if img_cv.ndim == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
    elif img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGB)
    else:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_cv)


def _build_transform(input_size: int = MODEL_INPUT_SIZE) -> "T.Compose":
    """Return a torchvision transform matching DINOv3 preprocessing."""
    return T.Compose([
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# DINOv3 Embedder (via timm)
# ─────────────────────────────────────────────────────────────────────────────

class DINOv3Embedder:
    """Extract DINOv3 embeddings using a timm model.

    Parameters
    ----------
    model_name     : timm model identifier (see module docstring for options).
    device         : "cuda", "cpu", or None (auto).
    batch_size     : Images per forward pass.
    use_patch_mean : Whether to concatenate mean-patch token to CLS embedding.
    input_size     : Image size fed to the model (pixels). Default: 224.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 16,
        use_patch_mean: bool = USE_PATCH_MEAN,
        input_size: int = MODEL_INPUT_SIZE,
    ):
        if not TORCH_OK:
            raise RuntimeError("PyTorch required. pip install torch torchvision")
        if not TIMM_OK:
            raise RuntimeError("timm required. pip install timm")

        self.model_name     = model_name
        self.batch_size     = batch_size
        self.use_patch_mean = use_patch_mean
        self.transform      = _build_transform(input_size)

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[DINOv3Embedder] device={self.device}  model={model_name}")
        t0 = time.time()

        # Load model with features_only=False so we get forward_features()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,   # remove classifier head
        )
        self.model.eval()
        self.model.to(self.device)

        print(f"[DINOv3Embedder] Model loaded in {time.time()-t0:.1f}s")

        # Determine embedding dim via a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size,
                                device=self.device)
            out = self.model.forward_features(dummy)
            # ViT: out shape = (1, seq_len, D) — CLS is index 0
            # ConvNeXt: out shape = (1, C, H, W) — need global pool
            self._is_vit = out.ndim == 3
            if self._is_vit:
                cls_dim = out.shape[2]
            else:
                cls_dim = out.shape[1]

        self.embed_dim = cls_dim * 2 if use_patch_mean and self._is_vit else cls_dim
        print(f"[DINOv3Embedder] embed_dim={self.embed_dim}  "
              f"is_vit={self._is_vit}  use_patch_mean={use_patch_mean}")

    @torch.no_grad()
    def _embed_tensor_batch(self, pixel_values: "torch.Tensor") -> np.ndarray:
        """Embed a pre-processed batch tensor → (N, D) float32 numpy."""
        pixel_values = pixel_values.to(self.device)
        features = self.model.forward_features(pixel_values)  # (N, seq, D) or (N,C,H,W)

        if self._is_vit:
            cls_token  = features[:, 0, :]           # (N, D)
            if self.use_patch_mean:
                patch_mean = features[:, 1:, :].mean(dim=1)   # (N, D)
                emb = torch.cat([cls_token, patch_mean], dim=1)
            else:
                emb = cls_token
        else:
            # ConvNeXt: global average pool the spatial map
            emb = features.mean(dim=[2, 3])          # (N, C)

        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.cpu().float().numpy()

    def embed_paths(self, paths: List[str | Path]) -> np.ndarray:
        """Embed a list of image paths → (N, embed_dim) float32 array."""
        all_embeddings = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start: start + self.batch_size]
            images = [load_image_rgb(p) for p in batch_paths]
            batch_tensor = torch.stack([self.transform(img) for img in images])
            embs = self._embed_tensor_batch(batch_tensor)
            all_embeddings.append(embs)
            done = min(start + self.batch_size, len(paths))
            print(f"  Embedded {done}/{len(paths)} tiles …", end="\r", flush=True)
        print()
        return np.concatenate(all_embeddings, axis=0)

    def embed_single(self, path: str | Path) -> np.ndarray:
        """Embed one image → (embed_dim,) float32 vector."""
        return self.embed_paths([path])[0]


# ─────────────────────────────────────────────────────────────────────────────
# Tile index (FAISS or numpy cosine search)
# ─────────────────────────────────────────────────────────────────────────────

class TileIndex:
    """Flat exact cosine-similarity index over tile embeddings.

    Uses FAISS IndexFlatIP when available; pure-numpy fallback otherwise.
    Embeddings are expected to be **L2-normalised** (cosine sim = inner product).
    """

    def __init__(self, embeddings: np.ndarray, paths: List[str]):
        assert len(embeddings) == len(paths), "embeddings / paths length mismatch"
        self.paths     = [str(p) for p in paths]
        self.D         = embeddings.shape[1]
        self._emb      = embeddings.astype(np.float32)
        self._use_faiss = False

        if FAISS_OK:
            self._index = faiss.IndexFlatIP(self.D)
            self._index.add(self._emb)
            self._use_faiss = True

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) for top-k matches (cosine sim, descending)."""
        q = query.reshape(1, -1).astype(np.float32)
        if self._use_faiss:
            scores, indices = self._index.search(q, min(k, len(self.paths)))
            return scores[0], indices[0]
        else:
            sims = (self._emb @ q.T).ravel()
            top_idx = np.argsort(sims)[::-1][:k]
            return sims[top_idx], top_idx

    def save(self, path: str | Path):
        """Persist index to .npz file for re-use."""
        np.savez_compressed(str(path),
                            embeddings=self._emb,
                            paths=np.array(self.paths, dtype=object))
        print(f"[TileIndex] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TileIndex":
        """Restore a saved index."""
        data = np.load(str(path), allow_pickle=True)
        return cls(data["embeddings"], list(data["paths"]))


# ─────────────────────────────────────────────────────────────────────────────
# Collect tile paths
# ─────────────────────────────────────────────────────────────────────────────

def collect_tile_paths(tiles_dir: str | Path,
                       recursive: bool = False) -> List[Path]:
    """Return sorted list of PNG paths under tiles_dir."""
    tiles_dir = Path(tiles_dir)
    pattern   = "**/*.png" if recursive else "*.png"
    paths     = sorted(tiles_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No PNG tiles found in: {tiles_dir}\n"
            "Use --recursive if tiles are in sub-directories."
        )
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main retrieval function
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_top_k(
    query_path: str | Path,
    tiles_dir: str | Path,
    k: int = 20,
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    batch_size: int = 16,
    use_patch_mean: bool = USE_PATCH_MEAN,
    index_cache: Optional[str | Path] = None,
    recursive: bool = False,
) -> List[dict]:
    """Embed query + tiles with DINOv3 and return top-K tile matches.

    Returns
    -------
    List of dicts (sorted by score, descending):
        { "rank": int, "score": float, "path": str, "stem": str }
    """
    query_path = Path(query_path)
    tiles_dir  = Path(tiles_dir)

    embedder = DINOv3Embedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        use_patch_mean=use_patch_mean,
    )

    # ── Build or load tile index ─────────────────────────────────────────────
    if index_cache and Path(index_cache).exists():
        print(f"[retrieve_top_k] Loading cached index: {index_cache}")
        tile_index = TileIndex.load(index_cache)
    else:
        tile_paths = collect_tile_paths(tiles_dir, recursive=recursive)
        print(f"[retrieve_top_k] Embedding {len(tile_paths)} tiles …")
        t0        = time.time()
        tile_embs = embedder.embed_paths([str(p) for p in tile_paths])
        print(f"[retrieve_top_k] Tile embedding: {time.time()-t0:.1f}s")
        tile_index = TileIndex(tile_embs, [str(p) for p in tile_paths])
        if index_cache:
            Path(index_cache).parent.mkdir(parents=True, exist_ok=True)
            tile_index.save(index_cache)

    # ── Embed query ──────────────────────────────────────────────────────────
    print(f"[retrieve_top_k] Embedding query: {query_path.name}")
    q_emb = embedder.embed_single(str(query_path))

    # ── Rank ─────────────────────────────────────────────────────────────────
    scores, indices = tile_index.search(q_emb, k=k)

    results = [
        {
            "rank" : int(rank),
            "score": float(score),
            "path" : tile_index.paths[int(idx)],
            "stem" : Path(tile_index.paths[int(idx)]).stem,
        }
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1)
    ]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_results(
    query_path: str | Path,
    results: List[dict],
    out_dir: str | Path,
    n_cols: int = 5,
    thumb_px: int = 160,
) -> Path:
    """Save retrieval montage PNG.

    Layout: large query panel on the left | grid of top-K tiles on the right.
    Each tile cell shows rank, cosine score, and filename stem.
    """
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k         = len(results)
    n_rows    = (k + n_cols - 1) // n_cols

    def thumb(path, size=thumb_px):
        try:
            img = load_image_rgb(path)
            img = img.resize((size, size), Image.LANCZOS)
            return np.array(img)
        except Exception:
            return np.zeros((size, size, 3), dtype=np.uint8)

    query_arr  = np.array(
        load_image_rgb(query_path).resize((thumb_px * 2, thumb_px * 2), Image.LANCZOS)
    )
    tile_arrs  = [thumb(r["path"]) for r in results]

    # Figure dimensions (inches)
    fig_w = max((n_cols + 2.5) * thumb_px / 72, 14)
    fig_h = max(n_rows * thumb_px / 72 + 0.9, 5)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"DINOv3 Top-{k} Tile Retrieval  |  Query: {Path(query_path).name}",
        fontsize=11, y=0.99,
    )

    # Query panel (left 18 % of figure)
    ax_q = fig.add_axes([0.01, 0.05, 0.17, 0.88])
    ax_q.imshow(query_arr)
    ax_q.set_title("Query\n(UAV top-down)", fontsize=9)
    ax_q.axis("off")

    # Tile grid (right 80 %)
    cell_w = 0.79 / n_cols
    cell_h = 0.88 / max(n_rows, 1)

    for i, (r, arr) in enumerate(zip(results, tile_arrs)):
        row = i // n_cols
        col = i % n_cols
        left   = 0.20 + col * cell_w
        bottom = 0.05 + (n_rows - 1 - row) * cell_h
        ax = fig.add_axes([left, bottom, cell_w * 0.92, cell_h * 0.85])
        ax.imshow(arr)
        ax.set_title(
            f"#{r['rank']}  {r['score']:.4f}\n{r['stem'][:18]}",
            fontsize=6.5,
        )
        ax.axis("off")

    stem     = Path(query_path).stem
    out_path = out_dir / f"{stem}_top{k}_retrieval.png"
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualise] Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_results_json(results: List[dict], out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[results] JSON → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DINOv3 top-K satellite tile retrieval for UAV geo-localisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--query",  required=True,
                   help="Path to the UAV ortho PNG (query image)")
    p.add_argument("--tiles",  required=True,
                   help="Directory of satellite tile PNGs")
    p.add_argument("--topk",   type=int, default=20,
                   help="Number of top tiles to return")
    p.add_argument("--model",  default=DEFAULT_MODEL,
                   help="timm DINOv3 model name")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"],
                   help="Compute device (default: auto)")
    p.add_argument("--batch",  type=int, default=16,
                   help="Batch size for tile embedding")
    p.add_argument("--no-patch-mean", action="store_true",
                   help="Use only CLS token (faster, slightly less accurate)")
    p.add_argument("--out",    default=None,
                   help="Output directory (default: <query parent>/retrieval/)")
    p.add_argument("--index-cache", default=None,
                   help=".npz file to cache/load tile embeddings (speeds up re-runs)")
    p.add_argument("--recursive", action="store_true",
                   help="Search tiles directory recursively for PNGs")
    return p


def main():
    args    = _build_parser().parse_args()
    q_path  = Path(args.query)
    out_dir = Path(args.out) if args.out else q_path.parent / "retrieval"

    results = retrieve_top_k(
        query_path    = q_path,
        tiles_dir     = args.tiles,
        k             = args.topk,
        model_name    = args.model,
        device        = args.device,
        batch_size    = args.batch,
        use_patch_mean= not args.no_patch_mean,
        index_cache   = args.index_cache,
        recursive     = args.recursive,
    )

    # Console summary
    print(f"\n{'Rank':>4}  {'Score':>8}  Tile stem")
    print("-" * 60)
    for r in results:
        print(f"  {r['rank']:>3}  {r['score']:>8.4f}  {r['stem']}")

    # Save outputs
    save_results_json(results, out_dir / f"{q_path.stem}_top{args.topk}.json")
    visualise_results(q_path, results, out_dir, n_cols=5)
    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
