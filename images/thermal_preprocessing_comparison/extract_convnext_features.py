#!/usr/bin/env python3
"""Extract ImageNet-pretrained ConvNeXt-Tiny embeddings from UAV 1–99% displays."""
from pathlib import Path
import csv, hashlib, json, os
import numpy as np
from PIL import Image
import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'convnext_tiny_1_99'
OUT.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 8

# Put downloaded weights in writable, reusable storage, never beside the source images.
torch.hub.set_dir(os.environ.get('TORCH_HOME', '/tmp/thermal_torch_cache') + '/hub')
torch.set_num_threads(min(8, os.cpu_count() or 1))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = ConvNeXt_Tiny_Weights.DEFAULT
model = convnext_tiny(weights=weights).to(device).eval()

files = sorted(ROOT.glob('*.png'))
if not files:
    raise SystemExit(f'No PNGs found in {ROOT}')

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1,3,1,1)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1,3,1,1)
all_features, index_rows = [], []

@torch.inference_mode()
def infer(batch):
    x = torch.from_numpy(np.stack(batch)).to(device=device, dtype=torch.float32) / 255.0
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - mean.to(device)) / std.to(device)
    z = model.avgpool(model.features(x))
    # ConvNeXt's pretrained final LayerNorm, before the 1000-class classifier.
    z = model.classifier[0](z).flatten(1)
    return z.cpu().numpy().astype(np.float32)

pending, pending_rows = [], []
for path in files:
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim != 2 or a.dtype != np.uint16 or a.shape != (256, 320):
        raise ValueError(f'Unexpected input {path.name}: shape={a.shape}, dtype={a.dtype}')
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        image = np.zeros(a.shape, dtype=np.uint8)
    else:
        image = np.clip((a.astype(np.float32)-lo) * (255.0/(hi-lo)), 0, 255).round().astype(np.uint8)
    pending.append(image)
    pending_rows.append({
        'row_index': len(index_rows) + len(pending_rows),
        'filename': path.name,
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'p01_code': float(lo), 'p99_code': float(hi),
        'input_width': a.shape[1], 'input_height': a.shape[0],
        'input_preprocessing': 'per-image linear 1st–99th percentile stretch; grayscale repeated to RGB',
    })
    if len(pending) == BATCH_SIZE:
        all_features.extend(infer(pending))
        index_rows.extend(pending_rows)
        pending, pending_rows = [], []
if pending:
    all_features.extend(infer(pending))
    index_rows.extend(pending_rows)

features = np.stack(all_features).astype(np.float32)
np.save(OUT / 'convnext_tiny_embeddings.npy', features)
with (OUT / 'feature_index.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=index_rows[0].keys())
    writer.writeheader(); writer.writerows(index_rows)

# Record exact weight identity and output contract for downstream use.
report = f'''# ConvNeXt-Tiny feature extraction

- Images: {len(files)} source PNGs; sorted by filename in `feature_index.csv`.
- Model: torchvision ConvNeXt-Tiny, pretrained weights `{weights}` (ImageNet-1K V1).
- Output: `convnext_tiny_embeddings.npy`, shape `{features.shape}`, dtype `float32`; one pooled 768-dimensional feature vector per image.
- Feature stage: final ConvNeXt feature map → adaptive average pooling → pretrained final LayerNorm; classification layer omitted.
- Input: each original uint16 image independently mapped linearly from its 1st–99th percentile to 0–255, clipped at both ends; grayscale copied into three channels; ImageNet RGB mean/std normalization.
- Spatial handling: original 320×256 aspect and pixel dimensions retained; no resize or crop.
- Device: `{device}`; batch size {BATCH_SIZE}.
- `feature_index.csv` maps each embedding row to its source filename, source SHA-256, and actual stretch bounds.

These are ImageNet-pretrained features from contrast-enhanced grayscale thermal-like inputs. They are not temperature measurements or thermal-domain-trained embeddings. Their usefulness for a downstream task should be validated on labeled data; the per-frame stretch also means absolute frame-to-frame intensity offsets are not represented consistently. Original source pixels remain unchanged.
'''
(OUT / 'report.md').write_text(report)
print(f'Wrote {features.shape} embeddings to {OUT}')
print('embedding range', float(features.min()), float(features.max()), 'mean/std', float(features.mean()), float(features.std()))
