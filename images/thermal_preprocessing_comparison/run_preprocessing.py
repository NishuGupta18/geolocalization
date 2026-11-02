#!/usr/bin/env python3
"""Non-destructive visualization comparison for 16-bit UAV thermal-like PNGs."""
from pathlib import Path
from PIL import Image, PngImagePlugin, ImageDraw, ImageFont
import csv, json, hashlib
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
IMAGES = sorted(ROOT.glob('*.png'))
CSV_PATH = ROOT / 'metadata.csv'

with CSV_PATH.open(newline='') as f:
    metadata = {r['filename']: r for r in csv.DictReader(f)}

arrays = {}
rows = []
for path in IMAGES:
    with Image.open(path) as im:
        a = np.array(im)
        info = dict(im.info)
    if a.ndim != 2 or a.dtype != np.uint16:
        raise ValueError(f'Unexpected image representation: {path.name} {a.shape} {a.dtype}')
    arrays[path.name] = a
    q = np.percentile(a, [1, 50, 99])
    rows.append({
        'filename': path.name, 'format': 'PNG', 'width': a.shape[1], 'height': a.shape[0],
        'channels': 1, 'mode': 'I;16', 'dtype': str(a.dtype), 'min': int(a.min()),
        'p01': float(q[0]), 'median': float(q[1]), 'p99': float(q[2]), 'max': int(a.max()),
        'mean': float(a.mean()), 'std': float(a.std()), 'unique_values': int(np.unique(a).size),
        'zero_count': int(np.count_nonzero(a == 0)), 'uint16_max_count': int(np.count_nonzero(a == 65535)),
        'png_embedded_metadata_keys': '|'.join(sorted(info.keys())),
        'source_sha256': hashlib.sha256((ROOT / path.name).read_bytes()).hexdigest(),
    })

with (OUT / 'original_statistics.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# Choose four distinct frames spanning the collection's observed spatial contrast.
ranked = sorted(rows, key=lambda r: r['std'])
rep_q = [10, 35, 65, 90]
representatives = []
for p in rep_q:
    r = ranked[round((p / 100) * (len(ranked) - 1))]
    if r['filename'] not in [x['filename'] for x in representatives]:
        representatives.append(r)

methods = [
    ('native_range', 'Native uint16 range', 'Map uint16 code range 0–65535 linearly to 8-bit; retains the original global code scale for display.'),
    ('minmax', 'Min–max', 'Per-frame linear scaling from observed minimum–maximum to 0–255; uses the full range but is sensitive to outliers.'),
    ('percentile_1_99', 'Percentile 1–99%', 'Per-frame linear scaling between the 1st and 99th percentiles, clipping tails for stronger contrast.'),
    ('global_hist_eq', 'Global histogram equalization', 'Equalize the histogram of a 1–99% display mapping; expands common intensity differences but can exaggerate noise.'),
    ('mild_clahe', 'Mild CLAHE', 'CLAHE on the 1–99% display mapping (clipLimit=1.5, tiles=8×8); lifts local contrast with restrained clipping.'),
    ('bilateral_denoise', 'Mild bilateral denoise', 'Bilateral filter on the 1–99% display mapping (diameter=5, sigmaColor=12, sigmaSpace=3); smooths small fluctuations while retaining edges.'),
    ('unsharp_mask', 'Cautious unsharp mask', 'Unsharp mask on the 1–99% display mapping (amount=0.4, Gaussian sigma=1.0); adds edge acutance, with possible halos/noise.'),
    ('contrast_denoise', 'Contrast + denoise', '1–99% display mapping followed by mild bilateral filtering (diameter=5, sigmaColor=12, sigmaSpace=3).'),
]

def percentile_map(a):
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo: return np.zeros(a.shape, np.uint8), float(lo), float(hi)
    return np.clip((a.astype(np.float32)-lo) * (255.0/(hi-lo)), 0, 255).round().astype(np.uint8), float(lo), float(hi)

def minmax_map(a):
    lo, hi = int(a.min()), int(a.max())
    if hi <= lo: return np.zeros(a.shape, np.uint8)
    return np.clip((a.astype(np.float32)-lo) * (255.0/(hi-lo)), 0, 255).round().astype(np.uint8)

def create_variants(a):
    base, lo, hi = percentile_map(a)
    native = np.rint(a.astype(np.float32) * (255.0/65535.0)).astype(np.uint8)
    mm = minmax_map(a)
    eq = cv2.equalizeHist(base)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)).apply(base)
    raw_bilateral = cv2.bilateralFilter(a.astype(np.float32), d=5, sigmaColor=60, sigmaSpace=3)
    bilateral, _, _ = percentile_map(raw_bilateral)
    blur = cv2.GaussianBlur(base, (0,0), 1.0)
    sharp = np.clip(base.astype(np.float32) + 0.4*(base.astype(np.float32)-blur.astype(np.float32)), 0, 255).round().astype(np.uint8)
    combined = cv2.bilateralFilter(base, d=5, sigmaColor=12, sigmaSpace=3)
    return {
        'native_range': native, 'minmax': mm, 'percentile_1_99': base,
        'global_hist_eq': eq, 'mild_clahe': clahe, 'bilateral_denoise': bilateral,
        'unsharp_mask': sharp, 'contrast_denoise': combined,
    }, lo, hi

# Export separate 8-bit display products and JSON provenance sidecars. Originals are read-only.
for rep in representatives:
    name = rep['filename']; a = arrays[name]
    variants, lo, hi = create_variants(a)
    for method, title, description in methods:
        dst = OUT / f'{Path(name).stem}__{method}.png'
        pngmeta = PngImagePlugin.PngInfo()
        pngmeta.add_text('SourceFile', name)
        pngmeta.add_text('PreprocessingMethod', title)
        pngmeta.add_text('SourceBitDepth', '16-bit unsigned grayscale')
        pngmeta.add_text('DisplayOnly', '8-bit visualization; do not use as calibrated temperature data')
        Image.fromarray(variants[method], mode='L').save(dst, pnginfo=pngmeta)
        (OUT / f'{Path(name).stem}__{method}.json').write_text(json.dumps({
            'source_file': name, 'source_sha256': rep['source_sha256'],
            'source_format': 'PNG', 'source_dimensions': [a.shape[1], a.shape[0]],
            'source_channels': 1, 'source_dtype': 'uint16', 'source_pixel_range': [int(a.min()), int(a.max())],
            'source_embedded_png_metadata': {}, 'external_metadata_row': metadata.get(name),
            'output_file': dst.name, 'output_dtype': 'uint8', 'output_role': 'display-only visualization',
            'method': method, 'parameters': {
                'mapping_percentiles': [1,99] if method not in ['native_range','minmax'] else None,
                'mapping_bounds_for_percentile_methods': [lo,hi],
                'clahe': {'clipLimit':1.5,'tileGridSize':[8,8]} if method=='mild_clahe' else None,
                'bilateral': ({'d':5,'sigmaColor':60,'sigmaSpace':3,'input':'source uint16 codes before 1–99% stretch'} if method=='bilateral_denoise' else ({'d':5,'sigmaColor':12,'sigmaSpace':3,'input':'1–99% 8-bit display mapping'} if method=='contrast_denoise' else None)),
                'unsharp': {'amount':0.4,'gaussian_sigma':1.0} if method=='unsharp_mask' else None,
            },
            'calibration_note': 'The metadata CSV provides navigation and visible-camera intrinsics/distortion, but no thermal calibration coefficients, temperature scale, emissivity, or NoData declaration.'
        }, indent=2) + '\n')

# Labeled comparison grid. Methods occupy a dedicated left column; frames have a top header.
cell_w, cell_h, row_h, label_w = 320, 256, 282, 198
margin, top_h = 8, 44
grid = Image.new('RGB', (label_w + len(representatives)*cell_w + margin, top_h + len(methods)*row_h + margin), 'white')
draw = ImageDraw.Draw(grid)
for ci, rep in enumerate(representatives):
    x = label_w + ci*cell_w
    draw.text((x+3, 12), f"{rep['filename'][:8]} | SD {rep['std']:.0f}", fill='black')
for ri, (method, title, _) in enumerate(methods):
    y = top_h + ri*row_h
    draw.text((8, y+9), title, fill='black')
    for ci, rep in enumerate(representatives):
        a=arrays[rep['filename']]
        vv, _, _ = create_variants(a)
        x = label_w + ci*cell_w
        grid.paste(Image.fromarray(vv[method]).convert('RGB'), (x, y))
grid.save(OUT / 'comparison_grid.png')

# Concise report with aggregate stats and explicit interpretation.
raw = np.concatenate([a.ravel() for a in arrays.values()])
stat_arr = np.array([[r['min'], r['max'], r['mean'], r['std']] for r in rows], dtype=float)
rep_lines = '\n'.join(f"- `{r['filename']}`: range {r['min']}–{r['max']}; mean {r['mean']:.1f}; SD {r['std']:.1f}; p01/median/p99 {r['p01']:.0f}/{r['median']:.0f}/{r['p99']:.0f}." for r in representatives)
report = f'''# UAV thermal preprocessing comparison

## Scope and source inspection

Processed {len(IMAGES)} PNGs in this folder as a non-destructive, display-only comparison. No TGM or SGM was used. All source images are 320×256, one-channel grayscale PNGs decoded as unsigned 16-bit (`uint16`); no image has embedded PNG metadata. The accompanying `metadata.csv` has one row per image with GPS/altitude/attitude and camera intrinsics/distortion. It contains no thermal calibration coefficients, temperature units, emissivity/atmospheric parameters, or invalid-pixel declaration.

Across the collection, source values span {int(raw.min())}–{int(raw.max())}; aggregate median {np.median(raw):.1f}, mean {raw.mean():.1f}, SD {raw.std():.1f}. There are {sum(r['zero_count'] for r in rows)} zero-valued and {sum(r['uint16_max_count'] for r in rows)} 65535-valued pixels. No NaN/Inf can occur in the integer PNGs. No explicit NoData sentinel is documented; zero and 65535 are absent, but undocumented invalid values cannot be ruled out. Per-file measurements are in `original_statistics.csv`.

### Initial representative pixel statistics

{rep_lines}

## Raw or rendered?

These are not conventional 8-bit/colorized thermal exports: they contain single-channel 16-bit integer codes with a narrow observed range centered near 20,000 and no embedded display palette or metadata. That is consistent with sensor counts or a radiometric-style grayscale export. PNG storage and numeric range alone cannot prove the files are raw detector counts or calibrated radiometric temperatures; they might already have undergone an undocumented linear transform. There is no evidence here that pixel codes convert to °C/°F. Treat source pixels as uncalibrated measurements until the supplier confirms the encoding and provides the camera calibration/temperature conversion.

## Visualization pipeline

The grid in `comparison_grid.png` includes four frames chosen at the 10th, 35th, 65th and 90th percentile of per-frame standard deviation (to cover low through high scene contrast). Separate 8-bit PNGs are saved for each frame and method, with JSON sidecars containing source provenance and the matching CSV row. Source files are untouched. Original PNGs carry no metadata to copy; the supplier CSV metadata is retained in each sidecar.

Methods and purpose:

- **Native uint16 range:** direct 0–65535-to-8-bit mapping. Preserves a single global code scale for display, but looks nearly flat because the observed codes occupy a small part of the uint16 range.
- **Min–max:** uses all observed values; improves visibility but outlier pixels set the scale and pixel extrema can dominate.
- **1–99% stretch:** expands the central 98% of each frame; the clearest simple baseline, at the cost of clipping roughly 1% at each tail.
- **Global histogram equalization:** redistributes global gray levels; may make edges stand out but can make noise and intensity distributions look unnatural.
- **Mild CLAHE:** clipLimit 1.5, 8×8 tiles; addresses spatially uneven/local contrast, with some risk of local noise amplification or tile artifacts.
- **Bilateral denoise:** diameter 5, sigmaColor 60 source-code units, sigmaSpace 3 before 1–99% display scaling; suppresses small fluctuations while retaining boundaries, but can erase fine texture.
- **Unsharp mask:** amount 0.4, Gaussian sigma 1.0 after stretch; increases edge acutance, with possible halos and amplified noise.
- **Contrast + denoise:** 1–99% stretch then a mild bilateral filter (diameter 5, sigmaColor 12, sigmaSpace 3 in 8-bit display units); combines visibility improvement and slight smoothing.

All variants are 8-bit display products whose numerical values are altered. None should be used for temperature analysis or represented as calibrated radiometric values. If the source files are radiometric, preserve and analyze their original `uint16` codes (or a vendor-verified conversion to temperature); visualization enhancement is separate from value-preserving processing.

## Assessment

In the comparison, the 1–99% stretch is the best default for making scene structure readable while keeping a simple monotonic relationship to each source frame. Mild CLAHE can reveal local detail in locally flat areas but is more likely to overstate texture/noise. The combined stretch plus mild bilateral filter is the preferable option when visible granular noise distracts from boundaries; use it cautiously because it can smooth fine thermal features. Histogram equalization and sharpening are less conservative. None repairs blur, low spatial resolution, atmospheric effects, or poor focus if those are the actual cause of haze.

For production inspection, first compare `percentile_1_99` and `contrast_denoise` against the unchanged source, and keep the original pixel array as the quantitative record. Validate any fixed cross-frame contrast window against the supplier's scale/calibration before comparing temperatures across frames.
'''
(OUT/'report.md').write_text(report)
print('representatives:')
for r in representatives: print(r['filename'],r['std'],r['min'],r['max'])
print('wrote',OUT)
