# geolocalization

## DINOv3 satellite-tile validation

Run retrieval evaluation against the UAV GPS labels in `images/metadata.csv`:

```bash
HF_HUB_OFFLINE=1 python3 pipeline/validate_dinov3.py \
  --model vit_base_patch16_dinov3.lvd1689m \
  --tiles satellite_tile/tiles-20261103T143243Z-1-001/tiles \
  --tile-db satellite_tile/satellite_tiles.db \
  --out pipeline/outputs/dinov3_validation_vitb \
  --index-cache pipeline/outputs/dinov3_validation_vitb/tile_index.npz
```

The validator uses the tile-center coordinates in `satellite_tiles.db`,
excludes tiles with `valid_pct=0`, and reports top-1 tile-center distance and
recall within several geographic radii for top-1 and top-K. The first run
embeds all valid satellite tiles; later runs reuse the embedding cache.

### Current validation result

Using `vit_base_patch16_dinov3.lvd1689m` on CPU, all 500 UAV images, and 494
valid satellite tiles:

| Metric | Result |
| --- | ---: |
| Top-1 tile-center error, median | 1,074 m |
| Top-1 tile-center error, mean | 1,452 m |
| Top-1 recall within 250 m | 10.0% |
| Top-10 recall within 250 m | 49.6% |
| Top-10 recall within 500 m | 72.2% |

For reference, the nearest valid tile center by GPS is a median 94 m from a
query, so the map coverage is adequate and the larger retrieval error reflects
the visual ranking. This is a cross-domain test: the UAV images are 16-bit
grayscale thermal frames converted to RGB by the current loader, while the
satellite tiles are RGB. The satellite-pretrained DINOv3 ViT-L weights were
not available offline; the measured model is the cached general-purpose
DINOv3 ViT-B checkpoint. Detailed per-query rankings are in
`pipeline/outputs/dinov3_validation_vitb/predictions.json`; aggregate metrics
are in `pipeline/outputs/dinov3_validation_vitb/metrics.json`.
