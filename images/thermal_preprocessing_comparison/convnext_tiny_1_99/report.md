# ConvNeXt-Tiny feature extraction

- Images: 500 source PNGs; sorted by filename in `feature_index.csv`.
- Model: torchvision ConvNeXt-Tiny, pretrained weights `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` (ImageNet-1K V1).
- Output: `convnext_tiny_embeddings.npy`, shape `(500, 768)`, dtype `float32`; one pooled 768-dimensional feature vector per image.
- Feature stage: final ConvNeXt feature map → adaptive average pooling → pretrained final LayerNorm; classification layer omitted.
- Input: each original uint16 image independently mapped linearly from its 1st–99th percentile to 0–255, clipped at both ends; grayscale copied into three channels; ImageNet RGB mean/std normalization.
- Spatial handling: original 320×256 aspect and pixel dimensions retained; no resize or crop.
- Device: `cpu`; batch size 8.
- `feature_index.csv` maps each embedding row to its source filename, source SHA-256, and actual stretch bounds.

These are ImageNet-pretrained features from contrast-enhanced grayscale thermal-like inputs. They are not temperature measurements or thermal-domain-trained embeddings. Their usefulness for a downstream task should be validated on labeled data; the per-frame stretch also means absolute frame-to-frame intensity offsets are not represented consistently. Original source pixels remain unchanged.
