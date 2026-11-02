# Grad-CAM examples for ConvNeXt-Tiny features

The contact sheet `gradcam_examples.png` shows four representative inputs, the Grad-CAM maps, and overlays. Each map targets that image's top-1 class from the pretrained ImageNet-1K classifier. The activation map is computed from the final ConvNeXt stage, then resized from its 8×10 feature grid to the input size for display.

**Interpretation:** warm colors indicate areas with stronger positive contribution to the selected ImageNet class score under Grad-CAM. This is a rough model-attention explanation, not a learned thermal importance map, a segmentation, a temperature map, or a scientific causal explanation. ConvNeXt-Tiny was pretrained on ImageNet natural images and was not trained or fine-tuned on these UAV thermal-like files. Its predicted class may be irrelevant. Treat the overlays as a way to inspect where this particular pretrained model responds, not as evidence that those areas matter to the real-world scene.

The input uses the same per-image 1–99% grayscale stretch as feature extraction, repeated across RGB. Original source arrays remain unchanged. Per-image overlays and color heatmaps are saved beside the comparison sheet.
