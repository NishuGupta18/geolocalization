# UAV thermal preprocessing comparison

## Scope and source inspection

Processed 500 PNGs in this folder as a non-destructive, display-only comparison. No TGM or SGM was used. All source images are 320×256, one-channel grayscale PNGs decoded as unsigned 16-bit (`uint16`); no image has embedded PNG metadata. The accompanying `metadata.csv` has one row per image with GPS/altitude/attitude and camera intrinsics/distortion. It contains no thermal calibration coefficients, temperature units, emissivity/atmospheric parameters, or invalid-pixel declaration.

Across the collection, source values span 19181–25153; aggregate median 20832.0, mean 20796.7, SD 374.3. There are 0 zero-valued and 0 65535-valued pixels. No NaN/Inf can occur in the integer PNGs. No explicit NoData sentinel is documented; zero and 65535 are absent, but undocumented invalid values cannot be ruled out. Per-file measurements are in `original_statistics.csv`.

### Initial representative pixel statistics

- `a664fc5aa38347b98d13c1886b46baf6.png`: range 20521–21676; mean 20848.6; SD 106.8; p01/median/p99 20619/20844/21125.
- `532d401b7c144eb285a06fdd005ca053.png`: range 20395–21704; mean 20885.2; SD 172.6; p01/median/p99 20561/20848/21407.
- `40758301d38d46dc846f990eee4b19bc.png`: range 19948–22346; mean 20478.9; SD 225.5; p01/median/p99 20090/20438/21266.
- `124eaa29330f4570974d7ca70bb114a1.png`: range 19918–22163; mean 20548.8; SD 292.9; p01/median/p99 20131/20475/21409.

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
