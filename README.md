# SCUNet Image Denoise Pipeline

Local AI denoising and restoration pipeline for images using **SCUNet**
and **ONNX Runtime**.

The pipeline is intended for images that are noisy, lightly compressed,
or rough around fine edges. It works **at the same resolution** and
does not perform any actual upscaling.

## Model

Model in use:

``` text
scunet_color_real_psnr.onnx
```

If the model uses external weights, both files must be in the same
directory:

``` text
scunet_color_real_psnr.onnx
scunet_color_real_psnr.onnx.data
```

SCUNet stands for **Swin-Conv-UNet** and is a
restoration/denoising model for real-world color images.

The `real_psnr` variant is relatively conservative and therefore a good
fit for:

-   maps
-   scans
-   historical graphics
-   photographs
-   JPEG artifacts
-   light image noise
-   frayed or uneven edges

## Requirements

-   Python 3
-   Pillow
-   NumPy
-   ONNX Runtime

Using a virtual Python environment is recommended.

## Installation

Create a project folder:

``` bash
mkdir ai-scunet
cd ai-scunet
```

Create a virtual environment:

``` bash
python3 -m venv venv
```

Activate it:

``` bash
source venv/bin/activate
```

Install dependencies:

``` bash
pip install onnxruntime pillow numpy
```

## Folder Structure

``` text
ai-scunet/
├── scunet.py
├── scunet_color_real_psnr.onnx
├── scunet_color_real_psnr.onnx.data
├── input.jpg
└── output.jpg
```

The `.onnx.data` file is only required if the downloaded ONNX model
stores its weights externally.

## Processing

``` text
JPEG / PNG
     ↓
SCUNet
     ↓
Denoise + Restoration
     ↓
JPEG / PNG
```

The image resolution is preserved.

## Tiled Processing

Large images should not be processed in a single SCUNet pass. The
script therefore processes the image in tiles.

Default:

``` python
TILE_SIZE = 512
OVERLAP = 32
```

This means:

-   each tile is at most `512 × 512 px`
-   adjacent tiles overlap by `32 px`
-   overlapping areas are blended smoothly
-   visible tile boundaries are avoided

The edge tiles are positioned so they also keep the full tile size and
only overlap more strongly with the previous tile.

## Padding

SCUNet uses several downsampling and transformer stages. The input
dimensions therefore need to be cleanly divisible for certain internal
operations.

The script pads tiles to a multiple of `64` when needed:

``` python
pad_h = (64 - h % 64) % 64
pad_w = (64 - w % 64) % 64
```

`reflect` is used as the padding mode. After inference, the padding is
removed again.

## Running the Script

With a fixed input defined in the script:

``` bash
python3 scunet.py
```

Example:

``` python
INPUT = "roman-expansion.jpg"
OUTPUT = "roman-expansion_scunet.jpg"
```

## JPEG Output

For high-quality JPEG output:

``` python
Image.fromarray(output_uint8).save(
    OUTPUT,
    format="JPEG",
    quality=98,
    subsampling=0,
    optimize=True
)
```

For maximum lossless quality, PNG can be used instead.

## Recommended Settings

Normal:

``` python
TILE_SIZE = 512
OVERLAP = 32
```

With limited RAM:

``` python
TILE_SIZE = 256
OVERLAP = 32
```

A larger tile size means fewer inference passes, but higher memory
usage. A smaller tile size reduces memory usage, but adds more
overhead.

## What SCUNet Does Not Do

This pipeline is not generative upscaling. In particular, it does not
try to invent new high-resolution content.

Not intended for:

-   2x or 4x upscaling
-   generative detail synthesis
-   Stable Diffusion restoration
-   face reconstruction

Models such as SwinIR, Real-ESRGAN, or other super-resolution models
would be responsible for that.

SCUNet is better suited for:

``` text
Original
   ↓
Denoise
   ↓
Artifact Reduction
   ↓
Edge Cleanup
   ↓
Original Resolution
```

## Notes on Maps and Graphics

For maps, labels, and fine lines, a conservative restoration model is
often more useful than a GAN-based upscaler.

Aggressive super-resolution models can alter letters, reinterpret
lines, distort small symbols, or generate artificial textures.

SCUNet `color_real_psnr` is therefore particularly well suited as a
first cleanup step.

## Performance

The current pipeline uses:

``` python
providers=["CPUExecutionProvider"]
```

ONNX Runtime therefore runs on the CPU.

Tiled processing mainly prevents extreme RAM usage and swapping.

## Troubleshooting

### `The input tensor cannot be reshaped`

This usually means that an image or tile dimension does not fit SCUNet's
internal blocks.

The current pipeline prevents this by using:

-   full-size edge tiles
-   padding to a multiple of 64

### Very high memory usage

Reduce the tile size:

``` python
TILE_SIZE = 256
```

### Visible boundaries between tiles

Increase overlap to `48` or `64`. `32 px` is usually a good starting
point.

## Summary

``` text
SCUNet color_real_psnr
+ ONNX Runtime
+ 512 px tiles
+ 32 px overlap
+ reflect padding
= local 1:1 image restoration pipeline
```

It is especially well suited when an image should become **cleaner and
smoother without changing its resolution or basic content**.
