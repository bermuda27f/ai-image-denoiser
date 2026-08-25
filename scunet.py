from PIL import Image
import numpy as np
import onnxruntime as ort
import argparse
from pathlib import Path

MODEL = "scunet_color_real_psnr.onnx"
OUTPUT = "output.jpg"

TILE_SIZE = 512
OVERLAP = 32

# Modell laden
session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SCUNet denoising on an input image."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="roman-expansion.jpg",
        help="Input image path (default: roman-expansion.jpg)"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output image path (default: <input>_scunet.jpg)"
    )
    return parser.parse_args()

def run_scunet(tile):
    """
    tile: HWC float32, Werte 0..1
    """

    h, w, _ = tile.shape

    # Robuster für SCUNet: auf Vielfache von 64 padd'en
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64

    if pad_h or pad_w:
        tile = np.pad(
            tile,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect"
        )

    # HWC -> NCHW
    x = np.transpose(tile, (2, 0, 1))
    x = np.expand_dims(x, axis=0).astype(np.float32)

    result = session.run(
        [output_name],
        {input_name: x}
    )[0]

    # NCHW -> HWC
    result = result[0]
    result = np.transpose(result, (1, 2, 0))

    # Padding wieder entfernen
    result = result[:h, :w, :]

    return np.clip(result, 0, 1)


def make_weight_mask(height, width, overlap):
    """
    Weiches Gewicht für Tile-Ränder.
    """

    mask_y = np.ones(height, dtype=np.float32)
    mask_x = np.ones(width, dtype=np.float32)

    fade_y = min(overlap, height // 2)
    fade_x = min(overlap, width // 2)

    if fade_y > 0:
        ramp = np.linspace(0.001, 1.0, fade_y, dtype=np.float32)
        mask_y[:fade_y] = ramp
        mask_y[-fade_y:] = ramp[::-1]

    if fade_x > 0:
        ramp = np.linspace(0.001, 1.0, fade_x, dtype=np.float32)
        mask_x[:fade_x] = ramp
        mask_x[-fade_x:] = ramp[::-1]

    mask = mask_y[:, None] * mask_x[None, :]

    return mask[:, :, None]


def make_positions(length, tile_size, step):
    """
    Erzeugt Tile-Positionen so, dass das letzte Tile immer
    volle tile_size hat und exakt am Bildrand endet.
    """

    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, step))

    last_position = length - tile_size

    if positions[-1] != last_position:
        positions.append(last_position)

    return sorted(set(positions))


args = parse_args()
input_path = args.input

if args.output is None:
    input_stem = Path(input_path).stem
    OUTPUT = f"{input_stem}_scunet.jpg"
else:
    OUTPUT = args.output

# Bild laden
img = Image.open(input_path).convert("RGB")
img_np = np.asarray(img).astype(np.float32) / 255.0

height, width, _ = img_np.shape

print(f"Input: {width}x{height}")
print(f"Tile size: {TILE_SIZE}")
print(f"Overlap: {OVERLAP}")

# Ergebnis- und Gewichtsspeicher
output = np.zeros_like(img_np, dtype=np.float32)
weights = np.zeros((height, width, 1), dtype=np.float32)

step = TILE_SIZE - OVERLAP

# Tile-Positionen
x_positions = make_positions(width, TILE_SIZE, step)
y_positions = make_positions(height, TILE_SIZE, step)

total_tiles = len(x_positions) * len(y_positions)
tile_counter = 0

print(f"Tiles: {len(x_positions)} x {len(y_positions)} = {total_tiles}")

for y in y_positions:
    for x in x_positions:

        x2 = min(x + TILE_SIZE, width)
        y2 = min(y + TILE_SIZE, height)

        tile = img_np[y:y2, x:x2]

        tile_counter += 1

        print(
            f"Tile {tile_counter}/{total_tiles} "
            f"({x}:{x2}, {y}:{y2}) "
            f"{tile.shape[1]}x{tile.shape[0]}"
        )

        restored = run_scunet(tile)

        mask = make_weight_mask(
            restored.shape[0],
            restored.shape[1],
            OVERLAP
        )

        output[y:y2, x:x2] += restored * mask
        weights[y:y2, x:x2] += mask


# gewichtetes Zusammenfügen
output = output / np.maximum(weights, 1e-8)

output = np.clip(output, 0, 1)

output_uint8 = (output * 255.0).round().astype(np.uint8)

Image.fromarray(output_uint8).save(
    OUTPUT,
    format="JPEG",
    quality=95,      # 95–100
    subsampling=0,   # keine Chroma-Subsampling-Artefakte
    optimize=True
)

print(f"Done: {OUTPUT}")
