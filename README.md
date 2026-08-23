# SCUNet Image Denoise Pipeline

Lokale AI-Denoise- und Restoration-Pipeline für Bilder mit **SCUNet**
und **ONNX Runtime**.

Die Pipeline ist für Bilder gedacht, die verrauscht, leicht komprimiert
oder an feinen Kanten unsauber sind. Sie arbeitet **bei gleicher
Auflösung** und führt kein eigentliches Upscaling durch.

## Modell

Verwendetes Modell:

``` text
scunet_color_real_psnr.onnx
```

Falls das Modell externe Gewichte verwendet, müssen beide Dateien im
selben Verzeichnis liegen:

``` text
scunet_color_real_psnr.onnx
scunet_color_real_psnr.onnx.data
```

SCUNet steht für **Swin-Conv-UNet** und ist ein
Restoration-/Denoising-Modell für reale Farbbilder.

Die `real_psnr`-Variante ist relativ konservativ und eignet sich daher
gut für:

-   Karten
-   Scans
-   historische Grafiken
-   Fotografien
-   JPEG-Artefakte
-   leichtes Bildrauschen
-   ausgefranste oder unruhige Kanten

## Voraussetzungen

-   Python 3
-   Pillow
-   NumPy
-   ONNX Runtime

Optional empfiehlt sich eine virtuelle Python-Umgebung.

## Installation

Projektordner anlegen:

``` bash
mkdir ai-scunet
cd ai-scunet
```

Virtuelle Umgebung erstellen:

``` bash
python3 -m venv venv
```

Aktivieren:

``` bash
source venv/bin/activate
```

Abhängigkeiten installieren:

``` bash
pip install onnxruntime pillow numpy
```

## Ordnerstruktur

``` text
ai-scunet/
├── scunet.py
├── scunet_color_real_psnr.onnx
├── scunet_color_real_psnr.onnx.data
├── input.jpg
└── output.jpg
```

Die `.onnx.data`-Datei ist nur erforderlich, wenn das heruntergeladene
ONNX-Modell seine Gewichte extern speichert.

## Verarbeitung

``` text
JPEG / PNG
     ↓
SCUNet
     ↓
Denoise + Restoration
     ↓
JPEG / PNG
```

Die Bildauflösung bleibt erhalten.

## Tiled Processing

Große Bilder sollten nicht in einem einzigen SCUNet-Durchlauf
verarbeitet werden. Das Script verarbeitet das Bild deshalb in Tiles.

Standard:

``` python
TILE_SIZE = 512
OVERLAP = 32
```

Das bedeutet:

-   jedes Teilbild ist maximal `512 × 512 px`
-   benachbarte Tiles überlappen sich um `32 px`
-   die Überlappungsbereiche werden weich miteinander verrechnet
-   sichtbare Tile-Grenzen werden dadurch vermieden

Die Rand-Tiles werden so positioniert, dass sie ebenfalls die volle
Tile-Größe besitzen und lediglich stärker mit dem vorherigen Tile
überlappen.

## Padding

SCUNet verwendet mehrere Downsampling- und Transformer-Stufen. Die
Eingabedimensionen müssen deshalb für bestimmte interne Operationen
sauber teilbar sein.

Das Script padded Tiles bei Bedarf auf ein Vielfaches von `64`:

``` python
pad_h = (64 - h % 64) % 64
pad_w = (64 - w % 64) % 64
```

Als Padding-Modus wird `reflect` verwendet. Nach der Inferenz wird das
Padding wieder entfernt.

## Script ausführen

Bei fest definiertem Input im Script:

``` bash
python3 scunet.py
```

Beispiel:

``` python
INPUT = "roman-expansion.jpg"
OUTPUT = "roman-expansion_scunet.jpg"
```

## JPEG-Ausgabe

Für hochwertige JPEG-Ausgabe:

``` python
Image.fromarray(output_uint8).save(
    OUTPUT,
    format="JPEG",
    quality=98,
    subsampling=0,
    optimize=True
)
```

Für maximale verlustfreie Qualität kann stattdessen PNG verwendet
werden.

## Empfohlene Einstellungen

Normal:

``` python
TILE_SIZE = 512
OVERLAP = 32
```

Bei wenig verfügbarem RAM:

``` python
TILE_SIZE = 256
OVERLAP = 32
```

Eine größere Tile-Größe bedeutet weniger Inference-Durchläufe, aber
höheren Speicherverbrauch. Eine kleinere Tile-Größe reduziert den
Speicherverbrauch, erzeugt aber mehr Overhead.

## Was SCUNet nicht macht

Diese Pipeline ist kein generatives Upscaling. Sie versucht insbesondere
nicht, neue hochauflösende Inhalte zu erfinden.

Nicht vorgesehen sind:

-   2× oder 4× Upscaling
-   generative Detailerzeugung
-   Stable-Diffusion-Restoration
-   Face Reconstruction

Dafür wären Modelle wie SwinIR, Real-ESRGAN oder andere
Super-Resolution-Modelle zuständig.

SCUNet eignet sich besser für:

``` text
Original
   ↓
Denoise
   ↓
Artifact Reduction
   ↓
Edge Cleanup
   ↓
Originalauflösung
```

## Hinweise zu Karten und Grafiken

Bei Karten, Beschriftungen und feinen Linien ist ein konservatives
Restoration-Modell häufig sinnvoller als ein GAN-basierter Upscaler.

Aggressive Super-Resolution-Modelle können Buchstaben verändern, Linien
neu interpretieren, kleine Symbole verfälschen oder künstliche Texturen
erzeugen.

SCUNet `color_real_psnr` eignet sich deshalb besonders gut als erster
Cleanup-Schritt.

## Performance

Die aktuelle Pipeline verwendet:

``` python
providers=["CPUExecutionProvider"]
```

ONNX Runtime läuft damit über die CPU.

Tiled Processing verhindert dabei vor allem extremen RAM-Verbrauch und
Swap.

## Troubleshooting

### `The input tensor cannot be reshaped`

Das bedeutet meistens, dass eine Bild- oder Tile-Dimension nicht zu den
internen SCUNet-Blöcken passt.

Die aktuelle Pipeline verhindert dies durch:

-   volle Rand-Tiles
-   Padding auf ein Vielfaches von 64

### Sehr hoher Speicherverbrauch

Tile-Größe reduzieren:

``` python
TILE_SIZE = 256
```

### Sichtbare Grenzen zwischen Tiles

Overlap auf `48` oder `64` erhöhen. `32 px` ist normalerweise ein guter
Ausgangspunkt.

## Kurzfassung

``` text
SCUNet color_real_psnr
+ ONNX Runtime
+ 512 px Tiles
+ 32 px Overlap
+ Reflect Padding
= lokale 1:1 Image-Restoration-Pipeline
```

Sie eignet sich besonders gut, wenn ein Bild **sauberer und ruhiger
werden soll, ohne seine Auflösung oder seinen grundlegenden Inhalt zu
verändern**.
