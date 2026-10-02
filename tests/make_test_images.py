"""Creates synthetic 512×512 test images carrying various AI labels.

Usage: python -m tests.make_test_images [target_dir]
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

C2PA_TEST_IMAGE_URL = (
    "https://raw.githubusercontent.com/contentauth/c2pa-python/main/tests/fixtures/C.jpg"
)

A1111_PARAMETERS = (
    "a photo of a red fox in a snowy forest, highly detailed\n"
    "Negative prompt: blurry, lowres\n"
    "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1234567890, Size: 512x512, "
    "Model hash: 31e35c80fc, Model: sd_xl_base_1.0, Version: v1.6.0"
)

IPTC_XMP = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
    xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
    Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia">
   <photoshop:Credit>Made with Google AI</photoshop:Credit>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def base_image(seed: int = 0, size: int = 512) -> Image.Image:
    """Structured image (gradient + noise) so DWT watermarks embed reliably."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    r = x / size * 255
    g = y / size * 255
    b = 128 + 60 * np.sin(x / 23.0) * np.cos(y / 31.0)
    arr = np.stack([r, g, b], axis=-1) + rng.normal(0, 18, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def make_a1111(target: Path) -> Path:
    info = PngInfo()
    info.add_text("parameters", A1111_PARAMETERS)
    path = target / "a1111.png"
    base_image(1).save(path, pnginfo=info)
    return path


def make_iptc(target: Path) -> Path:
    path = target / "iptc_google.jpg"
    base_image(2).save(path, "JPEG", quality=90, xmp=IPTC_XMP.encode("utf-8"))
    return path


def make_exif(target: Path) -> Path:
    exif = Image.Exif()
    exif[0x0131] = "Midjourney"
    path = target / "exif_midjourney.jpg"
    base_image(3).save(path, "JPEG", quality=90, exif=exif)
    return path


def make_sdxl(target: Path) -> Path:
    import cv2

    from app.detectors import SDXL_BITS, load_watermark_module

    wm = load_watermark_module()
    encoder = wm.WatermarkEncoder()
    encoder.set_watermark("bits", SDXL_BITS)
    bgr = cv2.cvtColor(np.asarray(base_image(4)), cv2.COLOR_RGB2BGR)
    marked = encoder.encode(bgr, "dwtDct")
    path = target / "sdxl_watermark.png"
    Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)).save(path)
    return path


def make_unmarked(target: Path) -> Path:
    path = target / "unmarked.jpg"
    base_image(5).save(path, "JPEG", quality=90)
    return path


def download_c2pa_test_image(target: Path, timeout: float = 20) -> Path:
    path = target / "c2pa_C.jpg"
    if not path.exists():
        with urllib.request.urlopen(C2PA_TEST_IMAGE_URL, timeout=timeout) as response:
            path.write_bytes(response.read())
    return path


def make_all(target: Path) -> dict[str, Path]:
    target.mkdir(parents=True, exist_ok=True)
    return {
        "a1111": make_a1111(target),
        "iptc": make_iptc(target),
        "exif": make_exif(target),
        "sdxl": make_sdxl(target),
        "unmarked": make_unmarked(target),
    }


if __name__ == "__main__":
    target_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "test_images")
    for name, path in make_all(target_dir).items():
        print(f"{name:12} {path}")
    try:
        print(f"{'c2pa':12} {download_c2pa_test_image(target_dir)}")
    except OSError as exc:
        print(f"Could not download C2PA test image: {exc}", file=sys.stderr)
