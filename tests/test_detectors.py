"""Tests for detectors, report, API, CLI and access protection.  Run: python -m pytest"""

from __future__ import annotations

import io
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app import auth, detectors
from app import main as main_module
from app.cli import main as cli_main
from app.detectors import (
    LEVEL_AI,
    LEVEL_HINT,
    LEVEL_INFO,
    VERDICT_AI,
    VERDICT_HINT,
    VERDICT_NONE,
    analyze,
    decode_user_comment,
    detect_generators,
    read_xmp_field,
    truncate,
)
from app.main import app
from app.report import DISCLAIMER, to_text
from tests import make_test_images as mti


@pytest.fixture(scope="session")
def images(tmp_path_factory) -> dict[str, Path]:
    return mti.make_all(tmp_path_factory.mktemp("test_images"))


@pytest.fixture(scope="session")
def c2pa_image(tmp_path_factory) -> Path:
    try:
        return mti.download_c2pa_test_image(tmp_path_factory.mktemp("c2pa"))
    except OSError as exc:
        pytest.skip(f"C2PA test image not downloadable: {exc}")


def check(path: Path):
    return analyze(path.read_bytes(), path.name)


def png_bytes(seed: int = 12, size: int = 64, info: PngInfo | None = None) -> bytes:
    buffer = io.BytesIO()
    mti.base_image(seed, size).save(buffer, "PNG", pnginfo=info)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Expected verdicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "verdict", "level"),
    [
        ("a1111", VERDICT_AI, "ai"),
        ("iptc", VERDICT_AI, "ai"),
        ("sdxl", VERDICT_AI, "ai"),
        ("exif", VERDICT_HINT, "hint"),
        ("unmarked", VERDICT_NONE, "none"),
    ],
)
def test_verdicts(images, name, verdict, level):
    result = check(images[name])
    assert result.errors == []
    assert result.verdict == verdict, to_text(result)
    assert result.verdict_level == level
    assert result.size == (512, 512)


def test_c2pa_test_image(c2pa_image):
    result = check(c2pa_image)
    assert result.errors == []
    assert result.verdict == VERDICT_HINT, to_text(result)
    c2pa = [f for f in result.findings if f.source == "c2pa"]
    assert c2pa and c2pa[0].level == LEVEL_HINT
    assert "algorithmicMedia" in c2pa[0].title
    details = c2pa[0].details
    assert details["signed_by"] == "C2PA Test Signing Cert"
    assert details["actions"] == [
        {
            "action": "c2pa.created",
            "digital_source_type": "algorithmicMedia",
            "software_agent": "Make Test Images 0.26.0",
        }
    ]
    assert details["source_types"] == [
        {"type": "algorithmicMedia", "meaning": detectors.IPTC_SOURCE_TYPES["algorithmicMedia"][1]}
    ]
    assert details["validation_state"] == "Valid"
    assert details["validation"].startswith("Structurally valid")
    assert details["validation_codes"] == ["signingCredential.untrusted"]
    assert details["validation_messages"] == [
        "Signing certificate is not from a trusted issuer (not on the trust list)"
    ]


def test_a1111_details(images):
    result = check(images["a1111"])
    finding = next(f for f in result.findings if f.level == LEVEL_AI)
    assert finding.source == "png_text"
    assert finding.details["model"] == "sd_xl_base_1.0"
    assert finding.details["seed"] == "1234567890"
    assert finding.details["prompt"].startswith("a photo of a red fox")
    assert finding.details["negative_prompt"] == "blurry, lowres"
    assert "sd_xl_base_1.0" in finding.title


def test_iptc_and_xmp(images):
    result = check(images["iptc"])
    iptc = next(f for f in result.findings if f.source == "iptc_xmp")
    assert iptc.level == LEVEL_AI and iptc.details["value"] == "trainedAlgorithmicMedia"
    xmp = next(f for f in result.findings if f.source == "xmp")
    assert xmp.level == LEVEL_HINT
    assert xmp.details["photoshop:Credit"] == "Made with Google AI"
    assert "Made with Google AI" in xmp.details["detected_generators"]
    assert result.findings[0].level == LEVEL_AI  # sorting: ai first


def test_exif_midjourney(images):
    result = check(images["exif"])
    exif = next(f for f in result.findings if f.source == "exif")
    assert exif.level == LEVEL_HINT
    assert exif.details["Software"] == "Midjourney"
    assert exif.details["detected_generators"] == ["Midjourney"]


def test_sdxl_watermark_bit_errors(images):
    result = check(images["sdxl"])
    finding = next(f for f in result.findings if f.source == "watermark")
    assert "SDXL" in finding.title
    assert finding.details["bit_errors"] <= finding.details["max_bit_errors"] == detectors.SDXL_MAX_BIT_ERRORS


def test_sd_v1_watermark():
    import cv2

    wm = detectors.load_watermark_module()
    encoder = wm.WatermarkEncoder()
    encoder.set_watermark("bytes", b"StableDiffusionV1")
    # 136 bits are repeated less often than 48 bits; dwtDct needs a larger image for that.
    bgr = cv2.cvtColor(np.asarray(mti.base_image(0, size=1536)), cv2.COLOR_RGB2BGR)
    buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(encoder.encode(bgr, "dwtDct"), cv2.COLOR_BGR2RGB)).save(buffer, "PNG")
    result = analyze(buffer.getvalue(), "sd1.png")
    titles = [f.title for f in result.findings]
    assert any(f.type == "sd_watermark" for f in result.findings), titles
    assert "Invisible Stable Diffusion watermark (v1/v2)" in titles
    assert "[KI] Unsichtbares Stable-Diffusion-Wasserzeichen (v1/v2)" in to_text(result)
    assert result.verdict == VERDICT_AI


def test_small_images_skip_watermark_check():
    buffer = io.BytesIO()
    mti.base_image(8, size=200).save(buffer, "PNG")
    ctx = detectors.Context(buffer.getvalue(), Image.open(buffer), "PNG", "image/png")
    assert detectors.check_watermarks(ctx) == []


# ---------------------------------------------------------------------------
# Individual functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Created with DALL·E 3", "DALL·E"),
        ("Adobe Photoshop 25.0 (Generative Fill)", "Generative Fill"),
        ("Made with Google AI", "Made with Google AI"),
        ("FLUX.1 [dev]", "FLUX"),
        ("Leonardo.Ai", "Leonardo.Ai"),
        ("ComfyUI", "ComfyUI"),
        ("Google Imagen 3", "Imagen"),
        ("Bing Image Creator", "Bing Image Creator"),
    ],
)
def test_generator_detection(text, expected):
    assert expected in detect_generators(text)


@pytest.mark.parametrize(
    "text", ["Adobe Photoshop Lightroom", "Canon EOS R5", "una imagen de la playa", "Grokking", "midjourneys"]
)
def test_no_generators(text):
    assert detect_generators(text) == []


def test_truncate():
    assert truncate("abc") == "abc"
    shortened = truncate("x" * 700, 500)
    assert shortened.startswith("x" * 500) and shortened.endswith("[200 characters truncated]")


def test_user_comment_decoding():
    assert decode_user_comment(b"ASCII\x00\x00\x00Hallo") == "Hallo"
    assert decode_user_comment(b"UNICODE\x00" + "Grüße".encode("utf-16-le"), big_endian=False) == "Grüße"
    assert decode_user_comment(b"UNICODE\x00" + "Steps: 20".encode("utf-16-be")) == "Steps: 20"
    assert decode_user_comment(b"\x00" * 8 + "ÄÖÜ".encode()) == "ÄÖÜ"


def test_xmp_notations():
    packet = """<x:xmpmeta><rdf:Description xmp:CreatorTool="Tool A">
      <dc:creator><rdf:Seq><rdf:li>Anna</rdf:li><rdf:li>Ben</rdf:li></rdf:Seq></dc:creator>
      <dc:description><rdf:Alt><rdf:li xml:lang="x-default">Ein &amp; Bild</rdf:li></rdf:Alt></dc:description>
      <Iptc4xmpExt:AISystemUsed>Midjourney</Iptc4xmpExt:AISystemUsed>
    </rdf:Description></x:xmpmeta>"""
    assert read_xmp_field(packet, "xmp:CreatorTool") == ["Tool A"]
    assert read_xmp_field(packet, "dc:creator") == ["Anna", "Ben"]
    assert read_xmp_field(packet, "dc:description") == ["Ein & Bild"]
    buffer = io.BytesIO()
    mti.base_image(9, 64).save(buffer, "JPEG", xmp=packet.encode())
    result = analyze(buffer.getvalue(), "x.jpg")
    xmp = next(f for f in result.findings if f.source == "xmp")
    assert xmp.level == LEVEL_AI


def test_algorithmic_media_in_c2pa_is_hint():
    finding = detectors._evaluate_manifest(
        "m",
        {
            "assertions": [
                {
                    "label": "c2pa.actions.v2",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "digitalSourceType": detectors.IPTC_URI + "algorithmicMedia",
                                "softwareAgent": {"name": "Tool"},
                            }
                        ]
                    },
                }
            ]
        },
        active=True,
    )
    assert finding.level == LEVEL_HINT


def test_c2pa_generator_at_least_hint():
    finding = detectors._evaluate_manifest(
        "m", {"claim_generator_info": [{"name": "Adobe Firefly", "version": "3"}]}, active=True
    )
    assert finding.level == LEVEL_HINT
    finding = detectors._evaluate_manifest("m", {"claim_generator": "Adobe Lightroom"}, active=True)
    assert finding.level == LEVEL_INFO


def test_c2pa_fallback_for_broken_manifest():
    data = b"\xff\xd8\xff\xe0" + b"\x00" * 32 + b"jumbxxxxc2pa" + b"\x00" * 32 + b"\xff\xd9"
    findings = detectors.check_c2pa(detectors.Context(data, None, "JPEG", "image/jpeg"))
    assert len(findings) == 1 and findings[0].level == LEVEL_HINT
    assert findings[0].type == "c2pa_unreadable"
    assert findings[0].title == "C2PA manifest detected but unreadable/invalid"


def test_png_comfyui_and_novelai():
    info = PngInfo()
    info.add_text("prompt", json.dumps({"4": {"inputs": {"ckpt_name": "flux1-dev.safetensors"}}}))
    info.add_text("Software", "NovelAI")
    info.add_text("Comment", "{}")
    result = analyze(png_bytes(10, info=info), "x.png")
    titles = [f.title for f in result.findings if f.level == LEVEL_AI]
    assert any("ComfyUI" in t and "flux1-dev" in t for t in titles)
    assert any("NovelAI" in t for t in titles)


def test_failing_detector_does_not_block_others(images, monkeypatch):
    def broken(ctx):
        raise RuntimeError("intentional")

    monkeypatch.setattr(detectors, "DETECTORS", [("Broken", broken), *detectors.DETECTORS])
    result = check(images["a1111"])
    assert result.verdict == VERDICT_AI
    assert [asdict(e) for e in result.errors] == [
        {
            "code": "detector_failed",
            "message": "Detector 'Broken' failed",
            "detector": "Broken",
            "exception": "RuntimeError: intentional",
        }
    ]
    assert "Prüfer Broken: RuntimeError: intentional" in to_text(result)


def test_not_an_image():
    result = analyze(b"this is not an image", "text.txt")
    assert result.verdict == VERDICT_NONE
    assert [e.code for e in result.errors] == ["image_decode_failed"]
    assert result.errors[0].message.startswith("Image could not be decoded")
    assert "Bild konnte nicht dekodiert werden: UnidentifiedImageError" in to_text(result)


def test_report(images):
    text = to_text(check(images["iptc"]))
    assert "Datei:    iptc_google.jpg" in text
    assert "Ergebnis: KI-Kennzeichnung gefunden" in text
    assert "[KI] IPTC Digital Source Type: trainedAlgorithmicMedia" in text
    assert "[HINWEIS]" in text
    assert "SynthID" in text and "Screenshot" in text
    assert DISCLAIMER.split()[0] in text


def test_report_uses_german_labels(images, c2pa_image):
    text = to_text(check(c2pa_image))
    assert "Signiert von: C2PA Test Signing Cert" in text and "signed_by" not in text
    assert "Quelle: C2PA" in text
    assert "validation_state" not in text and "validation_codes" not in text
    assert "(signingCredential.untrusted)" in text
    text = to_text(check(images["sdxl"]))
    assert "Bitfehler: " in text and "Toleranz (max. Bitfehler): 4" in text
    assert "Quelle: Wasserzeichen" in text


def test_report_renders_structured_c2pa_details_in_german(c2pa_image):
    text = to_text(check(c2pa_image))
    assert "c2pa.created – Quelle: algorithmicMedia (Rein algorithmisch erzeugt" in text
    assert "Software: Make Test Images 0.26.0" in text
    assert "Validierung: Strukturell gültig" in text
    assert "Signaturzertifikat stammt nicht von einer vertrauenswürdigen Stelle" in text
    assert "(signingCredential.untrusted)" in text
    assert "Structurally valid" not in text and "trust list" not in text


def test_report_translates_truncation_and_iptc_meaning(images):
    result = check(images["iptc"])
    result.findings[0].details["raw_note"] = truncate("y" * 650)
    text = to_text(result)
    assert "Bedeutung: Mit einem trainierten KI-Modell erzeugt (generative KI)" in text
    flat = re.sub(r"\s+", " ", text)  # the note may be wrapped across lines
    assert "[150 Zeichen gekürzt]" in flat and "characters truncated" not in flat


def test_c2pa_unknown_source_type_and_fallback_explanation_in_report():
    finding = detectors._evaluate_manifest(
        "m",
        {"assertions": [{"label": "c2pa.actions", "data": {"actions": [
            {"action": "c2pa.created", "digitalSourceType": detectors.IPTC_URI + "somethingNew"}
        ]}}]},
        active=True,
    )
    assert finding.details["source_types"] == [{"type": "somethingNew", "meaning": None}]
    result = detectors.Result(file="x", findings=[finding])
    assert "somethingNew: unbekannter Wert" in to_text(result)

    data = b"\xff\xd8\xff\xe0" + b"jumbxxxxc2pa" + b"\xff\xd9"
    broken = detectors.check_c2pa(detectors.Context(data, None, "JPEG", "image/jpeg"))[0]
    assert broken.details["explanation"].startswith("The file contains")
    assert "Die Datei enthält JUMBF" in to_text(detectors.Result(file="x", findings=[broken]))


def test_titles_and_errors_are_english_report_is_german(images, c2pa_image):
    german = re.compile(r"[äöüÄÖÜß]|\b(Metadaten|Wasserzeichen|Generierungs\w*|Modell|nennen|erzeugt)\b")
    expected_de = {
        "a1111": "[KI] Generierungsparameter von AUTOMATIC1111 / Forge (Modell: sd_xl_base_1.0)",
        "iptc": "[HINWEIS] XMP-Metadaten nennen einen KI-Generator",
        "exif": "[HINWEIS] EXIF-Metadaten nennen einen KI-Generator",
        "sdxl": "[KI] Unsichtbares SDXL-Wasserzeichen",
    }
    for name, path in [*images.items(), ("c2pa", c2pa_image)]:
        result = check(path)
        for finding in result.findings:
            assert finding.type and not german.search(finding.title), (name, finding.title)
        if name in expected_de:
            assert expected_de[name] in to_text(result)
    text = to_text(check(c2pa_image))
    assert "[HINWEIS] C2PA Content Credentials (aktives Manifest): algorithmicMedia" in text


def test_api_errors_are_structured(client):
    data = client.post("/analyze", files={"file": ("x.txt", b"no image", "text/plain")}).json()
    assert data["errors"][0]["code"] == "image_decode_failed"
    assert set(data["errors"][0]) == {"code", "message", "detector", "exception"}
    assert "Bild konnte nicht dekodiert werden" in data["report"]


def test_detail_values_are_english(images, c2pa_image):
    german = re.compile(r"[äöüÄÖÜß]|\b(Quelle|Beschreibung|Zeichen|gekürzt|unbekannt|Signatur)\b")
    for path in [*images.values(), c2pa_image]:
        for finding in check(path).findings:
            dumped = json.dumps(finding.details, ensure_ascii=False)
            assert not german.search(dumped), (path.name, finding.source, dumped)


def test_detail_keys_are_english(images, c2pa_image):
    raw_metadata_keys = set(detectors.XMP_FIELDS) | set(detectors.EXIF_TAGS.values()) | {"UserComment"}
    for path in [*images.values(), c2pa_image]:
        for finding in check(path).findings:
            if finding.source == "png_text" and finding.level == LEVEL_INFO:
                continue  # original PNG chunk names
            for key in finding.details:
                assert key in raw_metadata_keys or re.fullmatch(r"[a-z][a-z0-9_]*", key), (path.name, key)


def test_report_multiline_indentation(images):
    lines = to_text(check(images["a1111"])).splitlines()
    i = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Rohdaten:"))
    assert lines[i + 1].startswith(" " * len("    Rohdaten: "))


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    for name in ("API_KEY", "API_KEY_FILE", "SESSION_HOURS", "COOKIE_SECURE"):
        monkeypatch.delenv(name, raising=False)
    return TestClient(app)


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_api_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'name="file"' in response.text and "prefers-color-scheme: dark" in response.text
    assert 'id="dropzone"' in response.text and '"drop"' in response.text


def test_api_analyze_json(client, images):
    with open(images["a1111"], "rb") as f:
        response = client.post("/analyze", files={"file": ("a1111.png", f, "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "AI label found" and data["verdict_level"] == "ai"
    assert "Ergebnis: KI-Kennzeichnung gefunden" in data["report"]
    assert data["size"] == {"width": 512, "height": 512}
    assert "[KI]" in data["report"]
    assert data["findings"][0]["level"] == LEVEL_AI
    assert data["findings"][0]["source"] == "png_text"
    assert list(data["findings"][0]) == ["type", "source", "level", "title", "details"]
    assert data["findings"][0]["type"] == "generation_parameters"
    assert data["findings"][0]["title"] == "Generation parameters from AUTOMATIC1111 / Forge (model: sd_xl_base_1.0)"


def test_api_analyze_text(client, images):
    with open(images["exif"], "rb") as f:
        response = client.post("/analyze?format=text", files={"file": ("e.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "Ergebnis: Hinweise auf KI-Erzeugung gefunden (keine standardisierte Kennzeichnung)" in response.text


def test_api_invalid_format(client, images):
    with open(images["exif"], "rb") as f:
        response = client.post("/analyze?format=xml", files={"file": ("e.jpg", f, "image/jpeg")})
    assert response.status_code == 422


def test_api_check_html_escaped(client):
    info = PngInfo()
    info.add_text("Comment", "<script>alert(1)</script>")
    response = client.post("/check", files={"file": ("<b>x</b>.png", png_bytes(11, info=info), "image/png")})
    assert response.status_code == 200
    assert "<script>alert" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert "<pre>" in response.text


def test_api_upload_limit(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    large = b"\x00" * (1024 * 1024 + 10)
    response = client.post("/analyze", files={"file": ("large.bin", large, "application/octet-stream")})
    assert response.status_code == 413
    assert response.json()["detail"] == "File too large. The maximum is 1 MB (MAX_UPLOAD_MB)."
    response = client.post("/check", files={"file": ("large.bin", large, "application/octet-stream")})
    assert response.status_code == 413
    small = client.post("/analyze", files={"file": ("small.bin", b"\x00" * 1000, "application/octet-stream")})
    assert small.status_code == 200


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_exit_codes(images, capsys):
    assert cli_main([str(images["unmarked"]), str(images["exif"])]) == 0
    assert "Ergebnis:" in capsys.readouterr().out
    assert cli_main([str(images["unmarked"]), str(images["a1111"]), "--json"]) == 2
    data = json.loads(capsys.readouterr().out)
    assert [d["verdict_level"] for d in data] == ["none", "ai"]
    assert cli_main(["/does/not/exist.jpg"]) == 1


# ---------------------------------------------------------------------------
# Access protection (API_KEY)
# ---------------------------------------------------------------------------

KEY = "secret-123"


@pytest.fixture
def protected(client, monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.setattr(main_module, "FAILED_LOGIN_DELAY", 0)
    return client


def session_cookie(client):
    return client.cookies.get(auth.SESSION_COOKIE)


def test_no_api_key_no_protection(client):
    assert 'id="dropzone"' in client.get("/").text
    assert "Abmelden" not in client.get("/").text
    assert client.post("/analyze", files={"file": ("x.png", png_bytes(), "image/png")}).status_code == 200


def test_api_without_and_with_wrong_key(protected):
    response = protected.post("/analyze", files={"file": ("x.png", png_bytes(), "image/png")})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    response = protected.post(
        "/analyze", files={"file": ("x.png", png_bytes(), "image/png")}, headers={"X-API-Key": "wrong"}
    )
    assert response.status_code == 401


@pytest.mark.parametrize("header", [{"X-API-Key": KEY}, {"Authorization": f"Bearer {KEY}"}])
def test_api_with_key(protected, header):
    response = protected.post(
        "/analyze?format=text", files={"file": ("x.png", png_bytes(), "image/png")}, headers=header
    )
    assert response.status_code == 200 and "Ergebnis:" in response.text
    response = protected.post("/check", files={"file": ("x.png", png_bytes(), "image/png")}, headers=header)
    assert response.status_code == 200 and "Prüfbericht" in response.text


def test_health_and_docs_stay_open(protected):
    assert protected.get("/health").status_code == 200
    schema = protected.get("/openapi.json").json()
    assert {"APIKeyHeader", "HTTPBearer"} <= schema["components"]["securitySchemes"].keys()


def test_key_checked_before_upload(protected, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    large = b"\x00" * (2 * 1024 * 1024)
    response = protected.post("/analyze", files={"file": ("l.bin", large, "application/octet-stream")})
    assert response.status_code == 401  # not 413: the upload was never processed


def test_frontend_login(protected):
    index = protected.get("/")
    assert 'action="/login"' in index.text and 'id="dropzone"' not in index.text

    response = protected.post("/check", files={"file": ("x.png", png_bytes(), "image/png")})
    assert response.status_code == 401 and 'action="/login"' in response.text

    wrong = protected.post("/login", data={"key": "wrong"}, follow_redirects=False)
    assert wrong.status_code == 401 and "ungültig" in wrong.text
    assert session_cookie(protected) is None

    right = protected.post("/login", data={"key": KEY}, follow_redirects=False)
    assert right.status_code == 303 and right.headers["location"] == "/?lang=de"
    set_cookie = right.headers["set-cookie"].lower()
    assert "httponly" in set_cookie and "samesite=lax" in set_cookie
    token = session_cookie(protected)
    assert token and KEY not in token

    index = protected.get("/")
    assert 'id="dropzone"' in index.text and "Abmelden" in index.text
    response = protected.post("/check", files={"file": ("x.png", png_bytes(), "image/png")})
    assert response.status_code == 200 and "Prüfbericht" in response.text

    protected.post("/logout", follow_redirects=False)
    assert session_cookie(protected) is None
    assert 'action="/login"' in protected.get("/").text


def test_session_expiry_and_key_rotation(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.setenv("SESSION_HOURS", "1")
    token = auth.create_session(KEY, now=1_000_000)
    assert auth.session_valid(token, now=1_000_000 + 3500)
    assert not auth.session_valid(token, now=1_000_000 + 3700)
    timestamp, _, signature = token.partition(".")
    assert not auth.session_valid(f"{int(timestamp) + 1}.{signature}", now=1_000_000)  # tampered time
    assert not auth.session_valid("broken", now=1_000_000)
    monkeypatch.setenv("API_KEY", "new-key")
    assert not auth.session_valid(token, now=1_000_000 + 10)


def test_multiple_keys_and_file(tmp_path, monkeypatch):
    path = tmp_path / "keys"
    path.write_text("from-file\n\n  second  \n", encoding="utf-8")
    monkeypatch.setenv("API_KEY", "a, b")
    monkeypatch.setenv("API_KEY_FILE", str(path))
    assert auth.api_keys() == ["a", "b", "from-file", "second"]
    assert auth.key_valid("second") and not auth.key_valid("c")


def test_unreadable_key_file_locks(monkeypatch, tmp_path):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("API_KEY_FILE", str(tmp_path / "missing"))
    assert auth.protection_enabled()  # fail closed
    with pytest.raises(auth.ConfigurationError):
        with TestClient(app):  # startup aborts
            pass
    client = TestClient(app)
    assert client.post("/analyze", files={"file": ("x.png", png_bytes(), "image/png")}).status_code == 401


# ---------------------------------------------------------------------------
# Languages (web UI, report)
# ---------------------------------------------------------------------------

GERMAN_WORDS = re.compile(r"[äöüÄÖÜß]|\b(Datei|Ergebnis|Quelle|Bild|prüfen|Hinweis|Funde|Fehler)\b")


def test_english_report_has_no_german(images, c2pa_image):
    for path in [*images.values(), c2pa_image]:
        text = to_text(check(path), "en")
        assert not GERMAN_WORDS.search(text), (path.name, GERMAN_WORDS.search(text))
        assert "Verdict: " in text and "Note: A missing label" in text
    text = to_text(check(images["sdxl"]), "en")
    assert "[AI] Invisible SDXL watermark" in text and "Source: Watermark" in text
    assert "Image could not be decoded: UnidentifiedImageError" in to_text(analyze(b"x", "x"), "en")


def test_report_unknown_language_falls_back_to_german(images):
    assert to_text(check(images["exif"]), "fr") == to_text(check(images["exif"]))


def test_ui_default_german_and_accept_language(client):
    page = client.get("/").text
    assert '<html lang="de">' in page and "Bild prüfen" in page
    page = client.get("/", headers={"Accept-Language": "en-GB,en;q=0.9,de;q=0.5"}).text
    assert '<html lang="en">' in page and "Check image" in page
    page = client.get("/", headers={"Accept-Language": "fr-FR, de;q=0.8"}).text
    assert '<html lang="de">' in page


def test_ui_language_switch_sets_cookie(client):
    response = client.get("/?lang=en")
    assert '<html lang="en">' in response.text
    assert "Drag an image here" in response.text and '"locale": "en-US"' in response.text
    assert '<span aria-current="true" lang="en">EN</span>' in response.text
    assert '<a href="/?lang=de"' in response.text
    assert "ui_lang=en" in response.headers["set-cookie"]
    # cookie is remembered, explicit choice beats Accept-Language
    assert '<html lang="en">' in client.get("/", headers={"Accept-Language": "de"}).text
    assert '<html lang="de">' in client.get("/?lang=de").text
    assert '<html lang="de">' in client.get("/").text
    assert "set-cookie" not in client.get("/?lang=xx").headers


def test_ui_check_report_in_selected_language(client, images):
    with open(images["exif"], "rb") as f:
        response = client.post("/check", files={"file": ("e.jpg", f, "image/jpeg")}, data={"lang": "en"})
    assert '<html lang="en">' in response.text
    assert "<h1>Report</h1>" in response.text and "Verdict: Indications of AI generation" in response.text
    assert "Check another image" in response.text
    with open(images["exif"], "rb") as f:
        response = client.post("/check", files={"file": ("e.jpg", f, "image/jpeg")})
    assert "<h1>Prüfbericht</h1>" in response.text and "Ergebnis: Hinweise" in response.text


def test_ui_too_large_message_localised(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    large = b"\x00" * (1024 * 1024 + 10)
    response = client.post("/check", files={"file": ("l.bin", large, "application/octet-stream")}, data={"lang": "en"})
    assert response.status_code == 413 and "File too large. The maximum is 1 MB" in response.text
    response = client.post("/check", files={"file": ("l.bin", large, "application/octet-stream")})
    assert response.status_code == 413 and "Datei zu groß" in response.text


def test_api_report_language(client, images):
    with open(images["a1111"], "rb") as f:
        data = client.post("/analyze?lang=en", files={"file": ("a.png", f, "image/png")}).json()
    assert "\nAI label check (Art. 50(2) EU AI Act)\n" in data["report"]
    assert "[AI] Generation parameters" in data["report"]
    with open(images["a1111"], "rb") as f:
        text = client.post("/analyze?format=text&lang=en", files={"file": ("a.png", f, "image/png")}).text
    assert "Verdict: AI label found" in text
    with open(images["a1111"], "rb") as f:
        assert client.post("/analyze?lang=fr", files={"file": ("a.png", f, "image/png")}).status_code == 422


def test_login_flow_in_english(protected):
    page = protected.get("/?lang=en").text
    assert "<h1>Log in</h1>" in page and 'name="lang" value="en"' in page
    wrong = protected.post("/login", data={"key": "wrong", "lang": "en"}, follow_redirects=False)
    assert wrong.status_code == 401 and "The key is invalid." in wrong.text
    right = protected.post("/login", data={"key": KEY, "lang": "en"}, follow_redirects=False)
    assert right.headers["location"] == "/?lang=en"
    page = protected.get("/?lang=en").text
    assert "Log out" in page and 'id="dropzone"' in page
    response = protected.post("/logout", data={"lang": "en"}, follow_redirects=False)
    assert response.headers["location"] == "/?lang=en"
    expired = protected.post("/check", files={"file": ("x.png", png_bytes(), "image/png")})
    assert expired.status_code == 401 and "Not logged in or session expired" in expired.text


def test_cli_language(images, capsys):
    cli_main([str(images["exif"]), "--lang", "en"])
    assert "Verdict: Indications of AI generation" in capsys.readouterr().out


def test_check_result_contains_both_languages_for_in_place_switch(client, images):
    with open(images["sdxl"], "rb") as f:
        page = client.post("/check", files={"file": ("s.png", f, "image/png")}, data={"lang": "en"}).text
    de_start, en_start = page.index('data-lang-block="de"'), page.index('data-lang-block="en"')
    de_block = page[de_start:en_start]
    en_block = page[en_start:]
    assert '<html lang="en">' in page
    assert " hidden>" in de_block[:200] and " hidden>" not in en_block[:200]
    assert "[KI] Unsichtbares SDXL-Wasserzeichen" in de_block and "Weiteres Bild prüfen" in de_block
    assert "[AI] Invisible SDXL watermark" in en_block and "Check another image" in en_block
    assert 'data-switch-lang="en"' in de_block and 'data-switch-lang="de"' in en_block
    assert 'href="/?lang=de"' in en_block  # fallback without JavaScript
    assert "document.cookie" in page
    # the start page keeps the normal links (no in-place switch needed)
    assert "data-switch-lang" not in client.get("/").text


def test_report_lines_fit_web_ui_box(images, c2pa_image):
    from app.report import LINE_WIDTH

    for path in [*images.values(), c2pa_image]:
        for lang in ("de", "en"):
            longest = max(to_text(check(path), lang).splitlines(), key=len)
            assert len(longest) <= LINE_WIDTH, (path.name, lang, longest)


# ---------------------------------------------------------------------------
# Labelling recommendation (EU AI Act, Art. 50)
# ---------------------------------------------------------------------------

from app import recommendation as rec  # noqa: E402


@pytest.mark.parametrize(
    ("name", "category", "confidence"),
    [
        ("a1111", "ai_generated", "high"),
        ("iptc", "ai_generated", "high"),
        ("sdxl", "ai_generated", "high"),
        ("exif", "ai_generated", "medium"),  # only a generator name in EXIF
        ("unmarked", None, None),
    ],
)
def test_recommendation_for_test_images(images, name, category, confidence):
    recommendation = check(images[name]).recommendation
    if category is None:
        assert recommendation is None
    else:
        assert (recommendation.category, recommendation.confidence) == (category, confidence)
        assert recommendation.labelling == "recommended"


def test_no_recommendation_for_algorithmic_media(c2pa_image):
    # algorithmicMedia = procedural, no trained model -> no AI involvement evidenced
    assert check(c2pa_image).recommendation is None


def _c2pa_result(*actions, generators=None):
    details = {"actions": [dict(a) for a in actions]}
    if generators:
        details["detected_generators"] = generators
    finding = detectors.Finding("c2pa_active_manifest", "c2pa", "ai", "t", details)
    return detectors.Result(file="x", findings=[finding])


def test_c2pa_created_by_ai_and_edited_is_generated():
    # e.g. Google: c2pa.created + c2pa.edited (SynthID), both trainedAlgorithmicMedia
    result = _c2pa_result(
        {"action": "c2pa.created", "digital_source_type": "trainedAlgorithmicMedia"},
        {"action": "c2pa.edited", "digital_source_type": "trainedAlgorithmicMedia"},
    )
    recommendation = rec.recommend(result)
    assert recommendation.category == "ai_generated" and recommendation.confidence == "high"
    assert recommendation.evidence[0].value == "c2pa.created (trainedAlgorithmicMedia)"


def test_c2pa_photo_edited_with_ai_is_modified():
    result = _c2pa_result(
        {"action": "c2pa.created", "digital_source_type": "digitalCapture"},
        {"action": "c2pa.edited", "digital_source_type": "trainedAlgorithmicMedia"},
    )
    assert rec.recommend(result).category == "ai_modified"


def test_composite_with_ai_beats_weak_generator_name():
    # Photoshop Generative Fill: composite source type; "Adobe Firefly" only as a name
    result = _c2pa_result(
        {"action": "c2pa.edited", "digital_source_type": "compositeWithTrainedAlgorithmicMedia"},
        generators=["Adobe Firefly"],
    )
    recommendation = rec.recommend(result)
    assert (recommendation.category, recommendation.confidence) == ("ai_modified", "high")


def test_generative_fill_name_only_is_modified_medium():
    finding = detectors.Finding("xmp_generator", "xmp", "hint", "t", {"detected_generators": ["Generative Fill"]})
    recommendation = rec.recommend(detectors.Result(file="x", findings=[finding]))
    assert (recommendation.category, recommendation.confidence) == ("ai_modified", "medium")


def test_algorithmically_enhanced_is_assisted_and_optional():
    finding = detectors.Finding(
        "iptc_source_type", "iptc_xmp", "hint", "t", {"value": "algorithmicallyEnhanced"}
    )
    result = detectors.Result(file="x", findings=[finding])
    recommendation = rec.recommend(result)
    assert (recommendation.category, recommendation.labelling) == ("ai_assisted", "optional")
    result.recommendation = recommendation
    text = to_text(result)
    assert "KI-unterstützt (AI-assisted)" in text and "voraussichtlich nicht erforderlich" in text


def test_recommendation_in_api_and_reports(client, images):
    with open(images["a1111"], "rb") as f:
        data = client.post("/analyze", files={"file": ("a.png", f, "image/png")}).json()
    r = data["ai_label_recommendation"]
    assert list(data).index("ai_label_recommendation") == list(data).index("verdict_level") + 1
    assert r["category"] == "ai_generated" and r["label"] == "AI-generated"
    assert r["labelling"] == "recommended" and r["confidence"] == "high"
    assert r["evidence"][0]["kind"] == "generation_data" and r["evidence"][0]["finding_type"] == "generation_parameters"
    assert r["legal_basis"] == "Art. 50(2) and (4) EU AI Act" and "not legal advice" in r["note"]
    assert "Einstufung: KI-generiert (AI-generated)" in data["report"]
    with open(images["unmarked"], "rb") as f:
        data = client.post("/analyze?lang=en", files={"file": ("u.jpg", f, "image/jpeg")}).json()
    assert data["ai_label_recommendation"] is None and "Labelling recommendation" not in data["report"]


def test_recommendation_box_on_result_page(client, images):
    with open(images["exif"], "rb") as f:
        page = client.post("/check", files={"file": ("e.jpg", f, "image/jpeg")}).text
    assert page.count('class="recommendation ai_generated"') == 2  # one per language block
    assert "Empfehlung nach KI-Verordnung" in page and "Recommendation under the EU AI Act" in page
    assert "Sicherheit: mittel (nur Indizien)" in page and "Confidence: medium (indications only)" in page
    with open(images["unmarked"], "rb") as f:
        page = client.post("/check", files={"file": ("u.jpg", f, "image/jpeg")}).text
    assert 'class="recommendation' not in page


def test_badge_in_api_text_output(client, images):
    with open(images["a1111"], "rb") as f:
        text = client.post("/analyze?format=text", files={"file": ("a.png", f, "image/png")}).text
    lines = text.splitlines()
    assert lines[0].startswith("┌") and lines[3].startswith("└")
    assert lines[1] == f"│ {'KI-GENERIERT (AI-GENERATED)':<{len(lines[1]) - 4}} │"
    assert "Kennzeichnung: empfohlen · Sicherheit: hoch" in lines[2]
    assert len({len(line) for line in lines[:4]}) == 1  # frame is aligned
    assert lines[5].startswith("KI-Kennzeichnungsprüfung")
    with open(images["a1111"], "rb") as f:
        data = client.post("/analyze?lang=en", files={"file": ("a.png", f, "image/png")}).json()
    assert data["report"].startswith("┌") and "│ AI-GENERATED" in data["report"]
    assert "Labelling: recommended · Confidence: high" in data["report"]


def test_badge_wraps_long_text_and_is_missing_without_recommendation(images):
    from app.report import LINE_WIDTH, badge_lines

    assisted = rec.Recommendation("ai_assisted", "AI-assisted", "optional", "high")
    for lang in ("de", "en"):
        lines = badge_lines(assisted, lang)
        assert len(lines) > 4 and all(len(line) <= LINE_WIDTH for line in lines)
        assert len({len(line) for line in lines}) == 1
    assert not to_text(check(images["unmarked"])).startswith("┌")


def test_no_text_badge_on_web_result_page(client, images):
    with open(images["a1111"], "rb") as f:
        page = client.post("/check", files={"file": ("a.png", f, "image/png")}).text
    assert "┌" not in page and 'class="recommendation ai_generated"' in page


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import base64  # noqa: E402

from mcp import Client  # noqa: E402

from app import mcp_server  # noqa: E402


def _call(server, name, arguments):
    async def run():
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def _tool_names(server):
    async def run():
        async with Client(server) as client:
            return [t.name for t in (await client.list_tools()).tools]

    return asyncio.run(run())


def test_mcp_tools_without_image_dir(monkeypatch):
    monkeypatch.delenv("MCP_IMAGE_DIR", raising=False)
    assert _tool_names(mcp_server.build_server()) == ["analyze_image"]


def test_mcp_analyze_image_base64(images):
    data = base64.b64encode(images["a1111"].read_bytes()).decode()
    result = _call(mcp_server.build_server(), "analyze_image", {"image_base64": data, "filename": "a.png", "lang": "en"})
    assert not result.is_error
    content = result.structured_content
    assert content["verdict_level"] == "ai"
    assert content["ai_label_recommendation"]["category"] == "ai_generated"
    assert content["report"].startswith("┌") and "Verdict: AI label found" in content["report"]
    # data: URL prefix is accepted too
    result = _call(mcp_server.build_server(), "analyze_image", {"image_base64": "data:image/png;base64," + data})
    assert result.structured_content["verdict_level"] == "ai"


def test_mcp_analyze_image_errors(monkeypatch):
    result = _call(mcp_server.build_server(), "analyze_image", {"image_base64": "!!!notbase64"})
    assert result.is_error and "Invalid base64 data" in result.content[0].text
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")
    result = _call(mcp_server.build_server(), "analyze_image", {"image_base64": "A" * 4000})
    assert result.is_error and "File too large" in result.content[0].text


def test_mcp_image_dir_tools(images, monkeypatch):
    root = images["a1111"].parent
    monkeypatch.setenv("MCP_IMAGE_DIR", str(root))
    server = mcp_server.build_server()
    assert _tool_names(server) == ["analyze_image", "list_image_files", "analyze_image_file"]
    listing = _call(server, "list_image_files", {}).structured_content
    assert "exif_midjourney.jpg" in listing["files"] and listing["truncated"] is False
    result = _call(server, "analyze_image_file", {"path": "exif_midjourney.jpg"}).structured_content
    assert result["verdict_level"] == "hint" and result["file"] == "exif_midjourney.jpg"
    assert "Ergebnis: Hinweise" in result["report"]


@pytest.mark.parametrize("path", ["../outside.jpg", "/etc/passwd", "missing.jpg"])
def test_mcp_image_dir_refuses_paths(images, monkeypatch, path, tmp_path):
    root = images["a1111"].parent
    (root.parent / "outside.jpg").write_bytes(images["unmarked"].read_bytes())
    monkeypatch.setenv("MCP_IMAGE_DIR", str(root))
    result = _call(mcp_server.build_server(), "analyze_image_file", {"path": path})
    assert result.is_error
    assert ("outside the image directory" in result.content[0].text) or ("not found" in result.content[0].text)


def test_mcp_symlink_out_of_image_dir_is_refused(images, monkeypatch, tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    secret = tmp_path / "secret.png"
    secret.write_bytes(images["a1111"].read_bytes())
    (root / "link.png").symlink_to(secret)
    monkeypatch.setenv("MCP_IMAGE_DIR", str(root))
    result = _call(mcp_server.build_server(), "analyze_image_file", {"path": "link.png"})
    assert result.is_error and "outside the image directory" in result.content[0].text


def test_mcp_transport_security_settings(monkeypatch):
    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)
    settings = mcp_server.transport_security()
    assert settings.enable_dns_rebinding_protection and "localhost:*" in settings.allowed_hosts
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "checker.example.com, localhost:*")
    settings = mcp_server.transport_security()
    assert settings.allowed_hosts == ["checker.example.com", "localhost:*"]
    assert "https://checker.example.com" in settings.allowed_origins
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "*")
    assert not mcp_server.transport_security().enable_dns_rebinding_protection


@pytest.fixture(scope="module")
def live_server():
    """Runs the real app (incl. /mcp) with uvicorn on a free port."""
    import socket
    import threading
    import time as _time

    import uvicorn

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        _time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_mcp_over_http_with_api_key(live_server, images, monkeypatch):
    import httpx2

    from mcp.client.streamable_http import streamable_http_client

    monkeypatch.setenv("API_KEY", KEY)
    url = f"{live_server.replace('127.0.0.1', 'localhost')}/mcp"
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    assert httpx2.post(url, json=payload).status_code == 401  # no key → rejected before MCP

    async def run():
        async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {KEY}"}, timeout=60) as http:
            async with Client(streamable_http_client(url, http_client=http)) as client:
                names = [t.name for t in (await client.list_tools()).tools]
                data = base64.b64encode(images["sdxl"].read_bytes()).decode()
                result = await client.call_tool("analyze_image", {"image_base64": data, "filename": "s.png"})
                return names, result

    names, result = asyncio.run(run())
    assert "analyze_image" in names
    assert not result.is_error and result.structured_content["verdict_level"] == "ai"


def test_mcp_http_rejects_foreign_host_header(live_server, monkeypatch):
    import httpx2

    monkeypatch.delenv("API_KEY", raising=False)
    response = httpx2.post(
        f"{live_server}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        headers={"Host": "evil.example.com", "Accept": "application/json, text/event-stream"},
    )
    assert response.status_code in (400, 421)  # DNS rebinding protection
