"""Detectors for machine-readable AI labels in image files.

Each detector receives a ``Context`` (raw bytes, opened Pillow image if decodable,
MIME type) and returns a list of ``Finding`` objects. ``analyze`` runs all detectors
independently of each other.

Everything in ``Result`` is English (verdict, finding ``type``/``source``/``level``/
``title``, ``details`` keys and values, ``errors``). Original metadata field names and
values from the file (e.g. ``photoshop:Credit``) are kept as-is. ``report`` renders the German plain-text
report from these English data.
"""

from __future__ import annotations

import html
import io
import json
import re
import sys
import types
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Callable

from PIL import Image

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

LEVEL_AI = "ai"  # unambiguous, standardised AI label
LEVEL_HINT = "hint"  # points to AI generation, but not standardised
LEVEL_INFO = "info"  # provenance data without AI relation
LEVEL_RANK = {LEVEL_AI: 0, LEVEL_HINT: 1, LEVEL_INFO: 2}

# Finding sources (machine-readable; report.SOURCE_LABELS has the German labels)
SOURCE_C2PA = "c2pa"
SOURCE_IPTC = "iptc_xmp"
SOURCE_XMP = "xmp"
SOURCE_EXIF = "exif"
SOURCE_EXIF_USER_COMMENT = "exif_user_comment"
SOURCE_PNG_TEXT = "png_text"
SOURCE_WATERMARK = "watermark"

# English verdicts; German texts live in report.VERDICTS_DE
VERDICT_AI = "AI label found"
VERDICT_HINT = "Indications of AI generation found (no standardised label)"
VERDICT_NONE = "No machine-readable AI label found"


# Finding types (machine-readable; report.FINDING_TITLES_DE has the German titles)
TYPE_C2PA_ACTIVE = "c2pa_active_manifest"
TYPE_C2PA_INGREDIENT = "c2pa_ingredient_manifest"
TYPE_C2PA_UNREADABLE = "c2pa_unreadable"
TYPE_IPTC_SOURCE_TYPE = "iptc_source_type"
TYPE_XMP_AI_SYSTEM = "xmp_ai_system"
TYPE_XMP_GENERATOR = "xmp_generator"
TYPE_XMP_METADATA = "xmp_metadata"
TYPE_EXIF_GENERATOR = "exif_generator"
TYPE_EXIF_METADATA = "exif_metadata"
TYPE_GENERATION_PARAMETERS = "generation_parameters"
TYPE_COMFYUI_WORKFLOW = "comfyui_workflow"
TYPE_NOVELAI = "novelai"
TYPE_GENERATOR_METADATA = "generator_metadata"
TYPE_PNG_TEXT_GENERATOR = "png_text_generator"
TYPE_PNG_TEXT = "png_text"
TYPE_SD_WATERMARK = "sd_watermark"
TYPE_SDXL_WATERMARK = "sdxl_watermark"

# Error codes
ERROR_IMAGE_DECODE = "image_decode_failed"
ERROR_DETECTOR = "detector_failed"


@dataclass
class Finding:
    type: str
    source: str
    level: str
    title: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisError:
    code: str
    message: str
    detector: str | None = None  # source id of the failing detector
    exception: str | None = None  # "<ExceptionType>: <text>"


@dataclass
class Result:
    file: str
    format: str | None = None
    size: tuple[int, int] | None = None  # (width, height) in pixels
    findings: list[Finding] = field(default_factory=list)
    errors: list[AnalysisError] = field(default_factory=list)
    file_size_bytes: int = 0
    # recommendation.Recommendation or None (set by analyze; only if AI involvement is evidenced)
    recommendation: Any = None

    @property
    def verdict_level(self) -> str:
        """Machine-readable verdict: ``ai``, ``hint`` or ``none``."""
        levels = {f.level for f in self.findings}
        if LEVEL_AI in levels:
            return LEVEL_AI
        if LEVEL_HINT in levels:
            return LEVEL_HINT
        return "none"

    @property
    def verdict(self) -> str:
        return {LEVEL_AI: VERDICT_AI, LEVEL_HINT: VERDICT_HINT}.get(self.verdict_level, VERDICT_NONE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "format": self.format,
            "size": {"width": self.size[0], "height": self.size[1]} if self.size else None,
            "file_size_bytes": self.file_size_bytes,
            "verdict": self.verdict,
            "verdict_level": self.verdict_level,
            "ai_label_recommendation": self.recommendation.to_dict() if self.recommendation else None,
            "findings": [
                {"type": f.type, "source": f.source, "level": f.level, "title": f.title, "details": f.details}
                for f in self.findings
            ],
            "errors": [asdict(e) for e in self.errors],
        }


def highest_level(*levels: str | None) -> str:
    present = [lv for lv in levels if lv in LEVEL_RANK]
    if not present:
        return LEVEL_INFO
    return min(present, key=LEVEL_RANK.__getitem__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_VALUE_LENGTH = 500


def truncate(value: Any, length: int = MAX_VALUE_LENGTH) -> str:
    """Truncates long values and states how many characters were omitted."""
    text = str(value).strip()
    if len(text) <= length:
        return text
    return f"{text[:length].rstrip()} … [{len(text) - length} characters truncated]"


# Generator detection: (display name, regex). Word boundaries, case-insensitive.
# Ambiguous words (e.g. "Imagen" = Spanish for "image", "Runway") only match in
# unambiguous spellings or as a standalone value.
_GENERATOR_PATTERNS: list[tuple[str, str]] = [
    ("DALL·E", r"\bdall[\s·\-.]?e\b"),
    ("OpenAI", r"\bopen\s?ai\b"),
    ("ChatGPT", r"\bchat\s?gpt\b"),
    ("GPT-4o", r"\bgpt[\s-]?4o\b"),
    ("GPT Image", r"\bgpt[\s-]image\b"),
    ("Midjourney", r"\bmid[\s-]?journey\b"),
    ("Stable Diffusion", r"\bstable[\s_-]?diffusion\b"),
    ("Stability AI", r"\bstability[\s.]?ai\b"),
    ("SDXL", r"\bsd[\s_-]?xl\b"),
    ("ComfyUI", r"\bcomfy\s?ui\b"),
    ("A1111", r"\b(?:a1111|automatic1111)\b"),
    ("InvokeAI", r"\binvoke\s?ai\b"),
    ("Fooocus", r"\bfooocus\b"),
    ("NovelAI", r"\bnovel\s?ai\b"),
    ("Adobe Firefly", r"\b(?:adobe\s+)?firefly\b"),
    ("Generative Fill", r"\bgenerative[\s_-]+fill\b"),
    ("Imagen", r"\bgoogle\s+imagen\b|\bimagen[\s-]?\d\b|^\s*imagen\s*$"),
    ("Made with Google AI", r"\bmade\s+with\s+google\s+ai\b"),
    ("Gemini", r"\bgemini\b"),
    ("Nano Banana", r"\bnano[\s-]?banana\b"),
    ("Bing Image Creator", r"\bbing\s+image\s+creator\b"),
    ("Microsoft Designer", r"\bmicrosoft\s+designer\b"),
    ("Leonardo.Ai", r"\bleonardo[\s.]?ai\b"),
    ("Ideogram", r"\bideogram\b"),
    ("Black Forest Labs", r"\bblack\s+forest\s+labs\b"),
    ("FLUX", r"\bflux[.\s-]?[12](?:\.\d)?\b"),
    ("Runway", r"\brunway[\s-]?(?:ml|ai|gen[\s-]?\d)\b|^\s*runway\s*$"),
    ("Krea", r"\bkrea(?:\.ai|\s+ai)?\b"),
    ("Playground AI", r"\bplayground[\s.]?(?:ai|v\d)\b"),
    ("DreamStudio", r"\bdream\s?studio\b"),
    ("Recraft", r"\brecraft\b"),
    ("Grok", r"\bgrok\b"),
    ("Meta AI", r"\bmeta\s+ai\b|\bimagine\s+with\s+meta\b"),
]
_GENERATORS = [
    (name, re.compile(pattern, re.IGNORECASE | re.MULTILINE)) for name, pattern in _GENERATOR_PATTERNS
]


def detect_generators(*texts: Any) -> list[str]:
    """Returns the names of all AI generators mentioned in the texts (without duplicates)."""
    found: list[str] = []
    for text in texts:
        if text is None:
            continue
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False) if isinstance(text, (dict, list)) else str(text)
        for name, regex in _GENERATORS:
            if name not in found and regex.search(text):
                found.append(name)
    return found


# ---------------------------------------------------------------------------
# IPTC Digital Source Type – vocabulary
# ---------------------------------------------------------------------------

IPTC_URI = "http://cv.iptc.org/newscodes/digitalsourcetype/"

# name -> (level, English meaning); German texts live in report.IPTC_MEANINGS_DE
IPTC_SOURCE_TYPES: dict[str, tuple[str, str]] = {
    # ai
    "trainedAlgorithmicMedia": (LEVEL_AI, "Created using a trained AI model (generative AI)"),
    "compositeWithTrainedAlgorithmicMedia": (
        LEVEL_AI,
        "Composite containing elements created with generative AI (e.g. Generative Fill)",
    ),
    # hint
    "algorithmicMedia": (LEVEL_HINT, "Created purely algorithmically, without training data (e.g. procedural)"),
    "compositeSynthetic": (LEVEL_HINT, "Composite containing synthetic elements"),
    "algorithmicallyEnhanced": (LEVEL_HINT, "Algorithmically enhanced or modified"),
    "dataDrivenMedia": (LEVEL_HINT, "Data-driven media"),
    # info
    "digitalCapture": (LEVEL_INFO, "Digital capture (camera)"),
    "computationalCapture": (LEVEL_INFO, "Computational capture (e.g. smartphone HDR)"),
    "digitalArt": (LEVEL_INFO, "Digital art created by a human"),
    "virtualRecording": (LEVEL_INFO, "Recording of a virtual environment"),
    "composite": (LEVEL_INFO, "Composite of several elements"),
    "compositeCapture": (LEVEL_INFO, "Composite of several captures"),
    "screenCapture": (LEVEL_INFO, "Screen capture"),
    "negativeFilm": (LEVEL_INFO, "Digitised film negative"),
    "positiveFilm": (LEVEL_INFO, "Digitised slide/positive film"),
    "print": (LEVEL_INFO, "Digitised print"),
    "minorHumanEdits": (LEVEL_INFO, "Minor human edits"),
    "humanEdits": (LEVEL_INFO, "Human edits"),
}
_IPTC_LOWER = {k.lower(): k for k in IPTC_SOURCE_TYPES}


def iptc_source_type(value: str) -> tuple[str, str, str] | None:
    """Maps a value (URI or short form) to (name, level, English meaning)."""
    short = value.strip().rstrip("/").rsplit("/", 1)[-1]
    name = _IPTC_LOWER.get(short.lower())
    if name is None:
        return None
    level, description = IPTC_SOURCE_TYPES[name]
    return name, level, description


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

_MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "MPO": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "TIFF": "image/tiff",
    "GIF": "image/gif",
    "AVIF": "image/avif",
    "HEIF": "image/heif",
}


def sniff_format(data: bytes) -> tuple[str | None, str | None]:
    """Format from magic bytes – fallback when Pillow cannot open the file."""
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG", "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG", "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP", "image/webp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF", "image/tiff"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF", "image/gif"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"avif", b"avis"):
            return "AVIF", "image/avif"
        if brand in (b"heic", b"heix", b"mif1", b"msf1", b"heim", b"heis"):
            return "HEIF", "image/heif"
    if data.lstrip()[:5] in (b"<?xml", b"<svg ") or b"<svg" in data[:512]:
        return "SVG", "image/svg+xml"
    return None, None


@dataclass
class Context:
    data: bytes
    image: Image.Image | None
    format: str | None
    mime: str | None


# ---------------------------------------------------------------------------
# 1. C2PA / Content Credentials
# ---------------------------------------------------------------------------

# English texts; German counterparts live in report.C2PA_VALIDATION_DE / C2PA_STATE_DE.
C2PA_VALIDATION_TEXTS: dict[str, str] = {
    "signingCredential.untrusted": "Signing certificate is not from a trusted issuer (not on the trust list)",
    "signingCredential.expired": "Signing certificate had expired at signing time",
    "signingCredential.revoked": "Signing certificate has been revoked",
    "signingCredential.invalid": "Signing certificate is invalid",
    "signingCredential.ocsp.unknown": "Revocation status of the certificate (OCSP) is unknown",
    "assertion.dataHash.mismatch": "Image data was modified after signing (hash mismatch)",
    "assertion.bmffHash.mismatch": "Media data was modified after signing (BMFF hash mismatch)",
    "assertion.boxesHash.mismatch": "Media data was modified after signing (box hash mismatch)",
    "assertion.hashedURI.mismatch": "An assertion was modified after signing",
    "assertion.missing": "A referenced assertion is missing",
    "assertion.action.ingredientMismatch": "An action refers to a non-matching ingredient",
    "claimSignature.mismatch": "Manifest signature is invalid (manifest was modified)",
    "claimSignature.missing": "Manifest signature is missing",
    "claimSignature.outsideValidity": "Signature is outside the certificate's validity period",
    "claim.missing": "Manifest claim is missing",
    "claim.malformed": "Manifest claim is malformed",
    "manifest.inaccessible": "Manifest is not accessible",
    "manifest.unreferenced": "Manifest is not referenced",
    "timeStamp.mismatch": "Timestamp does not match the signature",
    "timeStamp.untrusted": "Timestamp is not from a trusted authority",
    "timeStamp.outsideValidity": "Timestamp is outside the certificate's validity period",
    "general.error": "General validation error",
}

C2PA_STATE_TEXTS: dict[str, str] = {
    "Trusted": "Valid and signed by a trusted issuer",
    "Valid": "Structurally valid (signature intact, but issuer not confirmed as trusted)",
    "Invalid": "Invalid (manifest damaged, modified or signature broken)",
}

C2PA_UNREADABLE_EXPLANATION = (
    "The file contains JUMBF/C2PA structures that could not be evaluated "
    "(damaged, truncated or unsupported)."
)


def _c2pa_validation_text(entry: dict[str, Any]) -> str:
    code = str(entry.get("code", ""))
    return C2PA_VALIDATION_TEXTS.get(code) or str(entry.get("explanation") or code)


def _software_name(agent: Any) -> str | None:
    if isinstance(agent, str):
        return agent
    if isinstance(agent, dict):
        name = agent.get("name")
        version = agent.get("version")
        if name:
            return f"{name} {version}" if version else str(name)
    return None


def _evaluate_manifest(label: str, manifest: dict[str, Any], active: bool) -> Finding:
    details: dict[str, Any] = {"manifest": label}
    generator_texts: list[str] = []

    if manifest.get("claim_generator"):
        details["claim_generator"] = truncate(manifest["claim_generator"], 300)
        generator_texts.append(manifest["claim_generator"])
    infos = manifest.get("claim_generator_info")
    if isinstance(infos, dict):
        infos = [infos]
    if isinstance(infos, list) and infos:
        names = [n for n in (_software_name(i) for i in infos) if n]
        if names:
            details["claim_generator_info"] = [truncate(n, 300) for n in names]
            generator_texts.extend(names)

    signature = manifest.get("signature_info") or {}
    if signature.get("issuer"):
        details["signed_by"] = signature["issuer"]
    if signature.get("common_name") and signature.get("common_name") != signature.get("issuer"):
        details["certificate_cn"] = signature["common_name"]
    if signature.get("time"):
        details["signature_time"] = signature["time"]

    actions: list[dict[str, str]] = []
    source_types: dict[str, tuple[str, str]] = {}
    unknown_sources: list[str] = []
    for assertion in manifest.get("assertions") or []:
        if not str(assertion.get("label", "")).startswith("c2pa.actions"):
            continue
        for action in (assertion.get("data") or {}).get("actions") or []:
            entry = {"action": str(action.get("action", "?"))}
            dst = action.get("digitalSourceType")
            if dst:
                match = iptc_source_type(str(dst))
                if match:
                    name, level, meaning = match
                    source_types[name] = (level, meaning)
                    entry["digital_source_type"] = name
                else:
                    short = str(dst).rstrip("/").rsplit("/", 1)[-1]
                    if short not in unknown_sources:
                        unknown_sources.append(short)
                    entry["digital_source_type"] = short
            agent = _software_name(action.get("softwareAgent"))
            if agent:
                entry["software_agent"] = agent
                generator_texts.append(agent)
            if action.get("description"):
                entry["description"] = truncate(action["description"], 200)
            actions.append(entry)
    if actions:
        details["actions"] = actions
    if source_types or unknown_sources:
        details["source_types"] = [
            {"type": n, "meaning": m} for n, (_, m) in source_types.items()
        ] + [{"type": n, "meaning": None} for n in unknown_sources]

    generators = detect_generators(*generator_texts)
    if generators:
        details["detected_generators"] = generators

    # Level is derived from the IPTC table (algorithmicMedia must not end up as "info").
    level = highest_level(*(lv for lv, _ in source_types.values()))
    if generators:
        level = highest_level(level, LEVEL_HINT)

    title = "C2PA Content Credentials (active manifest)" if active else "C2PA manifest (previous version/ingredient)"
    if source_types:
        title += ": " + ", ".join(source_types)
    return Finding(TYPE_C2PA_ACTIVE if active else TYPE_C2PA_INGREDIENT, SOURCE_C2PA, level, title, details)


def check_c2pa(ctx: Context) -> list[Finding]:
    import c2pa

    raw_marker = b"jumb" in ctx.data and b"c2pa" in ctx.data
    try:
        if not ctx.mime:
            raise ValueError("unknown file format for C2PA")
        with c2pa.Reader(ctx.mime, io.BytesIO(ctx.data)) as reader:
            store = json.loads(reader.json())
            state = reader.get_validation_state()
    except Exception as exc:  # no manifest is the normal case, not an error
        if raw_marker:
            return [
                Finding(
                    TYPE_C2PA_UNREADABLE,
                    SOURCE_C2PA,
                    LEVEL_HINT,
                    "C2PA manifest detected but unreadable/invalid",
                    {
                        "explanation": C2PA_UNREADABLE_EXPLANATION,
                        "error_message": truncate(f"{type(exc).__name__}: {exc}", 300),
                    },
                )
            ]
        return []

    manifests = store.get("manifests") or {}
    active = store.get("active_manifest")
    findings: list[Finding] = []
    for label in sorted(manifests, key=lambda lbl: lbl != active):
        finding = _evaluate_manifest(label, manifests[label], label == active)
        if label == active:
            statuses = [e for e in store.get("validation_status") or [] if e.get("code")]
            # Machine-readable, language-independent: state as reported by c2pa
            # ("Trusted" / "Valid" / "Invalid") and the raw C2PA status codes.
            if state:
                finding.details["validation_state"] = str(state)
                finding.details["validation"] = C2PA_STATE_TEXTS.get(str(state), str(state))
            # validation_messages is aligned index by index with validation_codes.
            finding.details["validation_codes"] = [str(e["code"]) for e in statuses]
            if statuses:
                finding.details["validation_messages"] = [_c2pa_validation_text(e) for e in statuses]
        findings.append(finding)
    return findings


# ---------------------------------------------------------------------------
# XMP extraction
# ---------------------------------------------------------------------------

_XMP_PACKET = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.DOTALL)


def extract_xmp(ctx: Context) -> list[str]:
    packets: list[str] = []
    for match in _XMP_PACKET.finditer(ctx.data):
        packets.append(match.group(0).decode("utf-8", errors="replace"))
    if ctx.image is not None:
        # Important for compressed PNG iTXt chunks, which the raw-byte regex cannot see.
        for key in ("XML:com.adobe.xmp", "xmp"):
            value = ctx.image.info.get(key)
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if isinstance(value, str) and value.strip():
                packets.append(value)
    return list(dict.fromkeys(packets))


# ---------------------------------------------------------------------------
# 2. IPTC Digital Source Type (XMP)
# ---------------------------------------------------------------------------

_DST_URI = re.compile(r"digitalsourcetype/([A-Za-z]+)", re.IGNORECASE)
_DST_SHORT = re.compile(
    r"DigitalSourceType\s*=\s*[\"']([A-Za-z]+)[\"']|DigitalSourceType>\s*([A-Za-z]+)\s*<",
    re.IGNORECASE,
)


def check_iptc_source_type(ctx: Context) -> list[Finding]:
    values: list[str] = []
    for packet in extract_xmp(ctx):
        values += _DST_URI.findall(packet)
        values += [a or b for a, b in _DST_SHORT.findall(packet)]
    findings: list[Finding] = []
    seen: set[str] = set()
    for value in values:
        match = iptc_source_type(value)
        if not match or match[0] in seen:
            continue
        name, level, description = match
        seen.add(name)
        findings.append(
            Finding(
                TYPE_IPTC_SOURCE_TYPE,
                SOURCE_IPTC,
                level,
                f"IPTC Digital Source Type: {name}",
                {"value": name, "meaning": description, "uri": IPTC_URI + name},
            )
        )
    return findings


# ---------------------------------------------------------------------------
# 3. Further XMP fields
# ---------------------------------------------------------------------------

XMP_FIELDS = [
    "xmp:CreatorTool",
    "photoshop:Credit",
    "photoshop:Source",
    "dc:creator",
    "dc:description",
    "dc:rights",
    "tiff:Software",
    "Iptc4xmpExt:AISystemUsed",
    "Iptc4xmpExt:AISystemVersionUsed",
    "Iptc4xmpExt:AIPromptInformation",
    "plus:DataMining",
]
XMP_AI_FIELDS = {
    "Iptc4xmpExt:AISystemUsed",
    "Iptc4xmpExt:AISystemVersionUsed",
    "Iptc4xmpExt:AIPromptInformation",
}
_RDF_LI = re.compile(r"<rdf:li\b[^>]*?(?:/>|>(.*?)</rdf:li>)", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def _xmp_text(raw: str) -> str:
    return html.unescape(_TAG.sub("", raw)).strip()


def read_xmp_field(packet: str, name: str) -> list[str]:
    """Reads an XMP field written as attribute or element (incl. rdf:Alt/Seq/Bag with rdf:li)."""
    escaped = re.escape(name)
    values: list[str] = []
    for m in re.finditer(rf"\s{escaped}\s*=\s*(\"([^\"]*)\"|'([^']*)')", packet):
        values.append(html.unescape(m.group(2) if m.group(2) is not None else m.group(3)).strip())
    for m in re.finditer(rf"<{escaped}\b([^>]*?)(?:/>|>(.*?)</{escaped}\s*>)", packet, re.DOTALL):
        attrs, content = m.group(1), m.group(2)
        if content is None:  # <field rdf:resource="..."/>
            resource = re.search(r"rdf:resource\s*=\s*[\"']([^\"']*)", attrs)
            if resource:
                values.append(html.unescape(resource.group(1)))
            continue
        items = _RDF_LI.findall(content)
        if items:
            values += [_xmp_text(li) for li in items]
        else:
            values.append(_xmp_text(content))
    return [v for v in dict.fromkeys(values) if v]


def check_xmp(ctx: Context) -> list[Finding]:
    found: dict[str, list[str]] = {}
    for packet in extract_xmp(ctx):
        for name in XMP_FIELDS:
            for value in read_xmp_field(packet, name):
                values = found.setdefault(name, [])
                if value not in values:
                    values.append(value)
    if not found:
        return []

    details: dict[str, Any] = {}
    for name, values in found.items():
        shortened = [truncate(v) for v in values]
        details[name] = shortened[0] if len(shortened) == 1 else shortened
    generators = detect_generators(*(v for values in found.values() for v in values))
    ai_fields = XMP_AI_FIELDS & found.keys()

    if ai_fields:
        kind, level, title = TYPE_XMP_AI_SYSTEM, LEVEL_AI, "XMP: IPTC fields describing the AI system used"
    elif generators:
        kind, level, title = TYPE_XMP_GENERATOR, LEVEL_HINT, "XMP metadata names an AI generator"
    else:
        kind, level, title = TYPE_XMP_METADATA, LEVEL_INFO, "XMP metadata"
    if generators:
        details["detected_generators"] = generators
    return [Finding(kind, SOURCE_XMP, level, title, details)]


# ---------------------------------------------------------------------------
# 4. EXIF
# ---------------------------------------------------------------------------

EXIF_TAGS = {
    0x010F: "Make",
    0x0110: "Model",
    0x0131: "Software",
    0x010E: "ImageDescription",
    0x013B: "Artist",
    0x8298: "Copyright",
}
EXIF_IFD = 0x8769
EXIF_USER_COMMENT = 0x9286


def _bytes_to_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).replace("\x00", "").strip()


def decode_user_comment(value: Any, big_endian: bool | None = None) -> str:
    """Decodes EXIF UserComment including its 8-byte charset prefix (ASCII/UNICODE/JIS/empty)."""
    if isinstance(value, str):
        return value.replace("\x00", "").strip()
    if not isinstance(value, (bytes, bytearray)):
        return str(value)
    prefix, payload = bytes(value[:8]), bytes(value[8:])
    if prefix == b"UNICODE\x00":
        if payload[:2] in (b"\xfe\xff", b"\xff\xfe"):
            text = payload.decode("utf-16", errors="replace")
        else:
            if big_endian is None:
                # Heuristic: for mostly Latin text the zero byte comes first (BE) or second (LE).
                even = payload[0::2].count(0)
                odd = payload[1::2].count(0)
                big_endian = even > odd
            text = payload.decode("utf-16-be" if big_endian else "utf-16-le", errors="replace")
    elif prefix == b"ASCII\x00\x00\x00":
        text = payload.decode("ascii", errors="replace")
    elif prefix == b"JIS\x00\x00\x00\x00\x00":
        text = payload.decode("shift_jis", errors="replace")
    elif prefix == b"\x00" * 8:
        text = payload.decode("utf-8", errors="replace")
    else:
        text = bytes(value).decode("utf-8", errors="replace")
    return text.replace("\x00", "").strip()


_A1111_MARKERS = re.compile(r"\b(?:Steps|Sampler|CFG scale|Seed):", re.IGNORECASE)


def check_exif(ctx: Context) -> list[Finding]:
    if ctx.image is None:
        return []
    exif = ctx.image.getexif()
    details: dict[str, Any] = {}
    for tag, name in EXIF_TAGS.items():
        value = exif.get(tag)
        if value not in (None, b"", ""):
            text = _bytes_to_text(value)
            if text:
                details[name] = truncate(text)

    comment = ""
    try:
        ifd = exif.get_ifd(EXIF_IFD)
    except Exception:
        ifd = {}
    if EXIF_USER_COMMENT in ifd:
        big_endian = {">": True, "<": False}.get(getattr(exif, "endian", None) or "")
        comment = decode_user_comment(ifd[EXIF_USER_COMMENT], big_endian)
        if comment:
            details["UserComment"] = truncate(comment)

    if not details:
        return []

    findings: list[Finding] = []
    # A1111 stores generation parameters in UserComment for JPEG/WebP.
    if comment and len(_A1111_MARKERS.findall(comment)) >= 2:
        findings.append(_a1111_finding(comment, SOURCE_EXIF_USER_COMMENT))

    generators = detect_generators(*details.values())
    if generators:
        details["detected_generators"] = generators
        findings.append(Finding(TYPE_EXIF_GENERATOR, SOURCE_EXIF, LEVEL_HINT, "EXIF metadata names an AI generator", details))
    else:
        findings.append(Finding(TYPE_EXIF_METADATA, SOURCE_EXIF, LEVEL_INFO, "EXIF metadata", details))
    return findings


# ---------------------------------------------------------------------------
# 5. PNG text chunks
# ---------------------------------------------------------------------------


def _a1111_finding(parameters: str, source: str) -> Finding:
    details: dict[str, Any] = {}
    lines = parameters.strip().splitlines()
    last = next((ln for ln in reversed(lines) if _A1111_MARKERS.search(ln)), "")
    last_idx = lines.index(last) if last in lines else None
    neg_idx = next((i for i, ln in enumerate(lines) if ln.startswith("Negative prompt:")), None)
    prompt_end = neg_idx if neg_idx is not None else (last_idx if last_idx is not None else len(lines))
    prompt = "\n".join(lines[:prompt_end]).strip()
    if prompt:
        details["prompt"] = truncate(prompt, 300)
    if neg_idx is not None:
        negative = "\n".join(lines[neg_idx:last_idx])
        details["negative_prompt"] = truncate(negative.removeprefix("Negative prompt:").strip(), 300)
    for key, name in (
        ("Model", "model"),
        ("Model hash", "model_hash"),
        ("Steps", "steps"),
        ("Sampler", "sampler"),
        ("CFG scale", "cfg_scale"),
        ("Seed", "seed"),
        ("Size", "size"),
        ("Version", "version"),
    ):
        m = re.search(rf"(?:^|,\s*){re.escape(key)}:\s*([^,\n]+)", last)
        if m:
            details[name] = m.group(1).strip()
    version = details.get("version", "")
    tool = "Stable Diffusion WebUI Forge" if version.lower().startswith("f") else "AUTOMATIC1111 / Forge"
    details = {"tool": tool, **details}
    title = f"Generation parameters from {tool}"
    if details.get("model"):
        title += f" (model: {details['model']})"
    details["raw"] = truncate(parameters, 600)
    return Finding(TYPE_GENERATION_PARAMETERS, source, LEVEL_AI, title, details)


def _json_or_none(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _comfy_models(prompt: Any) -> list[str]:
    models: list[str] = []
    if isinstance(prompt, dict):
        for node in prompt.values():
            inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
            for key in ("ckpt_name", "unet_name", "model_name", "lora_name"):
                value = inputs.get(key) if isinstance(inputs, dict) else None
                if isinstance(value, str) and value not in models:
                    models.append(value)
    return models


_TOOL_KEYS = {
    "invokeai_metadata": "InvokeAI",
    "invokeai_graph": "InvokeAI",
    "sd-metadata": "InvokeAI (legacy, sd-metadata)",
    "dream": "InvokeAI / lstein Dream (legacy)",
    "fooocus_scheme": "Fooocus",
    "generation_data": "Stable Diffusion based generator (generation_data)",
}


def check_png_text(ctx: Context) -> list[Finding]:
    if ctx.image is None or ctx.image.format != "PNG":
        return []
    texts: dict[str, str] = {
        str(k): _bytes_to_text(v)
        for k, v in getattr(ctx.image, "text", {}).items()
        if k != "XML:com.adobe.xmp"
    }
    if not texts:
        return []
    lower = {k.lower(): k for k in texts}
    consumed: set[str] = set()
    findings: list[Finding] = []

    # A1111 / Forge
    if "parameters" in lower:
        key = lower["parameters"]
        if _A1111_MARKERS.search(texts[key]):
            findings.append(_a1111_finding(texts[key], SOURCE_PNG_TEXT))
            consumed.add(key)

    # ComfyUI
    comfy_details: dict[str, Any] = {}
    for name in ("prompt", "workflow"):
        if name in lower:
            key = lower[name]
            parsed = _json_or_none(texts[key])
            if isinstance(parsed, (dict, list)):
                consumed.add(key)
                comfy_details[f"{name}_json"] = truncate(texts[key], 400)
                if name == "prompt":
                    models = _comfy_models(parsed)
                    if models:
                        comfy_details["models"] = models
    if comfy_details:
        title = "ComfyUI workflow embedded"
        if comfy_details.get("models"):
            title += f" (model: {comfy_details['models'][0]})"
        findings.append(Finding(TYPE_COMFYUI_WORKFLOW, SOURCE_PNG_TEXT, LEVEL_AI, title, comfy_details))

    # NovelAI
    software_key = lower.get("software")
    if software_key and texts[software_key].lower().startswith("novelai"):
        details: dict[str, Any] = {"software": texts[software_key]}
        consumed.add(software_key)
        for extra in ("source", "description", "comment", "title"):
            if extra in lower:
                details[lower[extra]] = truncate(texts[lower[extra]], 400)
                consumed.add(lower[extra])
        findings.append(Finding(TYPE_NOVELAI, SOURCE_PNG_TEXT, LEVEL_AI, "Created by NovelAI", details))

    # Further known generator keys
    for name, tool in _TOOL_KEYS.items():
        if name in lower:
            key = lower[name]
            consumed.add(key)
            findings.append(
                Finding(
                    TYPE_GENERATOR_METADATA,
                    SOURCE_PNG_TEXT,
                    LEVEL_AI,
                    f"Generation data from {tool}",
                    {"tool": tool, "key": key, "content": truncate(texts[key], 600)},
                )
            )

    rest = {k: truncate(v) for k, v in texts.items() if k not in consumed}
    if rest:
        generators = detect_generators(*rest.values())
        if generators:
            rest["detected_generators"] = generators
            findings.append(Finding(TYPE_PNG_TEXT_GENERATOR, SOURCE_PNG_TEXT, LEVEL_HINT, "PNG text fields name an AI generator", rest))
        else:
            findings.append(Finding(TYPE_PNG_TEXT, SOURCE_PNG_TEXT, LEVEL_INFO, "PNG text fields", rest))
    return findings


# ---------------------------------------------------------------------------
# 6. Invisible watermarks (invisible-watermark, dwtDct)
# ---------------------------------------------------------------------------

WATERMARK_MIN_SIDE = 256
SD_WATERMARK_PREFIX = b"StableDiffusionV"
SDXL_WATERMARK = 0b101100111110110010010000011110111011000110011110
SDXL_BITS = [int(b) for b in bin(SDXL_WATERMARK)[2:]]
SDXL_MAX_BIT_ERRORS = 4


@lru_cache(maxsize=1)
def load_watermark_module() -> types.ModuleType:
    """Imports imwatermark without PyTorch.

    ``imwatermark/rivaGan.py`` imports ``torch`` at module level although only
    dwtDct is used. An empty placeholder module is enough for the import.
    """
    if "torch" not in sys.modules:
        sys.modules["torch"] = types.ModuleType("torch")
    import imwatermark

    return imwatermark


def _to_bgr(image: Image.Image):
    import cv2
    import numpy as np

    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def check_watermarks(ctx: Context) -> list[Finding]:
    if ctx.image is None or min(ctx.image.size) < WATERMARK_MIN_SIDE:
        return []
    wm = load_watermark_module()
    bgr = _to_bgr(ctx.image)
    findings: list[Finding] = []

    sd = wm.WatermarkDecoder("bytes", 136).decode(bgr, "dwtDct")
    if isinstance(sd, (bytes, bytearray)) and bytes(sd).startswith(SD_WATERMARK_PREFIX):
        payload = bytes(sd).rstrip(b"\x00").decode("ascii", errors="replace")
        findings.append(
            Finding(
                TYPE_SD_WATERMARK,
                SOURCE_WATERMARK,
                LEVEL_AI,
                "Invisible Stable Diffusion watermark (v1/v2)",
                {
                    "method": "invisible-watermark, dwtDct, 136-bit",
                    "decoded_payload": payload,
                },
            )
        )

    bits = [int(bool(b)) for b in wm.WatermarkDecoder("bits", 48).decode(bgr, "dwtDct")]
    bit_errors = sum(a != b for a, b in zip(bits, SDXL_BITS)) + abs(len(bits) - len(SDXL_BITS))
    if bit_errors <= SDXL_MAX_BIT_ERRORS:
        findings.append(
            Finding(
                TYPE_SDXL_WATERMARK,
                SOURCE_WATERMARK,
                LEVEL_AI,
                "Invisible SDXL watermark",
                {
                    "method": "invisible-watermark, dwtDct, 48-bit",
                    "bit_errors": bit_errors,
                    "max_bit_errors": SDXL_MAX_BIT_ERRORS,
                },
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Overall analysis
# ---------------------------------------------------------------------------

DETECTORS: list[tuple[str, Callable[[Context], list[Finding]]]] = [
    (SOURCE_C2PA, check_c2pa),
    (SOURCE_IPTC, check_iptc_source_type),
    (SOURCE_XMP, check_xmp),
    (SOURCE_EXIF, check_exif),
    (SOURCE_PNG_TEXT, check_png_text),
    (SOURCE_WATERMARK, check_watermarks),
]


def _exception_text(exc: Exception) -> str:
    return truncate(f"{type(exc).__name__}: {exc}", 300)


def analyze(data: bytes, filename: str = "image") -> Result:
    """Runs all detectors; a failing detector does not block the others."""
    result = Result(file=filename, file_size_bytes=len(data))
    fmt, mime = sniff_format(data)

    image: Image.Image | None = None
    try:
        image = Image.open(io.BytesIO(data))
        fmt = image.format or fmt
        mime = _MIME_BY_FORMAT.get(image.format or "", mime)
        result.size = image.size
    except Exception as exc:
        result.errors.append(
            AnalysisError(
                ERROR_IMAGE_DECODE,
                "Image could not be decoded; only raw-byte checks were run",
                exception=_exception_text(exc),
            )
        )
    result.format = fmt

    ctx = Context(data=data, image=image, format=fmt, mime=mime)
    for name, detector in DETECTORS:
        try:
            result.findings.extend(detector(ctx))
        except Exception as exc:
            result.errors.append(
                AnalysisError(
                    ERROR_DETECTOR, f"Detector '{name}' failed", detector=name, exception=_exception_text(exc)
                )
            )

    result.findings.sort(key=lambda f: LEVEL_RANK.get(f.level, 99))

    from .recommendation import recommend  # local import: recommendation depends on this module

    result.recommendation = recommend(result)
    return result
