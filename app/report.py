"""Human-readable plain-text report for a ``Result``, in German (default) or English.

The German report translates the English ``Result`` data via its own tables keyed by
language-independent codes; the English report uses the data largely as-is.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

from . import detectors as d
from . import recommendation as rec
from .detectors import (
    LEVEL_AI,
    LEVEL_HINT,
    LEVEL_INFO,
    SOURCE_C2PA,
    SOURCE_EXIF,
    SOURCE_EXIF_USER_COMMENT,
    SOURCE_IPTC,
    SOURCE_PNG_TEXT,
    SOURCE_WATERMARK,
    SOURCE_XMP,
    C2PA_UNREADABLE_EXPLANATION,
    Finding,
    Result,
)

LANGUAGES = ("de", "en")
DEFAULT_LANGUAGE = "de"

VERDICTS_DE = {
    LEVEL_AI: "KI-Kennzeichnung gefunden",
    LEVEL_HINT: "Hinweise auf KI-Erzeugung gefunden (keine standardisierte Kennzeichnung)",
    "none": "Keine maschinenlesbare KI-Kennzeichnung gefunden",
}

LEVEL_PREFIX = {LEVEL_AI: "[KI]", LEVEL_HINT: "[HINWEIS]", LEVEL_INFO: "[INFO]"}
LEVEL_PREFIX_EN = {LEVEL_AI: "[AI]", LEVEL_HINT: "[HINT]", LEVEL_INFO: "[INFO]"}

DISCLAIMER = (
    "Hinweis: Eine fehlende Kennzeichnung beweist nicht, dass ein Bild echt ist. "
    "Metadaten gehen bei Screenshots, beim Neu-Speichern und beim Upload in soziale "
    "Netzwerke häufig verloren. Proprietäre Wasserzeichen (z. B. Google SynthID, "
    "Meta Video Seal) lassen sich nur mit Werkzeugen der jeweiligen Anbieter prüfen."
)

DISCLAIMER_EN = (
    "Note: A missing label does not prove that an image is authentic. Metadata is often "
    "lost through screenshots, re-saving and uploads to social networks. Proprietary "
    "watermarks (e.g. Google SynthID, Meta Video Seal) can only be checked with the "
    "respective vendors' tools."
)

# Fixed texts of the report per language
REPORT_TEXTS = {
    "de": {
        "heading": "KI-Kennzeichnungsprüfung (Art. 50 Abs. 2 EU AI Act)",
        "file": "Datei",
        "format": "Format",
        "size": "Größe",
        "verdict": "Ergebnis",
        "unknown": "unbekannt",
        "pixels": "Pixel",
        "findings": "Funde",
        "source": "Quelle",
        "nothing_found": "Keine Herkunfts-Metadaten, Generator-Angaben oder bekannten Wasserzeichen gefunden.",
        "errors": "Fehler bei der Prüfung:",
        "disclaimer": DISCLAIMER,
    },
    "en": {
        "heading": "AI label check (Art. 50(2) EU AI Act)",
        "file": "File",
        "format": "Format",
        "size": "Size",
        "verdict": "Verdict",
        "unknown": "unknown",
        "pixels": "pixels",
        "findings": "Findings",
        "source": "Source",
        "nothing_found": "No provenance metadata, generator information or known watermarks found.",
        "errors": "Errors during the check:",
        "disclaimer": DISCLAIMER_EN,
    },
}

SOURCE_LABELS_EN = {
    SOURCE_C2PA: "C2PA",
    SOURCE_IPTC: "IPTC (XMP)",
    SOURCE_XMP: "XMP",
    SOURCE_EXIF: "EXIF",
    SOURCE_EXIF_USER_COMMENT: "EXIF (UserComment)",
    SOURCE_PNG_TEXT: "PNG text",
    SOURCE_WATERMARK: "Watermark",
}

DETAIL_LABELS_EN = {
    "manifest": "Manifest",
    "claim_generator": "Created with (claim_generator)",
    "claim_generator_info": "Created with (claim_generator_info)",
    "signed_by": "Signed by",
    "certificate_cn": "Certificate (CN)",
    "signature_time": "Signature time",
    "actions": "Actions",
    "source_types": "Source types (IPTC Digital Source Type)",
    "validation": "Validation",
    "validation_messages": "Validation messages",
    "explanation": "Explanation",
    "error_message": "Error message",
    "value": "Value",
    "meaning": "Meaning",
    "uri": "URI",
    "detected_generators": "Detected AI generators",
    "prompt": "Prompt",
    "negative_prompt": "Negative prompt",
    "model": "Model",
    "model_hash": "Model hash",
    "models": "Models",
    "steps": "Steps",
    "sampler": "Sampler",
    "cfg_scale": "CFG scale",
    "seed": "Seed",
    "size": "Size",
    "version": "Version",
    "raw": "Raw data",
    "prompt_json": "prompt (JSON)",
    "workflow_json": "workflow (JSON)",
    "software": "Software",
    "key": "Key",
    "content": "Content",
    "method": "Method",
    "decoded_payload": "Decoded payload",
    "bit_errors": "Bit errors",
    "max_bit_errors": "Tolerance (max. bit errors)",
}

SOURCE_LABELS = {
    SOURCE_C2PA: "C2PA",
    SOURCE_IPTC: "IPTC (XMP)",
    SOURCE_XMP: "XMP",
    SOURCE_EXIF: "EXIF",
    SOURCE_EXIF_USER_COMMENT: "EXIF (UserComment)",
    SOURCE_PNG_TEXT: "PNG-Text",
    SOURCE_WATERMARK: "Wasserzeichen",
}

# German labels for the English ``details`` keys. Keys not listed here (original
# metadata field names such as ``photoshop:Credit`` or PNG chunk names) are shown as-is.
DETAIL_LABELS = {
    # C2PA
    "manifest": "Manifest",
    "claim_generator": "Erzeugt mit (claim_generator)",
    "claim_generator_info": "Erzeugt mit (claim_generator_info)",
    "signed_by": "Signiert von",
    "certificate_cn": "Zertifikat (CN)",
    "signature_time": "Signaturzeit",
    "actions": "Aktionen",
    "source_types": "Quellarten (IPTC Digital Source Type)",
    "validation": "Validierung",
    "validation_messages": "Validierungsmeldungen",
    "explanation": "Erläuterung",
    "error_message": "Fehlermeldung",
    # IPTC
    "value": "Wert",
    "meaning": "Bedeutung",
    "uri": "URI",
    # generators / generation parameters
    "detected_generators": "Erkannte KI-Generatoren",
    "prompt": "Prompt",
    "negative_prompt": "Negativ-Prompt",
    "model": "Modell",
    "model_hash": "Modell-Hash",
    "models": "Modelle",
    "steps": "Schritte",
    "sampler": "Sampler",
    "cfg_scale": "CFG Scale",
    "seed": "Seed",
    "size": "Größe",
    "version": "Version",
    "raw": "Rohdaten",
    "prompt_json": "prompt (JSON)",
    "workflow_json": "workflow (JSON)",
    "software": "Software",
    "key": "Schlüssel",
    "content": "Inhalt",
    # watermarks
    "method": "Verfahren",
    "decoded_payload": "Gelesener Inhalt",
    "bit_errors": "Bitfehler",
    "max_bit_errors": "Toleranz (max. Bitfehler)",
}

# Machine-readable keys that the plain-text report omits because a German
# counterpart is already shown ("validation" / "validation_messages").
REPORT_HIDDEN_KEYS = {"validation_state", "validation_codes", "tool"}

# German titles per finding type; callables receive the finding's details.
TOOL_NAMES_DE = {
    "sd-metadata": "InvokeAI (ältere Version, sd-metadata)",
    "dream": "InvokeAI / lstein Dream (ältere Version)",
    "generation_data": "Stable-Diffusion-basierter Generator (generation_data)",
}


def _c2pa_title(base: str, details: dict[str, Any]) -> str:
    known = [t["type"] for t in details.get("source_types") or [] if isinstance(t, dict) and t.get("meaning")]
    return f"{base}: {', '.join(known)}" if known else base


def _with_model(base: str, model: Any) -> str:
    return f"{base} (Modell: {model})" if model else base


FINDING_TITLES_DE: dict[str, Any] = {
    d.TYPE_C2PA_ACTIVE: lambda x: _c2pa_title("C2PA Content Credentials (aktives Manifest)", x),
    d.TYPE_C2PA_INGREDIENT: lambda x: _c2pa_title("C2PA-Manifest (frühere Version/Zutat)", x),
    d.TYPE_C2PA_UNREADABLE: "C2PA-Manifest erkannt, aber nicht lesbar/ungültig",
    d.TYPE_IPTC_SOURCE_TYPE: lambda x: f"IPTC Digital Source Type: {x.get('value')}",
    d.TYPE_XMP_AI_SYSTEM: "XMP: IPTC-Felder zum verwendeten KI-System",
    d.TYPE_XMP_GENERATOR: "XMP-Metadaten nennen einen KI-Generator",
    d.TYPE_XMP_METADATA: "XMP-Metadaten",
    d.TYPE_EXIF_GENERATOR: "EXIF-Metadaten nennen einen KI-Generator",
    d.TYPE_EXIF_METADATA: "EXIF-Metadaten",
    d.TYPE_GENERATION_PARAMETERS: lambda x: _with_model(f"Generierungsparameter von {x.get('tool')}", x.get("model")),
    d.TYPE_COMFYUI_WORKFLOW: lambda x: _with_model("ComfyUI-Workflow eingebettet", (x.get("models") or [None])[0]),
    d.TYPE_NOVELAI: "Von NovelAI erzeugt",
    d.TYPE_GENERATOR_METADATA: lambda x: "Generierungsdaten von "
    + TOOL_NAMES_DE.get(str(x.get("key", "")).lower(), str(x.get("tool"))),
    d.TYPE_PNG_TEXT_GENERATOR: "PNG-Textfelder nennen einen KI-Generator",
    d.TYPE_PNG_TEXT: "PNG-Textfelder",
    d.TYPE_SD_WATERMARK: "Unsichtbares Stable-Diffusion-Wasserzeichen (v1/v2)",
    d.TYPE_SDXL_WATERMARK: "Unsichtbares SDXL-Wasserzeichen",
}


def finding_title_de(finding: Finding) -> str:
    title = FINDING_TITLES_DE.get(finding.type)
    if title is None:
        return finding.title
    return title(finding.details) if callable(title) else title


def error_de(error: d.AnalysisError) -> str:
    suffix = f": {error.exception}" if error.exception else ""
    if error.code == d.ERROR_IMAGE_DECODE:
        return f"Bild konnte nicht dekodiert werden{suffix}"
    if error.code == d.ERROR_DETECTOR:
        return f"Prüfer {SOURCE_LABELS.get(error.detector or '', error.detector)}{suffix}"
    return f"{error.message}{suffix}"

# German texts for the English values in ``details`` (keyed by language-independent codes).
IPTC_MEANINGS_DE = {
    "trainedAlgorithmicMedia": "Mit einem trainierten KI-Modell erzeugt (generative KI)",
    "compositeWithTrainedAlgorithmicMedia": (
        "Komposition, die mit generativer KI erzeugte Elemente enthält (z. B. Generative Fill)"
    ),
    "algorithmicMedia": "Rein algorithmisch erzeugt (ohne Trainingsdaten, z. B. prozedural)",
    "compositeSynthetic": "Komposition mit synthetischen Elementen",
    "algorithmicallyEnhanced": "Algorithmisch verbessert bzw. verändert",
    "dataDrivenMedia": "Datengetrieben erzeugte Medien",
    "digitalCapture": "Digitale Aufnahme (Kamera)",
    "computationalCapture": "Rechnergestützte Aufnahme (z. B. Smartphone-HDR)",
    "digitalArt": "Digital von Menschen erstellte Kunst",
    "virtualRecording": "Aufzeichnung einer virtuellen Umgebung",
    "composite": "Komposition aus mehreren Elementen",
    "compositeCapture": "Komposition aus mehreren Aufnahmen",
    "screenCapture": "Bildschirmaufnahme",
    "negativeFilm": "Digitalisiertes Filmnegativ",
    "positiveFilm": "Digitalisiertes Dia/Positiv",
    "print": "Digitalisierter Druck/Abzug",
    "minorHumanEdits": "Geringfügige menschliche Bearbeitung",
    "humanEdits": "Menschliche Bearbeitung",
}

C2PA_VALIDATION_DE = {
    "signingCredential.untrusted": (
        "Signaturzertifikat stammt nicht von einer vertrauenswürdigen Stelle (nicht in der Vertrauensliste)"
    ),
    "signingCredential.expired": "Signaturzertifikat war zum Signaturzeitpunkt abgelaufen",
    "signingCredential.revoked": "Signaturzertifikat wurde widerrufen",
    "signingCredential.invalid": "Signaturzertifikat ist ungültig",
    "signingCredential.ocsp.unknown": "Widerrufsstatus des Zertifikats (OCSP) ist unbekannt",
    "assertion.dataHash.mismatch": "Bilddaten wurden nach der Signatur verändert (Hash stimmt nicht überein)",
    "assertion.bmffHash.mismatch": "Mediendaten wurden nach der Signatur verändert (BMFF-Hash stimmt nicht überein)",
    "assertion.boxesHash.mismatch": "Mediendaten wurden nach der Signatur verändert (Box-Hash stimmt nicht überein)",
    "assertion.hashedURI.mismatch": "Eine Assertion wurde nach der Signatur verändert",
    "assertion.missing": "Eine referenzierte Assertion fehlt",
    "assertion.action.ingredientMismatch": "Aktion verweist auf eine nicht passende Zutat",
    "claimSignature.mismatch": "Signatur des Manifests ist ungültig (Manifest wurde verändert)",
    "claimSignature.missing": "Signatur des Manifests fehlt",
    "claimSignature.outsideValidity": "Signatur liegt außerhalb der Gültigkeit des Zertifikats",
    "claim.missing": "Manifest-Claim fehlt",
    "claim.malformed": "Manifest-Claim ist fehlerhaft",
    "manifest.inaccessible": "Manifest ist nicht zugänglich",
    "manifest.unreferenced": "Manifest wird nicht referenziert",
    "timeStamp.mismatch": "Zeitstempel passt nicht zur Signatur",
    "timeStamp.untrusted": "Zeitstempel stammt nicht von einer vertrauenswürdigen Stelle",
    "timeStamp.outsideValidity": "Zeitstempel liegt außerhalb der Zertifikatsgültigkeit",
    "general.error": "Allgemeiner Fehler bei der Validierung",
}

C2PA_STATE_DE = {
    "Trusted": "Gültig und vertrauenswürdig signiert",
    "Valid": "Strukturell gültig (Signatur intakt, Aussteller aber nicht als vertrauenswürdig bestätigt)",
    "Invalid": "Ungültig (Manifest beschädigt, verändert oder Signatur fehlerhaft)",
}

C2PA_UNREADABLE_EXPLANATION_DE = (
    "Die Datei enthält JUMBF-/C2PA-Strukturen, die sich nicht auswerten ließen "
    "(beschädigt, abgeschnitten oder nicht unterstützt)."
)

_TRUNCATED = re.compile(r"\[(\d+) characters truncated\]")

INDENT = "    "
LINE_WIDTH = 88  # total line length incl. indentation; fits the <pre> box of the web UI
WRAP_WIDTH = LINE_WIDTH


def _number(n: float, digits: int = 1) -> str:
    """German number format (1.234,5)."""
    return f"{n:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_file_size(num_bytes: int, lang: str = DEFAULT_LANGUAGE) -> str:
    number = _number if lang == "de" else (lambda n: f"{n:,.1f}")
    if num_bytes < 1024:
        return f"{num_bytes} Bytes" if lang == "de" else f"{num_bytes} bytes"
    if num_bytes < 1024 * 1024:
        return f"{number(num_bytes / 1024)} KB"
    return f"{number(num_bytes / 1024 / 1024)} MB"


def _wrap(text: str, width: int = WRAP_WIDTH) -> list[str]:
    """Wraps long lines at spaces; existing line breaks are preserved."""
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        lines += textwrap.wrap(paragraph, width, break_long_words=True, break_on_hyphens=False) or [""]
    return lines


def _german(value: Any) -> Any:
    """Translates the generic parts of English values (truncation note), recursively."""
    if isinstance(value, str):
        return _TRUNCATED.sub(r"[\1 Zeichen gekürzt]", value)
    if isinstance(value, list):
        return [_german(v) for v in value]
    if isinstance(value, dict):
        return {k: _german(v) for k, v in value.items()}
    return value


def _source_type_de(name: str | None) -> str:
    return IPTC_MEANINGS_DE.get(name or "", "unbekannter Wert")


def _action_de(action: dict[str, Any]) -> str:
    parts = [str(action.get("action", "?"))]
    if action.get("digital_source_type"):
        dst = action["digital_source_type"]
        parts.append(f"Quelle: {dst} ({_source_type_de(dst)})")
    if action.get("software_agent"):
        parts.append(f"Software: {action['software_agent']}")
    if action.get("description"):
        parts.append(f"Beschreibung: {action['description']}")
    return " – ".join(parts)


def _german_value(key: str, value: Any, details: dict[str, Any]) -> Any:
    """German rendering of a structured English ``details`` value."""
    if key == "meaning" and details.get("value") in IPTC_MEANINGS_DE:
        return IPTC_MEANINGS_DE[details["value"]]
    if key == "validation":
        return C2PA_STATE_DE.get(details.get("validation_state", ""), value)
    if key == "validation_messages":
        codes = details.get("validation_codes") or []
        return [
            f"{C2PA_VALIDATION_DE.get(code, message)} ({code})" if code else message
            for code, message in zip(codes + [""] * (len(value) - len(codes)), value)
        ]
    if key == "actions" and isinstance(value, list):
        return [_action_de(a) if isinstance(a, dict) else a for a in value]
    if key == "source_types" and isinstance(value, list):
        return [f"{t.get('type')}: {_source_type_de(t.get('type'))}" if isinstance(t, dict) else t for t in value]
    if key == "explanation" and value == C2PA_UNREADABLE_EXPLANATION:
        return C2PA_UNREADABLE_EXPLANATION_DE
    return value


def finding_details_de(finding: Finding) -> list[tuple[str, Any]]:
    """(German label, German value) pairs for the plain-text report."""
    return [
        (DETAIL_LABELS.get(key, key), _german(_german_value(key, value, finding.details)))
        for key, value in finding.details.items()
        if key not in REPORT_HIDDEN_KEYS
    ]


def _english_value(key: str, value: Any, details: dict[str, Any]) -> Any:
    """English rendering of structured ``details`` values (lists of objects become lines)."""
    if key == "validation_messages":
        codes = details.get("validation_codes") or []
        return [
            f"{message} ({code})" if code else message
            for code, message in zip(codes + [""] * (len(value) - len(codes)), value)
        ]
    if key == "actions" and isinstance(value, list):
        lines = []
        for a in value:
            if not isinstance(a, dict):
                lines.append(a)
                continue
            parts = [str(a.get("action", "?"))]
            if a.get("digital_source_type"):
                dst = a["digital_source_type"]
                meaning = d.IPTC_SOURCE_TYPES.get(dst, (None, "unknown value"))[1]
                parts.append(f"source: {dst} ({meaning})")
            if a.get("software_agent"):
                parts.append(f"software: {a['software_agent']}")
            if a.get("description"):
                parts.append(f"description: {a['description']}")
            lines.append(" – ".join(parts))
        return lines
    if key == "source_types" and isinstance(value, list):
        return [
            f"{t.get('type')}: {t.get('meaning') or 'unknown value'}" if isinstance(t, dict) else t for t in value
        ]
    return value


def finding_details_en(finding: Finding) -> list[tuple[str, Any]]:
    """(English label, English value) pairs for the plain-text report."""
    return [
        (DETAIL_LABELS_EN.get(key, key), _english_value(key, value, finding.details))
        for key, value in finding.details.items()
        if key not in REPORT_HIDDEN_KEYS
    ]


def error_en(error: d.AnalysisError) -> str:
    if error.code == d.ERROR_DETECTOR:
        message = f"Detector {SOURCE_LABELS_EN.get(error.detector or '', error.detector)}"
    elif error.code == d.ERROR_IMAGE_DECODE:
        message = "Image could not be decoded"
    else:
        message = error.message
    return f"{message}: {error.exception}" if error.exception else message


# ---------------------------------------------------------------------------
# Labelling recommendation (EU AI Act, Art. 50)
# ---------------------------------------------------------------------------

RECOMMENDATION_TEXTS = {
    "de": {
        "heading": "Empfehlung nach KI-Verordnung (Art. 50):",
        "category": "Einstufung",
        "labelling": "Kennzeichnung",
        "confidence": "Sicherheit",
        "evidence": "Begründung",
        "legal_basis": "Rechtsgrundlage",
        "categories": {
            rec.CATEGORY_GENERATED: "KI-generiert (AI-generated)",
            rec.CATEGORY_MODIFIED: "KI-bearbeitet (AI-modified)",
            rec.CATEGORY_ASSISTED: "KI-unterstützt (AI-assisted)",
        },
        "labellings": {
            rec.LABELLING_RECOMMENDED: "empfohlen",
            rec.LABELLING_OPTIONAL: "voraussichtlich nicht erforderlich "
            "(assistive Standardbearbeitung, Art. 50 Abs. 2 KI-VO); freiwillig möglich",
        },
        "confidences": {rec.CONFIDENCE_HIGH: "hoch", rec.CONFIDENCE_MEDIUM: "mittel (nur Indizien)"},
        "legal_basis_value": "Art. 50 Abs. 2 und 4 KI-VO (EU AI Act)",
        "weak": " – nur Indiz",
        "evidence_kinds": {
            rec.EVIDENCE_IPTC: "IPTC Digital Source Type „{v}“",
            rec.EVIDENCE_C2PA_ACTION: "C2PA-Aktion {v}",
            rec.EVIDENCE_GENERATION_DATA: "Eingebettete Generierungsdaten ({v})",
            rec.EVIDENCE_WATERMARK: "Unsichtbares Wasserzeichen ({v})",
            rec.EVIDENCE_AI_SYSTEM_FIELD: "IPTC-Feld zum verwendeten KI-System ({v})",
            rec.EVIDENCE_GENERATOR_NAME: "Generatorname „{v}“ in den Metadaten",
        },
        "note": "Empfehlung allein auf Basis maschinenlesbarer Metadaten, keine Rechtsberatung. "
        "Für Deepfakes ist eine sichtbare Offenlegung Pflicht (Art. 50 Abs. 4); assistive "
        "Standardbearbeitung ohne wesentliche Veränderung ist von der Kennzeichnungspflicht "
        "ausgenommen (Art. 50 Abs. 2).",
    },
    "en": {
        "heading": "Labelling recommendation (EU AI Act, Art. 50):",
        "category": "Category",
        "labelling": "Labelling",
        "confidence": "Confidence",
        "evidence": "Evidence",
        "legal_basis": "Legal basis",
        "categories": {
            rec.CATEGORY_GENERATED: "AI-generated",
            rec.CATEGORY_MODIFIED: "AI-modified",
            rec.CATEGORY_ASSISTED: "AI-assisted",
        },
        "labellings": {
            rec.LABELLING_RECOMMENDED: "recommended",
            rec.LABELLING_OPTIONAL: "probably not required (assistive standard editing, "
            "Art. 50(2) EU AI Act); voluntary labelling possible",
        },
        "confidences": {rec.CONFIDENCE_HIGH: "high", rec.CONFIDENCE_MEDIUM: "medium (indications only)"},
        "legal_basis_value": rec.LEGAL_BASIS,
        "weak": " – indication only",
        "evidence_kinds": {},  # English descriptions come with the evidence
        "note": rec.NOTE,
    },
}


def recommendation_summary(recommendation: rec.Recommendation, lang: str = DEFAULT_LANGUAGE) -> dict[str, str]:
    """Localised category, labelling and confidence texts (used by the report and the web UI)."""
    t = RECOMMENDATION_TEXTS[lang if lang in LANGUAGES else DEFAULT_LANGUAGE]
    return {
        "category": t["categories"].get(recommendation.category, recommendation.label),
        "labelling": t["labellings"].get(recommendation.labelling, recommendation.labelling),
        "confidence": t["confidences"].get(recommendation.confidence, recommendation.confidence),
    }


def badge_lines(recommendation: rec.Recommendation, lang: str = DEFAULT_LANGUAGE) -> list[str]:
    """Framed badge with the recommendation, e.g. for the top of the plain-text output."""
    t = RECOMMENDATION_TEXTS[lang if lang in LANGUAGES else DEFAULT_LANGUAGE]
    summary = recommendation_summary(recommendation, lang)
    content_width = LINE_WIDTH - 4  # "│ " + content + " │"
    rows = _wrap(summary["category"].upper(), content_width)
    rows += _wrap(
        f"{t['labelling']}: {summary['labelling']} · {t['confidence']}: {summary['confidence']}", content_width
    )
    width = max(len(r) for r in rows)
    return (
        ["┌" + "─" * (width + 2) + "┐"]
        + [f"│ {r:<{width}} │" for r in rows]
        + ["└" + "─" * (width + 2) + "┘"]
    )


def _recommendation_lines(recommendation: rec.Recommendation, lang: str) -> list[str]:
    t = RECOMMENDATION_TEXTS[lang]
    summary = recommendation_summary(recommendation, lang)
    evidence = []
    for e in recommendation.evidence:
        template = t["evidence_kinds"].get(e.kind)
        text = template.format(v=e.value) if template else e.description
        evidence.append(text + ("" if e.strong else t["weak"]))
    lines = [t["heading"]]
    lines += _detail_lines(t["category"], summary["category"], INDENT)
    lines += _detail_lines(t["labelling"], summary["labelling"], INDENT)
    lines += _detail_lines(t["confidence"], summary["confidence"], INDENT)
    lines += _detail_lines(t["evidence"], evidence, INDENT)
    lines += _detail_lines(t["legal_basis"], t["legal_basis_value"], INDENT)
    lines += [INDENT + line for line in _wrap(t["note"], LINE_WIDTH - len(INDENT))]
    return lines


def _detail_lines(key: str, value: Any, indent: str) -> list[str]:
    head = f"{indent}{key}: "
    if isinstance(value, (list, tuple)):
        if not value:
            return [head + "–"]
        lines = [head.rstrip()]
        for item in value:
            parts = _wrap(str(item), max(40, LINE_WIDTH - len(indent) - 4))
            lines.append(f"{indent}  - {parts[0]}")
            lines += [f"{indent}    {p}" for p in parts[1:]]
        return lines
    if isinstance(value, dict):
        lines = [head.rstrip()]
        for k, v in value.items():
            lines += _detail_lines(str(k), v, indent + "  ")
        return lines
    parts = _wrap(str(value), max(40, LINE_WIDTH - len(head)))
    continuation = " " * len(head)
    return [head + parts[0]] + [continuation + p for p in parts[1:]]


def to_text(result: Result, lang: str = DEFAULT_LANGUAGE, badge: bool = True) -> str:
    """Plain-text report. ``badge`` puts a framed recommendation badge at the top (if there is
    a recommendation); the web UI disables it because it shows its own HTML badge."""
    if lang not in LANGUAGES:
        lang = DEFAULT_LANGUAGE
    t = REPORT_TEXTS[lang]
    german = lang == "de"
    width = max(len(t[k]) for k in ("file", "format", "size", "verdict")) + 2

    def head(key: str, value: str) -> str:
        return f"{t[key] + ':':<{width}}{value}"

    lines = []
    if badge and result.recommendation:
        lines += badge_lines(result.recommendation, lang) + [""]
    lines += [t["heading"], "=" * 52]
    lines.append(head("file", result.file))
    lines.append(head("format", result.format or t["unknown"]))
    size = f"{result.size[0]} × {result.size[1]} {t['pixels']}" if result.size else t["unknown"]
    if result.file_size_bytes:
        size += f", {format_file_size(result.file_size_bytes, lang)}"
    lines.append(head("size", size))
    verdict = VERDICTS_DE.get(result.verdict_level, result.verdict) if german else result.verdict
    lines.append(head("verdict", verdict))
    lines.append("")

    if result.recommendation:
        lines += _recommendation_lines(result.recommendation, lang)
        lines.append("")

    prefixes = LEVEL_PREFIX if german else LEVEL_PREFIX_EN
    sources = SOURCE_LABELS if german else SOURCE_LABELS_EN
    if result.findings:
        lines.append(f"{t['findings']} ({len(result.findings)}):")
        for finding in result.findings:
            title = finding_title_de(finding) if german else finding.title
            details = finding_details_de(finding) if german else finding_details_en(finding)
            lines.append("")
            lines.append(f"{prefixes.get(finding.level, '[?]')} {title}")
            lines.append(f"{INDENT}{t['source']}: {sources.get(finding.source, finding.source)}")
            for label, value in details:
                lines += _detail_lines(label, value, INDENT)
    else:
        lines.append(t["nothing_found"])

    if result.errors:
        lines.append("")
        lines.append(t["errors"])
        for error in result.errors:
            parts = _wrap(error_de(error) if german else error_en(error), LINE_WIDTH - 4)
            lines.append(f"  - {parts[0]}")
            lines += [f"    {p}" for p in parts[1:]]

    lines.append("")
    lines.append("-" * 52)
    lines += _wrap(t["disclaimer"])
    return "\n".join(lines) + "\n"
