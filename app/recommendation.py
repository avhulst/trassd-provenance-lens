"""Labelling recommendation under the EU AI Act (Art. 50) derived from the findings.

Classifies the degree of AI involvement into

- ``ai_generated``: the image as a whole was created by a (generative) AI system,
- ``ai_modified``: an existing image was manipulated with generative AI (e.g. Generative Fill),
- ``ai_assisted``: AI-based enhancement only (e.g. denoising, upscaling) – typically covered
  by the exception for assistive standard editing in Art. 50(2),

and returns ``None`` when no AI involvement is evidenced. Only machine-readable evidence
is used; the result is a recommendation, not legal advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import detectors as d

CATEGORY_GENERATED = "ai_generated"
CATEGORY_MODIFIED = "ai_modified"
CATEGORY_ASSISTED = "ai_assisted"
CATEGORY_PRIORITY = (CATEGORY_GENERATED, CATEGORY_MODIFIED, CATEGORY_ASSISTED)

LABELS_EN = {
    CATEGORY_GENERATED: "AI-generated",
    CATEGORY_MODIFIED: "AI-modified",
    CATEGORY_ASSISTED: "AI-assisted",
}

LABELLING_RECOMMENDED = "recommended"
LABELLING_OPTIONAL = "optional"

CONFIDENCE_HIGH = "high"  # backed by a standardised label or unambiguous generator data
CONFIDENCE_MEDIUM = "medium"  # only indications such as generator names in metadata

LEGAL_BASIS = "Art. 50(2) and (4) EU AI Act"
NOTE = (
    "Recommendation based on machine-readable metadata only; not legal advice. A visible "
    "disclosure is mandatory for deep fakes (Art. 50(4)); assistive standard editing that does "
    "not substantially alter the image is exempt from the marking obligation (Art. 50(2))."
)

# Evidence kinds (report.py renders them in German/English)
EVIDENCE_IPTC = "iptc_source_type"
EVIDENCE_C2PA_ACTION = "c2pa_action"
EVIDENCE_GENERATION_DATA = "generation_data"
EVIDENCE_WATERMARK = "watermark"
EVIDENCE_AI_SYSTEM_FIELD = "ai_system_field"
EVIDENCE_GENERATOR_NAME = "generator_name"

_GENERATED_FINDING_TYPES = {
    d.TYPE_GENERATION_PARAMETERS: EVIDENCE_GENERATION_DATA,
    d.TYPE_COMFYUI_WORKFLOW: EVIDENCE_GENERATION_DATA,
    d.TYPE_NOVELAI: EVIDENCE_GENERATION_DATA,
    d.TYPE_GENERATOR_METADATA: EVIDENCE_GENERATION_DATA,
    d.TYPE_SD_WATERMARK: EVIDENCE_WATERMARK,
    d.TYPE_SDXL_WATERMARK: EVIDENCE_WATERMARK,
    d.TYPE_XMP_AI_SYSTEM: EVIDENCE_AI_SYSTEM_FIELD,
}

# IPTC digital source type -> (category, strong evidence?)
_IPTC_CATEGORIES = {
    "trainedAlgorithmicMedia": (CATEGORY_GENERATED, True),
    "compositeWithTrainedAlgorithmicMedia": (CATEGORY_MODIFIED, True),
    "compositeSynthetic": (CATEGORY_MODIFIED, False),
    "algorithmicallyEnhanced": (CATEGORY_ASSISTED, True),
}

# Generator names that indicate editing of an existing image rather than creation
_MODIFYING_GENERATORS = {"Generative Fill"}

_GENERATOR_DETAIL_TYPES = {
    d.TYPE_C2PA_ACTIVE,
    d.TYPE_C2PA_INGREDIENT,
    d.TYPE_XMP_GENERATOR,
    d.TYPE_EXIF_GENERATOR,
    d.TYPE_PNG_TEXT_GENERATOR,
}


@dataclass
class Evidence:
    kind: str
    value: str
    finding_type: str
    source: str
    strong: bool
    description: str


@dataclass
class Recommendation:
    category: str
    label: str
    labelling: str
    confidence: str
    evidence: list[Evidence] = field(default_factory=list)
    legal_basis: str = LEGAL_BASIS
    note: str = NOTE

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "label": self.label,
            "labelling": self.labelling,
            "confidence": self.confidence,
            "evidence": [
                {
                    "kind": e.kind,
                    "value": e.value,
                    "finding_type": e.finding_type,
                    "source": e.source,
                    "description": e.description,
                }
                for e in self.evidence
            ],
            "legal_basis": self.legal_basis,
            "note": self.note,
        }


def _describe(kind: str, value: str) -> str:
    return {
        EVIDENCE_IPTC: f"IPTC digital source type '{value}'",
        EVIDENCE_C2PA_ACTION: f"C2PA action {value}",
        EVIDENCE_GENERATION_DATA: f"Embedded generation data ({value})",
        EVIDENCE_WATERMARK: f"Invisible watermark ({value})",
        EVIDENCE_AI_SYSTEM_FIELD: f"IPTC field naming the AI system used ({value})",
        EVIDENCE_GENERATOR_NAME: f"Generator name '{value}' in the metadata",
    }.get(kind, value)


def _evidence_from(finding: d.Finding) -> list[tuple[str, Evidence]]:
    found: list[tuple[str, Evidence]] = []

    def add(category: str, kind: str, value: str, strong: bool) -> None:
        found.append(
            (category, Evidence(kind, value, finding.type, finding.source, strong, _describe(kind, value)))
        )

    details = finding.details
    if finding.type in _GENERATED_FINDING_TYPES:
        kind = _GENERATED_FINDING_TYPES[finding.type]
        if kind == EVIDENCE_AI_SYSTEM_FIELD:
            value = str(details.get("Iptc4xmpExt:AISystemUsed") or "Iptc4xmpExt:AISystemUsed")
        elif kind == EVIDENCE_WATERMARK:
            value = "Stable Diffusion" if finding.type == d.TYPE_SD_WATERMARK else "SDXL"
        else:
            value = str(details.get("tool") or {d.TYPE_COMFYUI_WORKFLOW: "ComfyUI", d.TYPE_NOVELAI: "NovelAI"}.get(
                finding.type, finding.type
            ))
        add(CATEGORY_GENERATED, kind, value, True)

    if finding.type == d.TYPE_IPTC_SOURCE_TYPE:
        value = str(details.get("value"))
        if value in _IPTC_CATEGORIES:
            category, strong = _IPTC_CATEGORIES[value]
            add(category, EVIDENCE_IPTC, value, strong)

    if finding.type in (d.TYPE_C2PA_ACTIVE, d.TYPE_C2PA_INGREDIENT):
        for action in details.get("actions") or []:
            if not isinstance(action, dict):
                continue
            dst = action.get("digital_source_type")
            if dst not in _IPTC_CATEGORIES:
                continue
            category, strong = _IPTC_CATEGORIES[dst]
            # An AI-sourced edit of an image that was not itself created by AI is a modification.
            if category == CATEGORY_GENERATED and action.get("action") != "c2pa.created":
                category = CATEGORY_MODIFIED
            add(category, EVIDENCE_C2PA_ACTION, f"{action.get('action')} ({dst})", strong)

    if finding.type in _GENERATOR_DETAIL_TYPES:
        for name in details.get("detected_generators") or []:
            category = CATEGORY_MODIFIED if name in _MODIFYING_GENERATORS else CATEGORY_GENERATED
            add(category, EVIDENCE_GENERATOR_NAME, str(name), False)
    return found


def recommend(result: d.Result) -> Recommendation | None:
    """Derives the labelling recommendation; ``None`` if no AI involvement is evidenced."""
    by_category: dict[str, list[Evidence]] = {c: [] for c in CATEGORY_PRIORITY}
    for finding in result.findings:
        for category, evidence in _evidence_from(finding):
            if all(e.description != evidence.description for e in by_category[category]):
                by_category[category].append(evidence)

    # Strong evidence beats mere indications; within the same strength the more
    # far-reaching category wins (generated > modified > assisted).
    chosen = next((c for c in CATEGORY_PRIORITY if any(e.strong for e in by_category[c])), None)
    if chosen is None:
        chosen = next((c for c in CATEGORY_PRIORITY if by_category[c]), None)
    if chosen is None:
        return None

    evidence = sorted(by_category[chosen], key=lambda e: not e.strong)
    return Recommendation(
        category=chosen,
        label=LABELS_EN[chosen],
        labelling=LABELLING_OPTIONAL if chosen == CATEGORY_ASSISTED else LABELLING_RECOMMENDED,
        confidence=CONFIDENCE_HIGH if evidence[0].strong else CONFIDENCE_MEDIUM,
        evidence=evidence,
    )
