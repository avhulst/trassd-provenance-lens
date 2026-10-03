"""MCP server (Model Context Protocol) for Claude Desktop, Mistral Vibe and other MCP clients.

Two transports:

- **stdio** – ``python -m app.mcp_server``; the client starts the process itself
  (e.g. ``docker run -i --rm ... trassd-provenance-lens python -m app.mcp_server``).
- **Streamable HTTP** – mounted into the FastAPI app at ``/mcp`` (see ``app.main``),
  protected by the same ``API_KEY`` as the REST API.

Tools:

- ``analyze_image`` – image as base64 (works with every transport).
- ``list_image_files`` / ``analyze_image_file`` – only when ``MCP_IMAGE_DIR`` is set: read
  images from that directory (e.g. a read-only volume). Paths cannot escape it.
"""

from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from . import __version__
from .detectors import analyze
from .report import DEFAULT_LANGUAGE, to_text

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif", ".heic", ".heif", ".avif"}
MAX_LISTED_FILES = 500

INSTRUCTIONS = """\
Checks images for machine-readable AI labels (EU AI Act, Art. 50(2)): C2PA Content \
Credentials, IPTC Digital Source Type, XMP/EXIF metadata, PNG generation data and invisible \
Stable Diffusion watermarks.

The result contains `verdict_level` (ai / hint / none), the `findings` and – only if AI \
involvement is evidenced – `ai_label_recommendation` with `category` (ai_generated / \
ai_modified / ai_assisted) and `labelling`. `report` is a ready-made human-readable report \
(German by default, `lang="en"` for English).

Always point out: a missing label does not prove authenticity (metadata is lost through \
screenshots, re-saving and social media uploads; proprietary watermarks such as SynthID \
cannot be checked here), and the recommendation is not legal advice. Analyse the original file \
where possible."""

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def upload_limit_bytes() -> int:
    try:
        return int(float(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024)
    except ValueError:
        return 50 * 1024 * 1024


def image_dir() -> Path | None:
    value = os.environ.get("MCP_IMAGE_DIR", "").strip()
    return Path(value).resolve() if value else None


def _result(data: bytes, filename: str, lang: str) -> dict[str, Any]:
    result = analyze(data, filename)
    return {**result.to_dict(), "report": to_text(result, lang)}


def _check_size(size: int) -> None:
    limit = upload_limit_bytes()
    if size > limit:
        raise ToolError(f"File too large: {size} bytes; the maximum is {limit} bytes (MAX_UPLOAD_MB).")


def resolve_image_path(path: str) -> Path:
    """Resolves ``path`` inside MCP_IMAGE_DIR; refuses anything outside it."""
    root = image_dir()
    if root is None:
        raise ToolError("Reading files is disabled (MCP_IMAGE_DIR is not set).")
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ToolError(f"Path is outside the image directory {root}: {path}")
    if not resolved.is_file():
        raise ToolError(f"File not found: {path}")
    return resolved


def build_server() -> MCPServer:
    server = MCPServer(
        name="trassd-provenance-lens",
        title="AI Label Checker",
        instructions=INSTRUCTIONS,
        version=__version__,
    )

    @server.tool(
        title="Analyze image (base64)",
        description=(
            "Checks an image for machine-readable AI labels. Pass the image file content as "
            "base64 (no data: prefix needed). Returns the analysis as JSON plus a "
            "human-readable report."
        ),
        annotations=_READ_ONLY,
    )
    def analyze_image(
        image_base64: str,
        filename: str = "image",
        lang: Literal["de", "en"] = DEFAULT_LANGUAGE,
    ) -> dict[str, Any]:
        payload = image_base64.strip()
        if payload.startswith("data:") and "," in payload:
            payload = payload.split(",", 1)[1]
        _check_size(len(payload) * 3 // 4)
        try:
            data = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as exc:  # expected input errors reach the model as ToolError
            raise ToolError(f"Invalid base64 data: {exc}") from None
        return _result(data, filename, lang)

    if image_dir() is not None:

        @server.tool(
            title="List image files",
            description=(
                "Lists image files in the configured image directory (MCP_IMAGE_DIR), optionally "
                "in a subdirectory. Paths are relative to that directory."
            ),
            annotations=_READ_ONLY,
        )
        def list_image_files(subdirectory: str = "") -> dict[str, Any]:
            root = image_dir()
            base = (root / subdirectory).resolve() if subdirectory else root
            if base != root and root not in base.parents:
                raise ToolError(f"Path is outside the image directory: {subdirectory}")
            if not base.is_dir():
                raise ToolError(f"Directory not found: {subdirectory}")
            files = sorted(
                str(p.relative_to(root))
                for p in base.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            return {
                "directory": str(base.relative_to(root)) if base != root else ".",
                "files": files[:MAX_LISTED_FILES],
                "truncated": len(files) > MAX_LISTED_FILES,
            }

        @server.tool(
            title="Analyze image file",
            description=(
                "Checks an image file from the configured image directory (MCP_IMAGE_DIR) for "
                "machine-readable AI labels. `path` is relative to that directory (see "
                "list_image_files). Returns the analysis as JSON plus a human-readable report."
            ),
            annotations=_READ_ONLY,
        )
        def analyze_image_file(path: str, lang: Literal["de", "en"] = DEFAULT_LANGUAGE) -> dict[str, Any]:
            resolved = resolve_image_path(path)
            _check_size(resolved.stat().st_size)
            return _result(resolved.read_bytes(), resolved.name, lang)

    return server


def transport_security() -> TransportSecuritySettings:
    """DNS rebinding protection for the HTTP transport.

    ``MCP_ALLOWED_HOSTS`` (comma-separated Host header values, ``*`` as port wildcard, e.g.
    ``checker.example.com,localhost:*``) – default: localhost only. ``*`` disables the check.
    """
    raw = os.environ.get("MCP_ALLOWED_HOSTS", "").strip()
    if raw == "*":
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    hosts = [h.strip() for h in raw.split(",") if h.strip()] or ["localhost:*", "127.0.0.1:*", "[::1]:*"]
    origins = [f"{scheme}://{h}" for h in hosts for scheme in ("http", "https")]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins
    )


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
