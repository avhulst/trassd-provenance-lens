"""HTTP interface (FastAPI).

The JSON API is English. The web UI is available in German and English (language switch
in the header, remembered in a cookie); the plain-text report follows the UI language.
"""

from __future__ import annotations

import html
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Callable, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Security, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.security import APIKeyHeader, HTTPBearer
from starlette.requests import HTTPConnection

from . import __version__, auth
from .detectors import analyze
from .mcp_server import build_server as build_mcp_server
from .mcp_server import transport_security as mcp_transport_security
from .report import DEFAULT_LANGUAGE, LANGUAGES, recommendation_summary, to_text


# MCP server (Streamable HTTP) at /mcp – same access protection as the REST API.
# Base64 inflates images by ~4/3, plus JSON-RPC overhead.
mcp_server = build_mcp_server()
_mcp_app = mcp_server.streamable_http_app(
    streamable_http_path="/mcp",
    max_request_body_size=int(float(os.environ.get("MAX_UPLOAD_MB", "50") or 50) * 1024 * 1024 * 4 / 3) + 64 * 1024,
    transport_security=mcp_transport_security(),
    host="0.0.0.0",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    auth.api_keys()  # aborts startup if API_KEY_FILE is set but not readable
    async with mcp_server.session_manager.run():
        yield


app = FastAPI(
    title="AI Label Checker",
    description=(
        "Checks images for machine-readable AI labels (Art. 50(2) EU AI Act). "
        "The `report` field / text output is German by default (`lang=en` for English). "
        "An MCP server (Streamable HTTP) is available at `/mcp`. "
        "If `API_KEY` is set, `/analyze`, `/check` and `/mcp` require the key "
        "as `X-API-Key` header or `Authorization: Bearer …`."
    ),
    version=__version__,
    lifespan=lifespan,
)

# Only for the OpenAPI docs ("Authorize" in /docs); enforcement happens in the middleware.
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False, description="Access key (API_KEY)")
_BEARER = HTTPBearer(auto_error=False, description="Access key (API_KEY) as bearer token")
PROTECTED_PATHS = {"/analyze", "/check", "/mcp"}
FAILED_LOGIN_DELAY = 1.0  # seconds, slows down key guessing
READ_CHUNK = 1024 * 1024
LANG_COOKIE = "ui_lang"
LANG_COOKIE_MAX_AGE = 365 * 24 * 3600

# ---------------------------------------------------------------------------
# UI texts
# ---------------------------------------------------------------------------

UI_TEXTS: dict[str, dict[str, str]] = {
    "de": {
        "app_name": "KI-Kennzeichnungsprüfer",
        "language": "Sprache",
        "check_heading": "Bild prüfen",
        "intro": "Prüft ein Originalbild auf maschinenlesbare KI-Kennzeichnungen: C2PA Content "
        "Credentials, IPTC Digital Source Type, XMP/EXIF-Metadaten, PNG-Generierungsdaten und "
        "unsichtbare Stable-Diffusion-Wasserzeichen.",
        "drop_html": "<strong>Bild hierher ziehen</strong> oder klicken, um eine Datei auszuwählen",
        "drop_again_html": "<strong>Anderes Bild hierher ziehen</strong> oder klicken",
        "submit": "Bild prüfen",
        "checking": "Wird geprüft …",
        "limit_note": "Maximal {mb} MB. Das Bild wird nur im Arbeitsspeicher verarbeitet und nicht gespeichert.",
        "too_large_js": " – zu groß, erlaubt sind {mb} MB",
        "number_locale": "de-DE",
        "report_heading": "Prüfbericht",
        "check_another": "Weiteres Bild prüfen",
        "error_heading": "Fehler",
        "too_large": "Datei zu groß. Erlaubt sind höchstens {mb} MB (MAX_UPLOAD_MB).",
        "login_heading": "Anmelden",
        "login_intro": "Dieser Dienst ist mit einem Zugangsschlüssel geschützt.",
        "key_label": "Zugangsschlüssel",
        "login": "Anmelden",
        "logout": "Abmelden",
        "invalid_key": "Der Schlüssel ist ungültig.",
        "session_expired": "Nicht angemeldet oder Sitzung abgelaufen. Bitte erneut anmelden.",
        "recommendation": "Empfehlung nach KI-Verordnung",
        "labelling": "Kennzeichnung",
        "confidence": "Sicherheit",
    },
    "en": {
        "app_name": "AI Label Checker",
        "language": "Language",
        "check_heading": "Check an image",
        "intro": "Checks an original image for machine-readable AI labels: C2PA Content "
        "Credentials, IPTC Digital Source Type, XMP/EXIF metadata, PNG generation data and "
        "invisible Stable Diffusion watermarks.",
        "drop_html": "<strong>Drag an image here</strong> or click to choose a file",
        "drop_again_html": "<strong>Drag another image here</strong> or click",
        "submit": "Check image",
        "checking": "Checking …",
        "limit_note": "Maximum {mb} MB. The image is processed in memory only and never stored.",
        "too_large_js": " – too large, the maximum is {mb} MB",
        "number_locale": "en-US",
        "report_heading": "Report",
        "check_another": "Check another image",
        "error_heading": "Error",
        "too_large": "File too large. The maximum is {mb} MB (MAX_UPLOAD_MB).",
        "login_heading": "Log in",
        "login_intro": "This service is protected with an access key.",
        "key_label": "Access key",
        "login": "Log in",
        "logout": "Log out",
        "invalid_key": "The key is invalid.",
        "session_expired": "Not logged in or session expired. Please log in again.",
        "recommendation": "Recommendation under the EU AI Act",
        "labelling": "Labelling",
        "confidence": "Confidence",
    },
}


def ui_language(conn: HTTPConnection, explicit: str | None = None) -> str:
    """Explicit choice (query/form) > cookie > Accept-Language > German."""
    for candidate in (explicit, conn.query_params.get("lang"), conn.cookies.get(LANG_COOKIE)):
        if candidate in LANGUAGES:
            return candidate
    for part in conn.headers.get("accept-language", "").split(","):
        code = part.split(";")[0].strip().lower()[:2]
        if code in LANGUAGES:
            return code
    return DEFAULT_LANGUAGE


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------


class UploadTooLarge(Exception):
    def __init__(self, limit_mb: float):
        self.limit_mb = limit_mb


def upload_limit_mb() -> float:
    try:
        return float(os.environ.get("MAX_UPLOAD_MB", "50"))
    except ValueError:
        return 50.0


def _read_upload(request: Request, upload: UploadFile) -> bytes:
    limit_mb = upload_limit_mb()
    limit = int(limit_mb * 1024 * 1024)
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > limit + 64 * 1024:  # allowance for multipart headers
        raise UploadTooLarge(limit_mb)
    chunks: list[bytes] = []
    total = 0
    while chunk := upload.file.read(READ_CHUNK):
        total += len(chunk)
        if total > limit:
            raise UploadTooLarge(limit_mb)
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; --bg:#f7f7f5; --fg:#1d1d1f; --muted:#5f6368; --card:#fff;
          --border:#d9d9d6; --accent:#1f5fbf; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#141518; --fg:#e8e8ea; --muted:#a0a3a8; --card:#1e2024; --border:#33363c; --accent:#7aa7ff; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width:860px; margin:0 auto; padding:20px 16px 48px; }}
  h1 {{ font-size:1.5rem; margin:0 0 .25rem; }}
  p {{ color:var(--muted); margin:.25rem 0 1.25rem; }}
  form, pre {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:20px; }}
  .topbar {{ display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;
            padding:0 0 14px; margin:0 0 24px; border-bottom:1px solid var(--border); }}
  .brand {{ color:var(--fg); font-weight:650; text-decoration:none; }}
  .topbar-actions {{ display:flex; align-items:center; gap:16px; font-size:.9rem; }}
  .topbar form {{ background:none; border:0; padding:0; }}
  .lang-switch {{ display:flex; border:1px solid var(--border); border-radius:999px; overflow:hidden; }}
  .lang-switch a, .lang-switch span {{ padding:3px 10px; text-decoration:none; color:var(--muted); }}
  .lang-switch a:hover {{ color:var(--fg); }}
  .lang-switch [aria-current] {{ background:var(--accent); color:#fff; font-weight:600; }}
  @media (prefers-color-scheme: dark) {{ .lang-switch [aria-current] {{ color:#0b1020; }} }}
  .dropzone {{ position:relative; display:flex; flex-direction:column; align-items:center; justify-content:center;
            gap:.35rem; min-height:190px; margin:0 0 16px; padding:24px 16px; text-align:center; cursor:pointer;
            border:2px dashed var(--border); border-radius:10px; transition:border-color .15s, background .15s; }}
  .dropzone:hover, .dropzone:focus-within {{ border-color:var(--accent); }}
  .dropzone.active {{ border-color:var(--accent); border-style:solid;
                  background:color-mix(in srgb, var(--accent) 10%, transparent); }}
  .dropzone .icon {{ width:40px; height:40px; color:var(--accent); }}
  .dropzone strong {{ font-weight:600; }}
  .dropzone .selected {{ color:var(--fg); overflow-wrap:anywhere; }}
  .dropzone .warning {{ color:#c5221f; }}
  @media (prefers-color-scheme: dark) {{ .dropzone .warning {{ color:#f28b82; }} }}
  .dropzone img {{ max-width:100%; max-height:160px; border-radius:6px; }}
  .dropzone input[type=file] {{ position:absolute; width:1px; height:1px; opacity:0; pointer-events:none; }}
  button:disabled {{ opacity:.5; cursor:not-allowed; }}
  button, a.button {{ background:var(--accent); color:#fff; border:0; border-radius:8px; padding:10px 18px;
           font:inherit; cursor:pointer; text-decoration:none; display:inline-block; }}
  @media (prefers-color-scheme: dark) {{ button, a.button {{ color:#0b1020; }} }}
  pre {{ white-space:pre-wrap; overflow-wrap:anywhere; font:13.5px/1.5 ui-monospace, Menlo, Consolas, monospace; }}
  small {{ color:var(--muted); }}
  button.link-button {{ background:none; color:var(--accent); padding:0; font-size:inherit; }}
  .login label {{ display:block; font-weight:600; margin:0 0 6px; }}
  .login input[type=password] {{ display:block; width:100%; margin:0 0 16px; padding:10px 12px; font:inherit;
            color:var(--fg); background:var(--bg); border:1px solid var(--border); border-radius:8px; }}
  .login input[type=password]:focus {{ outline:2px solid var(--accent); outline-offset:1px; }}
  .recommendation {{ display:flex; flex-direction:column; gap:2px; margin:0 0 16px; padding:14px 18px;
            background:var(--card); border:1px solid var(--border); border-left:4px solid var(--accent);
            border-radius:10px; }}
  .recommendation .rec-title {{ font-size:.85rem; color:var(--muted); }}
  .recommendation .rec-category {{ font-size:1.15rem; font-weight:650; }}
  .recommendation .rec-meta {{ color:var(--muted); font-size:.9rem; }}
  .recommendation.ai_generated {{ border-left-color:#d93025; }}
  .recommendation.ai_modified {{ border-left-color:#e37400; }}
  @media (prefers-color-scheme: dark) {{
    .recommendation.ai_generated {{ border-left-color:#f28b82; }}
    .recommendation.ai_modified {{ border-left-color:#fdd663; }}
  }}
  .error {{ color:#c5221f; margin:-6px 0 14px; }}
  @media (prefers-color-scheme: dark) {{ .error {{ color:#f28b82; }} }}
</style>
</head>
<body><main>
{header}
{content}
</main></body>
</html>
"""

# Drag & drop for the upload field. Without JavaScript the form still works via click.
# Texts come from the JSON block with id "ui-texts" so the script is language-independent.
UPLOAD_SCRIPT = """<script>
(() => {
  const t = JSON.parse(document.getElementById("ui-texts").textContent);
  const form = document.getElementById("upload-form");
  const dropzone = document.getElementById("dropzone");
  const input = document.getElementById("file");
  const label = document.getElementById("dropzone-file");
  const preview = document.getElementById("preview");
  const submit = document.getElementById("submit");
  const icon = dropzone.querySelector(".icon");
  const hint = document.getElementById("dropzone-text");
  const limit = parseFloat(form.dataset.limitMb) * 1024 * 1024;

  const formatSize = (b) => b < 1024 * 1024
    ? (b / 1024).toLocaleString(t.locale, {maximumFractionDigits: 1}) + " KB"
    : (b / 1024 / 1024).toLocaleString(t.locale, {maximumFractionDigits: 1}) + " MB";

  function render() {
    const file = input.files[0];
    label.classList.remove("warning");
    preview.hidden = true;
    icon.style.display = "";
    hint.innerHTML = file ? t.dropAgain : t.drop;
    if (preview.src) { URL.revokeObjectURL(preview.src); preview.removeAttribute("src"); }
    if (!file) { label.textContent = ""; submit.disabled = true; return; }
    label.textContent = file.name + " (" + formatSize(file.size) + ")";
    if (file.size > limit) {
      label.classList.add("warning");
      label.textContent += t.tooLarge;
      submit.disabled = true;
      return;
    }
    if (file.type.startsWith("image/")) {
      preview.src = URL.createObjectURL(file);
      preview.hidden = false;
      icon.style.display = "none";
    }
    submit.disabled = false;
  }

  input.addEventListener("change", render);

  // Prevents the browser from simply opening an image dropped next to the zone.
  ["dragover", "drop"].forEach((type) => window.addEventListener(type, (e) => e.preventDefault()));

  let depth = 0;
  dropzone.addEventListener("dragenter", (e) => { e.preventDefault(); depth++; dropzone.classList.add("active"); });
  dropzone.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; dropzone.classList.remove("active"); } });
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    depth = 0;
    dropzone.classList.remove("active");
    const files = e.dataTransfer.files;
    if (!files.length) return;
    const transfer = new DataTransfer();
    transfer.items.add(files[0]);
    input.files = transfer.files;
    render();
  });

  form.addEventListener("submit", () => {
    submit.disabled = true;
    submit.textContent = t.checking;
  });
  window.addEventListener("pageshow", () => { submit.textContent = t.submit; render(); });
})();
</script>"""


def _header(lang: str, show_logout: bool, in_place_switch: bool = False) -> str:
    """Page header. With ``in_place_switch`` the language links carry ``data-switch-lang`` so
    the bilingual-page script can switch without leaving the page (fallback: start page)."""
    t = UI_TEXTS[lang]
    switch = []
    for code in LANGUAGES:
        name = code.upper()
        if code == lang:
            switch.append(f'<span aria-current="true" lang="{code}">{name}</span>')
        else:
            extra = f' data-switch-lang="{code}"' if in_place_switch else ""
            switch.append(f'<a href="/?lang={code}" lang="{code}" hreflang="{code}"{extra}>{name}</a>')
    logout = (
        f'<form action="/logout" method="post"><input type="hidden" name="lang" value="{lang}">'
        f'<button type="submit" class="link-button">{t["logout"]}</button></form>'
        if show_logout
        else ""
    )
    return f"""<header class="topbar">
  <a class="brand" href="/">{t["app_name"]}</a>
  <div class="topbar-actions">
    <nav class="lang-switch" aria-label="{t["language"]}">{"".join(switch)}</nav>
    {logout}
  </div>
</header>"""


def _page(
    content: str,
    lang: str,
    status: int = 200,
    show_logout: bool = False,
    remember_lang: bool = False,
) -> HTMLResponse:
    response = HTMLResponse(
        PAGE.format(
            lang=lang,
            title=UI_TEXTS[lang]["app_name"],
            header=_header(lang, show_logout),
            content=content,
        ),
        status_code=status,
    )
    if remember_lang:
        response.set_cookie(LANG_COOKIE, lang, max_age=LANG_COOKIE_MAX_AGE, samesite="lax", path="/")
    return response


# Switches a bilingual page (see _bilingual_page) without reloading and remembers the choice.
SWITCH_SCRIPT = """<script>
(() => {
  document.querySelectorAll("[data-switch-lang]").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const lang = link.dataset.switchLang;
      let title = document.title;
      document.querySelectorAll("[data-lang-block]").forEach((block) => {
        block.hidden = block.dataset.langBlock !== lang;
        if (!block.hidden) title = block.dataset.title;
      });
      document.documentElement.lang = lang;
      document.title = title;
      document.cookie = "ui_lang=" + lang + "; max-age=31536000; path=/; samesite=lax";
    });
  });
})();
</script>"""


def _bilingual_page(
    render: Callable[[str], str], lang: str, status: int = 200, show_logout: bool = False
) -> HTMLResponse:
    """Renders header + content for every language; only ``lang`` is visible.

    Used for pages that cannot be re-created after a reload (the check result: the upload
    is not stored), so the language switch keeps the page instead of going back to start.
    """
    blocks = []
    for code in LANGUAGES:
        hidden = "" if code == lang else " hidden"
        blocks.append(
            f'<div data-lang-block="{code}" data-title="{html.escape(UI_TEXTS[code]["app_name"])}" lang="{code}"{hidden}>\n'
            f"{_header(code, show_logout, in_place_switch=True)}\n{render(code)}\n</div>"
        )
    return HTMLResponse(
        PAGE.format(
            lang=lang,
            title=UI_TEXTS[lang]["app_name"],
            header="",
            content="\n".join(blocks) + "\n" + SWITCH_SCRIPT,
        ),
        status_code=status,
    )


def _logged_in(request: Request) -> bool:
    return auth.protection_enabled() and auth.access_allowed(request)


def _login_page(lang: str, message: str = "", status: int = 200, remember_lang: bool = False) -> HTMLResponse:
    t = UI_TEXTS[lang]
    error = f'<p class="error" role="alert">{html.escape(message)}</p>' if message else ""
    return _page(
        f"""<h1>{t["login_heading"]}</h1>
<p>{t["login_intro"]}</p>
<form class="login" action="/login" method="post">
  <input type="hidden" name="lang" value="{lang}">
  <label for="key">{t["key_label"]}</label>
  <input type="password" id="key" name="key" autocomplete="current-password" required autofocus>
  {error}
  <button type="submit">{t["login"]}</button>
</form>""",
        lang,
        status,
        remember_lang=remember_lang,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.middleware("http")
async def access_control(request: Request, call_next):
    """Checks the key before the upload is read at all."""
    if request.url.path in PROTECTED_PATHS and not auth.access_allowed(request):
        if request.url.path == "/check":
            lang = ui_language(request)
            return _login_page(lang, UI_TEXTS[lang]["session_expired"], 401)
        return JSONResponse(
            {"detail": "Missing or invalid access key (X-API-Key or Authorization: Bearer)."},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


@app.post("/login", include_in_schema=False)
def login(request: Request, key: str = Form(""), lang: str = Form("")):
    lang = ui_language(request, lang)
    if not auth.protection_enabled():
        return RedirectResponse(f"/?lang={lang}", status_code=303)
    if not auth.key_valid(key):
        time.sleep(FAILED_LOGIN_DELAY)
        return _login_page(lang, UI_TEXTS[lang]["invalid_key"], 401)
    response = RedirectResponse(f"/?lang={lang}", status_code=303)
    response.set_cookie(
        auth.SESSION_COOKIE,
        auth.create_session(key),
        max_age=auth.session_duration_seconds(),
        httponly=True,
        samesite="lax",
        secure=auth.cookie_secure(request),
        path="/",
    )
    return response


@app.post("/logout", include_in_schema=False)
def logout(request: Request, lang: str = Form("")):
    lang = ui_language(request, lang)
    response = RedirectResponse(f"/?lang={lang}", status_code=303)
    response.delete_cookie(
        auth.SESSION_COOKIE, path="/", httponly=True, samesite="lax", secure=auth.cookie_secure(request)
    )
    return response


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request) -> HTMLResponse:
    lang = ui_language(request)
    remember = request.query_params.get("lang") in LANGUAGES
    if auth.protection_enabled() and not auth.access_allowed(request):
        return _login_page(lang, remember_lang=remember)
    t = UI_TEXTS[lang]
    limit = f"{upload_limit_mb():g}"
    js_texts = {
        "drop": t["drop_html"],
        "dropAgain": t["drop_again_html"],
        "submit": t["submit"],
        "checking": t["checking"],
        "tooLarge": t["too_large_js"].format(mb=limit),
        "locale": t["number_locale"],
    }
    # "</" must not appear inside the JSON script block.
    js_json = json.dumps(js_texts, ensure_ascii=False).replace("</", "<\\/")
    return _page(
        f"""<h1>{t["check_heading"]}</h1>
<p>{t["intro"]}</p>
<form id="upload-form" action="/check" method="post" enctype="multipart/form-data" data-limit-mb="{limit}">
  <input type="hidden" name="lang" value="{lang}">
  <label class="dropzone" id="dropzone">
    <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
         stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>
    </svg>
    <img id="preview" alt="" hidden>
    <span id="dropzone-text">{t["drop_html"]}</span>
    <span id="dropzone-file" class="selected" aria-live="polite"></span>
    <input type="file" id="file" name="file" accept="image/*" required>
  </label>
  <button type="submit" id="submit" disabled>{t["submit"]}</button>
  <p><small>{t["limit_note"].format(mb=limit)}</small></p>
</form>
<script id="ui-texts" type="application/json">{js_json}</script>
{UPLOAD_SCRIPT}""",
        lang,
        show_logout=_logged_in(request),
        remember_lang=remember,
    )


@app.post("/check", response_class=HTMLResponse)
def check(request: Request, file: UploadFile = File(...), lang: str = Form("")) -> HTMLResponse:
    lang = ui_language(request, lang)
    show_logout = _logged_in(request)

    def back(code: str) -> str:
        return f'<p><a class="button" href="/">{UI_TEXTS[code]["check_another"]}</a></p>'

    try:
        data = _read_upload(request, file)
    except UploadTooLarge as exc:
        limit = f"{exc.limit_mb:g}"

        def error_page(code: str) -> str:
            t = UI_TEXTS[code]
            message = html.escape(t["too_large"].format(mb=limit))
            return f"<h1>{t['error_heading']}</h1><p>{message}</p>{back(code)}"

        return _bilingual_page(error_page, lang, 413, show_logout)

    result = analyze(data, file.filename or "upload")

    def recommendation_box(code: str) -> str:
        if not result.recommendation:
            return ""
        t = UI_TEXTS[code]
        summary = {k: html.escape(v) for k, v in recommendation_summary(result.recommendation, code).items()}
        return (
            f'<div class="recommendation {html.escape(result.recommendation.category)}" role="note">'
            f'<span class="rec-title">{t["recommendation"]}</span>'
            f'<span class="rec-category">{summary["category"]}</span>'
            f'<span class="rec-meta">{t["labelling"]}: {summary["labelling"]} · '
            f'{t["confidence"]}: {summary["confidence"]}</span></div>'
        )

    def report_page(code: str) -> str:
        report = html.escape(to_text(result, code, badge=False))  # HTML badge is shown instead
        return (
            f"<h1>{UI_TEXTS[code]['report_heading']}</h1>\n{recommendation_box(code)}\n"
            f"<pre>{report}</pre>\n{back(code)}"
        )

    return _bilingual_page(report_page, lang, show_logout=show_logout)


@app.post("/analyze", dependencies=[Security(_API_KEY_HEADER), Security(_BEARER)])
def analyze_endpoint(
    request: Request,
    file: UploadFile = File(...),
    format: Literal["json", "text"] = Query("json", description="Output format"),
    lang: Literal["de", "en"] = Query(DEFAULT_LANGUAGE, description="Language of the report (`report` field / text output)"),
):
    try:
        data = _read_upload(request, file)
    except UploadTooLarge as exc:
        detail = UI_TEXTS["en"]["too_large"].format(mb=f"{exc.limit_mb:g}")
        raise HTTPException(status_code=413, detail=detail) from None
    result = analyze(data, file.filename or "upload")
    report = to_text(result, lang)
    if format == "text":
        return PlainTextResponse(report)
    return JSONResponse({**result.to_dict(), "report": report})


app.router.routes.extend(_mcp_app.routes)  # POST/GET/DELETE /mcp (MCP Streamable HTTP)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
