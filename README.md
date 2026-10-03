# AI Label Checker

Docker container that checks an **original image** for the most common machine-readable
AI labels (context: Art. 50(2) EU AI Act). It returns the result as a JSON API (English)
and as a human-readable report in German or English (web UI, plain text, CLI).

Each image gets one of three verdicts:

| `verdict_level` | `verdict` (API) | German report | Meaning |
|---|---|---|---|
| `ai` | AI label found | KI-Kennzeichnung gefunden | At least one unambiguous, standardised AI label (finding `level` `ai`) |
| `hint` | Indications of AI generation found (no standardised label) | Hinweise auf KI-Erzeugung gefunden (keine standardisierte Kennzeichnung) | Indications only, e.g. a known generator name in the metadata (`level` `hint`) |
| `none` | No machine-readable AI label found | Keine maschinenlesbare KI-Kennzeichnung gefunden | At most provenance data without AI relation (`level` `info`) |

## Labelling recommendation (EU AI Act, Art. 50)

If AI involvement is evidenced, the API (`ai_label_recommendation`), the report and the
result page additionally recommend how to label the image. Otherwise the field is `null`
and nothing is shown. The plain-text report (`?format=text`, the JSON `report` field and
the CLI) starts with a framed badge; the web UI shows a coloured badge above the report
instead:

```
┌─────────────────────────────────────────────┐
│ KI-GENERIERT (AI-GENERATED)                 │
│ Kennzeichnung: empfohlen · Sicherheit: hoch │
└─────────────────────────────────────────────┘
```

| `category` | Label (EN / DE) | Typical evidence | `labelling` |
|---|---|---|---|
| `ai_generated` | AI-generated / KI-generiert | IPTC `trainedAlgorithmicMedia`; C2PA `c2pa.created` with a trained source type; embedded generation data (A1111, ComfyUI, NovelAI, InvokeAI, …); SD/SDXL watermark; IPTC `AISystemUsed`; generator names such as Midjourney or DALL·E | `recommended` |
| `ai_modified` | AI-modified / KI-bearbeitet | IPTC `compositeWithTrainedAlgorithmicMedia` or `compositeSynthetic`; C2PA edit action (not `c2pa.created`) with a trained source type; "Generative Fill" | `recommended` |
| `ai_assisted` | AI-assisted / KI-unterstützt | IPTC `algorithmicallyEnhanced` (e.g. AI denoising/upscaling) | `optional` – probably exempt as assistive standard editing (Art. 50(2)) |

- **Priority:** strong evidence (standardised labels, generation data, watermarks) beats
  mere indications (generator names); with equal strength the more far-reaching category
  wins (generated > modified > assisted). An image created by AI and edited afterwards is
  therefore `ai_generated`.
- **`confidence`:** `high` if backed by strong evidence, `medium` if only indications.
- **`evidence`:** list of `{kind, value, finding_type, source, description}` explaining
  the decision (`kind`: `iptc_source_type`, `c2pa_action`, `generation_data`, `watermark`,
  `ai_system_field`, `generator_name`).
- `algorithmicMedia` (procedural, no trained model) alone yields no recommendation.
- The recommendation is based on machine-readable metadata only and is **not legal
  advice**. A visible disclosure is mandatory for deep fakes (Art. 50(4)).

## What is checked

| Detector (`source`) | What is read | `level` |
|---|---|---|
| **C2PA / Content Credentials** (`c2pa`) | Manifests via `c2pa-python`: `claim_generator(_info)`, signature (issuer, time), `c2pa.actions(.v2)` with `action`, `digitalSourceType`, `softwareAgent`; validation state and status codes | Highest level of the source types per the IPTC table; known generator ⇒ at least `hint`. Unreadable manifest (JUMBF/`c2pa` in the raw bytes) ⇒ `hint` |
| **IPTC Digital Source Type** (`iptc_xmp`) | `…/digitalsourcetype/<value>` in all XMP packets (including compressed PNG iTXt) | `ai`: `trainedAlgorithmicMedia`, `compositeWithTrainedAlgorithmicMedia` · `hint`: `algorithmicMedia`, `compositeSynthetic`, `algorithmicallyEnhanced`, `dataDrivenMedia` · `info`: all other values (`digitalCapture`, `screenCapture`, …) |
| **Further XMP fields** (`xmp`) | `xmp:CreatorTool`, `photoshop:Credit`/`Source`, `dc:creator`/`description`/`rights`, `tiff:Software`, `Iptc4xmpExt:AISystemUsed`/`AISystemVersionUsed`/`AIPromptInformation`, `plus:DataMining` – as attribute or element (`rdf:Alt/Seq/Bag`) | `ai` for IPTC AI fields, `hint` if a generator is detected, otherwise `info` |
| **EXIF** (`exif`, `exif_user_comment`) | Make, Model, Software, ImageDescription, Artist, Copyright, UserComment (incl. ASCII/UNICODE/JIS prefix) | `hint` if a generator is detected, otherwise `info`; A1111 parameters in UserComment ⇒ `ai` |
| **PNG text chunks** (`png_text`) | `parameters` (A1111/Forge incl. model), `prompt`/`workflow` (ComfyUI), `Software: NovelAI…`, `invokeai_metadata`, `invokeai_graph`, `sd-metadata`, `dream`, `fooocus_scheme`, `generation_data` | `ai`; other text fields `info`, or `hint` if a generator is detected |
| **Invisible watermarks** (`watermark`) | `invisible-watermark` (dwtDct): Stable Diffusion v1/v2 (`StableDiffusionV…`, 136 bits) and SDXL (48 bits, ≤ 4 bit errors tolerated) – only if the shorter side is ≥ 256 px | `ai` |

Detected generator names (regex with word boundaries, case-insensitive): DALL·E, OpenAI,
ChatGPT, GPT-4o, GPT Image, Midjourney, Stable Diffusion, Stability AI, SDXL, ComfyUI,
A1111, InvokeAI, Fooocus, NovelAI, Adobe Firefly, Generative Fill, Imagen,
"Made with Google AI", Gemini, Nano Banana, Bing Image Creator, Microsoft Designer,
Leonardo.Ai, Ideogram, Black Forest Labs, FLUX.1/2, Runway, Krea, Playground AI,
DreamStudio, Recraft, Grok, Meta AI.

Ambiguous words only match in unambiguous form: "Imagen" only as "Google Imagen",
"Imagen 3" etc. or as a standalone value (Spanish *imagen* = image); "Runway" only as
"RunwayML", "Runway Gen-3" etc. or standalone.

## Limitations

- **A missing label does not prove authenticity.** The check only finds what is embedded
  in machine-readable form and has survived.
- Metadata (C2PA, XMP, EXIF, PNG text) is usually lost through **screenshots, re-saving,
  format conversion and uploads to social networks**. Always check the original file.
- **Proprietary watermarks** such as Google SynthID or Meta Video Seal can only be checked
  with the vendors' own tools and are not detected here.
- The Stable Diffusion watermarks (dwtDct) are fragile: scaling, cropping and strong JPEG
  compression often destroy them. The 136-bit watermark of SD v1/v2 in particular is often
  no longer read correctly on small or heavily textured images.
- **C2PA trust list:** no trust list is configured. Valid signatures are therefore reported
  as structurally valid (`validation_state: "Valid"`) with the issuer not trusted
  (`signingCredential.untrusted`). The provenance data is evaluated anyway.
- Metadata can be forged. A detected label is strong evidence but not forensic proof
  (exception: validly signed C2PA manifests from a trusted issuer).
- No content-based AI detection (classifier) – deliberately, because such methods are
  unreliable and are not a *label* within the meaning of Art. 50(2).

## Build & run

Released images are on Docker Hub (`linux/amd64` and `linux/arm64`):

```bash
docker run -d --name trassd-provenance-lens -p 8000:8000 avhulst2/trassd-provenance-lens:latest
```

Or build it yourself:

```bash
docker build -t trassd-provenance-lens .
docker run -d --name trassd-provenance-lens -p 8000:8000 trassd-provenance-lens
```

The examples below use the local name `trassd-provenance-lens`; replace it with
`avhulst2/trassd-provenance-lens:<version>` to use the published image.

Web UI: <http://localhost:8000/> – upload via drag & drop or file picker, with preview
and dark mode, in German and English. The **DE | EN** switch in the header stores the
choice in a cookie (`ui_lang`, one year); without a choice the browser language
(`Accept-Language`) decides, falling back to German. The report on the result page uses
the same language. `/?lang=en` or `/?lang=de` links directly to a language. The result page
contains the report in both languages, so switching there stays on the result (the upload
itself is never stored); without JavaScript the switch falls back to the start page. The container runs as non-root user `app` (UID 10001) and has a
`HEALTHCHECK` on `/health`. Images are processed in memory only and never stored.

The image is about 430 MB. `invisible-watermark` is installed with `--no-deps`, and
`torch` is replaced by an empty module at import time. This saves about 1 GB of PyTorch,
which would only be needed for the unused RivaGAN method.

## API

| Method & path | Description |
|---|---|
| `GET /` | HTML upload page with drag & drop (login page when protection is enabled); `?lang=de\|en` selects the UI language |
| `POST /login`, `POST /logout` | Web UI login/logout (only relevant when protection is enabled) |
| `POST /check` | Form upload (fields `file`, optional `lang`), report as HTML page in the UI language |
| `POST /analyze?format=json` | Upload (field `file`), result as JSON incl. field `report` (default) |
| `POST /analyze?format=text` | Upload (field `file`), plain-text report |
| `…&lang=de\|en` | Language of `report` / the text output for `/analyze` (default `de`; JSON fields are always English) |
| `GET /health` | `{"status": "ok", "version": "…"}` – always reachable without key |
| `GET /docs` | OpenAPI docs (reachable without key; enter the key via "Authorize") |
| `POST/GET/DELETE /mcp` | MCP server (Streamable HTTP), see [MCP server](#mcp-server-claude-desktop-mistral-vibe-other-mcp-clients) |

Uploads that are too large are rejected with **HTTP 413**.

```bash
curl -s localhost:8000/health

curl -s -F file=@image.jpg "localhost:8000/analyze?format=text"

curl -s -F file=@image.jpg "localhost:8000/analyze?format=text&lang=en"

curl -s -F file=@image.png localhost:8000/analyze | jq '{verdict_level, findings: [.findings[] | {type, level, title}]}'

# with access protection enabled (see below) – both variants are equivalent
curl -s -H "X-API-Key: $API_KEY" -F file=@image.jpg localhost:8000/analyze
curl -s -H "Authorization: Bearer $API_KEY" -F file=@image.jpg localhost:8000/analyze
```

Example JSON (shortened):

```json
{
  "file": "image.png",
  "format": "PNG",
  "size": {"width": 512, "height": 512},
  "file_size_bytes": 659151,
  "verdict": "AI label found",
  "verdict_level": "ai",
  "ai_label_recommendation": {
    "category": "ai_generated",
    "label": "AI-generated",
    "labelling": "recommended",
    "confidence": "high",
    "evidence": [{"kind": "generation_data", "value": "AUTOMATIC1111 / Forge", "finding_type": "generation_parameters", "source": "png_text", "description": "Embedded generation data (AUTOMATIC1111 / Forge)"}],
    "legal_basis": "Art. 50(2) and (4) EU AI Act",
    "note": "Recommendation based on machine-readable metadata only; not legal advice. …"
  },
  "findings": [
    {
      "type": "generation_parameters",
      "source": "png_text",
      "level": "ai",
      "title": "Generation parameters from AUTOMATIC1111 / Forge (model: sd_xl_base_1.0)",
      "details": {"tool": "AUTOMATIC1111 / Forge", "prompt": "a photo of a red fox …", "model": "sd_xl_base_1.0", "seed": "1234567890", "raw": "…"}
    }
  ],
  "errors": [],
  "report": "KI-Kennzeichnungsprüfung (Art. 50 Abs. 2 EU AI Act)\n…"
}
```

### Response format

- **Language:** everything is English except `report`, the plain-text report in the
  language chosen with `lang` (default German). The German report builds its titles,
  labels, texts, verdicts and error messages from the English data.
- **Machine-readable fields:** use these for automated processing instead of the text
  fields.
  - `verdict_level`: `ai`, `hint` or `none`.
  - `level`: `ai`, `hint` or `info`.
  - `source`: `c2pa`, `iptc_xmp`, `xmp`, `exif`, `exif_user_comment`, `png_text`,
    `watermark`.
  - `type`: one per kind of finding, e.g. `c2pa_active_manifest`,
    `c2pa_ingredient_manifest`, `c2pa_unreadable`, `iptc_source_type`, `xmp_ai_system`,
    `xmp_generator`, `exif_generator`, `generation_parameters`, `comfyui_workflow`,
    `novelai`, `generator_metadata`, `sd_watermark`, `sdxl_watermark`.
- **`details`:** keys in `snake_case` (e.g. `signed_by`, `bit_errors`), descriptive values
  in English (e.g. `meaning`, `validation`, `validation_messages`).
  - `actions` (C2PA) is a list of objects: `action`, `digital_source_type`,
    `software_agent`, `description`.
  - `source_types` (C2PA) is a list of `{type, meaning}`; `meaning` is `null` for unknown
    values.
  - `validation_state` (C2PA): `Trusted`, `Valid` or `Invalid`, as reported by `c2pa`.
  - `validation_codes` (C2PA): raw C2PA status codes, empty when there are no issues;
    `validation_messages` is aligned with it index by index.
  - Original metadata from the file is kept as-is: field names such as
    `photoshop:Credit`, EXIF `Software` or PNG chunk names, and their contents.
- **`errors`:** list of objects `{code, message, detector, exception}`.
  - `image_decode_failed`: the image could not be decoded; only raw-byte checks ran.
  - `detector_failed`: a detector crashed; `detector` names it by its `source` id. Every
    detector runs in isolation, so one failure does not block the others.

The HTTP 413 message is English in the API (`"File too large. The maximum is 50 MB
(MAX_UPLOAD_MB)."`); the web UI shows it in the UI language.

## MCP server (Claude Desktop, Mistral Vibe, other MCP clients)

The checker is also available as an [MCP](https://modelcontextprotocol.io) server, so AI
assistants can call it as a tool.

| Tool | Available | Purpose |
|---|---|---|
| `analyze_image` | always | Image as base64 (`image_base64`, optional `filename`, `lang` `de`/`en`) |
| `list_image_files` | if `MCP_IMAGE_DIR` is set | Lists images in the mounted image directory |
| `analyze_image_file` | if `MCP_IMAGE_DIR` is set | Analyses an image from that directory (`path` relative to it) |

All tools are read-only and return the same JSON as `/analyze` (incl.
`ai_label_recommendation` and the `report` text). File access is confined to
`MCP_IMAGE_DIR`: `..`, absolute paths outside it and symlinks pointing outside are refused.

For chat assistants, **`analyze_image_file` is the practical tool**: language models
cannot reliably produce the base64 of a large image file, so mount the folder with your
images and let the assistant pick the file.

Two transports are available:

- **stdio** – the client starts the container itself: `python -m app.mcp_server`.
  Recommended for local use; no running server or key needed.
- **Streamable HTTP** at `http://<host>:8000/mcp` in the running container, protected by
  the same `API_KEY` (`Authorization: Bearer …` or `X-API-Key`).

### Claude Desktop

Config file: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows); restart Claude Desktop afterwards.

Local via stdio (adjust the image folder):

```json
{
  "mcpServers": {
    "trassd-provenance-lens": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/Users/me/Pictures:/images:ro",
        "-e", "MCP_IMAGE_DIR=/images",
        "trassd-provenance-lens", "python", "-m", "app.mcp_server"
      ]
    }
  }
}
```

Against the running container via HTTP (requires Node.js; `mcp-remote` bridges stdio to
HTTP and adds the key header):

```json
{
  "mcpServers": {
    "trassd-provenance-lens": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8000/mcp", "--header", "Authorization:${AUTH_HEADER}"],
      "env": { "AUTH_HEADER": "Bearer <your API key>" }
    }
  }
}
```

(`Authorization:${AUTH_HEADER}` without a space after the colon avoids argument-splitting
problems on Windows.) Over HTTP only `analyze_image` is available unless the container
also has `MCP_IMAGE_DIR` set and the images mounted, e.g. with the project folder
`images/` (excluded from the image via `.dockerignore`):

```bash
docker run -d --name trassd-provenance-lens --restart unless-stopped -p 8000:8000 \
  -v "$PWD/secrets/api_key.txt:/run/secrets/api_key:ro" -e API_KEY_FILE=/run/secrets/api_key \
  -v "$PWD/images:/images:ro" -e MCP_IMAGE_DIR=/images \
  trassd-provenance-lens
```

Images dropped into `images/` are visible immediately, no restart needed.

### Mistral Vibe

In `~/.vibe/config.toml` (or `.vibe/config.toml` in a project; `VIBE_HOME` changes the
home directory). Tools appear as `trassd_provenance_lens_analyze_image` etc.

Via HTTP against the running container – the key is read from an environment variable:

```toml
[[mcp_servers]]
name = "trassd_provenance_lens"
transport = "streamable-http"
url = "http://localhost:8000/mcp"
api_key_env = "TRASSD_PROVENANCE_LENS_API_KEY"
api_key_header = "Authorization"
api_key_format = "Bearer {token}"
```

```bash
export TRASSD_PROVENANCE_LENS_API_KEY=$(cat secrets/api_key.txt)
```

Or locally via stdio with an image folder:

```toml
[[mcp_servers]]
name = "trassd_provenance_lens"
transport = "stdio"
command = "docker"
args = ["run", "-i", "--rm", "-v", "/Users/me/Pictures:/images:ro", "-e", "MCP_IMAGE_DIR=/images",
        "trassd-provenance-lens", "python", "-m", "app.mcp_server"]
startup_timeout_sec = 30
tool_timeout_sec = 120
```

### HTTP behind a domain

The HTTP endpoint only accepts `Host` headers listed in `MCP_ALLOWED_HOSTS` (protection
against DNS rebinding; default: `localhost`, `127.0.0.1`, `[::1]` on any port). Behind a
reverse proxy set e.g. `MCP_ALLOWED_HOSTS=checker.example.com`. Base64 uploads may be up
to `MAX_UPLOAD_MB` (plus base64 overhead).

## Command line

```bash
# mount images from the current directory read-only into the container
docker run --rm -v "$PWD:/images:ro" trassd-provenance-lens \
  python -m app.cli /images/photo.jpg /images/graphic.png

# JSON output (list, one object per image)
docker run --rm -v "$PWD:/images:ro" trassd-provenance-lens \
  python -m app.cli /images/photo.jpg --json
```

The plain-text output is the German report (`--lang en` for English); `--json` returns
the English JSON format.

Exit codes (e.g. for scripts/CI):

| Code | Meaning |
|---|---|
| `0` | No image with `verdict_level` `ai` |
| `2` | At least one image with `verdict_level` `ai` |
| `1` | At least one file was not readable (and no `ai` finding) |

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `MAX_UPLOAD_MB` | `50` | Maximum upload size in MB (decimals allowed); larger uploads get HTTP 413 |
| `API_KEY` | – | Access key; several keys comma-separated. Empty/unset = no protection |
| `API_KEY_FILE` | – | File with keys (one per line or comma-separated), e.g. a Docker secret. Adds to `API_KEY` |
| `SESSION_HOURS` | `12` | Validity of the browser login in hours |
| `COOKIE_SECURE` | – | `1` sets the Secure flag on the session cookie, e.g. behind an HTTPS reverse proxy |
| `MCP_IMAGE_DIR` | – | Directory the MCP tools `list_image_files` / `analyze_image_file` may read (e.g. a read-only volume); unset = file tools disabled |
| `MCP_ALLOWED_HOSTS` | localhost | Allowed `Host` headers for `/mcp` (comma-separated, `*` as port wildcard, `*` alone disables the check) |

```bash
docker run -d -p 8000:8000 -e MAX_UPLOAD_MB=20 trassd-provenance-lens
```

### Access protection with a key

Without `API_KEY` the service is open (e.g. for local use). With a key, `/analyze`,
`/check` and `/mcp` are protected; `/health` (Docker HEALTHCHECK) and `/docs` stay open.

```bash
# generate a long random key
export API_KEY=$(openssl rand -hex 32)
docker run -d --name trassd-provenance-lens -p 8000:8000 -e API_KEY trassd-provenance-lens

# or as a file / Docker secret (the key does not show up in `docker inspect`)
mkdir -p secrets && (umask 077; openssl rand -hex 32 > secrets/api_key.txt)
docker run -d --name trassd-provenance-lens --restart unless-stopped -p 8000:8000 \
  -v "$PWD/secrets/api_key.txt:/run/secrets/api_key:ro" \
  -e API_KEY_FILE=/run/secrets/api_key trassd-provenance-lens
```

- **API:** send the key as header `X-API-Key: …` or `Authorization: Bearer …`. Without a
  key or with a wrong one the service answers **HTTP 401**. The check runs in a middleware
  *before* the upload is processed.
- **Web UI:** log in once with the key. A signed session cookie is then used (`HttpOnly`,
  `SameSite=Lax`, HMAC-SHA256 over the login time). It does not contain the key itself and
  expires after `SESSION_HOURS`. "Abmelden" (log out) deletes it.
- **Key rotation:** a new `API_KEY` value or a changed file invalidates all existing
  browser sessions. `API_KEY_FILE` is re-read on every request, so rotation needs no
  restart. Several keys allow rotation without interruption.
- **Fail closed:** if `API_KEY_FILE` is set but not readable, the container does not start;
  if the file disappears at runtime, all requests are rejected.
- Failed logins are delayed by 1 second. Only a long random key really protects against
  guessing.
- The key is sent in clear text – on a network, run the service only behind **HTTPS**
  (reverse proxy) and set `COOKIE_SECURE=1`.
- `secrets/` is excluded via `.dockerignore`, so the key never ends up in the image.

## Development & tests

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps invisible-watermark==0.2.0
pip install -r requirements-dev.txt

python -m pytest                                # downloads the C2PA test image from GitHub (skipped offline)
python -m tests.make_test_images test_images    # create test images to try things out
uvicorn app.main:app --reload
```

Test images (`tests/make_test_images.py`, synthetic, 512 × 512):

| Test image | Expected `verdict_level` |
|---|---|
| A1111 PNG with `parameters` | `ai` |
| JPEG with IPTC `trainedAlgorithmicMedia` + credit "Made with Google AI" | `ai` |
| PNG with SDXL watermark | `ai` |
| JPEG with EXIF software "Midjourney" | `hint` |
| C2PA test image `C.jpg` from `c2pa-python` (`algorithmicMedia`) | `hint` |
| Unmarked JPEG | `none` |

## CI & releases

- **CI** (`.github/workflows/ci.yml`, every push to `main` and every pull request): unit
  tests, then the Docker image is built and tested with `.github/scripts/test-image.sh`
  (test suite inside the container, smoke test of health, API key, `/analyze` and `/mcp`).
- **Release** (`.github/workflows/release.yml`, when a GitHub release is published): the
  same image test, then a multi-arch build (`linux/amd64`, `linux/arm64`) pushed to
  [Docker Hub](https://hub.docker.com/r/avhulst2/trassd-provenance-lens) with SBOM and
  provenance attestations; the Docker Hub description is updated from `DOCKERHUB.md`.
- **Dependabot** (`.github/dependabot.yml`): weekly update PRs for Python packages, the
  base image and GitHub Actions.

Release tags must be semantic versions; the version is taken from the tag (build arg
`APP_VERSION`, shown by `/health`):

| Release tag | Docker tags |
|---|---|
| `v1.2.3` | `1.2.3`, `1.2`, `1`, `latest` |
| `v1.3.0-rc.1` (pre-release) | `1.3.0-rc.1` only |
| `v0.4.1` | `0.4.1`, `0.4`, `latest` (no `0` major tag) |

One-time setup in the GitHub repository (Settings → Secrets and variables → Actions):

- `DOCKERHUB_USERNAME` = `avhulst2`
- `DOCKERHUB_TOKEN` = Docker Hub personal access token (Account settings → Personal
  access tokens) with **Read, Write, Delete** – "Delete" is required to update the
  repository description.

Publishing a release:

```bash
git tag v1.2.0 && git push origin v1.2.0
gh release create v1.2.0 --generate-notes      # or via the GitHub web UI
```

A release created by another workflow with the default `GITHUB_TOKEN` does not trigger
the release workflow; use a personal token or start it manually (Actions → Release →
Run workflow, with the tag). The manual run can also re-publish an existing tag.

`DOCKERHUB.md` is the compact description for Docker Hub (limit 25,000 bytes) and links
to this README; update it when run options or configuration change.

## Project structure

```
├── .github/
│   ├── workflows/ci.yml       # tests + image test on push / pull request
│   ├── workflows/release.yml  # GitHub release → Docker Hub (amd64 + arm64)
│   ├── scripts/test-image.sh  # tests a built image (inside + smoke test)
│   └── dependabot.yml
├── Dockerfile
├── DOCKERHUB.md               # description shown on Docker Hub
├── requirements.txt
├── requirements-dev.txt
├── app/
│   ├── detectors.py   # all detectors + data model (Finding, Result, analyze)
│   ├── recommendation.py  # labelling recommendation (ai_generated / ai_modified / ai_assisted)
│   ├── report.py      # plain-text report (German / English)
│   ├── main.py        # FastAPI + web UI (German / English)
│   ├── auth.py        # access protection (API_KEY, session cookie)
│   ├── mcp_server.py  # MCP server (stdio + Streamable HTTP at /mcp)
│   └── cli.py         # command line
└── tests/
    ├── make_test_images.py
    └── test_detectors.py
```
