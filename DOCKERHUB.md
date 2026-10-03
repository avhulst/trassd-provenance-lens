# Trassd Provenance Lens

Checks an **original image** for the most common machine-readable AI labels (context:
Art. 50(2) EU AI Act) and recommends how to label it (AI-generated / AI-modified /
AI-assisted). Web UI in German and English, JSON API, CLI and MCP server (Claude
Desktop, Mistral Vibe, …).

**Full documentation:** <https://github.com/avhulst/trassd-provenance-lens>

## What is checked

- **C2PA / Content Credentials** – manifests, signature, actions, digital source type, validation
- **IPTC Digital Source Type** (XMP) – e.g. `trainedAlgorithmicMedia`
- **XMP / EXIF** – creator tool, credit, IPTC AI fields, generator names (Midjourney, DALL·E, Firefly, …)
- **PNG text chunks** – A1111/Forge, ComfyUI, NovelAI, InvokeAI, Fooocus
- **Invisible watermarks** – Stable Diffusion v1/v2 and SDXL (`invisible-watermark`, dwtDct)

A missing label does not prove authenticity; proprietary watermarks such as Google
SynthID cannot be checked. The labelling recommendation is not legal advice.

## Quick start

```bash
docker run -d --name trassd-provenance-lens -p 8000:8000 avhulst2/trassd-provenance-lens:latest
```

Web UI: <http://localhost:8000/> · API docs: <http://localhost:8000/docs>

```bash
curl -s -F file=@image.jpg "http://localhost:8000/analyze?format=text&lang=en"
curl -s -F file=@image.jpg http://localhost:8000/analyze | jq '{verdict_level, ai_label_recommendation}'
```

With access protection and an image folder for the MCP tools:

```bash
mkdir -p secrets images && (umask 077; openssl rand -hex 32 > secrets/api_key.txt)
docker run -d --name trassd-provenance-lens --restart unless-stopped -p 8000:8000 \
  -v "$PWD/secrets/api_key.txt:/run/secrets/api_key:ro" -e API_KEY_FILE=/run/secrets/api_key \
  -v "$PWD/images:/images:ro" -e MCP_IMAGE_DIR=/images \
  avhulst2/trassd-provenance-lens:latest
```

API clients then send `X-API-Key: <key>` or `Authorization: Bearer <key>`; the web UI asks
for the key once.

CLI: `docker run --rm -v "$PWD:/images:ro" avhulst2/trassd-provenance-lens python -m app.cli /images/photo.jpg --lang en`
(exit code 2 if an AI label was found).

MCP via stdio (e.g. Claude Desktop): `docker run -i --rm -v /path/to/images:/images:ro -e MCP_IMAGE_DIR=/images avhulst2/trassd-provenance-lens python -m app.mcp_server`
· MCP via HTTP: `http://localhost:8000/mcp`

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MAX_UPLOAD_MB` | `50` | Maximum upload size (HTTP 413 above) |
| `API_KEY` / `API_KEY_FILE` | – | Access key(s); unset = no protection |
| `SESSION_HOURS` | `12` | Validity of the web UI login |
| `COOKIE_SECURE` | – | `1` behind an HTTPS reverse proxy |
| `MCP_IMAGE_DIR` | – | Directory the MCP file tools may read |
| `MCP_ALLOWED_HOSTS` | localhost | Allowed `Host` headers for `/mcp` behind a domain |

## Tags

`latest` – newest release · `1`, `1.2`, `1.2.3` – semantic versions ·
platforms `linux/amd64`, `linux/arm64` · runs as non-root user (UID 10001) ·
health check on `/health`.
