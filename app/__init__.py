"""Checks images for machine-readable AI labels (Art. 50(2) EU AI Act)."""

import os

# Set at image build time from the release tag (Docker build arg APP_VERSION).
__version__ = os.environ.get("APP_VERSION") or "dev"
