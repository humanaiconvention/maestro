"""
Smoke tests — verify gateway modules import without errors.
No database or external services required.
"""
import os
import sys

# Set env vars before any app imports
os.environ.setdefault("MAESTRO_LAUNCH_MODE", "public_readonly")
os.environ.setdefault("USE_MOCK_ADAPTER", "true")

# TODO: remove after editable install — see pyproject.toml
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_gateway_main():
    import apps.gateway.main  # noqa: F401


def test_import_gateway_participation():
    import apps.gateway.participation  # noqa: F401


def test_import_gateway_db():
    import apps.gateway.db  # noqa: F401
