"""Shared path helpers for keeping SGT artifacts inside the SGT project tree."""

import os


PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")


def project_root_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def project_results_path(*parts: str) -> str:
    return os.path.join(RESULTS_ROOT, *parts)
