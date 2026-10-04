"""Small, guarded Git updater used before the application imports its UI."""

from __future__ import annotations

import logging
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_COMMAND_TIMEOUT_SECONDS = 12
_DEPENDENCY_TIMEOUT_SECONDS = 900
_LOCAL_SETTINGS_FILES = {"config.json"}


def _has_source_edits(status_output: str) -> bool:
    """Return True for tracked edits other than app-managed local settings."""
    for line in status_output.splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        if path.replace("\\", "/") not in _LOCAL_SETTINGS_FILES:
            return True
    return False


def update_from_git(repo_dir: str | Path) -> bool:
    """Fast-forward a clean Git checkout to its configured upstream.

    Returns True only when files changed. Network errors, missing Git, detached
    checkouts, and local edits are reported and leave the current version intact.
    """
    repo = Path(repo_dir).resolve()
    if getattr(sys, "frozen", False) or not (repo / ".git").is_dir():
        return False
    if shutil.which("git") is None:
        logger.info("Update check skipped: Git is not installed.")
        return False

    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            env=environment,
            check=False,
        )

    try:
        upstream_result = git(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
        )
        if upstream_result.returncode != 0:
            logger.info("Update check skipped: the current branch has no upstream.")
            return False
        upstream = upstream_result.stdout.strip()

        fetch_result = git("fetch", "--quiet", "--prune")
        if fetch_result.returncode != 0:
            detail = fetch_result.stderr.strip() or "network unavailable"
            logger.info("Could not check for updates: %s", detail)
            return False

        counts_result = git("rev-list", "--left-right", "--count", f"HEAD...{upstream}")
        if counts_result.returncode != 0:
            logger.info("Could not compare the installed version with %s.", upstream)
            return False
        ahead, behind = (int(value) for value in counts_result.stdout.split())
        if behind == 0:
            logger.info("FieldScout is up to date.")
            return False
        if ahead:
            logger.warning(
                "Update available, but this checkout has %d unpushed commit(s); "
                "automatic update was skipped.",
                ahead,
            )
            return False

        status_result = git("status", "--porcelain", "--untracked-files=no")
        if status_result.returncode != 0 or _has_source_edits(status_result.stdout):
            logger.warning(
                "Update available, but tracked files have local edits; automatic "
                "update was skipped to protect those changes."
            )
            return False

        logger.info("Installing FieldScout update from %s…", upstream)
        merge_result = git("merge", "--ff-only", upstream)
        if merge_result.returncode != 0:
            detail = merge_result.stderr.strip() or merge_result.stdout.strip()
            logger.warning("Automatic update failed: %s", detail)
            return False

        logger.info("FieldScout updated successfully; restarting…")
        return True
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.info("Could not complete the update check: %s", exc)
        return False


def sync_dependencies(repo_dir: str | Path) -> bool:
    """Install changed requirements into the active virtual environment."""
    repo = Path(repo_dir).resolve()
    requirements = repo / "requirements.txt"
    if not requirements.is_file():
        return True

    if sys.prefix == sys.base_prefix:
        logger.warning(
            "Dependency update skipped because FieldScout is not running in a "
            "virtual environment. Relaunch it through run.bat or .venv."
        )
        return False

    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = Path(sys.prefix) / ".fieldscout-requirements.sha256"
    try:
        if marker.read_text(encoding="utf-8").strip() == digest:
            return True
    except OSError:
        pass

    environment = os.environ.copy()
    environment.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    uv_path = shutil.which("uv")

    try:
        logger.info("Installing updated FieldScout dependencies into .venv…")
        if uv_path:
            commands = [[
                uv_path,
                "pip",
                "install",
                "--torch-backend",
                "cpu",
                "--python",
                sys.executable,
                "-r",
                str(requirements),
            ]]
        else:
            commands = [
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ],
                [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
            ]

        for command in commands:
            result = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                capture_output=True,
                text=True,
                timeout=_DEPENDENCY_TIMEOUT_SECONDS,
                check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                logger.warning("Dependency update failed: %s", detail)
                return False

        marker.write_text(digest + "\n", encoding="utf-8")
        logger.info("FieldScout dependencies are up to date.")
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Dependency update failed: %s", exc)
        return False
