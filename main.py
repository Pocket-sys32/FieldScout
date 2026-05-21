"""
Cache Creek Game Camera Project – entry point.

Usage
─────
  GUI (default):
      python main.py

  CLI batch (no GUI, writes CSV + optional Sheet):
      python main.py --batch /path/to/videos

  Show version:
      python main.py --version
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fsvcc")


def _run_gui() -> None:
    try:
        from fsvcc_detector.gui import launch
        launch()
    except ImportError as exc:
        logger.error(
            "GUI dependencies missing: %s\n"
            "Run:  pip install customtkinter tkinterdnd2",
            exc,
        )
        sys.exit(1)


def _run_cli_batch(folder: Path) -> None:
    """Non-GUI batch mode — useful for scripting / scheduled tasks."""
    from fsvcc_detector.config import cfg
    from fsvcc_detector.pipeline import Pipeline
    from fsvcc_detector.sheets import SheetsWriter

    if not folder.exists():
        logger.error("Folder not found: %s", folder)
        sys.exit(1)

    pipeline = Pipeline()
    writer   = SheetsWriter(
        sheet_id             = cfg.sheet_id,
        service_account_path = cfg.service_account_resolved,
        worksheet_name       = cfg.worksheet_name,
        output_csv           = cfg.output_csv_resolved,
    )

    results = pipeline.process_folder(
        folder,
        sheets_writer=writer,
        video_callback=lambda pr: print(
            f"  {'OK' if not pr.error else 'ERR'}  {pr.video_path.name}"
            + (f"  → {pr.result.common_name}  (conf {pr.result.confidence:.0%})"
               if pr.result else "")
        ),
    )

    ok  = sum(1 for r in results if not r.error)
    err = sum(1 for r in results if r.error)
    print(f"\nDone.  {ok} processed, {err} error(s).")
    print(f"CSV backup: {cfg.output_csv_resolved}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fsvcc_detector",
        description="Cache Creek Game Camera Project – trail-cam species ID tool.",
    )
    parser.add_argument(
        "--batch", metavar="FOLDER",
        help="Run in CLI batch mode (no GUI) on the specified folder of .mov files.",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print version and exit.",
    )
    args = parser.parse_args()

    if args.version:
        from fsvcc_detector import __version__
        print(f"Cache Creek Game Camera Project  v{__version__}")
        return

    if args.batch:
        _run_cli_batch(Path(args.batch))
    else:
        _run_gui()


if __name__ == "__main__":
    main()
