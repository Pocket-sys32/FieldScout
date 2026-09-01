"""
Google Sheets + local CSV writer.

Sheet schema (one row per processed video)
──────────────────────────────────────────
Col A  Date             MM/DD/YYYY
Col B  Time             HH:MM:SS
Col C  Common Name      e.g. Raccoon
Col D  Scientific Name  e.g. Procyon lotor
Col E  Count            max animals seen simultaneously
Col F  Filename         just the file name, not the full path
Col G  Comments         auto-generated notes
Col H  Confidence       e.g. 0.873
Col I  Needs Review     TRUE / FALSE

Row 1 is assumed to be the header row; new rows are appended after the
last occupied row so the sheet never clobbers existing data.

Authentication
──────────────
Uses a Google Cloud service account JSON key.  The key file path is read
from `cfg.service_account_path`.  If the key file is missing the writer
falls back to CSV-only mode and logs a warning.

Thread safety
─────────────
`SheetsWriter.write()` is safe to call from a background thread; the
`gspread` session is created lazily and protected by a lock.
"""

from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .aggregate import VideoResult
from .config import cfg

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# Column header that MUST exist in row 1 of the worksheet
EXPECTED_HEADERS = [
    "Date",
    "Time",
    "Species Captured",
    "Scientific Name",
    "Numbers of same species present",
    "picture number",
    "Comment",
    "Confidence Int.",
    "Review",
]


class SheetsWriter:
    """
    Appends rows to an existing Google Sheet and maintains a local CSV backup.

    Parameters
    ----------
    sheet_id            : Google Sheet ID (the long string in the URL).
    service_account_path: Path to the service-account JSON key.
    worksheet_name      : Name of the tab to write to.
    output_csv          : Path to the local CSV backup file.
    """

    def __init__(
        self,
        sheet_id: str = "",
        service_account_path: str | Path = "service_account.json",
        worksheet_name: str = "Detections",
        output_csv: str | Path = "detections.csv",
    ) -> None:
        self.sheet_id             = sheet_id or cfg.sheet_id
        self.service_account_path = Path(service_account_path or cfg.service_account_path)
        self.worksheet_name       = worksheet_name or cfg.worksheet_name
        self.output_csv           = Path(output_csv or cfg.output_csv)

        self._gc        = None   # gspread.Client (lazy)
        self._worksheet = None   # gspread.Worksheet (lazy)
        self._csv_ready = False
        self._sheets_ok = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, result: VideoResult, video_path: str | Path) -> None:
        """
        Write one VideoResult row to the Google Sheet and the CSV backup.

        Parameters
        ----------
        result     : aggregated detection result for one video
        video_path : full path of the source .mov file
        """
        filename = Path(video_path).name
        row = self._build_row(result, filename)

        self._ensure_csv_header()
        self._append_csv(row)

        try:
            self._ensure_sheets()
            self._append_sheets(row)
        except Exception as exc:
            logger.warning(
                "Could not write to Google Sheet (%s). "
                "Row saved to CSV backup only.",
                exc,
            )

    def flush(self) -> None:
        """No-op (rows are written immediately); kept for interface symmetry."""

    def get_processed_filenames(self) -> set[str]:
        """
        Return normalised (lowercase) filenames already in the Sheet and/or CSV.
        """
        processed: set[str] = set()

        try:
            self._ensure_sheets()
            for name in self._worksheet.col_values(6)[1:]:
                name = name.strip()
                if name:
                    processed.add(name.lower())
            logger.info(
                "Loaded %d already-processed filename(s) from Google Sheet.",
                len(processed),
            )
        except Exception as exc:
            logger.warning(
                "Could not read processed filenames from Sheet (%s). "
                "Falling back to local CSV.",
                exc,
            )

        if self.output_csv.exists():
            try:
                with self.output_csv.open(newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) > 5 and row[5].strip():
                            processed.add(row[5].strip().lower())
            except Exception as exc:
                logger.warning("Could not read local CSV for skip list: %s", exc)

        return processed

    # ------------------------------------------------------------------
    # Row builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_row(result: VideoResult, filename: str) -> list[str]:
        if result.recorded_at:
            dt = result.recorded_at
            # OCR timestamps are already local camera time — do not shift timezone.
            if dt.tzinfo is not None and result.timestamp_source != "ocr":
                dt = dt.astimezone()
            date_str = dt.strftime("%m/%d/%Y")
            # Leading apostrophe forces Google Sheets to keep 24-hour text as-is.
            time_str = "'" + dt.strftime("%H:%M:%S")
        else:
            date_str = ""
            time_str = ""

        # Column order matches your Sheet:
        # Date | Time | Species Captured | Scientific Name |
        # Numbers of same species present | picture number |
        # Comment | Confidence Int. | Review
        return [
            date_str,
            time_str,
            result.common_name,
            result.scientific_name,
            str(result.count),
            filename,
            result.comments,
            f"{result.confidence:.3f}",
            "TRUE" if result.needs_review else "FALSE",
        ]

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _ensure_csv_header(self) -> None:
        if self._csv_ready:
            return
        with _lock:
            if self._csv_ready:
                return
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.output_csv.exists() or self.output_csv.stat().st_size == 0
            if write_header:
                with self.output_csv.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(EXPECTED_HEADERS)
                logger.debug("CSV header written to %s", self.output_csv)
            self._csv_ready = True

    def _append_csv(self, row: list[str]) -> None:
        with _lock:
            with self.output_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        logger.debug("CSV row written: %s", row[2])  # log species name

    # ------------------------------------------------------------------
    # Google Sheets helpers
    # ------------------------------------------------------------------

    def _ensure_sheets(self) -> None:
        """Lazily authenticate and cache the worksheet handle."""
        if self._sheets_ok:
            return
        with _lock:
            if self._sheets_ok:
                return
            self._connect()

    def _connect(self) -> None:
        if not self.sheet_id:
            raise ValueError("sheet_id is not configured.")
        if not self.service_account_path.exists():
            raise FileNotFoundError(
                f"Service account key not found: {self.service_account_path}\n"
                "See README.md → 'Google Sheets setup' for instructions."
            )
        try:
            import gspread                                        # type: ignore
            from google.oauth2.service_account import Credentials # type: ignore

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
            ]
            creds = Credentials.from_service_account_file(
                str(self.service_account_path), scopes=scopes
            )
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(self.sheet_id)

            # Find or create the target worksheet
            try:
                ws = sh.worksheet(self.worksheet_name)
            except gspread.exceptions.WorksheetNotFound:
                ws = sh.add_worksheet(self.worksheet_name, rows=1000, cols=20)
                ws.append_row(EXPECTED_HEADERS)
                logger.info("Created new worksheet '%s'", self.worksheet_name)

            # Validate header row
            headers = ws.row_values(1)
            if headers and headers != EXPECTED_HEADERS:
                logger.warning(
                    "Sheet header row doesn't match expected columns.\n"
                    "  Found   : %s\n"
                    "  Expected: %s",
                    headers,
                    EXPECTED_HEADERS,
                )

            self._gc        = gc
            self._worksheet = ws
            self._sheets_ok = True
            logger.info(
                "Connected to Google Sheet '%s' → tab '%s'",
                self.sheet_id[:8] + "…",
                self.worksheet_name,
            )

        except ImportError as exc:
            raise RuntimeError(
                "gspread or google-auth is not installed.  "
                "Run:  pip install gspread google-auth"
            ) from exc

    def _append_sheets(self, row: list[str]) -> None:
        with _lock:
            self._worksheet.append_row(
                row,
                value_input_option="USER_ENTERED",  # so TRUE/FALSE parse as booleans
            )
        logger.debug("Sheet row appended: %s", row[2])
