"""
Cache Creek Game Camera Project — main GUI window.

Design language: Apple macOS / iOS inspired.
  • Inter font (closest free equivalent to SF Pro), loaded via GDI for this
    process only — no system installation, zero compute overhead.
  • Frameless window with macOS-style traffic-light close / minimize buttons.
  • Neutral grey palette, sage-green accent, generous whitespace.
  • Light mode default; respects OS dark-mode preference.
"""

from __future__ import annotations

import ctypes
import logging
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import tkinter as tk

logger = logging.getLogger(__name__)

# ── Asset paths ────────────────────────────────────────────────────────────────
_ASSETS_FONTS = Path(__file__).parent.parent / "assets" / "fonts"

# ── Font bootstrap ─────────────────────────────────────────────────────────────
_FONT_FAMILY = "Segoe UI"   # overwritten to "Inter" if files load successfully


def _load_custom_fonts() -> None:
    """Load Inter TTF into GDI for this process only (AddFontResourceEx FR_PRIVATE).
    Runs once at import time; has no visible effect on compute after startup."""
    global _FONT_FAMILY
    if sys.platform != "win32":
        return
    try:
        gdi32 = ctypes.windll.gdi32
        FR_PRIVATE = 0x10
        loaded = 0
        for fname in ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf"):
            fp = _ASSETS_FONTS / fname
            if fp.exists():
                if gdi32.AddFontResourceExW(str(fp), FR_PRIVATE, None) > 0:
                    loaded += 1
        if loaded:
            _FONT_FAMILY = "Inter"
            logger.debug("Inter font loaded (%d weight(s))", loaded)
    except Exception as exc:
        logger.debug("Custom font unavailable (%s) — using Segoe UI", exc)


_load_custom_fonts()

# ── DnD ────────────────────────────────────────────────────────────────────────
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

# ── Colour palette ─────────────────────────────────────────────────────────────
_C: dict[str, str] = {
    "bg":            "#F5F5F7",
    "surface":       "#FFFFFF",
    "surface2":      "#F0F0F2",
    "border":        "#D8D8DC",
    "accent":        "#34A853",
    "accent_hover":  "#2D9147",
    "danger":        "#FF3B30",
    "danger_hover":  "#D9322B",
    "text_primary":  "#1D1D1F",
    "text_secondary":"#6E6E73",
    "text_tertiary": "#AEAEB2",
    "text_on_accent":"#FFFFFF",
    "log_bg":        "#FAFAFA",
    "log_text":      "#1D1D1F",
    "log_success":   "#1A7F37",
    "log_warn":      "#B45309",
    "log_error":     "#DC2626",
    "header_bg":     "#FFFFFF",
    "progress_bg":   "#E5E5EA",
    # macOS traffic-light authentic colours
    "tl_close":      "#FF5F57",
    "tl_close_h":    "#E0443D",
    "tl_min":        "#FEBC2E",
    "tl_min_h":      "#D99A00",
}


def _apply_dark_palette() -> None:
    _C.update({
        "bg":            "#1C1C1E",
        "surface":       "#2C2C2E",
        "surface2":      "#3A3A3C",
        "border":        "#48484A",
        "text_primary":  "#F2F2F7",
        "text_secondary":"#AEAEB2",
        "text_tertiary": "#636366",
        "log_bg":        "#1C1C1E",
        "log_text":      "#F2F2F7",
        "header_bg":     "#2C2C2E",
        "progress_bg":   "#3A3A3C",
    })


# ── Public entry point ────────────────────────────────────────────────────────

def launch() -> None:
    import customtkinter as ctk

    try:
        import darkdetect  # type: ignore
        if darkdetect.isDark():
            _apply_dark_palette()
    except Exception:
        pass

    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("green")

    if _DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    app = _App(root)
    app.build()
    root.mainloop()


# ── Win32 helpers ──────────────────────────────────────────────────────────────

def _win32_appwindow(root: tk.Misc) -> None:
    """After overrideredirect(True), restore the taskbar button via WS_EX_APPWINDOW."""
    if sys.platform != "win32":
        return
    try:
        GWL_EXSTYLE      = -20
        WS_EX_APPWINDOW  = 0x00040000
        WS_EX_TOOLWINDOW = 0x00000080
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        if not hwnd:
            return
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        style = (style & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        root.wm_withdraw()
        root.after(10, root.wm_deiconify)
    except Exception as exc:
        logger.debug("Win32 appwindow fix: %s", exc)


def _win32_minimize(root: tk.Misc) -> None:
    """Reliably minimise a frameless window."""
    if sys.platform != "win32":
        root.iconify()
        return
    try:
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
            return
    except Exception:
        pass
    root.iconify()


# ── Traffic-light title bar ────────────────────────────────────────────────────

class _TitleBar:
    """Draggable macOS-style title bar: two traffic-light circles + title + Settings.

    Uses plain tkinter (Canvas + Label) so it contributes essentially zero CPU load.
    """

    _H    = 48    # bar height
    _R    = 7     # circle radius
    _GAP  = 8     # gap between circles
    _PAD  = 16    # left padding

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        on_close,
        on_minimize,
        root: tk.Misc,
        on_settings,
    ) -> None:
        self._root  = root
        self._dx = self._dy = 0

        self.frame = tk.Frame(parent, bg=_C["header_bg"], height=self._H)
        self.frame.pack(fill="x")
        self.frame.pack_propagate(False)

        # ── Traffic-light dots ─────────────────────────────────────────────
        btn_w = self._PAD + (self._R * 2) * 2 + self._GAP + self._PAD
        self._cv = tk.Canvas(
            self.frame,
            width=btn_w, height=self._H,
            bg=_C["header_bg"], highlightthickness=0,
        )
        self._cv.pack(side="left")

        cy  = self._H // 2
        r   = self._R
        x1  = self._PAD + r
        x2  = x1 + r * 2 + self._GAP

        self._ov_c = self._cv.create_oval(
            x1-r, cy-r, x1+r, cy+r, fill=_C["tl_close"], outline="", tags="close"
        )
        self._ov_m = self._cv.create_oval(
            x2-r, cy-r, x2+r, cy+r, fill=_C["tl_min"], outline="", tags="minimize"
        )

        self._cv.tag_bind("close",    "<Enter>",    lambda _: self._hi("close", True))
        self._cv.tag_bind("close",    "<Leave>",    lambda _: self._hi("close", False))
        self._cv.tag_bind("close",    "<Button-1>", lambda _: on_close())
        self._cv.tag_bind("minimize", "<Enter>",    lambda _: self._hi("minimize", True))
        self._cv.tag_bind("minimize", "<Leave>",    lambda _: self._hi("minimize", False))
        self._cv.tag_bind("minimize", "<Button-1>", lambda _: on_minimize())

        # ── Title ──────────────────────────────────────────────────────────
        self._title = tk.Label(
            self.frame,
            text=title,
            font=(_FONT_FAMILY, 13, "bold"),
            fg=_C["text_primary"],
            bg=_C["header_bg"],
        )
        self._title.pack(side="left", fill="x", expand=True)

        # ── Settings link ──────────────────────────────────────────────────
        self._settings = tk.Label(
            self.frame,
            text="Settings",
            font=(_FONT_FAMILY, 12),
            fg=_C["text_secondary"],
            bg=_C["header_bg"],
            cursor="hand2",
            padx=14,
        )
        self._settings.pack(side="right", padx=(0, 8))
        self._settings.bind("<Button-1>", lambda _: on_settings())
        self._settings.bind("<Enter>",    lambda _: self._settings.configure(fg=_C["text_primary"]))
        self._settings.bind("<Leave>",    lambda _: self._settings.configure(fg=_C["text_secondary"]))

        # ── Drag bindings (anywhere on frame / title, not on buttons) ─────
        for w in (self.frame, self._title):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>",     self._drag_move)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _hi(self, which: str, entering: bool) -> None:
        if which == "close":
            self._cv.itemconfigure(
                self._ov_c, fill=_C["tl_close_h"] if entering else _C["tl_close"]
            )
        else:
            self._cv.itemconfigure(
                self._ov_m, fill=_C["tl_min_h"] if entering else _C["tl_min"]
            )

    def _drag_start(self, ev: tk.Event) -> None:
        self._dx = ev.x_root - self._root.winfo_x()
        self._dy = ev.y_root - self._root.winfo_y()

    def _drag_move(self, ev: tk.Event) -> None:
        self._root.geometry(f"+{ev.x_root - self._dx}+{ev.y_root - self._dy}")


# ── Main application ───────────────────────────────────────────────────────────

class _App:
    WINDOW_TITLE = "Cache Creek Game Camera Project"
    WIN_W, WIN_H = 860, 680

    def __init__(self, root: tk.Misc) -> None:
        self._root       = root
        self._folder_path: Optional[Path] = None
        self._mov_files:   list[Path] = []
        self._processing   = False
        self._stop_event   = threading.Event()
        self._ui_queue: queue.Queue = queue.Queue()

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self) -> None:
        import customtkinter as ctk

        root = self._root

        # Frameless + taskbar entry
        root.overrideredirect(True)
        root.geometry(f"{self.WIN_W}x{self.WIN_H}")
        root.minsize(740, 560)
        root.configure(bg=_C["bg"])
        root.after(30, lambda: _win32_appwindow(root))

        # ── Title bar ─────────────────────────────────────────────────────
        _TitleBar(
            parent      = root,
            title       = self.WINDOW_TITLE,
            on_close    = root.destroy,
            on_minimize = lambda: _win32_minimize(root),
            root        = root,
            on_settings = self._open_settings,
        )

        # 1 px separator under header
        tk.Frame(root, bg=_C["border"], height=1).pack(fill="x")

        # ── Content ───────────────────────────────────────────────────────
        content = ctk.CTkFrame(root, fg_color=_C["bg"], corner_radius=0)
        content.pack(fill="both", expand=True)

        # ── Drop zone card ────────────────────────────────────────────────
        self._drop_frame = ctk.CTkFrame(
            content,
            fg_color=_C["surface"],
            corner_radius=14,
            border_width=1,
            border_color=_C["border"],
        )
        self._drop_frame.pack(fill="x", padx=24, pady=(20, 0))

        drop_inner = ctk.CTkFrame(self._drop_frame, fg_color="transparent")
        drop_inner.pack(fill="x", padx=20, pady=16)

        self._drop_icon = ctk.CTkLabel(
            drop_inner, text="📂",
            font=ctk.CTkFont(size=28),
            text_color=_C["text_tertiary"],
        )
        self._drop_icon.pack(side="left", padx=(0, 12))

        text_col = ctk.CTkFrame(drop_inner, fg_color="transparent")
        text_col.pack(side="left", fill="x", expand=True)

        self._drop_title = ctk.CTkLabel(
            text_col,
            text="Drop a video folder here",
            font=ctk.CTkFont(family=_FONT_FAMILY, size=15, weight="bold"),
            text_color=_C["text_primary"],
            anchor="w",
        )
        self._drop_title.pack(anchor="w")

        self._drop_subtitle = ctk.CTkLabel(
            text_col,
            text="Supports  .mov  ·  .avi  ·  .mp4",
            font=ctk.CTkFont(family=_FONT_FAMILY, size=12),
            text_color=_C["text_tertiary"],
            anchor="w",
        )
        self._drop_subtitle.pack(anchor="w", pady=(2, 0))

        ctk.CTkButton(
            drop_inner, text="Browse",
            width=80, height=32,
            font=ctk.CTkFont(family=_FONT_FAMILY, size=13),
            fg_color=_C["surface2"],
            hover_color=_C["border"],
            text_color=_C["text_primary"],
            corner_radius=8,
            border_width=1,
            border_color=_C["border"],
            command=self._browse_folder,
        ).pack(side="right")

        if _DND_AVAILABLE:
            self._drop_frame.drop_target_register(DND_FILES)
            self._drop_frame.dnd_bind("<<Drop>>", self._on_drop)

        # ── Folder status ─────────────────────────────────────────────────
        self._folder_label = ctk.CTkLabel(
            content,
            text="No folder selected",
            font=ctk.CTkFont(family=_FONT_FAMILY, size=12),
            text_color=_C["text_tertiary"],
            anchor="w",
        )
        self._folder_label.pack(anchor="w", padx=28, pady=(6, 0))

        # ── Action buttons ────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(content, fg_color="transparent")
        btn_row.pack(fill="x", padx=24, pady=(12, 0))

        self._process_btn = ctk.CTkButton(
            btn_row,
            text="Process Videos",
            height=42,
            font=ctk.CTkFont(family=_FONT_FAMILY, size=15, weight="bold"),
            fg_color=_C["accent"],
            hover_color=_C["accent_hover"],
            text_color=_C["text_on_accent"],
            corner_radius=10,
            command=self._start_processing,
            state="disabled",
        )
        self._process_btn.pack(side="left", expand=True, fill="x", padx=(0, 8))

        self._stop_btn = ctk.CTkButton(
            btn_row,
            text="Stop",
            height=42,
            width=100,
            font=ctk.CTkFont(family=_FONT_FAMILY, size=15),
            fg_color=_C["surface2"],
            hover_color=_C["danger"],
            text_color=_C["text_secondary"],
            corner_radius=10,
            border_width=1,
            border_color=_C["border"],
            command=self._stop_processing,
            state="disabled",
        )
        self._stop_btn.pack(side="right")

        # ── Progress card ─────────────────────────────────────────────────
        prog_card = ctk.CTkFrame(
            content, fg_color=_C["surface"],
            corner_radius=12, border_width=1, border_color=_C["border"],
        )
        prog_card.pack(fill="x", padx=24, pady=(12, 0))

        prog_inner = ctk.CTkFrame(prog_card, fg_color="transparent")
        prog_inner.pack(fill="x", padx=16, pady=12)

        self._progress_label = ctk.CTkLabel(
            prog_inner, text="Ready to process",
            font=ctk.CTkFont(family=_FONT_FAMILY, size=12),
            text_color=_C["text_secondary"],
            anchor="w",
        )
        self._progress_label.pack(anchor="w", pady=(0, 6))

        self._progress_var = ctk.DoubleVar(value=0.0)
        self._progress_bar = ctk.CTkProgressBar(
            prog_inner,
            variable=self._progress_var,
            height=6,
            corner_radius=3,
            fg_color=_C["progress_bg"],
            progress_color=_C["accent"],
        )
        self._progress_bar.pack(fill="x")

        # ── Activity log card ─────────────────────────────────────────────
        log_card = ctk.CTkFrame(
            content, fg_color=_C["surface"],
            corner_radius=12, border_width=1, border_color=_C["border"],
        )
        log_card.pack(fill="both", expand=True, padx=24, pady=(12, 20))

        log_header = ctk.CTkFrame(log_card, fg_color="transparent", height=36)
        log_header.pack(fill="x", padx=16, pady=(10, 0))
        log_header.pack_propagate(False)

        ctk.CTkLabel(
            log_header, text="Activity",
            font=ctk.CTkFont(family=_FONT_FAMILY, size=12, weight="bold"),
            text_color=_C["text_secondary"],
            anchor="w",
        ).pack(side="left", anchor="w")

        self._clear_btn = ctk.CTkButton(
            log_header, text="Clear",
            width=52, height=22,
            font=ctk.CTkFont(family=_FONT_FAMILY, size=11),
            fg_color="transparent",
            hover_color=_C["surface2"],
            text_color=_C["text_tertiary"],
            corner_radius=6,
            command=self._clear_log,
        )
        self._clear_btn.pack(side="right")

        ctk.CTkFrame(log_card, fg_color=_C["border"], height=1, corner_radius=0).pack(fill="x")

        self._log_box = ctk.CTkTextbox(
            log_card,
            state="disabled",
            font=ctk.CTkFont(family=_FONT_FAMILY, size=12),
            fg_color=_C["log_bg"],
            text_color=_C["log_text"],
            corner_radius=0,
            wrap="word",
            border_width=0,
            scrollbar_button_color=_C["border"],
            scrollbar_button_hover_color=_C["text_tertiary"],
        )
        self._log_box.pack(fill="both", expand=True)

        self._log_box._textbox.tag_configure("success", foreground=_C["log_success"])
        self._log_box._textbox.tag_configure("warn",    foreground=_C["log_warn"])
        self._log_box._textbox.tag_configure("error",   foreground=_C["log_error"])
        self._log_box._textbox.tag_configure("dim",     foreground=_C["text_tertiary"])

        # ── Status bar ────────────────────────────────────────────────────
        status_bar = ctk.CTkFrame(root, fg_color=_C["header_bg"], corner_radius=0, height=24)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        ctk.CTkFrame(status_bar, fg_color=_C["border"], height=1, corner_radius=0).pack(
            fill="x", side="top"
        )

        self._status_bar = ctk.CTkLabel(
            status_bar, text="Ready",
            font=ctk.CTkFont(family=_FONT_FAMILY, size=11),
            text_color=_C["text_tertiary"],
            anchor="w",
        )
        self._status_bar.pack(side="left", padx=16, fill="y")

        root.after(100, self._poll_ui_queue)

    # ── Folder handling ──────────────────────────────────────────────────────

    def _browse_folder(self) -> None:
        import tkinter.filedialog as fd
        folder = fd.askdirectory(title="Select folder containing video files")
        if folder:
            self._set_folder(Path(folder))

    def _on_drop(self, event) -> None:
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        p = Path(raw)
        self._set_folder(p if p.is_dir() else p.parent)

    def _set_folder(self, folder: Path) -> None:
        self._folder_path = folder
        from .pipeline import _find_videos
        self._mov_files = _find_videos(folder)
        n = len(self._mov_files)

        if self._mov_files:
            self._drop_icon.configure(text="📁", text_color=_C["accent"])
            self._drop_title.configure(text=folder.name, text_color=_C["text_primary"])
            self._drop_subtitle.configure(text=str(folder), text_color=_C["text_tertiary"])
            self._folder_label.configure(
                text=f"{n} video file{'s' if n != 1 else ''} found",
                text_color=_C["accent"],
            )
            self._process_btn.configure(state="normal")
            self._log(f"Folder: {folder}  •  {n} video(s)", tag="dim")
        else:
            self._drop_icon.configure(text="⚠️",  text_color=_C["log_warn"])
            self._drop_title.configure(text="No videos found",    text_color=_C["log_warn"])
            self._drop_subtitle.configure(text=str(folder),       text_color=_C["text_tertiary"])
            self._folder_label.configure(
                text="No .mov / .avi / .mp4 files in this folder",
                text_color=_C["log_warn"],
            )
            self._process_btn.configure(state="disabled")

    # ── Processing ───────────────────────────────────────────────────────────

    def _start_processing(self) -> None:
        if not self._folder_path or not self._mov_files:
            return
        self._processing = True
        self._stop_event.clear()
        self._process_btn.configure(state="disabled")
        self._stop_btn.configure(
            state="normal",
            fg_color=_C["danger"],
            hover_color=_C["danger_hover"],
            text_color=_C["text_on_accent"],
        )
        self._progress_var.set(0.0)
        self._progress_label.configure(text="Starting…")
        self._log(f"Starting  •  {len(self._mov_files)} video(s)", tag="dim")
        threading.Thread(target=self._worker_thread, daemon=True).start()

    def _stop_processing(self) -> None:
        self._stop_event.set()
        self._stop_btn.configure(state="disabled")
        self._log("Stop requested — finishing current video…", tag="warn")

    def _worker_thread(self) -> None:
        from .config import cfg
        from .pipeline import Pipeline
        from .sheets import SheetsWriter
        try:
            pipeline = Pipeline()
            writer   = SheetsWriter(
                sheet_id             = cfg.sheet_id,
                service_account_path = cfg.service_account_resolved,
                worksheet_name       = cfg.worksheet_name,
                output_csv           = cfg.output_csv_resolved,
            )

            from .pipeline import _normalize_filename

            already_done: set[str] = set()
            try:
                already_done = writer.get_processed_filenames()
            except Exception as _skip_exc:
                self._ui_queue.put((
                    "log_tagged",
                    f"Could not load skip list: {_skip_exc}",
                    "warn",
                ))

            pending = [
                f for f in self._mov_files
                if _normalize_filename(f.name) not in already_done
            ]
            skipped = [f for f in self._mov_files if f not in pending]

            if skipped:
                for vid in skipped:
                    self._ui_queue.put((
                        "log_tagged",
                        f"Skipped (already in Sheet): {vid.name}",
                        "dim",
                    ))
            if not pending:
                self._ui_queue.put((
                    "log_tagged",
                    "All videos in this folder have already been processed.",
                    "dim",
                ))
                self._ui_queue.put(("done", 0))
                return

            total = len(pending)

            for v_idx, video_path in enumerate(pending):
                if self._stop_event.is_set():
                    break
                self._ui_queue.put(("status", video_path.name))

                def frame_cb(done: int, tot: int, msg: str, _i=v_idx, _p=video_path) -> None:
                    frac = (_i + done / max(tot, 1)) / total
                    self._ui_queue.put(("progress", frac,
                        f"{_p.name}  •  frame {done}/{tot}"))

                pr = pipeline.process_video(
                    video_path,
                    progress_callback=frame_cb,
                    save_crops=cfg.save_crops,
                )

                if pr.error:
                    self._ui_queue.put((
                        "log_tagged",
                        f"[{video_path.name}]  Error: {pr.error}",
                        "error",
                    ))
                else:
                    try:
                        writer.write(pr.result, pr.video_path)
                    except Exception as exc:
                        self._ui_queue.put((
                            "log_tagged",
                            f"Sheet write failed for {video_path.name}: {exc}",
                            "warn",
                        ))

                    tag  = "warn" if pr.result.needs_review else "success"
                    flag = "  ⚑ Review" if pr.result.needs_review else "  ✓"
                    self._ui_queue.put((
                        "log_tagged",
                        f"[{datetime.now():%H:%M:%S}]  {video_path.name}"
                        f"  →  {pr.result.common_name}"
                        f"  ({pr.result.confidence:.0%}){flag}",
                        tag,
                    ))

            self._ui_queue.put(("done", total))
        except Exception as exc:
            self._ui_queue.put(("error", str(exc)))

    # ── UI queue ─────────────────────────────────────────────────────────────

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                msg  = self._ui_queue.get_nowait()
                kind = msg[0]
                if kind in ("log", "log_tagged"):
                    self._write_log(msg[1], msg[2] if len(msg) > 2 else None)
                elif kind == "status":
                    self._status_bar.configure(text=msg[1])
                elif kind == "progress":
                    self._progress_var.set(msg[1])
                    self._progress_label.configure(text=msg[2])
                elif kind == "done":
                    self._progress_label.configure(
                        text=f"Done  •  {msg[1]} video(s) processed"
                    )
                    self._progress_var.set(1.0)
                    self._finish_processing()
                elif kind == "error":
                    self._write_log(f"Fatal error: {msg[1]}", "error")
                    self._finish_processing()
        except queue.Empty:
            pass
        self._root.after(100, self._poll_ui_queue)

    def _finish_processing(self) -> None:
        self._processing = False
        self._process_btn.configure(state="normal")
        self._stop_btn.configure(
            state="disabled",
            fg_color=_C["surface2"],
            hover_color=_C["danger"],
            text_color=_C["text_secondary"],
        )
        self._status_bar.configure(text="Ready")

    # ── Settings dialog ───────────────────────────────────────────────────────

    def _open_settings(self) -> None:
        import customtkinter as ctk
        from .config import cfg

        dlg = ctk.CTkToplevel(self._root)
        dlg.title("Settings")
        dlg.geometry("580x620")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.configure(fg_color=_C["bg"])

        hdr = ctk.CTkFrame(dlg, fg_color=_C["header_bg"], corner_radius=0, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(
            hdr, text="Settings",
            font=ctk.CTkFont(family=_FONT_FAMILY, size=16, weight="bold"),
            text_color=_C["text_primary"],
        ).pack(side="left", padx=22)
        ctk.CTkFrame(dlg, fg_color=_C["border"], height=1, corner_radius=0).pack(fill="x")

        scroll = ctk.CTkScrollableFrame(dlg, fg_color=_C["bg"], corner_radius=0)
        scroll.pack(fill="both", expand=True)
        scroll.grid_columnconfigure(1, weight=1)

        sections = [
            ("Google Sheets", [
                ("Sheet ID",                    "sheet_id"),
                ("Worksheet tab name",          "worksheet_name"),
                ("Service account JSON path",   "service_account_path"),
            ]),
            ("Detection", [
                ("Animal confidence threshold",  "megadetector_confidence"),
                ("Species confidence threshold", "classifier_confidence_threshold"),
                ("Frame sample rate (fps)",      "frame_sample_rate"),
                ("Max frames per video",         "max_frames_per_video"),
            ]),
            ("Output", [
                ("Save animal crops",            "save_crops"),
                ("Crops output folder",          "crops_dir"),
                ("CSV backup path",              "output_csv"),
                ("Custom ONNX classifier",       "custom_classifier_path"),
            ]),
        ]

        vars_: dict[str, tk.StringVar] = {}
        row_i = 0
        for section_title, fields in sections:
            ctk.CTkLabel(
                scroll,
                text=section_title.upper(),
                font=ctk.CTkFont(family=_FONT_FAMILY, size=10, weight="bold"),
                text_color=_C["text_tertiary"],
                anchor="w",
            ).grid(row=row_i, column=0, columnspan=2, padx=22, pady=(16, 4), sticky="w")
            row_i += 1
            for label, attr in fields:
                ctk.CTkLabel(
                    scroll,
                    text=label,
                    anchor="w",
                    font=ctk.CTkFont(family=_FONT_FAMILY, size=13),
                    text_color=_C["text_primary"],
                ).grid(row=row_i, column=0, padx=(22, 8), pady=5, sticky="w")
                v = tk.StringVar(value=str(getattr(cfg, attr)))
                ctk.CTkEntry(
                    scroll,
                    textvariable=v,
                    fg_color=_C["surface"],
                    border_color=_C["border"],
                    text_color=_C["text_primary"],
                    font=ctk.CTkFont(family=_FONT_FAMILY, size=13),
                    corner_radius=8,
                ).grid(row=row_i, column=1, padx=(0, 22), pady=5, sticky="ew")
                vars_[attr] = v
                row_i += 1

        def _save() -> None:
            for attr, var in vars_.items():
                val = var.get().strip()
                if not val:
                    continue
                field_type = type(getattr(cfg, attr))
                try:
                    if field_type is float:
                        setattr(cfg, attr, float(val))
                    elif field_type is int:
                        setattr(cfg, attr, int(val))
                    elif field_type is bool:
                        setattr(cfg, attr, val.lower() in ("true", "1", "yes"))
                    else:
                        setattr(cfg, attr, val)
                except ValueError:
                    pass
            cfg.save()
            self._log("Settings saved.", tag="dim")
            dlg.destroy()

        footer = ctk.CTkFrame(dlg, fg_color=_C["header_bg"], corner_radius=0, height=56)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        ctk.CTkFrame(footer, fg_color=_C["border"], height=1, corner_radius=0).pack(
            fill="x", side="top"
        )
        ctk.CTkButton(
            footer, text="Save",
            width=100, height=34,
            font=ctk.CTkFont(family=_FONT_FAMILY, size=14, weight="bold"),
            fg_color=_C["accent"], hover_color=_C["accent_hover"],
            text_color=_C["text_on_accent"], corner_radius=8,
            command=_save,
        ).pack(side="right", padx=20)
        ctk.CTkButton(
            footer, text="Cancel",
            width=80, height=34,
            font=ctk.CTkFont(family=_FONT_FAMILY, size=14),
            fg_color="transparent", hover_color=_C["surface2"],
            text_color=_C["text_secondary"], corner_radius=8,
            command=dlg.destroy,
        ).pack(side="right", padx=(0, 8))

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _log(self, text: str, tag: str | None = None) -> None:
        if threading.current_thread() != threading.main_thread():
            self._ui_queue.put(("log", text, tag))
        else:
            self._write_log(text, tag)

    def _write_log(self, text: str, tag: str | None = None) -> None:
        self._log_box.configure(state="normal")
        self._log_box._textbox.insert("end", text + "\n", tag or "")
        self._log_box.configure(state="disabled")
        self._log_box.see("end")

    def _clear_log(self) -> None:
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")
