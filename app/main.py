from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from app import styles
from app.backend_bridge import resource_path
from app.components import GlobalStatusBar, SharedConsole
from app.constants import COLORS
from app.build_info import DISPLAY_VERSION
from app.ui_platform import configure_display_scaling, get_scroll_units


MAIN_TAB_ORDER = ("Live", "Dataset", "Training")


def enable_dpi_awareness():
    """Use the monitor's true pixel size on Windows display-scaled laptops."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except OSError:
            ctypes.windll.user32.SetProcessDPIAware()
    except (ImportError, AttributeError, OSError):
        pass


class LazyTab(ttk.Frame):
    """Notebook page that imports and creates its real tab only when opened."""

    def __init__(self, parent, label: str, factory):
        super().__init__(parent)
        self.label = label
        self.factory = factory
        self.content = None
        self._loading = False

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.loading_label = ttk.Label(
            self,
            text=f"Loading {label}...",
            anchor="center",
            style="SectionNote.TLabel",
        )
        self.loading_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

    def ensure_loaded(self):
        if self.content is not None or self._loading:
            return
        self._loading = True
        self.update_idletasks()
        try:
            self.content = self.factory(self)
            self.content.grid(row=0, column=0, sticky="nsew")
            self.loading_label.destroy()
        except Exception:
            self._loading = False
            raise

    def _delegate(self, name):
        if self.content is None:
            return None
        return getattr(self.content, name, None)

    def _toggle_log_panel(self):
        fn = self._delegate("_toggle_log_panel")
        if callable(fn):
            fn()

    def _save_settings(self):
        fn = self._delegate("_save_settings")
        if callable(fn):
            fn()

    def _load_saved_settings(self):
        fn = self._delegate("_load_saved_settings")
        if callable(fn):
            fn()


class MainApplication:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"Defect Detection Platform  —  v{DISPLAY_VERSION}")
        configure_display_scaling(self.root)
        self.root.minsize(1000, 600)
        self._maximize_window()

        self.scroll_targets = []
        self.shared_console = SharedConsole()
        self.logo_photo = None
        self.window_icon = None

        styles.configure_styles()
        self.root.configure(bg=COLORS["app_bg"])
        self._set_window_icon()
        self._build()

    def _build(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        self.notebook = ttk.Notebook(self.root, style="Dashboard.TNotebook")
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(8, 4))

        self.status_bar = GlobalStatusBar(
            self.root,
            toggle_log_callback=self._toggle_current_tab_console_log,
        )
        self.status_bar.grid(row=1, column=0, sticky="ew")

        tab_specs = (
            ("live_camera_tab", "Live", "  Live  ", self._create_live_tab),
            ("dataset_tab", "Dataset", "  Dataset  ", self._create_dataset_tab),
            ("training_tab", "Training", "  Training  ", self._create_training_tab),
        )
        for attribute, label, title, factory in tab_specs:
            tab = LazyTab(self.notebook, label, factory)
            setattr(self, attribute, tab)
            self.notebook.add(tab, text=title)

        self.notebook.select(self.live_camera_tab)
        self._last_selected_tab = self.live_camera_tab
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed, add="+")
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel, add="+")

        # Let Windows paint the main window before importing any optional heavy UI modules.
        self.root.after(50, self._load_initial_tab)

    def _load_initial_tab(self):
        self.live_camera_tab.ensure_loaded()

    def _create_live_tab(self, parent):
        from app.tabs.live_camera_tab import LiveCameraTab
        return LiveCameraTab(parent, self.scroll_targets, self.status_bar, self.shared_console)

    def _create_dataset_tab(self, parent):
        from app.tabs.dataset_tab import DatasetTab
        return DatasetTab(parent, self.scroll_targets, self.status_bar, self.shared_console)

    def _create_training_tab(self, parent):
        from app.tabs.training_tab import TrainingTab
        return TrainingTab(parent, self.scroll_targets, self.status_bar, self.shared_console)

    def _set_window_icon(self):
        """Set the title-bar/taskbar icon without adding a visible header."""
        try:
            with Image.open(resource_path("assets/logo.png")) as pil_img:
                icon = pil_img.convert("RGBA")
                icon.thumbnail((256, 256), Image.Resampling.LANCZOS)
                self.window_icon = ImageTk.PhotoImage(icon)
            self.root.iconphoto(True, self.window_icon)
        except (FileNotFoundError, OSError, tk.TclError):
            pass

    def _on_tab_changed(self, _event=None):
        current_id = self.notebook.select()
        if not current_id:
            return
        current_tab = self.root.nametowidget(current_id)
        previous_tab = getattr(self, "_last_selected_tab", None)
        if previous_tab is not None and previous_tab is not current_tab:
            previous_tab._save_settings()

        # Paint the selected page first, then perform its imports/initialization.
        self.root.after_idle(lambda tab=current_tab: self._activate_tab(tab))
        self._last_selected_tab = current_tab

    def _activate_tab(self, tab):
        if isinstance(tab, LazyTab):
            tab.ensure_loaded()
        tab._load_saved_settings()

    def _on_mousewheel(self, event):
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if not widget:
            return
        scroll_units = get_scroll_units(event)
        if scroll_units == 0:
            return
        for canvas in self.scroll_targets:
            if str(widget).startswith(str(canvas)):
                first, last = canvas.yview()
                if scroll_units < 0 and first <= 0.0:
                    canvas.yview_moveto(0.0)
                    break
                if scroll_units > 0 and last >= 1.0:
                    break
                canvas.yview_scroll(scroll_units, "units")
                break

    def _toggle_current_tab_console_log(self):
        current_tab_id = self.notebook.select()
        if not current_tab_id:
            return
        current_tab = self.root.nametowidget(current_tab_id)
        current_tab._toggle_log_panel()

    def _maximize_window(self):
        try:
            self.root.state("zoomed")
            return
        except tk.TclError:
            pass
        self.root.update_idletasks()
        self.root.geometry(
            f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0"
        )


def run():
    enable_dpi_awareness()
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()


if __name__ == "__main__":
    run()
