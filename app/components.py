from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import ttk
import os

from app.styles import COLORS


PATH_ENTRY_WIDTH = 36
HEADER_PROGRESS_LENGTH = 360
MAIN_PANE_WEIGHT = 5
CONSOLE_PANE_WEIGHT = 1


def display_text(value, fallback="—"):
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "none":
        return fallback
    return text


def enable_debounced_matplotlib_resize(canvas, delay_ms=120):
    """Throttle expensive Matplotlib redraws while a Tk pane is being dragged."""
    widget = canvas.get_tk_widget()
    state = {"after_id": None, "size": None}

    def apply_resize():
        state["after_id"] = None
        size = state["size"]
        if not size or not widget.winfo_exists():
            return

        width, height = size
        if width <= 1 or height <= 1:
            return

        class ResizeEventProxy:
            pass

        event = ResizeEventProxy()
        event.width = width
        event.height = height
        canvas.resize(event)

    def queue_resize(event):
        state["size"] = (event.width, event.height)
        if state["after_id"] is not None:
            try:
                widget.after_cancel(state["after_id"])
            except tk.TclError:
                pass
        state["after_id"] = widget.after(delay_ms, apply_resize)

    # Replace Matplotlib's default per-pixel Configure callback. The widget still
    # follows the sash immediately; only the costly figure redraw is delayed.
    widget.bind("<Configure>", queue_resize)
    return queue_resize


class ScrollPanel(ttk.Frame):
    def __init__(self, parent, width, global_scroll_targets, show_scrollbar=True, **kwargs):
        super().__init__(parent, width=width, **kwargs)
        self.pack_propagate(False)
        canvas_holder = ttk.Frame(self)
        canvas_holder.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_holder, bg=COLORS["surface"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_holder, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        if show_scrollbar:
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(canvas_window, width=event.width),
        )
        self.inner.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        global_scroll_targets.append(self.canvas)


class CollapsibleGroup(ttk.Frame):
    def __init__(self, parent, title, expanded=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.pack(fill=tk.X, pady=(0, 5))
        self.is_open = expanded
        self.header = ttk.Button(self, style="CollapseHeader.TButton", command=self.toggle)
        self.header.pack(fill=tk.X)
        self.body = ttk.Frame(self, padding=8)
        self.title = title
        self._refresh()

    def _refresh(self):
        arrow = "▾" if self.is_open else "▸"
        self.header.config(text=f"{arrow}  {self.title}")
        if self.is_open:
            if not self.body.winfo_manager():
                self.body.pack(fill=tk.X, pady=0)
        else:
            self.body.pack_forget()

    def refresh(self):
        self._refresh()

    def toggle(self):
        self.is_open = not self.is_open
        self._refresh()


class DataDrivenForm(CollapsibleGroup):
    def __init__(self, parent, title, fields, field_width=None, expanded=True, **kwargs):
        super().__init__(parent, title, expanded=expanded, **kwargs)
        self.entries = {}
        total_columns = max((len(row) for row in fields), default=1)
        for idx in range(total_columns):
            self.body.columnconfigure(idx * 2 + 1, weight=1)
        for row_idx, row_items in enumerate(fields):
            for col_idx, item in enumerate(row_items):
                if item is None:
                    continue
                label_text, value = item[0], item[1]
                col_span = item[2] if len(item) > 2 else 1
                label_col = col_idx * 2
                input_col = label_col + 1
                ttk.Label(self.body, text=f"{label_text}:").grid(
                    row=row_idx,
                    column=label_col,
                    sticky="w",
                    padx=(0, 8),
                    pady=4,
                )
                if isinstance(value, list):
                    widget = ttk.Combobox(self.body, values=value, state="readonly")
                    if field_width:
                        widget.config(width=field_width)
                    widget.current(0)
                    widget.grid(
                        row=row_idx,
                        column=input_col,
                        sticky="w" if field_width else "ew",
                        pady=4,
                        padx=(0, 10),
                    )
                else:
                    widget = ttk.Entry(self.body)
                    if field_width:
                        widget.config(width=field_width)
                    widget.insert(0, str(value))
                    widget.grid(
                        row=row_idx,
                        column=input_col,
                        columnspan=col_span,
                        sticky="w" if field_width else "ew",
                        pady=4,
                        padx=(0, 10),
                    )
                self.entries[label_text] = widget


class PlainFormGroup(ttk.LabelFrame):
    def __init__(self, parent, title, fields, field_width=None, **kwargs):
        super().__init__(parent, text=title, padding=8, **kwargs)
        self.pack(fill=tk.X, pady=(0, 10))
        self.entries = {}
        total_columns = max((len(row) for row in fields), default=1)
        for idx in range(total_columns):
            self.columnconfigure(idx * 2 + 1, weight=1)
        for row_idx, row_items in enumerate(fields):
            for col_idx, item in enumerate(row_items):
                if item is None:
                    continue
                label_text, value = item[0], item[1]
                col_span = item[2] if len(item) > 2 else 1
                label_col = col_idx * 2
                input_col = label_col + 1
                ttk.Label(self, text=f"{label_text}:").grid(
                    row=row_idx,
                    column=label_col,
                    sticky="w",
                    padx=(0, 8),
                    pady=4,
                )
                if isinstance(value, list):
                    widget = ttk.Combobox(self, values=value, state="readonly")
                    if field_width:
                        widget.config(width=field_width)
                    widget.current(0)
                    widget.grid(
                        row=row_idx,
                        column=input_col,
                        sticky="w" if field_width else "ew",
                        pady=4,
                        padx=(0, 10),
                    )
                else:
                    widget = ttk.Entry(self)
                    if field_width:
                        widget.config(width=field_width)
                    widget.insert(0, str(value))
                    widget.grid(
                        row=row_idx,
                        column=input_col,
                        columnspan=col_span,
                        sticky="w" if field_width else "ew",
                        pady=4,
                        padx=(0, 10),
                    )
                self.entries[label_text] = widget


class EntryButtonGroup(CollapsibleGroup):
    def __init__(self, parent, title, rows, command_callback, required_labels=None, **kwargs):
        super().__init__(parent, title, expanded=True, **kwargs)
        self.body.columnconfigure(1, weight=1)
        self.entries = {}
        required_labels = set(required_labels or [])
        for row_idx, (label_text, button_text) in enumerate(rows):
            required_mark = " *" if label_text in required_labels else ""
            ttk.Label(self.body, text=f"{label_text}{required_mark}:").grid(
                row=row_idx,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=4,
            )
            entry = ttk.Entry(self.body, width=PATH_ENTRY_WIDTH)
            entry.grid(row=row_idx, column=1, sticky="ew", pady=4, padx=(0, 12))
            self.entries[label_text] = entry
            ttk.Button(
                self.body,
                text=button_text,
                width=3,
                command=lambda name=label_text: command_callback(name),
            ).grid(row=row_idx, column=2, padx=(8, 0), pady=4)
        if required_labels:
            ttk.Label(self.body, text="* required", style="SectionNote.TLabel").grid(
                row=len(rows),
                column=0,
                columnspan=3,
                sticky="w",
                pady=(2, 0),
            )


class LogWidget(ttk.Frame):
    def __init__(self, parent, height, **kwargs):
        self.max_lines = int(kwargs.pop("max_lines", 1000)) if "max_lines" in kwargs else 1000
        super().__init__(parent, **kwargs)
        self._mirrors = []
        self._shared_console = None
        self.text = tk.Text(
            self,
            height=height,
            state=tk.DISABLED,
            bg=COLORS["log_bg"],
            fg=COLORS["text"],
            bd=0,
            padx=8,
            pady=6,
            font=("TkFixedFont", 9),
        )
        self.text.pack(fill=tk.BOTH, expand=True)

    def _append(self, message):
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, message + "\n")
        line_count = int(float(self.text.index("end-1c").split(".")[0]))
        if line_count > self.max_lines:
            self.text.delete("1.0", f"{line_count - self.max_lines + 1}.0")
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)

    def _reset(self):
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.config(state=tk.DISABLED)

    def attach_mirror(self, mirror_widget):
        if mirror_widget is self or mirror_widget in self._mirrors:
            return
        self._mirrors.append(mirror_widget)
        content = self.text.get("1.0", tk.END).rstrip("\n")
        mirror_widget._reset()
        if content:
            for line in content.splitlines():
                mirror_widget._append(line)

    def log(self, message):
        if self._shared_console is not None:
            self._shared_console.log(message)
            return
        self._append(message)
        for mirror in self._mirrors:
            mirror._append(message)

    def clear(self):
        if self._shared_console is not None:
            self._shared_console.clear()
            return
        self._reset()
        for mirror in self._mirrors:
            mirror._reset()


class SharedConsole:
    def __init__(self):
        self._history = []
        self._widgets = []

    def attach(self, widget):
        if widget in self._widgets:
            return
        widget._shared_console = self
        self._widgets.append(widget)
        widget._reset()
        for line in self._history:
            widget._append(line)

    @staticmethod
    def _run_on_ui_thread(widget, callback, *args):
        try:
            widget.after(0, lambda: callback(*args))
        except (tk.TclError, RuntimeError):
            pass

    def log(self, message):
        text = str(message)
        self._history.append(text)
        for widget in tuple(self._widgets):
            self._run_on_ui_thread(widget, widget._append, text)

    def clear(self):
        self._history.clear()
        for widget in tuple(self._widgets):
            self._run_on_ui_thread(widget, widget._reset)


class ConsoleLogPanel(tk.Frame):
    def __init__(self, parent, height=6, **kwargs):
        super().__init__(
            parent,
            bg=COLORS["panel"],
            highlightbackground="#555555",
            highlightthickness=1,
            **kwargs,
        )
        notebook = ttk.Notebook(self, style="Dashboard.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        log_frame = ttk.Frame(notebook, padding=4)
        notebook.add(log_frame, text=" Console Log ")
        self.logger = LogWidget(log_frame, height=height)
        self.logger.pack(fill=tk.BOTH, expand=True)


class ModeHeader(ttk.Frame):
    def __init__(self, parent, title, buttons_config=None, center_builder=None, right_width=None, **kwargs):
        super().__init__(parent, padding=(4, 0, 4, 15), **kwargs)
        self.pack(fill=tk.X)
        self.columnconfigure(1, weight=1)
        ttk.Label(self, text=title, style="SectionHeader.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        if center_builder:
            center_frame = ttk.Frame(self)
            center_frame.grid(row=0, column=1, sticky="ew", padx=(16, 16))
            center_frame.columnconfigure(0, weight=1)
            center_builder(center_frame)
        if buttons_config:
            button_frame = ttk.Frame(self)
            if right_width:
                button_frame.configure(width=right_width, height=34)
                button_frame.grid_propagate(False)
                self.columnconfigure(2, minsize=right_width)
                button_frame.grid(row=0, column=2, sticky="ew")
            else:
                button_frame.grid(row=0, column=2, sticky="e")
            button_frame.rowconfigure(0, weight=1)
            for index, (label, command) in enumerate(buttons_config):
                style = "Accent.TButton" if index == 0 else "TButton"
                button_frame.columnconfigure(index, weight=1, uniform="header_buttons")
                ttk.Button(button_frame, text=label, style=style, command=command).grid(
                    row=0,
                    column=index,
                    sticky="nsew",
                    padx=(0 if index == 0 else 6, 0),
                )


class Toolbar(ttk.Frame):
    def __init__(self, parent, title, subtitle, buttons_config, **kwargs):
        super().__init__(parent, style="Toolbar.TFrame", padding=8, **kwargs)
        self.pack(fill=tk.X, pady=(6, 0))
        self.columnconfigure(1, weight=1)
        title_block = tk.Frame(self, bg=COLORS["panel"])
        title_block.grid(row=0, column=0, sticky="w")
        ttk.Label(title_block, text=title, style="ToolbarTitle.TLabel").pack(anchor="w")
        ttk.Label(title_block, text=subtitle, style="ToolbarNote.TLabel").pack(anchor="w")
        button_block = tk.Frame(self, bg=COLORS["panel"])
        button_block.grid(row=0, column=1, sticky="e")
        for index, (label, command) in enumerate(buttons_config):
            style = "Accent.TButton" if index == 0 else "TButton"
            ttk.Button(button_block, text=label, style=style, command=command).pack(
                side=tk.LEFT,
                padx=(6, 0),
            )


class SummaryCards(ttk.Frame):
    def __init__(self, parent, items, columns=3, **kwargs):
        super().__init__(parent, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)
        for column in range(columns):
            self.columnconfigure(column, weight=1)
        self.value_labels = {}
        for index, (title, value) in enumerate(items):
            row, column = divmod(index, columns)
            card = ttk.Frame(self, padding=8, style="Card.TFrame")
            card.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
            tk.Label(card, text=title, bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w")
            value_label = tk.Label(
                card,
                text=value,
                bg=COLORS["panel"],
                fg=COLORS["text"],
                font=("TkDefaultFont", 11, "bold"),
            )
            value_label.pack(anchor="w", pady=(6, 0))
            self.value_labels[title] = value_label

    def set_value(self, title, value):
        if title in self.value_labels:
            self.value_labels[title].config(text=value)


class SimpleTable(ttk.Frame):
    def __init__(self, parent, columns, rows, height=7, **kwargs):
        super().__init__(parent, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=height)
        for column in columns:
            self.tree.heading(column, text=column)
            self.tree.column(column, width=110, anchor="center")
        for row in rows:
            self.tree.insert("", tk.END, values=row)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def clear(self):
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)

    def append_row(self, row):
        item_id = self.tree.insert("", tk.END, values=row)

        self.tree.see(item_id)


class ImagePreview(ttk.Frame):
    def __init__(self, parent, empty_text="No image loaded", max_size=(720, 520), **kwargs):
        super().__init__(parent, **kwargs)
        self._max_size = max_size
        self._photo = None
        self._empty_text = empty_text
        self._current_image = None  # Keeps a reference to the original PIL Image

        self.label = tk.Label(
            self,
            text=empty_text,
            bg=COLORS["placeholder_bg"],
            fg=COLORS["muted"],
            relief=tk.FLAT,
            bd=0,
            anchor="center",
            justify="center",
            font=("TkDefaultFont", 10),
        )
        self.label.pack(fill=tk.BOTH, expand=True)

        # Bind to the Frame's resize event to scale the image dynamically
        self.bind("<Configure>", self._on_resize)

    def clear(self, text=None):
        self._photo = None
        self._current_image = None
        self.label.config(
            image="",
            text=text or self._empty_text,
            bg=COLORS["placeholder_bg"],
            fg=COLORS["muted"],
        )

    def show_image(self, image_path):
        if Image is None or ImageTk is None or ImageOps is None:
            self.clear("Pillow not installed.\nCannot preview image.")
            return

        try:
            with Image.open(image_path) as image:
                # Cache the fully oriented RGB image in memory
                self._current_image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception as e:
            self.clear(f"Failed to load image:\n{e}")
            return

        # Trigger a layout update to find out the current available width/height
        self.update_idletasks()
        self._render_image(self.winfo_width(), self.winfo_height())

    def _on_resize(self, event):
        # Only recalculate if an image is actively loaded
        if self._current_image is not None:
            self._render_image(event.width, event.height)

    def _render_image(self, max_w, max_h):
        # Guard against zero or microscopic container dimensions during rendering updates
        if max_w <= 10 or max_h <= 10:
            return

        img_w, img_h = self._current_image.size
        
        # Calculate the proper scaling factor to fit the current window frame space
        ratio = min(max_w / img_w, max_h / img_h)
        
        new_size = (int(img_w * ratio), int(img_h * ratio))
        
        # Avoid zero dimensions resizing errors
        if new_size[0] > 0 and new_size[1] > 0:
            resample_filter = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            preview_image = self._current_image.resize(new_size, resample_filter)
            
            self._photo = ImageTk.PhotoImage(preview_image)
            self.label.config(image=self._photo, text="", bg=COLORS["panel"])


class Placeholder(ttk.Frame):
    def __init__(self, parent, text, **kwargs):
        super().__init__(parent, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            self,
            text=text,
            bg=COLORS["placeholder_bg"],
            fg=COLORS["muted"],
            relief=tk.FLAT,
            bd=0,
            anchor="center",
            justify="center",
            font=("TkDefaultFont", 10),
        ).pack(fill=tk.BOTH, expand=True, ipadx=14, ipady=42)


class SliderRow(ttk.Frame):
    def __init__(
        self,
        parent,
        label_text,
        from_=0.0,
        to=1.0,
        initial_value=1.0,
        format_str="{:.2f}",
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self.pack(fill=tk.X, pady=4)
        self._fmt = format_str
        self._from = float(from_)
        self._to = float(to)
        ttk.Label(self, text=label_text, width=16).pack(side=tk.LEFT)
        self.var = tk.DoubleVar(value=initial_value)
        
        # Keep a reference to the scale widget so we can configure it later
        self.scale = ttk.Scale(
            self, 
            from_=from_, 
            to=to, 
            variable=self.var, 
            command=self._on_change,
            state="normal",
            cursor="arrow",
        )
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.entry_var = tk.StringVar(value=self._fmt.format(initial_value))
        self.entry = ttk.Entry(self, width=7, textvariable=self.entry_var)
        self.entry.pack(side=tk.RIGHT)
        self.entry.bind("<Return>", self._commit_entry)
        self.entry.bind("<FocusOut>", self._commit_entry)

    def _on_change(self, _):
        self.entry_var.set(self._fmt.format(self.var.get()))

    def _commit_entry(self, _event=None):
        try:
            value = float(self.entry_var.get())
        except ValueError:
            self._on_change(None)
            return
        value = min(max(value, self._from), self._to)
        self.var.set(value)
        self._on_change(None)

    def get(self):
        return self.var.get()

    # ADD THIS METHOD TO UPDATE BOUNDS ON THE FLY:
    def update_bounds(self, min_val, max_val):
        """Dynamically shifts the physical min/max slider constraints safely."""
        # Temporarily enable the widget to accept the layout alterations safely
        self.scale.config(state="normal")
        
        self._from = float(min_val)
        self._to = float(max_val)
        self.scale.config(from_=self._from, to=self._to)
        current_val = self.var.get()
        if current_val < self._from:
            self.var.set(self._from)
        elif current_val > self._to:
            self.var.set(self._to)
        self._on_change(None)
        
        # Lock it right back up
        self.scale.config(state="disabled")

    def set_value(self, target_value):
        """Programmatically updates the display value while keeping it locked to users."""
        self.scale.config(state="normal")
        self.var.set(float(target_value))
        self._on_change(None)
        self.scale.config(state="disabled")


class GlobalStatusBar(ttk.Frame):
    def __init__(self, parent, toggle_log_callback=None, **kwargs):
        # Fallback check: Redirect custom controllers to the true root window element
        if not hasattr(parent, "tk") and hasattr(parent, "root"):
            parent = parent.root

        super().__init__(parent, style="Toolbar.TFrame", padding=(8, 2), **kwargs)
        self.grid_propagate(False)
        self.configure(height=34)

        self._device_var = tk.StringVar(value="cpu")
        self._ckpt_var = tk.StringVar(value="No checkpoint loaded")
        self._dataset_var = tk.StringVar(value="No dataset root set")

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)

        self.info_frame = ttk.Frame(self, style="Toolbar.TFrame")
        self.info_frame.grid(row=0, column=0, sticky="ew")
        self.info_frame.columnconfigure(0, weight=1)
        self.info_frame.columnconfigure(1, weight=1)
        self.info_frame.columnconfigure(2, weight=1)

        self._value_labels = []

        # Grid info parameters so they can compress more gracefully on narrow screens.
        for column, (label, variable) in enumerate([
            ("Device", self._device_var),
            ("Checkpoint", self._ckpt_var),
            ("Dataset", self._dataset_var),
        ]):
            frame = ttk.Frame(self.info_frame, style="Toolbar.TFrame")
            frame.grid(row=0, column=column, sticky="ew", padx=(0, 10))
            frame.columnconfigure(1, weight=1)
            ttk.Label(frame, text=f"{label}:", style="ToolbarNote.TLabel").pack(side=tk.LEFT)
            value_label = ttk.Label(
                frame,
                textvariable=variable,
                style="ToolbarTitle.TLabel",
                anchor="w",
            )
            value_label.pack(side=tk.LEFT, padx=(4, 0), fill=tk.X, expand=True)
            self._value_labels.append(value_label)

        # Add the Log Console toggle button on the right edge of the bar panel
        if toggle_log_callback:
            self.toggle_btn = ttk.Button(
                self,
                text="Console Log",
                style="TButton",
                command=toggle_log_callback
            )
            self.toggle_btn.grid(row=0, column=1, sticky="e", padx=(8, 0))

        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        total_width = max(360, event.width)
        compact_width = 8
        dataset_width = max(12, int((total_width - 220) / 10))
        for index, value_label in enumerate(self._value_labels):
            width = dataset_width if index == 2 else compact_width
            value_label.configure(width=width, wraplength=0)

    def _ellipsize_middle(self, value, max_chars=36):
        if not value:
            return value
        if len(value) <= max_chars:
            return value
        keep = max(8, (max_chars - 3) // 2)
        return f"{value[:keep]}...{value[-keep:]}"

    def set_device(self, value):
        self._device_var.set(value)

    def set_ckpt(self, value):
        if not value or value == "No checkpoint loaded":
            self._ckpt_var.set(value)
            return
        self._ckpt_var.set(os.path.basename(value.rstrip("/\\")) or value)

    def set_dataset(self, value):
        if not value or value == "No dataset root set":
            self._dataset_var.set(value)
            return
        self._dataset_var.set(self._ellipsize_middle(value, max_chars=24))
