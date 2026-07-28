import tkinter as tk
from tkinter import ttk

from app.constants import COLORS


def configure_styles():
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    style.configure("TFrame", background=COLORS["surface"])
    style.configure("TLabel", background=COLORS["surface"], foreground=COLORS["text"])
    style.configure(
        "TLabelframe",
        background=COLORS["surface"],
        bordercolor=COLORS["border"],
        relief=tk.SOLID,
        borderwidth=1,
    )
    style.configure(
        "TLabelframe.Label",
        background=COLORS["surface"],
        foreground=COLORS["accent"],
        font=("TkDefaultFont", 10, "bold"),
    )

    style.configure("Dashboard.TNotebook", background=COLORS["app_bg"], borderwidth=0)
    style.configure(
        "Dashboard.TNotebook.Tab",
        background=COLORS["surface"],
        foreground=COLORS["muted"],
        padding=(18, 4),
        borderwidth=1,
    )
    style.map(
        "Dashboard.TNotebook.Tab",
        background=[("selected", COLORS["panel"])],
        foreground=[("selected", COLORS["accent"])],
        padding=[("selected", (18, 4)), ("!selected", (18, 4))],
    )

    style.configure(
        "TButton",
        padding=(10, 6),
        background=COLORS["panel"],
        foreground=COLORS["text"],
        bordercolor=COLORS["border"],
    )
    style.map("TButton", background=[("active", COLORS["accent_soft"])])
    style.configure(
        "Accent.TButton",
        background=COLORS["accent"],
        foreground="white",
        bordercolor=COLORS["accent"],
    )
    style.map(
        "Accent.TButton",
        background=[("active", "#004a99")],
        foreground=[("active", "white")],
    )
    style.configure(
        "Warn.TButton",
        background=COLORS["warning"],
        foreground="white",
        bordercolor=COLORS["warning"],
    )

    style.configure(
        "SectionHeader.TLabel",
        background=COLORS["surface"],
        foreground=COLORS["accent"],
        font=("TkDefaultFont", 11, "bold"),
    )
    style.configure(
        "SectionNote.TLabel",
        background=COLORS["surface"],
        foreground=COLORS["muted"],
    )
    style.configure(
        "Status.TLabel",
        background=COLORS["accent_soft"],
        foreground=COLORS["accent"],
        padding=(10, 5),
    )
    style.configure(
        "StatusOK.TLabel",
        background="#d4edda",
        foreground=COLORS["success"],
        padding=(6, 3),
    )
    style.configure("Card.TFrame", background=COLORS["panel"], relief=tk.SOLID, borderwidth=1)
    style.configure(
        "Toolbar.TFrame",
        background=COLORS["panel"],
        relief=tk.SOLID,
        borderwidth=1,
    )
    style.configure(
        "ToolbarTitle.TLabel",
        background=COLORS["panel"],
        foreground=COLORS["accent"],
        font=("TkDefaultFont", 10, "bold"),
    )
    style.configure(
        "ToolbarNote.TLabel",
        background=COLORS["panel"],
        foreground=COLORS["muted"],
    )
    style.configure(
        "CollapseHeader.TButton",
        padding=(10, 6),
        background=COLORS["accent_soft"],
        foreground=COLORS["accent"],
        bordercolor=COLORS["border"],
    )
    style.configure(
        "SummaryToggle.TRadiobutton",
        background=COLORS["panel"],
        foreground=COLORS["text"],
        bordercolor=COLORS["border"],
        relief=tk.RAISED,
        padding=(16, 6),
        indicatorrelief=tk.FLAT,
    )
    style.map(
        "SummaryToggle.TRadiobutton",
        background=[
            ("selected", COLORS["accent_soft"]),
            ("active", COLORS["accent_soft"]),
        ],
        foreground=[
            ("selected", COLORS["accent"]),
            ("disabled", COLORS["muted"]),
        ],
        relief=[("selected", tk.SUNKEN)],
    )
    style.configure(
        "Treeview",
        background=COLORS["panel"],
        fieldbackground=COLORS["panel"],
        foreground=COLORS["text"],
        rowheight=28,
        bordercolor=COLORS["border"],
    )
    style.configure(
        "Treeview.Heading",
        background=COLORS["accent_soft"],
        foreground=COLORS["accent"],
        relief=tk.FLAT,
    )
    style.map(
        "Treeview",
        background=[("selected", "#cfe6da")],
        foreground=[("selected", COLORS["text"])],
    )
