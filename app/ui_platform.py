import sys


def configure_display_scaling(root):
    """Align Tk scaling with the monitor DPI for more consistent sizing."""
    try:
        pixels_per_inch = float(root.winfo_fpixels("1i"))
    except Exception:
        return

    scaling = max(1.0, min(2.0, pixels_per_inch / 72.0))
    try:
        current = float(root.tk.call("tk", "scaling"))
    except Exception:
        current = None

    if current is None or abs(current - scaling) > 0.05:
        root.tk.call("tk", "scaling", scaling)


def get_sidebar_width(widget, preferred=400, minimum=320, maximum=460):
    """Size side panels from screen width instead of one fixed pixel value."""
    screen_width = widget.winfo_screenwidth()
    adaptive_width = int(screen_width * 0.26)
    return max(minimum, min(maximum, max(preferred, adaptive_width)))


def get_figure_dpi(widget, base_dpi=100):
    """Raise Matplotlib DPI on dense displays without exploding figure size."""
    try:
        scaling = float(widget.tk.call("tk", "scaling"))
    except Exception:
        scaling = 1.0
    return int(round(base_dpi * max(1.0, min(1.35, scaling))))


def get_scroll_units(event):
    """Normalize wheel events across Windows, macOS trackpads, and Linux."""
    button_num = getattr(event, "num", None)
    if button_num == 4:
        return -1
    if button_num == 5:
        return 1

    delta = getattr(event, "delta", 0)
    if not delta:
        return 0

    if sys.platform == "darwin":
        return -1 if delta > 0 else 1

    steps = int(delta / 120)
    if steps == 0:
        steps = 1 if delta > 0 else -1
    return -steps
