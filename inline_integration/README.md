# Inline Integration

This folder contains lightweight scripts for running model inference outside the desktop UI.

Key files:

- `model_integration.py` provides a programmatic prediction interface for external systems.
- `run_json_image_inference.py` is a small JSON/image inference runner built on top of `model_integration.py`.

Use this folder when another process needs to call the trained model directly without launching the Tkinter app.
