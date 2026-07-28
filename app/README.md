# App

This folder contains the Tkinter desktop user interface and source-side runtime helpers.

Key files:

- `main.py` builds the main application window and tab layout.
- `tabs/` contains the Live, Dataset, and Training tab implementations.
- `components.py`, `styles.py`, `constants.py`, and `ui_platform.py` define shared UI behavior and styling.
- `config_manager.py` loads and saves user settings from `config/config.json`.
- `backend_bridge.py` connects UI actions to backend training, validation, testing, and inference workflows.
- `checkpoint_registry.py`, `promotion_gate.py`, `promotion_package.py`, and related files manage model promotion state.

Most UI changes should start in `main.py`, `components.py`, or the relevant file in `tabs/`.
