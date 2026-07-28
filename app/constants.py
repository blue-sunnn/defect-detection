COLORS = {
    "app_bg": "#f0f0f0",
    "surface": "#f0f0f0",
    "panel": "#ffffff",
    "border": "#cccccc",
    "accent": "#005fb8",
    "accent_soft": "#e1e1e1",
    "text": "#000000",
    "muted": "#555555",
    "placeholder_bg": "#e8e8e8",
    "log_bg": "#ffffff",
    "warning": "#b25d2a",
    "success": "#1a7a4a",
}

CONFIG_PHASE_1 = [
    [("Optimizer", ["Adam", "AdamW", "SGD", "RMSprop"])],
    [("Learning Rate", "1e-3")],
    [("Epochs", "15")],
]

CONFIG_PHASE_1_STG1 = [
    [("Optimizer", ["AdamW", "Adam", "SGD", "RMSprop"])],
    [("Learning Rate", "1e-4")],
    [("Epochs", "50")],
]

CONFIG_PHASE_1_STG2 = [
    [("Optimizer", ["SGD", "AdamW", "Adam", "RMSprop", "None"])],
    [("Learning Rate", "1e-5")],
    [("Epochs", "50")],
]

CONFIG_PHASE_2_STG1 = [
    [("Optimizer", ["AdamW", "Adam", "SGD", "RMSprop"])],
    [("Learning Rate", "5e-6")],
    [("Epochs", "30")],
]

CONFIG_PHASE_2_STG2 = [
    [("Optimizer", ["None", "AdamW", "Adam", "SGD", "RMSprop"])],
    [("Learning Rate", "1e-6")],
    [("Epochs", "20")],
]
