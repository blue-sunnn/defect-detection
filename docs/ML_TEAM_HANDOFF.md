# ML Team Handoff

This document describes the implemented three-tab workflow in the Tkinter app:
`Live`, `Dataset`, and `Training`. It uses actual module names and persisted paths from
the repository.

## Main Tabs

`defect_detection/app/main.py` registers exactly these main tabs through
`MAIN_TAB_ORDER`: `Live`, `Dataset`, `Training`. Standalone Evaluation and Threshold
navigation has been removed. Reusable backend services remain in `defect_detection/backend/`
and are called from Training.

## Dataset

Dataset workflow lives in `defect_detection/app/tabs/dataset_tab.py`.

Inputs:

- NG source folder: board images expected to contain defects.
- JSON log: source records used to match image paths and proposed classes.
- Good source folder: known-good images.
- Model checkpoint: used for current-page prediction and review triage.
- Multiplier file: optional score-adjustment input for prediction.
- Destination Root: output dataset folder.

Review behavior:

- Rows can show `Needs review: <reason>`, `Needs review: class disagreement`,
  `Model agrees`, or `Awaiting prediction`.
- Current-page prediction updates predicted class and confidence for visible review rows.
- Review filters include `All`, `Needs review only`, and
  `Priority (Disagreements + Needs review)`.
- Rows that still need review require manual class override before dataset build.

Dataset build output:

- Reviewed images are copied under `Destination Root/<class_name>/`.
- Copied filenames use traceable names built from program, lot, side, and original filename.
- Current Dataset tab shows scan/build summaries in UI and console logs.
- No persisted Dataset manifest or summary file is currently written by this tab.

## Training

Training workflow lives in `defect_detection/app/tabs/training_tab.py`. Backend coordination
is in `defect_detection/app/backend_bridge.py`.

Main configuration:

- Dataset Root
- Fixed Validation Set
- Pretrained Model
- Phase 2 Dataset Root
- Output Folder
- Grad-CAM Output
- Model Name
- Version
- Image width and height
- Device
- Batch size
- Split percentage
- Phase 1 and Phase 2 optimizer, learning rate, epoch, loss, scheduler, augmentation,
  and regularization fields

Runtime output path:

- Run artifacts are written under `Output Folder/<model_name>_<version>/`.
- Checkpoints go under `<run>/checkpoints/`.
- Evaluation predictions go to `<run>/evaluation/predictions.jsonl`.
- Recommended thresholds go to `<run>/threshold/optimization/recommended_thresholds.json`.
- Saved active thresholds go to `<run>/threshold/active_thresholds.json`.
- Recommended multipliers go to `<run>/multiplier/optimization/recommended_multipliers.json`.
- Saved active multipliers go to `<run>/multiplier/active_multipliers.json`.
- Deployed flat multipliers go to `<run>/multiplier/defect_multipliers.json`.
- Promotion packages go to `Output Folder/promoted/<model_name>/<version>/`.

Best checkpoint selection:

- Training resolves best checkpoint from backend result fields in this order:
  `best_model_path`, `best_checkpoint_path`, then `final_model_path`.
- Successful training registers or updates checkpoint metadata through
  `defect_detection/app/checkpoint_registry.py`.
- Failed training does not trigger evaluation.

Automatic evaluation:

- Training calls the extracted evaluation service through `run_test_job`.
- Evaluation uses the selected checkpoint and the fixed validation dataset when available.
- Evaluation runs in a background worker. Tkinter UI updates are marshalled to the main thread.
- Grad-CAM is skipped by default for automatic evaluation.
- Training displays these determinate stages:
  `Evaluation 1/4 - Running validation inference`,
  `Evaluation 2/4 - Calculating metrics`,
  `Evaluation 3/4 - Building confusion matrix`,
  `Evaluation 4/4 - Generating report`.
- Batch progress during validation inference uses real batch counts.

Evaluation result shown in Training includes accuracy, macro precision, macro recall,
macro F1, misclassified image count, evaluation status, report path, confusion matrix path,
and prediction cache path.

## Cached Predictions

Prediction cache code is in `defect_detection/backend/prediction_cache.py`.

Default cache path:

```text
<run>/evaluation/predictions.jsonl
```

JSONL schema:

- First line: metadata row with `type: "metadata"`.
- Remaining lines: one `type: "record"` row per validation sample.

Metadata fields:

- `schema_version`
- `checkpoint_identity`
- `dataset_identity`
- `class_names`
- `model_name`
- `image_size`
- `preprocessing`

Record fields:

- `sample_id`
- `source_path`
- `true_index`
- `true_class`
- `predicted_index`
- `predicted_class`
- `probabilities`
- `confidence`
- `error`

Validity rules:

- Schema version must match `SCHEMA_VERSION`.
- Class order must match current class list exactly.
- Checkpoint identity signature must match current checkpoint path, size, and mtime.
- Dataset identity signature must match current validation image set.
- Probability vector length must match class count.
- Empty, partial, corrupt, or stale caches fail with explicit errors.
- Image bitmaps are not stored in the cache.
- Cache writes use a `.tmp` file followed by atomic replace.

Metrics, confusion matrix generation, misclassified-image reporting, threshold optimization,
multiplier optimization, and report generation consume this cache so inference runs once per
evaluation session.

## Thresholds

Threshold service code is in `defect_detection/backend/threshold_service.py`.
Persistence code is in `defect_detection/backend/threshold_persistence.py`.

Default optimization strategy:

```text
best_f1
```

Behavior:

- Threshold optimization consumes `PredictionResultCache`; it does not run inference.
- One recommendation is generated per defect class. `Pass` is excluded by default.
- Scores must be finite probabilities in `[0, 1]`.
- Class ordering must match prediction cache metadata.
- Classes with no positives, no negatives, or absent validation samples return warnings and
  no arbitrary threshold.

Saved threshold file:

```text
<run>/threshold/active_thresholds.json
```

Schema:

```json
{
  "schema_version": 1,
  "strategy": "best_f1",
  "checkpoint": "path/to/best.pt",
  "generated_at": "ISO-8601 timestamp",
  "classes": {
    "Scratch": {
      "recommended": 0.72,
      "active": 0.8,
      "mode": "manual",
      "status": "valid",
      "warning": null
    }
  }
}
```

Training UI behavior:

- Recommended and active thresholds are separate values.
- If no saved config exists, active values start equal to recommendations.
- Editing active threshold changes that class to `manual`.
- Restore-one and restore-all reset active values to recommended values and mode `auto`.
- Unsaved threshold edits are visible and block promotion.
- Save Thresholds writes the file atomically.
- Recalculation preserves saved manual overrides when valid.

Thresholds versus defect multipliers:

- Thresholds are per-class active cutoffs used for decisions and promotion readiness.
- Defect multipliers are deployment-time score adjustments generated from the same cached
  validation predictions.
- Training shows multipliers in `Phase 4B - Multipliers`. Recommended and active values are
  separate, manual edits are supported, and Save Active writes both the review config and the
  flat deployed `defect_multipliers.json`.
- Multipliers do not replace thresholds. For trained models where automatic multiplier
  recommendations completed, promotion requires saved multipliers and no unsaved multiplier
  edits. For external/custom fallback models, missing multipliers remain warning-only.

## Gate Rules

Promotion gate code is in `defect_detection/app/promotion_gate.py`.
Checkpoint registration gate code is in `defect_detection/app/checkpoint_registry.py`.

Reusable gate output includes `passed` plus a `rules` list. Each rule has a name, pass/fail
state, and supporting fields such as actual value, required value, path, status, or reason.

Promotion-blocking rules:

- `checkpoint_valid`
- `checkpoint_registered`
- `model_identity`
- `registration_gate`
- `minimum_accuracy`
- `maximum_escape_rate`, when configured
- `evaluation_complete`
- Required metrics: `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`,
  `misclassified_count`
- Required evaluation files: prediction cache, confusion matrix, and report
- Per-class metric requirements, when configured
- `thresholds_saved`
- `threshold_file_valid`
- `thresholds_no_unsaved_changes`
- `multiplier_recommendations`, when automatic multipliers are required
- `multipliers_saved`, when automatic multipliers are required
- `multiplier_file_valid`, when automatic multipliers are required
- `multipliers_no_unsaved_changes`, when automatic multipliers are required

Warning-only rule:

- `optional_multipliers`, when multipliers are not required and no deployed multiplier file is recorded

Accuracy-floor versus ratcheting-baseline behavior:

- Accuracy uses fixed floor from checkpoint requirements, defaulting to the registry floor.
- Escape rate is compared against the production baseline for the same model name and may not regress.
- The first registered checkpoint for a model name seeds that model's baseline; later promotion or rollback updates only that model's baseline.
- This preserves stable accuracy acceptance while keeping escape-rate regression guarded.

The gate recalculates after evaluation completion, threshold generation/save/edit,
multiplier generation/save/edit, checkpoint change, and restart recovery.

## Promotion

Promotion package code is in `defect_detection/app/promotion_package.py`.

Package path:

```text
<output_root>/promoted/<model_name>/<version>/
```

Required files:

- `model.pt`
- `labels.json`
- `thresholds.json`
- `metrics.json`
- `training_config.json`
- `gate_result.json`
- `model_metadata.json`
- `manifest.json`

Optional file:

- `defect_multipliers.json`

Promotion behavior:

- Promotion uses persisted recovered state, not only current-session fields.
- Existing promoted package version is not overwritten.
- Files are copied into a temporary directory, then finalized atomically.
- Missing required files fail with exact file names.
- `manifest.json` stores `sha256` and file size for packaged files.
- `model_metadata.json` stores model/version, checkpoint id, source checkpoint path,
  source checkpoint identity, architecture, image size, evaluation artifact paths, and
  threshold and multiplier mode counts.
- Packaged thresholds are active thresholds, not only recommendations.
- Packaged multipliers are the saved deployed flat multipliers when available.

## Restart Recovery

Recovery code is in `defect_detection/app/training_state_recovery.py`.

On project, model, or version selection, Training inspects registry records and persisted
artifacts. It does not depend only on in-memory fields from current session.

Precedence:

- Candidate checkpoint records are matched by model name/version metadata or by checkpoint
  path under the selected artifact directory.
- Most recent valid candidate wins when multiple records exist.
- Valid checkpoint without evaluation remains evaluatable after restart.
- Complete evaluation with saved thresholds can proceed to promotion after restart.
- Persisted predictions are reused when cache identity matches checkpoint, dataset, and class
  order.

Invalid-state handling:

- Missing or corrupt evaluation artifacts prevent evaluation from being marked complete.
- Corrupt threshold file prevents thresholds from being marked saved.
- Stale prediction cache is shown as stale and not reused.
- Partial runs are shown as partial, not promoted as ready.
- Promotion readiness is recalculated from recovered artifacts.
- Recovery log summarizes found checkpoint, evaluation, thresholds, and gate state.

## Live

Live UI lives in `defect_detection/app/tabs/live_camera_tab.py`.
Package discovery is in `defect_detection/app/live_model_package.py`.
Runtime worker logic is in `defect_detection/app/live_runtime.py`.

Preferred input:

- Select one promoted package folder.

Package discovery resolves:

- `model.pt`
- `labels.json`
- `thresholds.json`
- Optional `defect_multipliers.json`
- `model_metadata.json`
- `manifest.json`
- Model name
- Version
- Architecture
- Image size
- Class order
- Active thresholds

Live validates hashes, required files, JSON shape, threshold classes, metadata class order,
and optional multiplier class names before inference starts. Runtime uses active thresholds,
not recommended thresholds.

Custom model fallback:

- Standalone checkpoint loading is retained for developer use.
- Safe defaults use class names
  `Mousebite`, `Open`, `Pass`, `Pinhole`, `Protrusion`, `Short`, `Via`.
- Active thresholds default to `{}`.
- Defect multipliers default to neutral `0.0` for non-`Pass` classes.
- Architecture defaults to `custom`.
- Image size defaults to `(384, 384)`.

Runtime behavior:

- Camera capture, folder polling, preprocessing, and inference run off the Tkinter UI thread.
- UI updates are sent back through the Tkinter main thread.
- Work queue is bounded.
- When inference falls behind, oldest pending work is dropped.
- Folder polling has a reasonable minimum interval and skips duplicate files.
- Stop releases camera handles and worker resources.
- Only current preview and bounded recent history are retained.
- Prediction logs stay bounded by the current app runtime configuration.
- Real inference exceptions are logged and shown; generic `None` errors are avoided.

Output folders:

```text
<output_folder>/<relative_input_path>/Good/Pass/<image>
<output_folder>/<relative_input_path>/NG/<predicted_class>/<image>
<output_folder>/<relative_input_path>/inspection_results.csv
```

## Final Folder Structures

Training run:

```text
<output_root>/<model_name>_<version>/
├── checkpoints/
├── evaluation/
│   └── predictions.jsonl
├── test/
│   ├── plots/confusion_matrix.png
│   └── summaries/test_summary.txt
├── threshold/
│   ├── optimization/recommended_thresholds.json
│   └── active_thresholds.json
└── multiplier/
    ├── optimization/recommended_multipliers.json
    ├── active_multipliers.json
    └── defect_multipliers.json
```

Promoted package:

```text
<output_root>/promoted/<model_name>/<version>/
├── model.pt
├── labels.json
├── thresholds.json
├── metrics.json
├── training_config.json
├── gate_result.json
├── model_metadata.json
├── manifest.json
└── defect_multipliers.json
```

`defect_multipliers.json` is optional.

## Troubleshooting

Stale prediction cache:

- Cause: checkpoint, validation dataset, or class order changed.
- Fix: rerun evaluation from Training.

Corrupt `predictions.jsonl`:

- Cause: partial write, manual edit, or interrupted artifact copy.
- Fix: remove bad cache and rerun evaluation.

Invalid `active_thresholds.json`:

- Cause: bad JSON, unsupported schema, non-numeric active value, or value outside `[0, 1]`.
- Fix: regenerate recommendations, review manual overrides, then Save Thresholds.

Promotion blocked:

- Check every failed rule shown by Training.
- Common blockers: missing saved thresholds, unsaved threshold edits, stale evaluation,
  missing report, missing confusion matrix, or failed minimum accuracy.

Invalid promoted package:

- Check `manifest.json` hash errors and missing required files.
- Rebuild promotion package from Training; existing versions are preserved.

Live custom model:

- Use promoted package whenever possible.
- For standalone checkpoints, provide compatible class ordering and inference metadata through
  developer-supported paths. Thresholds and multipliers use safe defaults unless supplied by a
  promoted package.

## Manual Tests

These remain manual because they depend on hardware or external systems:

- Physical camera open, run, stop, and release.
- Long folder-watch soak test with real production image volume.
- GPU driver/device verification.
- Real promoted PyTorch checkpoint inference on deployment machine.
- External GUI JSON-string integration through `model_integration.predict`.

## Known Limitations

- Dataset tab currently does not persist a manifest or summary file for built datasets.
- Standalone custom model loading uses safe defaults; promoted package is the validated path.
- `test_json_string.py` is not present. The remaining external JSON smoke path is
  `defect_detection/run_json_image_inference.py`, which calls `model_integration.predict`
  and still requires manual integration coverage.
- Camera and hardware tests are not part of normal automated test suite.
- Existing promoted package versions are preserved; overwrite is intentionally refused.
