# UAV Localization Error Prediction (TartanAir)

Predicts frame-to-frame visual-localization error for a UAV from image,
depth, segmentation, and pose data, then explains each prediction with
SHAP and an optional local-LLM-generated report.

## Setup

```bash
pip install -r requirements.txt
```

Set the dataset location (previously hardcoded to a Windows path in every
script):

```bash
export UAV_DATASET_ROOT="/path/to/TartanAir/ArchVizTinyHouseDay/Data_easy"
```

The folder must contain one subfolder per sequence (`P000`, `P001`, ...),
each with `image_lcam_front/`, `depth_lcam_front/`, `seg_lcam_front/`, and
`pose_lcam_front.txt`. Sequence list and other constants live in
`config.py`.

## Pipeline (run in this order)

1. **`generate_all_features.py`** — extracts visual/semantic/depth/motion
   features for every frame of every sequence → `all_features.csv`
2. **`generate_all_labels.py`** — runs ORB matching + pose comparison to
   compute ground-truth localization error per frame → `all_localization_labels.csv`
3. **`merge_all_dataset.py`** — joins the two on `(sequence, frame)` →
   `training_dataset_all.csv`
4. **`train_model.py`** — trains the RandomForestRegressor, saves
   `rf_localization.pkl` / `feature_names.pkl`, and saves the held-out
   test row indices to `test_split_indices.pkl`
5. **`evaluation_model.py`** — re-scores the saved model on those same
   held-out rows (do this after step 4, before re-running `loso_evaluation.py`,
   see caveat below)
6. **`loso_evaluation.py`** — leave-one-sequence-out evaluation: the most
   trustworthy generalization estimate, since it never lets frames from
   the same trajectory leak between train and test. Retrains a final
   model on all data afterward and overwrites `rf_localization.pkl`.
6b. **`sequence_holdout_evaluation.py`** (optional) — trains on every
   sequence except one and tests only on that one (default: train on
   P001–P006, test on P000; change via `config.HOLDOUT_TEST_SEQUENCE` or
   `UAV_HOLDOUT_TEST_SEQUENCE`). Saves its own model
   (`rf_localization_holdout_<seq>.pkl`) separately so it never clobbers
   the model `graphs.py`/`test_system.py` use.
6c. **`time_split_evaluation.py`** (optional) — within each sequence,
   trains on the first ~80% of frames and tests on the last ~20%
   (`config.TIME_SPLIT_TEST_FRACTION` / `UAV_TIME_SPLIT_TEST_FRACTION`).
   Answers "given the start of this route, how well can the model predict
   error on the rest of it?" Also saves its own model file.
7. **`graphs.py`** — generates the full set of publication figures
   (distributions, correlations, SHAP plots, etc.) into `graphs/`
8. **`test_system.py`** — for one demo frame (`config.DEMO_SEQUENCE`/
   `DEMO_FRAME`), predicts localization error, computes SHAP, and passes
   the raw top-N SHAP features straight to a local Ollama LLM
   (`llm_engine.py`), which reasons over them directly to produce the
   report — feature names are made human-readable via `class_names.py`,
   but the LLM does its own interpretation rather than rephrasing a fixed
   per-feature template (the previous `reasoning_engine.py`, removed —
   see below)

## Known caveats (read before trusting the numbers)

- **Four ways this repo splits train/test data — they answer different
  questions, and their scores aren't directly comparable:**

  | Script | Split | What it tells you | R² (shipped CSVs) |
  |---|---|---|---|
  | `train_model.py` | Random 80/20 rows, all sequences mixed | Least trustworthy — near-duplicate adjacent frames can appear in both train and test | 0.918 |
  | `time_split_evaluation.py` | First 80% / last 20% of frames, per sequence | "Given the start of this route, how well can it predict the rest?" (same environment in both halves) | 0.765 |
  | `loso_evaluation.py` | Each sequence held out in turn, averaged | Most trustworthy overall generalization estimate | 0.79 (range 0.53–0.90) |
  | `sequence_holdout_evaluation.py` | Train on 6 sequences, test on 1 named sequence | Same idea as LOSO but for one specific sequence you care about (default: test on P000) | 0.816 (= the P000 fold of LOSO) |

  Notably, the "same environment" temporal split (0.765) does **not**
  score higher than the "brand-new environment" LOSO average (0.79) —
  which suggests the model isn't really relying on memorizing each room,
  more on frame-level motion/texture cues that vary regardless of
  environment.

- **`loso_evaluation.py`'s R² (~0.79, ranging 0.53–0.90 by sequence) is the
  honest generalization estimate.** `train_model.py`'s random-split R²
  (~0.92) is optimistic because nearby frames from the same flight can end
  up in both train and test.
- **`orb_features` dominates feature importance (~73%).** The label
  generator (`generate_all_labels.py`) derives `estimated_distance`
  directly from ORB match count (`matches / 2000`, clipped), so localization
  error is partly a function of feature-matching volume by construction.
  Semantic/depth features contribute comparatively little — worth
  revisiting if you want the model to lean more on scene understanding.
- **Segmentation class names (`class_names.py`) are unverified
  placeholders**, not the official TartanAir label legend for this scene.
  Treat any "Wall"/"Floor"/etc. mentioned in generated reports as a
  best guess pending validation against the real class legend.
- **Re-running `loso_evaluation.py` after `evaluation_model.py`** retrains
  on the *entire* dataset and overwrites the saved model, so
  `test_split_indices.pkl` no longer describes truly held-out rows for
  that model. Re-run `train_model.py` again if you need a fresh,
  consistent held-out split.
- **`llm_engine.py` / the report step in `test_system.py` requires a local
  Ollama server** with the model pulled (`ollama pull llama3.2`, or set
  `UAV_LLM_MODEL` to another model you've pulled). Without it, `test_system.py`
  will still print the prediction and SHAP-based facts, but the final
  report step will raise a clear `RuntimeError` instead of a raw
  connection traceback.

## Files removed in cleanup

The following were removed as duplicates or dead ends superseded by the
multi-sequence pipeline above — see git history if you need them back:

- `main.py`, `localization.py` — single-sequence exploratory versions of
  `generate_all_features.py` / `generate_all_labels.py`
- `merge_dataset.py` — single-sequence version of `merge_all_dataset.py`
- `pose_analysis.py` — standalone demo print-out of `pose_analyser.py`
- `uav_xai_system.py` — an inferior duplicate of `test_system.py` with its
  own inline (and inconsistent) class-name mapping instead of using
  `class_names.py`
- `semantic_labels.py` — a third, conflicting copy of the class-name
  mapping, never actually imported anywhere
- `xai.py` — crashed (`free(): invalid size`) because it evaluated SHAP
  using a stale `training_dataset.csv` feature set against a model that
  had since been retrained on `training_dataset_all.csv`'s different
  feature set. `graphs.py` already does this correctly (matching feature
  set + matching train/test split), so it's redundant as well as broken.
- `reasoning_engine.py` — converted SHAP output into a fixed per-feature
  template (canned "Domain Meaning" text) before handing it to the LLM.
  In testing this produced reports that directly contradicted the SHAP
  direction for a given frame (e.g. asserting "generally supports robust
  localization" for a feature whose SHAP contribution that frame was
  clearly negative). Replaced with a direct-SHAP-to-LLM path in
  `llm_engine.py` (`generate_report_direct`) that lets the model reason
  over the real numbers for that specific frame instead of reciting a
  static lookup table. The rank, direction, magnitude bucket, feature
  description, risk level, and opening summary sentence are still
  computed in Python and handed to the LLM as fixed facts — not because
  they're "canned text" the way `reasoning_engine.py`'s was, but because
  testing showed the local model (llama3.2) unreliable at deriving them
  itself (e.g. calling a 1.4%-of-image feature "large", or naming a
  small-magnitude feature as the primary driver). The LLM is left to
  freely reason only over the "why" and "recommendations", where being
  wrong is lower-stakes and hedging language is enforced by the prompt.
  A `sanitize_recommendations()` post-filter also strips any
  recommendation referencing equipment/algorithms not present in the
  given features, as a safety net for prompt rules the model didn't
  always follow.
- `features.csv`, `localization_labels.csv`, `training_dataset.csv` —
  generated outputs of the removed single-sequence scripts

## Architecture

```
dataset_loader.py ─┐
feature_extractor.py ─┤
semantic_extractor.py ─┼─► generate_all_features.py ─┐
depth_features.py ─┤                                  ├─► merge_all_dataset.py ─► train_model.py ─► evaluation_model.py
pose_analyser.py ─┘                                    │                                          └─► loso_evaluation.py
                    generate_all_labels.py ─────────────┘
                                                                                    │
                                                                                    ▼
                                                              graphs.py ◄── rf_localization.pkl
                                                              test_system.py ◄── class_names.py
                                                                            └──► llm_engine.py
```