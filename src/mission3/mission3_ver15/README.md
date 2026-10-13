# Mission 3 ver15

Standalone Colab experiment for broader symptom recall within a shared 180-minute work budget. Open `../mission3_ver15.ipynb`, select a GPU runtime, and run all cells using the existing `mission3_ver5_inputs.zip`. The notebook restores its own source files and uses the project's unchanged package versions.

## Experiment

Two fresh `klue/roberta-base` candidates use one forward view, no label attention, no supervised contrastive loss, the same Training split and seed, and positive-class weighted cross-entropy. The fixed label order and candidate weights are:

| Label | recall_moderate | recall_broader |
| --- | ---: | ---: |
| 고열 | 1.4 | 1.75 |
| 구토 | 1.8 | 2.25 |
| 두통 | 2.0 | 2.5 |
| 복통 | 1.2 | 1.25 |
| 어지러움 | 1.3 | 1.5 |
| 열상 | 1.0 | 1.0 |
| 오심 | 3.0 | 3.0 |
| 전신쇠약 | 1.5 | 1.75 |
| 호흡곤란 | 1.3 | 1.5 |

Training holdout Macro F1 selects the candidate and epoch. Subject to the remaining budget, the chosen configuration is freshly refitted on all Training examples. Official Validation is used only to evaluate the final selected model. Inputs remain `utterances[].text`; predictions remain categorical argmax. There is no post-hoc threshold calibration, score adjustment, annotation modification, external dataset, or reuse of previous experiment checkpoints. Higher weights may lower precision; actual overall and per-symptom F1 must be measured.

## Time accounting

The monotonic deadline starts after interactive Drive mounting and before source restoration/package installation. The same deadline is passed to the supervised installer, worker, and runtime. Package installation has an additional 20-minute cap. Evaluation and export reserve 1,200 seconds, with a 1.25 safety factor used for projected training time. The worker supervisor terminates work at the deadline if normal cooperative stopping cannot finish.

The runtime prioritizes completing up to three candidate fit epochs when each next epoch fits the training budget, even if this means skipping the full-Training refit on a slower GPU. It may also skip the second candidate. A completed fit-only checkpoint may be exported, with its actual `training_scope` recorded. Cooperative budget exhaustion reports `budget_exhausted`. An abrupt watchdog kill or Colab runtime reclaim may leave only the previous saved status, so missing final files are never a successful submission. A fixed time budget cannot guarantee full retraining or improved F1 on every Colab GPU. Prior L4 timings suggest approximately 140 minutes for the intended two-candidate path, but downloads, Drive I/O, GPU speed, and early stopping can change the completed work.

`AUTO_RELEASE_GPU=True` is the notebook default. After result display, on an installer/worker error, or at the shared deadline, the notebook requests Colab runtime release through [`google.colab.runtime.unassign()`](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py). Saved Drive results survive; `/content` scratch files may disappear. A daemon timer also requests release if the worker hangs. The timer and finalizer share a lock so they request release at most once. This optional setting is separate from the experiment settings and may be disabled for debugging; the worker time cap still applies. When disabled or if the release API fails, use **Runtime → Disconnect and delete runtime** manually. The real Colab release service cannot be verified by local tests.

## Outputs and resuming

Outputs are stored in a unique run directory under `results/mission3/ver15`:

- `candidate_summary.csv`: completed candidates and Training holdout scores.
- `final_selection.json`: selected configuration and Training selection evidence.
- `training_metrics.json` and `validation_metrics.json`: measured metrics.
- `runtime.json`: completion or interruption status and timing.
- `run_complete.json`: final completion marker and saved artifact hashes; absence means no completed run.
- `mission3.csv`: final predictions when inference completed.
- `mission3_ver15_model.ckpt`, referenced weights/assets, and `submission/`: portable local inference.

The manifest records `training_scope` (`all_training` or `fit_only`), `fitted_samples`, `selected_epochs`, `selected_candidate`, and any `refit_skipped_reason`. `runtime.json` distinguishes `complete`, `complete_budget_limited`, `budget_exhausted`, `interrupted`, and `failed`. The first two statuses represent completed final work; budget-limited completion can use fewer candidates or fit-only training. The remaining statuses require inspecting the saved artifacts before attempting submission.

Keep the entire run directory; the manifest alone is not a model. Preserve the executed notebook as well. Source, settings, and data identity must match before completed artifacts can be reused. Set `resume_run` to the original full run directory to request reuse. Incomplete epoch state is not resumed. Each explicit rerun starts a new time budget and consumes additional Colab resources.

Resuming a run already marked `complete` or `complete_budget_limited` only verifies and returns the saved results; it does not extend that finished experiment. To start another experiment, leave `resume_run` empty.

Previous version notebooks and results are not modified. This version makes no improvement claim until full data results are available.
