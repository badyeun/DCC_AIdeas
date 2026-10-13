# Mission 3 ver21

## Frozen ver19 features and a new nausea decision head

ver21 compares the unchanged ver19 baseline with `head_ce` and `head_ce_soft_f1`. Both new candidates start from the same original nausea two-class decision weights. All encoder, projection, utterance-evidence features, and the other eight symptom heads remain frozen. Cached features are reused across candidates, so repeated head optimization does not repeatedly run the encoder. The original text, targets, samples, and overflow windows remain; model input is only `utterances[].text`. Predictions remain raw two-class argmax. There is no threshold search, post-hoc score calibration, external training data, or label correction.

`head_ce` uses weighted cross entropy with the original nausea positive weight 3.0. `head_ce_soft_f1` adds 0.5 times soft-F1 loss. Defaults are seed 42, learning rate 0.001, weight decay 0, at most 80 full-fit feature updates, and evaluation every 5 updates. Soft-F1 is computed over the entire fit feature matrix in a real batch, not an average of minibatch F1 values or an accumulation substitute. New F1 and real Colab timings have not been measured locally. A large improvement is not guaranteed.

## Leakage controls and the unchanged baseline

The existing grouped Training split is verified before use. Candidate fitting and selection use features from the ver19 selection checkpoint trained on 23,360 fit samples, never the final model trained on all 29,200 Training samples. The 5,840 holdout samples are used only for candidate and epoch selection. `source_reference.json` and source-bundle checks bind the source weights, data, labels, split, and saved reference to the intended ver19 run. A missing, altered, or incompatible source is an error rather than an unchecked replacement.

Fit and holdout features are extracted in separate passes using the original batching, then restored to the original sample order. The original holdout decisions must reproduce before candidate fitting starts. Final export verifies actual model forward decisions against the portable reloaded model and checks that all other eight symptoms remain unchanged. Any nausea decision differences between CPU cached-head calculations and the final inference device are recorded as diagnostics; the actual reloaded model produces the final CSV.

Only a candidate that increases both holdout Macro F1 and nausea F1 can replace the baseline. Other eight symptom predictions must remain unchanged. This repeatedly used holdout is exploratory, not an untouched test set. Official Validation is evaluated once after selection and is never used for optimization or selection.

If a candidate improves and time permits, features are extracted using the separate final ver19 model trained on all 29,200 Training samples. The selected objective and number of updates train a fresh nausea head initialized from that final model's original head. The fit-model head is not transplanted onto final-model features. Existing encoder and other eight symptom outputs remain unchanged. If no candidate improves, or a complete new head cannot be produced within the budget, publication preserves the original full-Training ver19 model and clearly reports that no improvement was delivered. Original ver19 files are never modified.

Baseline publication still requires source verification, completed internal comparison, and final inference/publication to finish within the deadline. A timeout in those mandatory stages produces an incomplete run rather than a completed fallback result.

## Required files and execution

Open `mission3_ver21.ipynb` in Colab, select a GPU, and run all cells. Connect the shared Drive account containing both the existing `mission3_ver5_inputs.zip` and the complete existing ver19 result folder or its ZIP. The ver19 selection and full-Training weights, assets, and provenance files are required; the input ZIP alone is insufficient. Leave `source_run` blank to discover the existing ver19 result on Drive. If discovery fails, enter the complete Drive path to that folder or ZIP in `source_run`. No new pretrained model download is required. Runtime sources and reference metadata are embedded in the notebook. Dependencies match ver19 without additions or upgrades.

Run cells from the beginning after reconnecting or restarting the runtime. Running the last cell alone previously produced `NameError: runtime_dir` followed by a cleanup `NameError: release_gpu`. Preparation is now checked before execution. Missing or failed setup shows recovery instructions without launching training or masking the first error. Code restoration and successful installation must belong to the current runtime folder and deadline; a failed retry invalidates earlier success. If an earlier cell fails, fix its first error before rerunning all cells. The execution cell never creates a new time budget on its own.

Worker failures are recorded even when Colab does not display inherited subprocess output. Before requesting GPU release, the notebook shows the failed stage, exception type, safe diagnostic reason and code locations, and saves `ver21/_diagnostics/ver21_error_*.json` on Drive. The launcher also records import and settings-load failures. Reports exclude raw exception strings, dialogue content, per-record data paths, and local variables. An unclassified exception is reported as unclassified; the outer `CalledProcessError` alone is not treated as a diagnosis. If Drive cannot accept the report, its safe details remain in the cell output. Send that report for diagnosis instead of repeating training without knowing the cause.

By default results are saved in a new unique folder below `DCC_AIdeas/results/mission3/ver21`. The 180-minute shared deadline starts after Drive login and before dependency installation. Package installation has a 20-minute cap. A 25-minute reserve protects final inference and publication; timing estimates use a 1.25 safety factor. The watchdog bounds the worker process and requests Colab GPU release. These controls limit work; they do not guarantee completion of every stage on every GPU or Drive connection.

`AUTO_RELEASE_GPU=True` requests runtime release after results are displayed, on errors, or at the shared deadline. Results saved to Drive remain; `/content` scratch files may disappear. A failed release request prints manual Runtime > Disconnect and delete runtime instructions. Actual Colab release behavior requires a Colab run to verify.

## Results, fallback, and resuming

- `candidate_summary.csv`: baseline and completed head candidates on the Training holdout.
- `training_reference_comparison.json`: paired nausea error transitions and F1 changes.
- `final_selection.json`: selected objective, update count, and fallback reason when applicable.
- `training_metrics.json`: Training holdout metrics for the selected candidate.
- `validation_metrics.json` and `mission3.csv`: final model evaluation and predictions.
- `mission3_ver21_model.ckpt`: relative-path model manifest; preserve the complete run folder.
- `runtime.json`: operational status. `complete` indicates an improved candidate completed full-Training head fitting. `complete_not_improved` preserves the existing full ver19 model because neither new candidate passed selection. `complete_budget_limited` preserves the existing full ver19 model because candidate search or full-Training head fitting could not complete in budget, including a search stopped before its first evaluated candidate. Neither fallback is a performance improvement.
- `head_artifacts.json`: lists the active head sidecars belonging to the current export. Sidecars retained from an interrupted earlier attempt are not active when the current export falls back to the baseline. The model manifest and active-path record identify what was actually published.
- `run_complete.json`: final artifact integrity marker. A missing marker or final CSV means the run is not a completed submission result, even if some intermediate files exist.

To resume, set `resume_run` to the previous complete Drive run path. Compatible completed results are verified and returned without more training. An interrupted run may restart feature extraction and head fitting, not continue at its last update. Each explicit rerun starts a new budget and uses additional Colab resources. Keep `resume_run` empty for a fresh experiment. No raw patient text is exported in comparison summaries or displayed in notebook output.

Portable inference is available through `submission/inference.py`:

```text
python inference.py --checkpoint /path/to/run/mission3_ver21_model.ckpt --label_dir /path/to/labels --output /path/to/mission3.csv
```

The CSV header is `label file name,symptom`. Preserve all relative model assets when moving a run.

## Local verification

From the repository root:

```text
.venv\Scripts\python.exe -B src/mission3/mission3_ver21/build_notebook.py
.venv\Scripts\python.exe -B -m unittest discover -s src/mission3/mission3_ver21 -t src/mission3 -p "test_*.py"
```

Focused tests check the head objective, frozen outputs, source identity, notebook restoration, deadline sharing, GPU release requests, and inference/publication contracts. Synthetic checks establish execution behavior only; they do not establish actual F1 gains or a measured Colab duration.
