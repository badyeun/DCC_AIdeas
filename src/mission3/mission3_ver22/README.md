# Mission 3 ver22

## Cached utterance features and retrained nausea evidence

ver21 changed the final linear nausea head without improving the Training holdout. ver22 instead compares the unchanged ver19 baseline with `evidence_ce` and `evidence_ce_gce`, retraining the existing nausea utterance-evidence branch. Its local context projection, attention, and evidence output can change. Both candidates start from the same original ver19 evidence weights. The expensive encoder, global dialogue path, and other eight symptom heads remain frozen. Cached utterance representations are reused across candidates, so repeated evidence training does not repeatedly run the encoder. The original text, targets, samples, and overflow windows remain; model input is only `utterances[].text`. Predictions remain raw two-class argmax. There is no threshold search, post-hoc score calibration, external training data, or label correction.

`evidence_ce` uses weighted cross entropy with the original nausea positive weight 3.0. `evidence_ce_gce` uses the same CE for the first epoch and then an equal mix of CE and generalized cross entropy with q=0.7. Defaults are seed 42, batch size 128 cached dialogues, learning rate 0.0001, weight decay 0.01, at most 8 epochs, evaluation every epoch, and early stopping after 3 epochs without improvement. The original evidence dropout 0.1 is retained. GCE is a hypothesis about robustness to difficult targets, not a declaration that those labels are incorrect. No manual evidence labels or explicit question/negation classes are added. New F1 and real Colab timings have not been measured locally. A large improvement is not guaranteed.

## Leakage controls and the unchanged baseline

The existing grouped Training split is verified before use. Candidate fitting and selection use features from the ver19 selection checkpoint trained on 23,360 fit samples, never the final model trained on all 29,200 Training samples. The 5,840 holdout samples are used only for candidate and epoch selection. `source_reference.json` and source-bundle checks bind the source weights, data, labels, split, and saved reference to the intended ver19 run. A missing, altered, or incompatible source is an error rather than an unchecked replacement.

Fit and holdout features are extracted in separate passes using the original batching, then restored to the original sample order. The original holdout decisions must reproduce before candidate fitting starts. Final export verifies actual model forward decisions against the portable reloaded model and checks that all other eight symptoms remain unchanged. Any nausea decision differences between cached-evidence calculations and final model forward are recorded as diagnostics; the actual reloaded model produces the final CSV.

Only a candidate that increases both holdout Macro F1 and nausea F1 can replace the baseline. Other eight symptom predictions must remain unchanged. This repeatedly used holdout is exploratory, not an untouched test set. Official Validation is evaluated once after selection and is never used for optimization or selection.

Before full-Training refitting, the chosen cached candidate is checked through the actual fit-only model on the original holdout order. `candidate_confirmation.json` records any cached-versus-actual decision differences. Actual predictions replace cached metrics for final selection; a candidate that no longer improves cannot be accepted. A timeout before this confirmation retains the baseline with a budget-limited status.

The GCE objective follows [Zhang and Sabuncu, NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html). Its published noisy-label experiments do not establish an expected F1 gain on this Korean dataset. Here both objectives use the original labels without manual edits.

If a candidate improves and time permits, features are extracted using the separate final ver19 model trained on all 29,200 Training samples. The selected objective and epoch count retrain a fresh evidence branch initialized from that final model's original evidence weights. The fit-model evidence branch is not transplanted onto final-model features. Existing encoder and other eight symptom outputs remain unchanged. If no candidate improves, or a complete new evidence branch cannot be produced within the budget, publication preserves the original full-Training ver19 model and clearly reports that no improvement was delivered. Original ver19 files are never modified.

Baseline publication still requires source verification, completed internal comparison, and final inference/publication to finish within the deadline. A timeout in those mandatory stages produces an incomplete run rather than a completed fallback result.

## Required files and execution

Open `mission3_ver22.ipynb` in Colab, select a GPU, and run all cells. Connect the shared Drive account containing both the existing `mission3_ver5_inputs.zip` and the complete existing ver19 result folder or its ZIP. The ver19 selection and full-Training weights, assets, and provenance files are required; the input ZIP alone is insufficient. Leave `source_run` blank to discover the existing ver19 result on Drive. If discovery fails, enter the complete Drive path to that folder or ZIP in `source_run`. No new pretrained model download is required. Runtime sources and reference metadata are embedded in the notebook. Dependencies match ver19 without additions or upgrades.

Run cells from the beginning after reconnecting or restarting the runtime. Running the last cell alone previously produced `NameError: runtime_dir` followed by a cleanup `NameError: release_gpu`. Preparation is now checked before execution. Missing or failed setup shows recovery instructions without launching training or masking the first error. Code restoration and successful installation must belong to the current runtime folder and deadline; a failed retry invalidates earlier success. If an earlier cell fails, fix its first error before rerunning all cells. The execution cell never creates a new time budget on its own.

Worker failures are recorded even when Colab does not display inherited subprocess output. Before requesting GPU release, the notebook shows the failed stage, exception type, safe diagnostic reason and code locations, and saves `ver22/_diagnostics/ver22_error_*.json` on Drive. The launcher also records import and settings-load failures. Reports exclude raw exception strings, dialogue content, per-record data paths, and local variables. An unclassified exception is reported as unclassified; the outer `CalledProcessError` alone is not treated as a diagnosis. If Drive cannot accept the report, its safe details remain in the cell output. Send that report for diagnosis instead of repeating training without knowing the cause.

By default results are saved in a new unique folder below `DCC_AIdeas/results/mission3/ver22`. The 180-minute shared deadline starts after Drive login and before dependency installation. Package installation has a 20-minute cap. A 25-minute reserve protects final inference and publication; timing estimates use a 1.25 safety factor. The watchdog bounds the worker process and requests Colab GPU release. These controls limit work; they do not guarantee completion of every stage on every GPU or Drive connection.

`AUTO_RELEASE_GPU=True` requests runtime release after results are displayed, on errors, or at the shared deadline. Results saved to Drive remain; `/content` scratch files may disappear. A failed release request prints manual Runtime > Disconnect and delete runtime instructions. Actual Colab release behavior requires a Colab run to verify.

Large per-utterance feature caches stay in the Colab runtime below `/content/mission3_ver22/feature_caches/<run_id>/`, avoiding an extra multi-gigabyte Drive upload. Drive retains cache summaries, final model assets, metrics, and predictions. A compatible cache can be reused within the same runtime after identity checks; after runtime release, resuming may need to extract features again. Losing the temporary cache does not invalidate a completed saved model.

## Results, fallback, and resuming

- `candidate_summary.csv`: baseline and completed evidence candidates on the Training holdout.
- `candidate_confirmation.json`: actual fit-only model confirmation of the selected cached candidate.
- `training_reference_comparison.json`: paired nausea error transitions and F1 changes.
- `final_selection.json`: selected objective, epoch count, and fallback reason when applicable.
- `training_metrics.json`: Training holdout metrics for the selected candidate.
- `validation_metrics.json` and `mission3.csv`: final model evaluation and predictions.
- `mission3_ver22_model.ckpt`: relative-path model manifest; preserve the complete run folder.
- `runtime.json`: operational status. `complete` indicates an improved candidate completed full-Training evidence fitting. `complete_not_improved` preserves the existing full ver19 model because neither new candidate passed selection. `complete_budget_limited` preserves the existing full ver19 model because candidate search or full-Training evidence fitting could not complete in budget, including a search stopped before its first evaluated candidate. Neither fallback is a performance improvement.
- `final_evidence.npz`, `final_history.json`: new evidence weights and their full-Training learning history, when an improved candidate is exported.
- `evidence_artifacts.json`: lists the active evidence sidecars belonging to the current export. Sidecars retained from an interrupted earlier attempt are not active when the current export falls back to the baseline. The model manifest and active-path record identify what was actually published.
- `run_complete.json`: final artifact integrity marker. A missing marker or final CSV means the run is not a completed submission result, even if some intermediate files exist.

To resume, set `resume_run` to the previous ver22 Drive run path. Compatible completed results are verified and returned without more training. An interrupted run may restart feature extraction and evidence fitting, not continue at its last update. Each explicit rerun starts a new budget and uses additional Colab resources. Keep `resume_run` empty for a fresh experiment. No raw patient text is exported in comparison summaries or displayed in notebook output.

Portable inference is available through `submission/inference.py`:

```text
python inference.py --checkpoint /path/to/run/mission3_ver22_model.ckpt --label_dir /path/to/labels --output /path/to/mission3.csv
```

The CSV header is `label file name,symptom`. Preserve all relative model assets when moving a run.

## Local verification

From the repository root:

```text
.venv\Scripts\python.exe -B src/mission3/mission3_ver22/build_notebook.py
.venv\Scripts\python.exe -B -m unittest discover -s src/mission3/mission3_ver22 -t src/mission3 -p "test_*.py"
```

Focused tests check the evidence objective, frozen outputs, source identity, notebook restoration, deadline sharing, GPU release requests, and inference/publication contracts. Synthetic checks establish execution behavior only; they do not establish actual F1 gains or a measured Colab duration.
