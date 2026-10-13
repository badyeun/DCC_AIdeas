# Mission 3 ver17

## Experiment: Training-only task-adaptive pretraining

- baseline_ce: the ver15 classifier with moderate positive-class weights.
- tapt_ce: the same classifier after masked-language adaptation on Training Fit text.

Both start from klue/roberta-base revision 02f94ba5e3fcb7e2a58a390b8639b0fac974a8da. [TAPT](https://aclanthology.org/2020.acl-main.740/) is a hypothesis about learning task vocabulary and indirect expressions. It cannot recover information absent from the transcript or settle annotation-policy disagreements.

The default recipe uses 800 optimizer updates, batch four, accumulation four, learning rate 5e-5, mask probability 0.15, and 256-token windows. The first ten updates are included in the 800 and serve as a throughput pilot. Fit adaptation is capped at 15 minutes and full-Training adaptation at 20 minutes, both inside the shared 180-minute budget. These are caps, not measured L4 durations.

Candidate adaptation receives Fit text only: neither Holdout nor Validation text enters language-model training. A phase that does not complete its planned updates is never silently promoted as completed TAPT. Fixed update counts sample windows and do not guarantee every dialogue is seen in the MLM phase.

If TAPT wins, final refitting newly initializes from the public checkpoint, repeats the same adaptation recipe on full Training text, then supervises on every Training sample for the selected epochs. It does not continue the selected classifier or reuse incomplete adaptation. If baseline wins, the final model has no TAPT phase. Randomness is reset consistently after encoder preparation for supervised learning.

Completed adaptation reports record scope, updates, token counts, loss, and elapsed time. Partial MLM phases are not resumed. A time-limited run can finish using the completed baseline or another completed fit-only candidate; its actual method and scope are recorded.

## Execution and safety

Open the matching .ipynb in Colab, select a GPU, and run all cells using the existing mission3_ver5_inputs.zip. All runtime sources are embedded; no earlier result is required. Results go to unique run directories under DCC_AIdeas/results/mission3/ver17 in Drive.

Inputs remain utterances[].text. All original targets and samples remain unchanged. The nine two-class outputs use ordinary argmax, without thresholds, post-training calibration, external training corpora, or automatic relabeling. Both candidates use the same grouped Training split, seed, fixed KLUE revision, and moderate positive weights [1.4, 1.8, 2.0, 1.2, 1.3, 1.0, 3.0, 1.5, 1.3]. Candidate training uses at most three epochs with the original four-epoch learning-rate schedule horizon.

Training holdout Macro F1 selects the candidate and epoch. Official Validation is evaluated only after final selection. A selected full refit starts fresh and uses all Training targets for the selected number of epochs.

The shared 180-minute deadline starts after Drive login, before installation. Installation also has a 20-minute cap. The runtime reserves 20 minutes for finalization and uses measured time with a 1.25 safety factor. Slow hardware may require skipping a candidate or full refit. Only complete epochs and fully complete refits are published; incomplete refits never replace completed candidates. A fit-only fallback records its actual training_scope, fitted_samples, and refit_skipped_reason. A deadline cannot guarantee all phases finish on every GPU.

AUTO_RELEASE_GPU=True requests [Colab runtime release](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py) after result display, errors, or deadline. Drive results survive; /content scratch may disappear. If release fails, use Runtime > Disconnect and delete runtime manually. Actual Colab release service behavior and new-method GPU duration remain unverified locally.

ver16 and ver17 have independent budgets. Running both can consume up to six hours total. To conserve resources, run ver16 first and inspect its result before ver17.

## Saved results and resuming

Keep the entire result directory and executed notebook. Important files are candidate_summary.csv, final_selection.json, training_metrics.json, validation_metrics.json, runtime.json, mission3.csv, and mission3_ver17_model.ckpt with referenced weights/assets. The portable inference entry point is submission/inference.py.

Only complete or complete_budget_limited status plus run_complete.json represents completed final work. Missing final files or completion marker, budget_exhausted, failed, or interrupted mean incomplete work. Abrupt Colab reclamation can leave the previous saved status unchanged.

Set resume_run to a compatible run directory. Source, settings, data, split, and artifact hashes are checked before completed artifacts are reused. Partial epochs are not resumed. Resuming an already completed run only verifies and returns it without extending training. Each explicit rerun consumes a new budget and additional resources.

Previous versions and dependency requirements are unchanged. New full-data F1 improvement is unverified until the notebook is run.

