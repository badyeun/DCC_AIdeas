# Mission 3 ver18

## Experiment: global context with utterance evidence

- `baseline_ce`: the ver15 classifier with moderate positive-class weights.
- `utterance_evidence`: the same classifier plus a small nausea evidence branch over ordered utterances.

Both start from `klue/roberta-base` revision `02f94ba5e3fcb7e2a58a390b8639b0fac974a8da`. The evidence branch reuses the same encoder outputs rather than running another encoder pass. It pools token representations into original utterances, joins previous/current/next utterance representations in order, and learns which contextualized utterances provide evidence. An utterance split across windows is reunited, with inverse occurrence weights reducing duplicated overlap contributions. Every utterance and the original global context branch are retained.

The small evidence branch has hidden width 128 and dropout 0.1. Its zero-initialized output residual updates only the two nausea logits during training. The encoder is jointly trained, so other labels can change through shared representations. Special tokens and padding do not become evidence. Empty documents use the global prediction with a zero evidence residual. The manifest stores the architecture for strict reconstruction.

This tests whether localized symptom expressions and surrounding questions/answers help nausea classification. It does not assume that attention automatically understands negation, that missing keywords mean incorrect labels, or that existing labels should be replaced. It differs from ver14 token-level label attention by aggregating original utterances with ordered neighbors. CorNet, TAPT, GCE, keyword filtering, and post-training score adjustments are not added.

## Execution and safeguards

Open `mission3_ver18.ipynb` in Colab, select a GPU, and run all cells using the existing `mission3_ver5_inputs.zip`. Select the shared Drive account when prompted. All runtime sources are embedded; no earlier result is required. Results go to unique run directories under `DCC_AIdeas/results/mission3/ver18` in Drive.

Inputs remain `utterances[].text`. All original targets and samples remain unchanged. Utterance order and character offsets come from these text fields, not speaker or medical annotations. The nine two-class outputs use ordinary argmax, without thresholds, post-training calibration, external training corpora, or automatic relabeling. Both candidates use the same grouped Training split, seed, fixed KLUE revision, weighted cross entropy, and moderate positive weights `[1.4, 1.8, 2.0, 1.2, 1.3, 1.0, 3.0, 1.5, 1.3]`. Candidate training uses at most three epochs with the original four-epoch learning-rate schedule horizon. GPU execution is not guaranteed bitwise reproducible across runs.

Training holdout Macro F1 selects the candidate and epoch. Official Validation is evaluated only after final selection. A selected full refit starts fresh and uses all Training targets for the selected number of epochs. If the new branch is worse on the internal selection metric, the baseline can win; both candidates' internal results remain available.

The shared 180-minute deadline starts after Drive login, before installation. Installation also has a 20-minute cap. The runtime reserves 20 minutes for finalization and uses measured time with a 1.25 safety factor. Slow hardware may require skipping a candidate or full refit. Only complete epochs and fully complete refits are published; incomplete refits never replace completed candidates. A fit-only fallback records its actual `training_scope`, `fitted_samples`, and `refit_skipped_reason`. A deadline cannot guarantee all phases finish on every GPU.

`AUTO_RELEASE_GPU=True` requests Colab runtime release after result display, errors, or deadline. Drive results survive; `/content` scratch may disappear. If release fails, use Runtime > Disconnect and delete runtime manually. Actual Colab release service behavior and new-method GPU duration remain unverified locally.

## Saved results and resuming

Keep the entire result directory and executed notebook. Important files are `candidate_summary.csv`, `candidate_comparison.json`, `final_selection.json`, `training_metrics.json`, `validation_metrics.json`, `runtime.json`, `mission3.csv`, and `mission3_ver18_model.ckpt` with referenced weights/assets. The portable inference entry point is `submission/inference.py`.

`candidate_comparison.json` compares the two candidates on the same Training holdout: per-label precision/recall/F1 changes and paired counts of corrected, newly introduced, and persistent errors. It contains no dialogue text or sample identifiers. This diagnostic does not change the selection metric or tune on official Validation. If a candidate cannot complete within budget, the report records that comparison is unavailable.

Only `complete` or `complete_budget_limited` status plus `run_complete.json` represents completed final work. Missing final files or completion marker, `budget_exhausted`, `failed`, or `interrupted` mean incomplete work. Abrupt Colab reclamation can leave the previous saved status unchanged.

Set `resume_run` to a compatible run directory. Source, settings, data, split, and artifact hashes are checked before completed artifacts are reused. Partial epochs are not resumed. Resuming an already completed run only verifies and returns it without extending training. Each explicit rerun consumes a new budget and additional resources.

Previous versions and dependency requirements are unchanged. New full-data F1 improvement is unverified until the notebook is run.
