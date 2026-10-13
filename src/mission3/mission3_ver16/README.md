# Mission 3 ver16

## Experiment: CorNet-inspired symptom relationships

- baseline_ce: the ver15 classifier with moderate positive-class weights.
- cornet: the same classifier plus a small jointly trained label-correlation residual layer.

Both start from klue/roberta-base revision 02f94ba5e3fcb7e2a58a390b8639b0fac974a8da and use one encoder forward per view. The new layer reads nine text-derived label logit differences, with hidden width 32, depth one, dropout 0.1, and zero-initialized residual outputs. It never reads other gold symptoms as inputs. It is trained together with the model; it is not post-training calibration or a fixed rule that adds nausea when vomiting appears.

This is a small nine-label adaptation of [CorNet](https://www.kdd.org/kdd2020/accepted-papers/view/correlation-networks-for-extreme-multi-label-text-classification.html), not a reproduction of its extreme-label benchmarks. Correlation can also amplify false positives; improvement is not assumed. The model manifest stores the architecture for strict reconstruction.

## Execution and safety

Open the matching .ipynb in Colab, select a GPU, and run all cells using the existing mission3_ver5_inputs.zip. All runtime sources are embedded; no earlier result is required. Results go to unique run directories under DCC_AIdeas/results/mission3/ver16 in Drive.

Inputs remain utterances[].text. All original targets and samples remain unchanged. The nine two-class outputs use ordinary argmax, without thresholds, post-training calibration, external training corpora, or automatic relabeling. Both candidates use the same grouped Training split, seed, fixed KLUE revision, and moderate positive weights [1.4, 1.8, 2.0, 1.2, 1.3, 1.0, 3.0, 1.5, 1.3]. Candidate training uses at most three epochs with the original four-epoch learning-rate schedule horizon.

Training holdout Macro F1 selects the candidate and epoch. Official Validation is evaluated only after final selection. A selected full refit starts fresh and uses all Training targets for the selected number of epochs.

The shared 180-minute deadline starts after Drive login, before installation. Installation also has a 20-minute cap. The runtime reserves 20 minutes for finalization and uses measured time with a 1.25 safety factor. Slow hardware may require skipping a candidate or full refit. Only complete epochs and fully complete refits are published; incomplete refits never replace completed candidates. A fit-only fallback records its actual training_scope, fitted_samples, and refit_skipped_reason. A deadline cannot guarantee all phases finish on every GPU.

AUTO_RELEASE_GPU=True requests [Colab runtime release](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py) after result display, errors, or deadline. Drive results survive; /content scratch may disappear. If release fails, use Runtime > Disconnect and delete runtime manually. Actual Colab release service behavior and new-method GPU duration remain unverified locally.

ver16 and ver17 have independent budgets. Running both can consume up to six hours total. To conserve resources, run ver16 first and inspect its result before ver17.

## Saved results and resuming

Keep the entire result directory and executed notebook. Important files are candidate_summary.csv, final_selection.json, training_metrics.json, validation_metrics.json, runtime.json, mission3.csv, and mission3_ver16_model.ckpt with referenced weights/assets. The portable inference entry point is submission/inference.py.

Only complete or complete_budget_limited status plus run_complete.json represents completed final work. Missing final files or completion marker, budget_exhausted, failed, or interrupted mean incomplete work. Abrupt Colab reclamation can leave the previous saved status unchanged.

Set resume_run to a compatible run directory. Source, settings, data, split, and artifact hashes are checked before completed artifacts are reused. Partial epochs are not resumed. Resuming an already completed run only verifies and returns it without extending training. Each explicit rerun consumes a new budget and additional resources.

Previous versions and dependency requirements are unchanged. New full-data F1 improvement is unverified until the notebook is run.

