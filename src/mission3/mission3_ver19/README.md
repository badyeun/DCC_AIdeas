# Mission 3 ver19

## Experiment: a predefined nausea loss recipe

Both candidates use the exact ver18 global-plus-utterance architecture, including the same evidence branch, fixed KLUE revision, tokenization, grouped split method, initialization, optimizer, and positive weights. No trained checkpoint from an earlier run is reused.

- `baseline_ce`: existing weighted cross entropy for all nine labels.
- `nausea_gce`: epoch 1 uses the identical CE path; from epoch 2, only the nausea item uses 50% CE plus 50% generalized cross entropy (GCE), with fixed q=0.7. The other eight labels retain CE.

For the recorded target probability p, GCE=(1-p^q)/q. The mixed nausea gradient equals the weighted-CE gradient times 0.5+0.5*p^0.7. This reduces the influence of strongly contradicted targets while retaining at least half the original gradient. It also weakens learning from genuinely difficult positives and negatives and reduces the relative nausea task gradient. Improvement would not establish that label noise caused the previous errors. No label is corrected, removed, or replaced.

The stable implementation uses log_softmax and -expm1(q*log_p)/q. It never clamps probabilities, truncates high-loss examples, normalizes weights per sample, or changes the original batch-by-nine denominator. Positive weights are applied to the original target exactly as before. Inactive, warmup, and zero-mixture paths return the original CE result.

Source: [Zhang and Sabuncu, Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels, NeurIPS 2018](https://papers.neurips.cc/paper_files/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf), Eq. 6 and sections 3.2/4.1. One-epoch warmup and the CE/GCE mixture are this experiment's predefined application recipe, not a reproduction of the paper's results or robustness guarantees.

## Evidence and evaluation boundaries

The preparatory audit inspected only Training cases that remained wrong in ver15 through ver18: 299 false negatives and 429 false positives. Frozen lexical and neighboring-turn flags were computed across those cases. A redacted review is limited by visible context; absence of an expression does not prove missing evidence or incorrect annotation. These exploratory findings motivate an experiment, not automatic relabeling. Historical holdout inspection means this is not a fresh untouched audit set.

Both candidates start afresh from `klue/roberta-base` revision `02f94ba5e3fcb7e2a58a390b8639b0fac974a8da`. Inputs remain only `utterances[].text`; no speaker, diagnostic metadata, or gold co-symptom is a model feature. All original utterances, overflow windows, samples, and labels remain. Positive weights in the established label order are `[1.4, 1.8, 2.0, 1.2, 1.3, 1.0, 3.0, 1.5, 1.3]`.

The existing grouped Training split method uses seed 42 and holdout size 0.2. The actual indices are saved to `training_split.npz`. Candidate and epoch selection use Training holdout Macro F1 only, with first-declared candidate winning exact ties. Official Validation is evaluated only after final selection. The loss recipe is fixed before execution, with no q/mixture/threshold search. Model outputs remain nine two-class raw argmax decisions.

Candidate training uses at most three epochs and the original four-epoch learning-rate schedule horizon. A fresh all-Training refit repeats CE warmup from epoch 1. Configuration, histories, and manifests record the configured loss, active loss per epoch, and `trained_gce_epochs` for the saved checkpoint. If selection keeps epoch 1, GCE was not applied to that checkpoint. The shared encoder means other labels' predictions can change even though their loss terms remain CE.

## Run the standalone notebook

Open `mission3_ver19.ipynb` in Colab, select a GPU, and run all cells. Use the existing `mission3_ver5_inputs.zip` and the shared Drive account. All runtime modules, including diagnostics, are embedded; previous result files are not required. Results go to unique directories under `DCC_AIdeas/results/mission3/ver19`.

The 180-minute shared deadline begins after Drive login and before package installation. Installation has a 20-minute cap. Finalization reserves 20 minutes, and measured phase times use a 1.25 safety factor. Slow hardware may cause an additional candidate or fresh full refit to be skipped. Only completed epochs and fully completed refits can be published. Actual `training_scope`, `fitted_samples`, and fallback reason remain visible. This caps the job budget but cannot guarantee every phase finishes on every GPU.

`AUTO_RELEASE_GPU=True` requests Colab runtime release after result display, errors, or deadline. Drive artifacts remain; `/content` scratch can disappear. A failed request prints the manual Runtime > Disconnect and delete runtime instruction. New-method full-data F1, GPU runtime, and actual release behavior must be checked in Colab.

## Results and resuming

Keep the complete run folder and executed notebook. Important files include `candidate_summary.csv`, `candidate_comparison.json`, `final_selection.json`, `training_metrics.json`, `validation_metrics.json`, `runtime.json`, `mission3.csv`, and `mission3_ver19_model.ckpt` with its referenced weights/assets. Portable inference is `submission/inference.py`.

The comparison reports per-label metrics plus FN-to-TP, TP-to-FN, FP-to-TN, and TN-to-FP counts. Its `training_nausea_groups` contains aggregate comparisons for fixed expression matches, neighboring question/negation flags, and gold vomiting/dizziness without nausea. These overlapping groups are diagnostic only; they never control optimization, sample weights, labels, selection, or inference. No patient text or sample IDs appear in this aggregate report.

Only status `complete` or `complete_budget_limited` together with `run_complete.json` indicates final completion. Interrupted epochs and incomplete refits never replace complete candidates. Missing final artifacts or completion marker means incomplete work. Abrupt Colab reclamation may leave the previous status unchanged.

Set `resume_run` only to a compatible run directory. Source, settings, data, split, objective provenance, and artifact hashes are checked before reuse. A completed run is verified and returned without additional training. Partial epochs are not resumed. Each explicit rerun consumes a new budget. Older notebooks, results, and dependencies remain unchanged.
