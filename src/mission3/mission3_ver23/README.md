# Mission 3 ver23

## Continue the shared representation, with dropout consistency

Open `../mission3_ver23.ipynb` in Colab, select a GPU, and run all cells. The notebook contains its runtime sources. The existing `mission3_ver5_inputs.zip` and **complete ver19 result directory or ZIP** are required. Both the fit-only selection checkpoint and the all-Training final checkpoint must be present. Leave `source_run` empty for automatic Drive discovery or set it to the complete folder/ZIP path. Dependencies are unchanged from ver19.

ver23 starts from the verified ver19 checkpoint and trains the encoder's last four layers, shared projection, all nine symptom heads and nausea evidence path. Embeddings and the first eight encoder layers stay frozen. This allows the text representation and every symptom decision to change. ver21 trained only the last nausea head; ver22 trained only the evidence module with a frozen encoder. Neither improved the Training holdout.

Two small, predefined continuations start independently from the same original checkpoint:

| Recipe | Objective |
| --- | --- |
| `continue_gce` | Existing positive-weight CE, with the existing nausea CE/GCE mixture active from the first continuation epoch |
| `continue_rdrop` | The same supervised objective averaged over two dropout passes, plus symmetric KL consistency with coefficient 1 |

Both recipes use at most two continuation epochs. The candidate is also compared with a fixed equal-probability average of itself and the original ver19 model, followed by ordinary two-class argmax. No mixture weights are searched, no symptom-specific model is selected, and no thresholds or score offsets are fitted. The fixed average is an additional inference candidate, not a claimed improvement. It requires two model forwards at inference.

[R-Drop, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html) reduces disagreement between two dropout realizations of the same training input by penalizing both directions of KL divergence. Here it is applied to the nine categorical outputs during shared-encoder continuation. The paper does not establish an expected gain on this dataset. Legacy ver6 explored classifier R-Drop under a different frozen-feature and threshold-based pipeline; this is not a claim that R-Drop has never appeared in the project.

## What the previous records establish

The design review read 23 saved final-result records, 23 saved candidate metric records, source descriptions, runtime records and the previous Training-only probes. This is a historical artifact audit, not a rerun of every old checkpoint. The audit is saved under `outputs/mission3_ver23_design/history_audit.json` in the development workspace.

| Previous experiment | Evidence relevant to this design |
| --- | --- |
| ver1, ver3–9 | Their higher legacy scores include tuned thresholds or calibration and are not a compatible baseline under the current rule. No ver2 result artifact was found. |
| ver10–13 | Sparse classifiers, unweighted CE, label attention, supervised contrastive learning and KcELECTRA did not solve nausea recall. |
| ver14–15 | Positive-weight CE improved nausea and broader recall. The six-candidate ver14 run took about 453 minutes; repeating it exceeds the current budget. |
| ver16–19 | CorNet, short task-adaptive pretraining, utterance evidence and GCE produced modest changes. ver19 provides the pinned model and Training split for the new experiment. |
| ver20 | RoBERTa-large with the top eight layers trainable did not pass the joint Training improvement gate. Its exported diagnostic model used only the fit subset. |
| ver21–22 | Frozen-feature head/evidence retraining completed without improving the gate; final outputs preserved ver19. More epochs alone have no demonstrated benefit. |
| Previous lexical/ensemble probes | Balanced lexical SVM reached Training Macro F1 0.573575 and nausea F1 0.270833. The best tested categorical majority vote reached 0.644791/0.369551. Symptomwise model choice fell from 0.646483 to 0.641218 under cross-half checking. Soft probability averaging was not measured in those historical probes. |
| Fresh ver23 design probes | Training-only NB log-ratio logistic regression reached 0.550117/0.197211; joint neural258 and sparse balanced SVM reached 0.622519/0.301611. Both were below the original ver19 Training reference and were excluded from this notebook. These are actual local CPU probe measurements, not new Colab results. |

Some reviewed text and provided nausea labels appear to use different symptom boundaries. That is not proof of label error. No reviewed target is changed, removed, reweighted by a manual annotation, or supplemented with invented evidence labels. The reused Training holdout is exploratory and is not a new independent test.

## Data boundaries and selection

Inputs remain only `utterances[].text`, with the existing historical spelling fallback. All supplied utterances, samples, original labels and overflow windows are preserved. No external corpus or API is used. Predictions remain raw two-class argmax or raw argmax of the predefined equal-probability average.

Source hashes, original ordered split, target order and Training identity are checked before learning. The original fit-only model, trained on 23,360 samples, must reproduce the saved predictions on the 5,840-sample Training holdout. The model trained on all 29,200 samples is never used for this selection evaluation.

Candidate/epoch selection uses the Training holdout only and requires Macro F1 to increase while nausea F1 does not decrease from the verified baseline. Among eligible candidates, higher Macro F1 wins; exact ties retain the earlier candidate. All nine symptom results and paired errors are reported. There is no guarantee that another symptom individually improves or remains identical, because the representation and all heads now learn.

After selection, the chosen continuation is freshly applied to the separate all-Training ver19 model using all 29,200 Training examples for the fixed selected epoch count. Fit-only continuation weights are not transplanted onto a different model. If the fixed average was selected, the final pair comprises the original all-Training ver19 model and its matched all-Training continuation. Official Validation is evaluated after selection and does not select the recipe, epochs or inference policy.

If no candidate passes the gate, or a complete selected continuation cannot fit the budget, final publication preserves the original all-Training ver19 model and records why. A failed or interrupted mandatory verification/export is not a completed baseline publication. Existing source results are never modified.

## Three-hour budget and execution errors

The shared 180-minute deadline starts after interactive Drive login and before package installation. Installation also has a 20-minute cap. Finalization reserves 25 minutes; future training is projected from measured throughput with a 1.25 safety factor. Only completed epochs and completed final continuations can be published. A deadline limits resource use but cannot guarantee that every candidate and final refit complete on every Colab GPU.

Run all cells from the beginning after a runtime reset. Preparation checks prevent missing `runtime_dir` and stale installation state from launching work. Worker errors display the failed stage, safe reason and code locations, and save a report under `ver23/_diagnostics/ver23_error_*.json`. Reports omit raw exception strings, dialogue text, patient identifiers and local-variable contents. Send the diagnostic report for a failed run rather than rerunning without its cause.

`AUTO_RELEASE_GPU=True` requests Colab runtime release after results, errors or the deadline. Drive outputs remain; `/content` temporary files can disappear. If automatic release fails, use **Runtime → Disconnect and delete runtime**. Real Colab service behavior and complete GPU runtime remain to be checked in Colab.

## Outputs and resuming

Keep the entire result directory and executed notebook. Important artifacts include `candidate_summary.csv`, `training_reference_comparison.json`, `final_selection.json`, `training_metrics.json`, `validation_metrics.json`, `runtime.json`, `mission3.csv`, and `mission3_ver23_model.ckpt` with all referenced weights/assets. The portable entry point is `submission/inference.py`.

`run_complete.json` is written after completed final artifacts are saved and verified. Missing completion files or a failed/interrupted status means the run is incomplete. `complete_not_improved` means the baseline was retained after comparison; it is not a new-model gain. Inspect the saved training scope and selection reason alongside the status.

Keep the default `run_validation=True` for submission output. Setting it to `False` performs a small model reload check but does not produce Validation predictions or `mission3.csv`; that mode is not a completed submission.

Set `resume_run` only to a compatible completed ver23 run. Source, settings, data identity and completion hashes must match. A completed run is verified and returned without additional training. Incomplete-run resumption is explicitly rejected; clear `resume_run` to start a fresh run. A fresh run starts a new time budget and consumes more resources.

No F1 improvement or complete Colab duration has been verified before running this new experiment.
