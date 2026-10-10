# Label-side campaign. Everything scored at 0.5 on the shared holdout.
#
# The diagnosis that motivates this: a free threshold would buy only +0.006,
# so the cut is already near-optimal and the gain has to come from better
# probabilities. Three label-side levers are tested:
#   1. self-distillation  -- hard 0/1 targets assert certainty the annotation
#      does not have (even "메스꺼워요" carries 오심 only 63% of the time)
#   2. per-symptom loss weight -- macro F1 counts all nine equally, BCE does not
#   3. label-correlation mixer -- 오심 travels with 구토 and 어지러움
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:\DevBox\projects\DCC_AIdeas
$LOG = 'results\mission3\local\labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$BASE = @('--model','klue/roberta-base','--lr','2e-5','--epochs','3','--normalize','raw',
          '--batch_size','16','--accum','2','--log',$LOG)

function Go($label, $extra) {
    "===== $label  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src\mission3\local\train_local.py @BASE @extra 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}

# --- 1. teacher: four out-of-fold models over the train split -------------
foreach ($f in 0, 1, 2, 3) {
    Go "teacher fold $f" @('--seed','42','--folds','4','--fold',"$f",'--tag_suffix','_T')
}
& $PY src\mission3\local\stitch_oof.py
"[OOF_DONE] $(Get-Date -Format 'HH:mm:ss')"

# --- 2. students on soft targets -----------------------------------------
$SOFT = 'results\mission3\local\soft_targets_oof.npy'
foreach ($a in '0.3', '0.5', '0.7') {
    Go "distill alpha $a" @('--seed','42','--soft_targets',$SOFT,'--soft_alpha',$a,'--tag_suffix','_LS')
}
"[DISTILL_DONE] $(Get-Date -Format 'HH:mm:ss')"

# --- 3. loss-weight and correlation probes -------------------------------
Go "label_corr s42"  @('--seed','42','--label_corr','--tag_suffix','_LS')
Go "invfreq s42"     @('--seed','42','--label_weight','invfreq','--tag_suffix','_LS')
Go "hardboost s42"   @('--seed','42','--label_weight','custom:1,1,1.6,1,1,1,1.6,1.6,1','--tag_suffix','_LS')
Go "label_corr s1"   @('--seed','1','--label_corr','--tag_suffix','_LS')
"[LABELSIDE_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
