# Final submission ensemble: the validated recipe, trained on ALL 29,200 calls.
#
# Everything above this point held 20% of Training back as a local holdout, so
# the models only ever saw 23,360 calls. The data-size curve (50/65/80/100% of
# that split) extrapolates to about +0.004 from using the full set, and the
# rules allow it -- the holdout was our own choice, not a requirement.
#
# These runs CANNOT be scored locally: training on everything leaves nothing to
# score against. That is expected. The recipe was settled on the holdout first;
# this step only re-fits it on more data.
#
# Recipe, from 15 paired experiments:
#   klue/roberta-base, raw text, 3 epochs, lr 2e-5, pos_weight sqrt,
#   batch 16 x accum 2, mean and attention heads mixed, no EMA, no post-hoc
#   rule, plain probability averaging, threshold 0.5 for every symptom.

$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:\DevBox\projects\DCC_AIdeas

$LOG = 'results\mission3\local\final_train_log.md'

# Six mean-head and six attention-head members: the size curve flattened around
# ten to fourteen, and mixing heads was the only diversity that ever paid.
$plan = @()
foreach ($seed in 42, 1, 7, 2026, 7777, 31) { $plan += @{head='mean'; seed=$seed} }
foreach ($seed in 42, 1, 7, 2026, 7777, 31) { $plan += @{head='label_attention'; seed=$seed} }

foreach ($p in $plan) {
    $label = "FINAL $($p.head) seed $($p.seed)"
    "===== $label  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src\mission3\local\train_local.py --model klue/roberta-base `
        --head $p.head --lr 2e-5 --epochs 3 --normalize raw `
        --batch_size 16 --accum 2 --seed $p.seed `
        --full_train --save_model --tag_suffix '_FINAL' --log $LOG 2>&1 |
        Where-Object { $_ -notmatch 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated' }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}

"[FINAL_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
