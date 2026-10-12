# Distil a three-model ensemble into one model.
#
# Weight averaging just failed outright (-0.00913, worse than the weakest
# ingredient): different seeds randomise the classification head, so the three
# models sit in different basins and averaging their weights breaks both.
#
# Distillation does not touch the weights -- it trains a fresh model on the
# ensemble's own probabilities. If one model can carry the ensemble's gain,
# the submission drops from sixteen models to a handful.
#
# Teacher: probability average of the three holdout-split models (0.64235 raw,
# 0.64400 blended) whose ingredients score 0.63524-0.64071 alone. The gain to
# preserve is therefore about +0.003.
#
# Three mixing weights, because the right one is unknown: alpha is how much of
# the target comes from the teacher rather than the hard label. alpha 1.0 is
# pure distillation, 0.5 is half and half. Earlier self-distillation used 0.3
# and lost, but that was trying to beat a teacher, not replace an ensemble.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$L = @('--model','klue/roberta-base','--lr','2e-5','--epochs','3','--normalize','raw',
       '--batch_size','16','--accum','2','--pos_weight','sqrt5','--seed','42',
       '--soft_targets','results/mission3/local/soft_targets_ens3.npy','--log',$LOG)
function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen10.py @L @extra 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
Go "distil alpha 1.0" @('--soft_alpha','1.0','--tag_suffix','_DIST')
Go "distil alpha 0.7" @('--soft_alpha','0.7','--tag_suffix','_DIST')
Go "distil alpha 0.5" @('--soft_alpha','0.5','--tag_suffix','_DIST')
"[DIST_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
