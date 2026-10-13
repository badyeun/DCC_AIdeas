# Distillation with the target de-biased against pos_weight.
#
# Attempt 1 (pos_weight sqrt5, raw soft targets): monotone worse as alpha rose
#   -- 0.5 -> 0.62675, 0.7 -> 0.61903, 1.0 -> 0.60369, control 0.64071.
# Attempt 2 (pos_weight off): the direction flipped, alpha 1.0 beating its own
#   control by +0.0285 (0.63854 vs 0.61009). So the diagnosis held: the weight
#   was distorting the target, not distillation failing. But turning the weight
#   off costs more than distillation gains -- 0.63854 still trails the plain
#   weighted single model at 0.64071.
#
# This keeps the weight AND the teacher: the target becomes
# y* = p / (w - (w-1)p), whose weighted-loss optimum is exactly p. Verified
# numerically at w = 1, 2, 5 across p = 0.05..0.9 (error 0 in all 15 cases),
# and the de-biased targets average 0.1600 against a 0.1587 positive rate --
# the raw teacher probabilities averaged 0.2295.
#
# Three alphas, because how much to trust a teacher that is in-sample on the
# student's training data is not something the fix settles.
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
       '--soft_targets','results/mission3/local/soft_targets_ens3.npy',
       '--soft_debias','--tag_suffix','_D3','--log',$LOG)
function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen11.py @L @extra 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
Go "debias alpha 1.0" @('--soft_alpha','1.0')
Go "debias alpha 0.7" @('--soft_alpha','0.7')
Go "debias alpha 0.5" @('--soft_alpha','0.5')
"[DIST3_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
