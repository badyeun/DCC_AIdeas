# Distillation, with the loss interaction fixed.
#
# The first attempt got monotonically worse as the teacher's weight rose
# (alpha 0.5 -> 0.62675, 0.7 -> 0.61903, 1.0 -> 0.60369) against a hard-label
# control of 0.64071. That pattern is not "distillation does not help", it is
# "the target is being distorted".
#
# Why: the loss is BCE with pos_weight. With a hard label the weight only
# rebalances the two gradient terms -- the optimum stays at 0 or 1. With a
# SOFT target y it moves the optimum to w*y / (w*y + 1 - y). At w = 5 and
# y = 0.23 the model is being asked to predict 0.60, and the decision is taken
# at 0.5, so every soft target pushes predictions over the line.
#
# So: same distillation, pos_weight off, with a matched hard-label control at
# pos_weight off so the comparison is not confounded by removing the weight.
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
       '--batch_size','16','--accum','2','--pos_weight','none','--seed','42','--log',$LOG)
$SOFT = @('--soft_targets','results/mission3/local/soft_targets_ens3.npy')
function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen10.py @L @extra 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
Go "control: hard labels, pw none" @('--tag_suffix','_D2')
Go "distil alpha 1.0, pw none"     ($SOFT + @('--soft_alpha','1.0','--tag_suffix','_D2'))
Go "distil alpha 0.5, pw none"     ($SOFT + @('--soft_alpha','0.5','--tag_suffix','_D2'))
"[DIST2_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
