# Phase six: collect what distillation actually earned.
#
# Soft targets at alpha 0.3 raised the ranking ceiling to 0.64667 (+0.0017,
# the first arm to move it past the 0.0010 noise floor) but lost 0.008 at the
# fixed 0.5, because compressed targets plus pos_weight=sqrt push 9,583
# predictions out of 8,305 true. Its best single cut sits at 0.61.
#
# The cut cannot move, so the weighting does. Changing pos_weight from w1 to
# w2 shifts the boundary by log(w2/w1): sqrt averages 2.38 and pow025 averages
# 1.54, so log(1.54/2.38) = -0.44, which puts the effective cut at 0.608 --
# where this model wants it. Same formula for all nine symptoms; this is
# global class weighting, not a per-symptom threshold.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:\DevBox\projects\DCC_AIdeas
$LOG = 'results\mission3\local\labelside_log.md'
$SOFT = 'results\mission3\local\soft_targets_oof.npy'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$BASE = @('--model','klue/roberta-base','--lr','2e-5','--epochs','3','--normalize','raw',
          '--batch_size','16','--accum','2','--log',$LOG,'--tag_suffix','_LS')
function Go($label, $extra) {
    "===== $label  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src\mission3\local\train_frozen4.py @BASE @extra 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}
Go "증류0.3 + pw025 s42" @('--seed','42','--soft_targets',$SOFT,'--soft_alpha','0.3','--pos_weight','pow025')
Go "증류0.3 + pw035 s42" @('--seed','42','--soft_targets',$SOFT,'--soft_alpha','0.3','--pos_weight','pow035')
Go "증류0.3 + pw025 s1"  @('--seed','1', '--soft_targets',$SOFT,'--soft_alpha','0.3','--pos_weight','pow025')
Go "증류0.15 s42"        @('--seed','42','--soft_targets',$SOFT,'--soft_alpha','0.15')
Go "pw025 단독 s42 (대조)" @('--seed','42','--pos_weight','pow025')
"[PHASE6_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
