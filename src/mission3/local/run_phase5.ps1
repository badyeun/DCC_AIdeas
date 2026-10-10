# Phase five: optimise the metric instead of a proxy for it.
#
# Every arm so far minimised BCE and left the 0.5 cut to land where it may.
# The things that lost -- pos_weight, per-symptom loss weights, the logit
# mixer -- were all attempts to nudge that landing from outside. A soft macro
# F1 term puts the objective inside the loss: counts built from the sigmoid
# outputs, so a symptom's rarity is handled by the F1 formula rather than by
# a weight we guess. Added to BCE, not replacing it: alone it gives no
# gradient to a symptom missing from the batch, and 두통 (9.8%) often is.
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
          '--batch_size','16','--accum','2','--log',$LOG,'--tag_suffix','_LS')
function Go($label, $extra) {
    "===== $label  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src\mission3\local\train_frozen3.py @BASE @extra 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}
Go "softf1 0.5 s42" @('--seed','42','--softf1','0.5')
Go "softf1 1.0 s42" @('--seed','42','--softf1','1.0')
Go "softf1 0.5 s1"  @('--seed','1', '--softf1','0.5')
"[PHASE5_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
