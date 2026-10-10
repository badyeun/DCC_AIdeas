# Phase ten: roberta-large was only ever tried at lr 2e-5, inherited from the
# base-model sweep, in a single seed, with no ensemble. The base model's own
# optimum (2e-5) was found AFTER large's one run, and a larger model usually
# wants a lower rate, not the same one. This is the same gap already admitted
# once this session (attention head's learning rate copied from the mean
# head's sweep) -- now on the backbone that scored closest to roberta-base.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$BASE = @('--model','klue/roberta-large','--epochs','3','--normalize','raw',
          '--batch_size','8','--accum','4','--grad_checkpoint','--log',$LOG,'--tag_suffix','_LS')
function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen5.py @BASE @extra 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
Go "large lr1e-5 s42" @('--lr','1e-5','--seed','42')
Go "large lr1.5e-5 s42" @('--lr','1.5e-5','--seed','42')
Go "large lr1e-5 s1" @('--lr','1e-5','--seed','1')
Go "large lr1e-5 s7" @('--lr','1e-5','--seed','7')
"[PHASE10_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
