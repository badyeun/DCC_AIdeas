# Confirm roberta-large as a twelfth family member.
#
# It was dropped twice on standalone grounds -- "ties base and costs 3.7x" --
# without ever measuring what it contributes to the ensemble. That was the
# same mistake maxlen256 taught this morning: standalone quality does not
# predict ensemble contribution (correlation +0.22 across nine candidates).
#
# Measured now, it is the only candidate that pays, and it pays whether added
# or swapped in:  add +0.00113, replace-weakest +0.00111. But that is one
# seed against a 0.0026 noise floor, so two more.
#
# The settings below are the ones that produced that member, read back from
# its history file rather than remembered: lr 1.5e-5, batch 4 x accum 8.
# An earlier version of this script had lr 2e-5 and batch 8 x accum 4, which
# would have measured a different model.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$L = @('--model','klue/roberta-large','--lr','1.5e-5','--epochs','3','--normalize','raw',
       '--batch_size','4','--accum','8','--grad_checkpoint','--log',$LOG,'--tag_suffix','_LGC')
function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen9.py @L @extra 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
Go "large s1"        @('--seed','1')
Go "large s7"        @('--seed','7')
Go "large attn s42"  @('--seed','42','--head','label_attention')
"[LGC_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
