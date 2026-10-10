# Check the two new pooling heads before trusting them in a submission.
#
# Family diversity is the only lever still paying (+0.0016 across 200 random
# draws), so more families are worth having -- but only families of equal
# quality. Backbone swaps failed exactly here: they added diversity and lost
# more in quality than they gained. CLS and max pooling are measured against
# the mean head on the same seeds and the same holdout first.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen8.py @extra --model klue/roberta-base --lr 2e-5 --epochs 3 --normalize raw --batch_size 16 --accum 2 --log $LOG --tag_suffix '_LS' 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
foreach ($s in 42, 1) {
    Go "cls head s$s" @('--head','cls','--seed',"$s")
    Go "max head s$s" @('--head','max','--seed',"$s")
}
"[HEADS_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
