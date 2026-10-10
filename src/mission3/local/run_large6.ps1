# roberta-large, finished properly this time.
#
# It was stopped at 3 epochs while its score was still climbing steeply:
# 0.63243 -> 0.63447 -> 0.64061. Calling it "tied with base and 3.7x the
# cost" was a verdict on an undertrained model. At lr 1e-5 a large encoder
# needs more passes than the base model's three.
#
# Six epochs, three seeds. train_local.py scores every epoch and keeps the
# best, so one run shows both where the curve peaks and what it peaks at.
# Judged twice: as a standalone recipe, and as an ensemble family against
# the eleven-member submission.
#
# The tag records lr and epochs this time -- neither is in the automatic tag,
# and that already caused one silent overwrite tonight.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
foreach ($s in 42, 1, 7) {
    "===== large lr1e-5 6ep s$s start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen8.py --model klue/roberta-large `
        --lr 1e-5 --epochs 6 --normalize raw --batch_size 8 --accum 4 `
        --grad_checkpoint --seed $s --log $LOG --tag_suffix '_lr1e5e6_LS' 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== large lr1e-5 6ep s$s end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
"[LARGE6_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
