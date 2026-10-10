# Out-of-fold teacher over ALL 29,200 calls, for the submission retrain.
#
# The teacher run so far covered only the 23,360-call train split, so the
# holdout rows still carry hard labels. A submission model trains on
# everything; feeding it a mix of soft and hard targets would train on two
# different objectives. --full_train makes the fold split cover the whole set.
#
# Only run this if phase 6 shows distillation is worth shipping.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:\DevBox\projects\DCC_AIdeas
$LOG = 'results\mission3\local\labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
foreach ($f in 0, 1, 2, 3) {
    "===== full-OOF fold $f  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src\mission3\local\train_frozen4.py --model klue/roberta-base --lr 2e-5 `
        --epochs 3 --normalize raw --batch_size 16 --accum 2 --seed 42 `
        --full_train --folds 4 --fold $f --tag_suffix '_TF' --log $LOG 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== full-OOF fold $f  종료 $(Get-Date -Format 'HH:mm:ss') ====="
}
"[FULLOOF_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
