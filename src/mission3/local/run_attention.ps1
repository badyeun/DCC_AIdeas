$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; $env:TRANSFORMERS_VERBOSITY = 'error'
Set-Location C:\DevBox\projects\DCC_AIdeas
$LOG = 'results\mission3\local\train_log.md'
# Paired against the existing mean-pooling runs: same seeds, same holdout,
# same raw text -- the head is the only thing that changes.
foreach ($seed in 42, 1, 7) {
    "===== label_attention / seed $seed  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src\mission3\local\train_local.py --model klue/roberta-base --head label_attention `
        --normalize raw --epochs 3 --batch_size 16 --accum 2 --lr 3e-5 --seed $seed --log $LOG 2>&1 |
        Where-Object { $_ -notmatch 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated' }
    "===== label_attention / seed $seed  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}
"===== attention 실험 종료 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
