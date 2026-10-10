$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; $env:TRANSFORMERS_VERBOSITY = 'error'
Set-Location C:\DevBox\projects\DCC_AIdeas
$LOG = 'results\mission3\local\train_log.md'

# none (1.0) -> 0.619, sqrt (1.8-3.0) -> 0.638: stronger weighting is better,
# so sweep further along that direction. Seed 42 first for every scheme, so a
# one-seed read of the whole curve arrives before any scheme gets three seeds.
foreach ($seed in 42, 1, 7) {
    foreach ($pw in 'pow065', 'pow080', 'full') {
        "===== pos_weight=$pw / seed $seed  시작 $(Get-Date -Format 'HH:mm:ss') ====="
        & $PY src\mission3\local\train_local.py --model klue/roberta-base --pos_weight $pw `
            --normalize raw --epochs 3 --batch_size 16 --accum 2 --lr 3e-5 --seed $seed --log $LOG 2>&1 |
            Where-Object { $_ -notmatch 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated' }
        "===== pos_weight=$pw / seed $seed  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
    }
}
"===== pos_weight 실험 종료 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
