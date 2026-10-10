$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; $env:TRANSFORMERS_VERBOSITY = 'error'
Set-Location C:\DevBox\projects\DCC_AIdeas
$LOG = 'results\mission3\local\train_log.md'
# Layer-wise LR decay is the last standard fine-tuning technique this pipeline
# never used; ver4 ran 0.95. Paired against the existing lr 2e-5 runs.
foreach ($d in '0.95', '0.9') {
  foreach ($seed in 42, 1, 7) {
    if ($d -eq '0.9' -and $seed -ne 42) { continue }   # 0.9 only needs a direction read
    "===== llrd=$d / seed $seed  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src\mission3\local\train_local.py --model klue/roberta-base --lr 2e-5 --epochs 3 `
        --normalize raw --batch_size 16 --accum 2 --seed $seed --llrd $d --log $LOG 2>&1 |
        Where-Object { $_ -notmatch 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated' }
    "===== llrd=$d / seed $seed  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
  }
}
"[PHASE_D_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
