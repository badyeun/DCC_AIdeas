$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; $env:TRANSFORMERS_VERBOSITY = 'error'
Set-Location C:\DevBox\projects\DCC_AIdeas
$LOG = 'results\mission3\local\train_log.md'

# Learning rate was never swept -- every run so far used 3e-5. Lower rates
# usually want more epochs, so each rate below 3e-5 also gets a 4-epoch run;
# otherwise the sweep would just measure "undertrained at a low rate".
$plan = @(
    @{lr='1e-5'; ep=3}, @{lr='1e-5'; ep=5},
    @{lr='2e-5'; ep=3}, @{lr='2e-5'; ep=4},
    @{lr='5e-5'; ep=3}, @{lr='8e-5'; ep=3}
)
foreach ($p in $plan) {
    $tag = "_lr$($p.lr)e$($p.ep)"
    "===== lr=$($p.lr) epochs=$($p.ep) / seed 42  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src\mission3\local\train_local.py --model klue/roberta-base --lr $p.lr --epochs $p.ep `
        --normalize raw --batch_size 16 --accum 2 --seed 42 --tag_suffix $tag --log $LOG 2>&1 |
        Where-Object { $_ -notmatch 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated' }
    "===== lr=$($p.lr) epochs=$($p.ep)  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}
"===== 학습률 탐색 종료 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
