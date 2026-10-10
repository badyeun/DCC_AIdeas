$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'; $env:TRANSFORMERS_VERBOSITY = 'error'
Set-Location C:\DevBox\projects\DCC_AIdeas
$LOG = 'results\mission3\local\train_log.md'
$S = 'src\mission3\local\train_local.py'

function Run($label, $a) {
    "===== $label  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY $S @a 2>&1 | Where-Object { $_ -notmatch 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated' }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}

$base = @('--model','klue/roberta-base','--lr','2e-5','--epochs','3','--normalize','raw','--batch_size','16','--accum','2','--log',$LOG)

# C1: EMA. Cheap, standard, and the one remaining knob that changes what gets
# evaluated rather than what gets learned.
foreach ($seed in 42, 1, 7) { Run "EMA 0.999 / seed $seed" ($base + @('--seed',"$seed",'--ema','0.999')) }

# C2/C3: loss shape. pos_weight showed a real inverted-U, so the loss curve
# itself may matter; ver6 only tested these on frozen features.
foreach ($seed in 42, 1) { Run "ASL / seed $seed" ($base + @('--seed',"$seed",'--loss','asl','--gamma_neg','2')) }
foreach ($seed in 42, 1) { Run "Focal / seed $seed" ($base + @('--seed',"$seed",'--loss','focal','--gamma_neg','2')) }

"[PHASE_C_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

