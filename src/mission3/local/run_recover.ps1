# Recovery + remaining label-side work, all against the frozen trainer.
#
# Teacher fold 3 died when train_local.py was edited under it, which left the
# stitch with three folds of four and sank the three distillation runs. This
# redoes that chain and then carries on with the auxiliary-head and R-Drop arms.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:\DevBox\projects\DCC_AIdeas
$LOG = 'results\mission3\local\labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$TRAIN = 'src\mission3\local\train_frozen.py'
$BASE = @('--model','klue/roberta-base','--lr','2e-5','--epochs','3','--normalize','raw',
          '--batch_size','16','--accum','2','--log',$LOG)

# Wait for the queue still running to go idle. Absence of python.exe is the
# signal; two consecutive clear checks guard against the gap between runs.
$clear = 0
while ($clear -lt 2) {
    if (@(Get-Process python -ErrorAction SilentlyContinue).Count -eq 0) { $clear++ } else { $clear = 0 }
    Start-Sleep -Seconds 20
}
"[RECOVER_START] $(Get-Date -Format 'HH:mm:ss')"

function Go($label, $extra) {
    "===== $label  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY $TRAIN @BASE @extra 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}

Go "teacher fold 3 (재실행)" @('--seed','42','--folds','4','--fold','3','--tag_suffix','_T')
& $PY src\mission3\local\stitch_oof.py
if (-not (Test-Path 'results\mission3\local\soft_targets_oof.npy')) {
    "[ABORT] 소프트 타깃 생성 실패 — 증류 건너뜀"
} else {
    "[OOF_DONE] $(Get-Date -Format 'HH:mm:ss')"
    $SOFT = 'results\mission3\local\soft_targets_oof.npy'
    foreach ($a in '0.3', '0.5', '0.7') {
        Go "distill alpha $a" @('--seed','42','--soft_targets',$SOFT,'--soft_alpha',$a,'--tag_suffix','_LS')
    }
}
"[DISTILL_DONE] $(Get-Date -Format 'HH:mm:ss')"

Go "aux combo40 w0.3 s42" @('--seed','42','--aux','combo:40','--aux_weight','0.3','--tag_suffix','_LS')
Go "aux combo40 w0.3 s1"  @('--seed','1', '--aux','combo:40','--aux_weight','0.3','--tag_suffix','_LS')
Go "aux count w0.3 s42"   @('--seed','42','--aux','count','--aux_weight','0.3','--tag_suffix','_LS')
Go "rdrop 0.5 s42"        @('--seed','42','--rdrop','0.5','--tag_suffix','_LS')
Go "rdrop 1.0 s42"        @('--seed','42','--rdrop','1.0','--tag_suffix','_LS')
"[ALL_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
