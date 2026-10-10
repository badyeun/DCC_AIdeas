# Label-side, part two: inject the label-set structure as a training-only task.
#
# Only 188 of the 511 possible symptom sets ever occur, every call carries at
# least one symptom, and 어지러움+오심 is the ninth most common pair. Nine
# independent sigmoids cannot represent any of that. A softmax over whole
# combinations can, and it is dropped at inference, so it shapes the encoder
# without touching the 0.5 decision rule.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:\DevBox\projects\DCC_AIdeas
$LOG = 'results\mission3\local\labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$BASE = @('--model','klue/roberta-base','--lr','2e-5','--epochs','3','--normalize','raw',
          '--batch_size','16','--accum','2','--log',$LOG,'--tag_suffix','_LS')
function Go($label, $extra) {
    "===== $label  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src\mission3\local\train_local.py @BASE @extra 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}
Go "aux combo40 w0.3 s42" @('--seed','42','--aux','combo:40','--aux_weight','0.3')
Go "aux combo40 w0.3 s1"  @('--seed','1', '--aux','combo:40','--aux_weight','0.3')
Go "aux combo40 w1.0 s42" @('--seed','42','--aux','combo:40','--aux_weight','1.0')
Go "aux count w0.3 s42"   @('--seed','42','--aux','count','--aux_weight','0.3')
Go "aux combo20 w0.3 s42" @('--seed','42','--aux','combo:20','--aux_weight','0.3')
"[LABELSIDE2_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
