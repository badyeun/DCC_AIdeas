# Phase four: two levers the earlier rounds never touched.
#
# 1. Oversampling. pos_weight was swept five ways, but that only rescales the
#    loss -- the model still sees 두통 in 9.9% of calls. Drawing rare-symptom
#    calls more often changes what it actually sees: 두통 rises to 14.9% and
#    오심 to 15.2% under "full".
# 2. The distillation control. If soft targets win, we need to know whether it
#    was the teacher's knowledge or merely softer targets. These targets carry
#    zero teacher knowledge -- every call gets that symptom's base rate.
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
    & $PY src\mission3\local\train_frozen2.py @BASE @extra 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}
Go "oversample full s42" @('--seed','42','--oversample','full')
Go "oversample full s1"  @('--seed','1', '--oversample','full')
Go "oversample sqrt s42" @('--seed','42','--oversample','sqrt')
Go "증류 대조군 const a0.3 s42" @('--seed','42','--soft_targets',
    'results\mission3\local\soft_targets_constant.npy','--soft_alpha','0.3')
"[PHASE4_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
