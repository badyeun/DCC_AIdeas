# Phase three: close a gap in the submission's own evidence.
#
# The 12 submitted checkpoints are six mean-head and six attention-head, all
# at lr 2e-5. But the holdout only ever saw the attention head at lr 3e-5 --
# the learning-rate sweep was run on the mean head alone and the result was
# carried over. If the attention head prefers a different rate, half the
# submission is mis-tuned and nothing measured so far would have shown it.
#
# Paired against the existing attn lr3e-5 runs on the same seeds.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:\DevBox\projects\DCC_AIdeas
$LOG = 'results\mission3\local\labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$BASE = @('--model','klue/roberta-base','--epochs','3','--normalize','raw',
          '--batch_size','16','--accum','2','--log',$LOG,'--head','label_attention')
function Go($label, $extra) {
    "===== $label  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src\mission3\local\train_frozen.py @BASE @extra 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}
foreach ($s in 42, 1, 7) {
    Go "attn lr2e-5 s$s" @('--seed',"$s",'--lr','2e-5','--tag_suffix','_lr2e5_LS')
}
"[PHASE3_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
