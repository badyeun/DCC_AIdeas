# The four knobs that were queued overnight and never ran: the large sweep
# and the diverse rebuild took priority. None of them has ever been touched
# in this project.
#
# They are judged twice. First as a recipe change -- does the single model
# score better? Second, and more usefully now, as a FAMILY: the ensemble sits
# at the size plateau, so the only thing left that pays is a member that is
# just as good and wrong about different calls.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$BASE = @('--model','klue/roberta-base','--lr','2e-5','--epochs','3','--normalize','raw',
          '--batch_size','16','--accum','2','--log',$LOG,'--tag_suffix','_LS')
function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen8.py @BASE @extra 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
foreach ($s in 42, 1) {
    Go "maxlen256 s$s"  @('--seed',"$s",'--max_length','256')
    Go "cosine s$s"     @('--seed',"$s",'--scheduler','cosine')
    Go "dropout0.2 s$s" @('--seed',"$s",'--dropout','0.2')
    Go "warmup0.2 s$s"  @('--seed',"$s",'--warmup','0.2')
}
"[KNOBS_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
