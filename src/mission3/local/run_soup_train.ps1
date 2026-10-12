# Train the holdout counterparts needed to test weight averaging honestly.
#
# Goal: fewer models, same score. The only lever that actually removes models
# rather than rearranging them is a weight soup -- averaging the weights of
# models that share an architecture and a head into ONE model. If a soup of
# three base-mean models matches the three-model ensemble, every same-recipe
# group in the submission collapses to a single checkpoint.
#
# Souping cannot be judged on the shipped weights: those are trained on all
# 29,200 calls, so there is no honest data left to score them on. These three
# are the holdout-split twins of roberta-base_raw_s{42,1,7}_FINAL -- same
# recipe read back from history (lr 2e-5, 3 epochs, batch 16 x 2, mean head,
# raw text, pos_weight sqrt5), with --save_model so the weights exist to
# average.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$L = @('--model','klue/roberta-base','--lr','2e-5','--epochs','3','--normalize','raw',
       '--batch_size','16','--accum','2','--pos_weight','sqrt5','--save_model',
       '--tag_suffix','_SOUP','--log',$LOG)
function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen10.py @L @extra 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
Go "holdout mean s42" @('--seed','42')
Go "holdout mean s1"  @('--seed','1')
Go "holdout mean s7"  @('--seed','7')
"[SOUP_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
