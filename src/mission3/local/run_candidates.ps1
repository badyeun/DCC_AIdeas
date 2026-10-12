# Candidate encoders the campaign never tried, in expected-value order.
#
# The mentor's two points drive this: twelve members is unusual, and some may
# not earn their place; and medically fine-tuned models are worth a look.
# The user wants the current eleven kept as they are, so these are tested as
# ADDITIONS, not replacements.
#
# Judged by ensemble contribution to the shipped eleven, not by standalone
# score. maxlen256 taught that lesson: it passed a "similar quality, enough
# disagreement" proxy and still cost -0.0011 when actually added.
#
# Excluded after a config check: albert-kor-base (256 positions, our median
# call is 230 tokens), xlm-roberta-base and KoE5 (250k vocab, the same memory
# thrashing that killed kf-deberta on an 8GB card).
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'

function Go($label, $model, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen8.py --model $model --lr 2e-5 --epochs 3 `
        --normalize raw --seed 42 --log $LOG @extra 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}

$B = @('--batch_size','16','--accum','2','--tag_suffix','_CAND')
Go "km-bert (medical)"  'madatnlp/km-bert'                    $B
Go "bert-kor-base"      'kykim/bert-kor-base'                 $B
Go "KoSimCSE-roberta"   'BM-K/KoSimCSE-roberta'               $B
Go "roberta-small"      'klue/roberta-small'                  $B
"[CAND_WAVE1] $(Get-Date -Format 'HH:mm:ss')"

# BigBird and large need the smaller batch plus checkpointing.
$L = @('--batch_size','8','--accum','4','--grad_checkpoint','--tag_suffix','_CAND')
Go "kobigbird-base"     'monologg/kobigbird-bert-base'        $L
Go "klue-roberta-large-tapt" 'SangwonYoon/klue-roberta-large-tapt' $L
"[CAND_WAVE2] $(Get-Date -Format 'HH:mm:ss')"
"[CAND_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
