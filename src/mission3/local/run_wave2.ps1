# Wave two: diversity axes INSIDE klue/roberta-base.
#
# The pattern across eight backbone swaps is unambiguous -- km-bert (medical,
# -0.0118), KcELECTRA (-0.0100), KoELECTRA (-0.0140), KR-ELECTRA (-0.0080),
# bert-base (-0.0036), large (~0), and two more. Each adds plenty of
# disagreement and loses more in quality than it gains. Every family that
# actually paid is a klue/roberta-base variant: a different head, a different
# input, a different pretraining continuation, a different rate.
#
# So these stay on that backbone and vary what has never been varied:
#   pool_layer   read an earlier transformer layer, which is less tuned to the
#                training labels and wrong about different calls
#   freeze       adapt only the upper layers, leaving general features alone
#   meanmax      concatenate mean and max pooling into one head
#
# Judged by contribution to the shipped eleven, never by standalone score.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen9.py --model klue/roberta-base --lr 2e-5 `
        --epochs 3 --normalize raw --batch_size 16 --accum 2 --seed 42 `
        --log $LOG --tag_suffix '_W2' @extra 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
Go "meanmax head"      @('--head','meanmax')
Go "pool layer 10"     @('--pool_layer','10')
Go "pool layer 8"      @('--pool_layer','8')
Go "freeze bottom 4"   @('--freeze_layers','4')
Go "freeze bottom 6"   @('--freeze_layers','6')
Go "attn + layer 10"   @('--head','label_attention','--pool_layer','10')
"[W2_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
