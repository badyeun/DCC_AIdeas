# Phase nine: domain-adaptive pretraining, then fine-tune on top of it.
#
# The per-symptom probe shows 오심 is the only label the model cannot separate
# inside its own evidence group (AUC 0.567); the other eight sit at 0.72-0.98.
# They are learnable, so the limit there is how well the encoder reads
# transcribed emergency speech -- not the annotation. klue/roberta-base was
# pretrained on written Korean; continuing its masked-LM training on the call
# transcripts closes that gap before a single label is read.
#
# Holdout transcripts are excluded from the MLM step, so the encoder never
# sees the surface of the calls it is scored on.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$ADAPTED = 'results/mission3/local/domain_adapted'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'

"===== MLM domain adaptation start $(Get-Date -Format 'HH:mm:ss') ====="
& $PY src/mission3/local/domain_pretrain.py --epochs 3 2>&1 | Where-Object { $_ -notmatch $NOISE }
"===== MLM end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="

if (Test-Path "$ADAPTED/config.json") {
    foreach ($s in 42, 1, 7) {
        "===== DAPT finetune mean s$s start $(Get-Date -Format 'HH:mm:ss') ====="
        & $PY src/mission3/local/train_frozen5.py --model $ADAPTED --lr 2e-5 --epochs 3 --normalize raw --batch_size 16 --accum 2 --seed $s --log $LOG --tag_suffix '_DAPT_LS' 2>&1 | Where-Object { $_ -notmatch $NOISE }
        "===== DAPT finetune mean s$s end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
    }
    "===== DAPT finetune attn s42 start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen5.py --model $ADAPTED --head label_attention --lr 2e-5 --epochs 3 --normalize raw --batch_size 16 --accum 2 --seed 42 --log $LOG --tag_suffix '_DAPT_LS' 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== DAPT finetune attn s42 end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
} else {
    "[ABORT9] domain-adapted model was not produced"
}
"[PHASE9_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
