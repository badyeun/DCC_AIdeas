# Iteration 1: does backbone diversity beat seed diversity in the ensemble?
#
# Last night the only reliable gain was ensembling, but every member was the
# same klue/roberta-base with a different seed, and the gain saturated around
# nine members. Different pretrained models make different mistakes, so a
# same-sized ensemble of different backbones should ensemble better -- that is
# the hypothesis this queue tests.
#
# Raw text throughout: the paired A/B showed normalisation does nothing
# (-0.0003 over four pairs), so it is one less variable here.

$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:\DevBox\projects\DCC_AIdeas

$LOG = 'results\mission3\local\train_log.md'
$script = 'src\mission3\local\train_local.py'

# Korean encoders with genuinely different pretraining corpora.
# KcELECTRA/KcBERT are trained on comments and spoken-style text, which is
# closer to a 119 call than KLUE's news/wiki mix.
$backbones = @(
    'beomi/KcELECTRA-base-v2022',
    'monologg/koelectra-base-v3-discriminator',
    'klue/bert-base',
    'beomi/kcbert-base'
)

foreach ($model in $backbones) {
    "===== $model  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY $script --model $model --normalize raw --epochs 3 `
        --batch_size 16 --accum 2 --lr 3e-5 --seed 42 --log $LOG 2>&1 | Where-Object {
        $_ -notmatch 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
    }
    "===== $model  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
    ""
}

"===== 백본 비교 종료 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
