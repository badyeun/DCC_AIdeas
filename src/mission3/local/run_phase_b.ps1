# Phase B: confirm the learning-rate finding, then test bagging.
#
# The sweep put 2e-5 above the inherited 3e-5 by +0.0028 on one seed, which is
# inside the usual seed spread -- so it gets two more seeds before anything is
# built on it. Bagging follows because it is the one diversity source that
# should not cost member quality: every member still sees 80% of the data,
# unlike the weak backbones that failed in experiment 6.

$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:\DevBox\projects\DCC_AIdeas

$LOG = 'results\mission3\local\train_log.md'
$script = 'src\mission3\local\train_local.py'

function Run($label, $arguments) {
    "===== $label  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY $script @arguments 2>&1 | Where-Object {
        $_ -notmatch 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
    }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}

# --- B1: two more seeds at lr 2e-5, paired against the existing 3e-5 runs. ---
foreach ($seed in 1, 7) {
    Run "lr2e-5 / seed $seed" @(
        '--model', 'klue/roberta-base', '--lr', '2e-5', '--epochs', '3',
        '--normalize', 'raw', '--batch_size', '16', '--accum', '2',
        '--seed', "$seed", '--tag_suffix', '_lr2e-5e3', '--log', $LOG)
}

# --- B2: bagging. Same recipe, each member on a different 80% subsample. ---
foreach ($bag in 1, 2, 3, 4, 5) {
    Run "bag$bag (80% 부분집합)" @(
        '--model', 'klue/roberta-base', '--lr', '2e-5', '--epochs', '3',
        '--normalize', 'raw', '--batch_size', '16', '--accum', '2',
        '--seed', '42', '--train_frac', '0.8', '--bag_seed', "$bag", '--log', $LOG)
}

# --- B3: data-size curve, to extrapolate what training on 100% would buy. ---
foreach ($frac in '0.5', '0.65') {
    Run "데이터 $frac" @(
        '--model', 'klue/roberta-base', '--lr', '2e-5', '--epochs', '3',
        '--normalize', 'raw', '--batch_size', '16', '--accum', '2',
        '--seed', '42', '--train_frac', $frac, '--bag_seed', '0', '--log', $LOG)
}

"===== phase B 종료 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
