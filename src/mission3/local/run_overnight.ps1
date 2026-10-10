# Local GPU experiment queue -- rerun after fixing the split bug.
#
# The first attempt grouped the holdout split on the PREPROCESSED text, so the
# raw and normalised arms were scored on different calls (only 19% overlap) and
# could not be compared. split() now runs on the raw transcripts before any
# normalisation, so every arm below sees the identical 5,840-call holdout.
#
# Ordered by information per minute: the paired base A/B first, because that is
# the question we actually need answered; the slower large runs after it.

$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:\DevBox\projects\DCC_AIdeas

$LOG = 'results\mission3\local\train_log.md'
$script = 'src\mission3\local\train_local.py'

function Step($label, $arguments) {
    "===== $label  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY $script @arguments 2>&1 | Where-Object {
        $_ -notmatch 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
    }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
    ""
}

function Base($norm, $seed, $epochs = 3, $suffix = '') {
    $arguments = @(
        '--model', 'klue/roberta-base', '--normalize', $norm, '--epochs', "$epochs",
        '--batch_size', '16', '--accum', '2', '--lr', '3e-5', '--seed', "$seed",
        '--log', $LOG)
    # PowerShell drops an empty string when splatting to a native command, which
    # would leave "--tag_suffix" dangling and argparse would reject the call.
    if ($suffix) { $arguments += @('--tag_suffix', $suffix) }
    Step "base / $norm / seed $seed / ${epochs}ep" $arguments
}

# --- 1. Paired base A/B, three seeds. ~15 min each, ~90 min total. ---
foreach ($seed in 42, 1, 7) {
    Base 'raw' $seed
    Base 'synonym+filler' $seed
}

# --- 2. Does the finding hold on the larger encoder? ~50 min each. ---
foreach ($norm in 'raw', 'synonym+filler') {
    Step "large / $norm / seed 42" @(
        '--model', 'klue/roberta-large', '--normalize', $norm, '--epochs', '3',
        '--batch_size', '4', '--accum', '8', '--eval_batch_size', '16',
        '--lr', '1.5e-5', '--seed', '42', '--grad_checkpoint', '--log', $LOG)
}

# --- 3. Extras, only if the night still has room. ---
Base 'synonym' 42                     # synonym merging without filler removal
Base 'synonym+filler' 42 6 '_e6'      # both normalised arms were still climbing
Base 'raw' 42 6 '_e6'

"===== 전체 종료 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
