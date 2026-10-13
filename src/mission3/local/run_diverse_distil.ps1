# Diverse distillation: give each student a different teacher.
#
# The shared-teacher version failed for a measured reason. Three students
# distilled from one teacher each beat their matched control (+0.00109 mean
# over three paired seeds) but their ENSEMBLE lost to the teacher's ensemble
# (-0.00109, 15% win), because distillation pulled them toward each other:
# internal disagreement fell from 0.0299 to 0.0238. The individual gain and
# the diversity loss cancelled exactly.
#
# So: split the sixteen into disjoint teachers and give each student its own.
# Each student still absorbs several models, but they absorb DIFFERENT ones,
# which is the thing the shared teacher destroyed. The split is by family, not
# by score -- base recipes in one half, the large family plus the remaining
# base in the other -- so nothing is chosen from a holdout number.
#
# Stage 1 re-runs the teacher pass saving per-model Training probabilities
# (the first pass saved only their average), after which any split is free.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_OFFLINE = '1'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
function Go($label, $cmd) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    "      args: $($cmd.Count)"
    & $PY @cmd 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
# Stage 1 -- per-model Training probabilities for the sixteen.
Go "per-model teacher probs" (@('src/mission3/local/make_teacher.py',
    '--dir','results/mission3/local/submission_final16',
    '--out','soft_targets_ens16_again.npy',
    '--per_model_dir','results/mission3/local/train_probs16'))
"[D6_STAGE1] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
# Stage 2 -- Validation probabilities for every distilled student so far.
Go "student Validation probs" (@('src/mission3/local/val_probs.py'))
"[D6_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
