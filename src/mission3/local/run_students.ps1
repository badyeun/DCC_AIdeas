# More distilled students, so they can be ensembled.
#
# The question this answers: if one de-biased distilled model already carries
# a three-model ensemble's gain (0.64241 vs teacher 0.64235, control 0.64071),
# does an ENSEMBLE of such students go further? Each student has absorbed the
# teacher, and different seeds and heads still disagree, so there is diversity
# left to average over. Born-again networks are known to pass their teacher.
#
# F1 first, efficiency second -- so this stops optimising for model count and
# goes after the score. Six students, varied the way the main ensemble's
# families are varied (seed, head, architecture), because within-recipe seeds
# alone saturate fast.
#
# Teacher throughout is the shipped sixteen (soft_targets_ens16.npy), so every
# student here must be scored on the official Validation set: the teacher saw
# all of Training including the holdout.
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
$TR = 'src/mission3/local/train_frozen11.py'
# Shared: distil from the sixteen, de-biased, alpha fixed at 0.5 by the probe.
$S = @('--epochs','3','--normalize','raw','--pos_weight','sqrt5',
       '--soft_targets','results/mission3/local/soft_targets_ens16.npy',
       '--soft_debias','--soft_alpha','0.5','--full_train','--save_model',
       '--tag_suffix','_S16','--log',$LOG)
$B = @($TR,'--model','klue/roberta-base','--lr','2e-5','--batch_size','16','--accum','2')
$L = @($TR,'--model','klue/roberta-large','--lr','1.5e-5','--batch_size','4','--accum','8',
       '--grad_checkpoint','--save_at_epoch','2','--stop_at_save')
Go "student base s1"        (($B + @('--seed','1')) + $S)
Go "student base attn s42"  (($B + @('--seed','42','--head','label_attention')) + $S)
Go "student large s1"       (($L + @('--seed','1')) + $S)
Go "student large attn s42" (($L + @('--seed','42','--head','label_attention')) + $S)
"[D5_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
