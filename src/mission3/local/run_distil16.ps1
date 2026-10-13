# Distil the sixteen-member ensemble into a single model.
#
# The three-model probe worked once the target was de-biased against
# pos_weight: one model scored 0.64241 against a teacher ensemble's 0.64235
# and its own matched control's 0.64071. That is the full ensemble gain
# carried by one checkpoint -- the first thing in this campaign that reduced
# model count without losing score.
#
# Stage 1 replicates the probe at two more seeds. One seed proves nothing when
# single-model seed spread is 0.0023 and the effect is 0.0017.
#
# Stage 2 and 3 are the real experiment: teacher is the shipped sixteen
# (2,903M parameters, Validation 0.65746), students are one base and one large
# trained on all 29,200 calls.
#
# Evaluation must be the official Validation set, not the holdout. The teacher
# was trained on all of Training including the holdout, so a holdout score for
# the student would be inflated by what leaks through the teacher. Validation
# is the only data the teacher never saw. alpha stays 0.5, fixed by the probe,
# so Validation decides nothing here.
#
# Every Go call wraps its argument array in parentheses. Without them
# PowerShell parses `Go "x" @(...) + $S` as a call plus a separate expression
# and $S vanishes -- the student would train with no soft targets, no
# --full_train and no --save_model, silently. Verified before writing this.
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
$P3 = @($TR,'--model','klue/roberta-base','--lr','2e-5','--epochs','3','--normalize','raw',
        '--batch_size','16','--accum','2','--pos_weight','sqrt5',
        '--soft_targets','results/mission3/local/soft_targets_ens3.npy',
        '--soft_debias','--soft_alpha','0.5','--tag_suffix','_D3','--log',$LOG)
Go "probe seed 1" ($P3 + @('--seed','1'))
Go "probe seed 7" ($P3 + @('--seed','7'))
"[D4_STAGE1] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Go "teacher from 16" (@('src/mission3/local/make_teacher.py',
                        '--dir','results/mission3/local/submission_final16',
                        '--out','soft_targets_ens16.npy'))
"[D4_STAGE2] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$S = @('--epochs','3','--normalize','raw','--pos_weight','sqrt5','--seed','42',
       '--soft_targets','results/mission3/local/soft_targets_ens16.npy',
       '--soft_debias','--soft_alpha','0.5','--full_train','--save_model','--log',$LOG)
Go "student base" (@($TR,'--model','klue/roberta-base','--lr','2e-5',
                     '--batch_size','16','--accum','2','--tag_suffix','_S16') + $S)
Go "student large" (@($TR,'--model','klue/roberta-large','--lr','1.5e-5',
                      '--batch_size','4','--accum','8','--grad_checkpoint',
                      '--save_at_epoch','2','--stop_at_save','--tag_suffix','_S16') + $S)
"[D4_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
