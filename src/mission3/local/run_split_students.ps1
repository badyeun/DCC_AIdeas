# Students with different teachers.
#
# The shared-teacher ensemble failed for a measured reason: three students
# from one teacher each beat their control (+0.00109 mean, three paired seeds)
# but their ensemble lost to the teacher's (-0.00109, 15% win) because
# distillation pulled them together -- internal disagreement 0.0299 -> 0.0238.
#
# Here each student gets a different teacher, split by architecture: the
# eleven base members in one, the five large in the other. That split is the
# ensemble's own structure, not something read off a holdout number, and the
# two halves are the most dissimilar pair available.
#
# Two students per teacher so the pairing is not a single draw.
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
Go "split teachers" (@('src/mission3/local/split_teachers.py'))
$TR = 'src/mission3/local/train_frozen11.py'
$C = @('--epochs','3','--normalize','raw','--pos_weight','sqrt5',
       '--soft_debias','--soft_alpha','0.5','--full_train','--save_model','--log',$LOG)
$B = @($TR,'--model','klue/roberta-base','--lr','2e-5','--batch_size','16','--accum','2')
$TA = @('--soft_targets','results/mission3/local/soft_targets_base11.npy','--tag_suffix','_TA')
$TB = @('--soft_targets','results/mission3/local/soft_targets_large5.npy','--tag_suffix','_TB')
Go "student <- base11 s42"  (($B + @('--seed','42')) + $TA + $C)
Go "student <- large5 s42"  (($B + @('--seed','42')) + $TB + $C)
Go "student <- base11 s1"   (($B + @('--seed','1'))  + $TA + $C)
Go "student <- large5 s1"   (($B + @('--seed','1'))  + $TB + $C)
"[D7_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
