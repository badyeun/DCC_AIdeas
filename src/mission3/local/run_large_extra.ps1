# The two remaining large-class members of the chosen 16.
#
# Composition chosen by nested validation (decide on half A, measure on half
# B, 300 draws): current 11 + five large-class members scored +0.00362 on the
# unseen half and was picked 269/300 times. It is also the no-selection
# option -- every large-class candidate, nothing cherry-picked. Holdout
# 0.64666 -> 0.65038.
#
# run_large_full.ps1 produces three of the five. These are the other two.
#
# Settings read back from each model's own history file:
#   large norm : lr 1.5e-5, batch 4 x 8, synonym+filler, peaked epoch 3
#   tapt       : lr 2e-5,   batch 8 x 4, raw,            peaked epoch 3
# Both peak at the final epoch, so a 3-epoch run saves the right weights --
# unlike plain large, which peaks at 2.
#
# Stage 1 exists because the large-norm measurement predates --pos_weight:
# its history has no such key, so "same recipe" is an assumption, not a fact.
# Re-running it on the holdout under the current trainer either reproduces
# 0.63798 or tells me the measured member is not the one I would ship.
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
    & $PY src/mission3/local/train_frozen9.py @extra --log $LOG 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
$NORM = @('--model','klue/roberta-large','--lr','1.5e-5','--epochs','3',
          '--normalize','synonym+filler','--batch_size','4','--accum','8',
          '--grad_checkpoint','--pos_weight','sqrt5','--seed','42')
$TAPT = @('--model','SangwonYoon/klue-roberta-large-tapt','--lr','2e-5','--epochs','3',
          '--normalize','raw','--batch_size','8','--accum','4','--grad_checkpoint',
          '--pos_weight','sqrt5','--seed','42',
          '--arch_name','SangwonYoon/klue-roberta-large-tapt')
# Stage 1 -- reproduce the large-norm holdout number under the current trainer.
Go "norm holdout verify" ($NORM + @('--tag_suffix','_LGXV'))
"[LGX_STAGE1] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
# Stage 2 -- shippable full-data weights for both.
Go "full norm s42" ($NORM + @('--full_train','--save_model','--tag_suffix','_LGF'))
Go "full tapt s42" ($TAPT + @('--full_train','--save_model','--tag_suffix','_LGF'))
"[LGX_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
