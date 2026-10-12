# More DISTINCT large configurations.
#
# Mechanism found today: among large-class members, distinct configurations
# pay and near-duplicates cost. The three large variants excluded from the
# chosen five have 0.0153 mutual disagreement (two of them 0.0000 -- the same
# model twice), against 0.0364 inside the chosen five, and adding any of them
# costs -0.0014 to -0.0020. So the lever is not more large seeds, it is more
# large RECIPES.
#
# These six are each distinct on an axis the chosen five do not cover: head,
# seed-on-a-head, learning rate, normalised text with attention, and the TAPT
# backbone under a second head and a second seed.
#
# Three epochs each, because the peak epoch is unknown for these and the
# history file records every epoch. Holdout runs only -- whichever earn a
# place get a full-data run afterwards at their own peak epoch.
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
    & $PY src/mission3/local/train_frozen9.py @extra --log $LOG --epochs 3 `
        --grad_checkpoint --pos_weight sqrt5 --tag_suffix '_LGW' 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
$LRG = @('--model','klue/roberta-large','--batch_size','4','--accum','8')
$TPT = @('--model','SangwonYoon/klue-roberta-large-tapt','--batch_size','8','--accum','4')
Go "large cls s42"      ($LRG + @('--lr','1.5e-5','--normalize','raw','--head','cls','--seed','42'))
Go "large attn s1"      ($LRG + @('--lr','1.5e-5','--normalize','raw','--head','label_attention','--seed','1'))
Go "large mean lr2.5e5" ($LRG + @('--lr','2.5e-5','--normalize','raw','--seed','42'))
Go "large norm attn"    ($LRG + @('--lr','1.5e-5','--normalize','synonym+filler','--head','label_attention','--seed','42'))
Go "tapt attn s42"      ($TPT + @('--lr','2e-5','--normalize','raw','--head','label_attention','--seed','42'))
Go "tapt mean s1"       ($TPT + @('--lr','2e-5','--normalize','raw','--seed','1'))
"[LGW_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
