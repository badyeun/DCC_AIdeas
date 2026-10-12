# The three exploration runs that the re-ship queue preempted.
#
# The first three of this wave all failed (-0.00096 to +0.00014 as a sixth
# large member), so the large axis looks saturated at five and shippable
# weights matter more than more exploration. These three are still worth the
# GPU once the re-ship is done: two of them vary the TAPT backbone, which is
# one of the five that pays, and seed diversity inside a contributing family
# is what already earns base two slots.
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
Go "large norm attn" ($LRG + @('--lr','1.5e-5','--normalize','synonym+filler','--head','label_attention','--seed','42'))
Go "tapt attn s42"   ($TPT + @('--lr','2e-5','--normalize','raw','--head','label_attention','--seed','42'))
Go "tapt mean s1"    ($TPT + @('--lr','2e-5','--normalize','raw','--seed','1'))
"[LGW2_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
