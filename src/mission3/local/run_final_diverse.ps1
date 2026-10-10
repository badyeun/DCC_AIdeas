# Rebuild the submission ensemble as seven families instead of one recipe.
#
# Why: the size curve is flat past ten members, so more of the same buys
# nothing, but across 200 random draws an ensemble taking one member per
# family beat the current homogeneous six every single time (+0.0016).
# Member quality is the same either way; only the mutual disagreement
# differs (0.0295 -> 0.0358). That is the condition that made head mixing
# work in round one.
#
# Families shipped, eleven members:
#   base mean lr2e-5  x2   reused from the existing FINAL checkpoints
#   base attn lr2e-5  x2   reused
#   base mean lr3e-5  x1   new
#   base attn lr3e-5  x1   new
#   normalised text   x2   new  (checkpoint records its own variant)
#   domain adapted    x2   new  (MLM over all 29,200, one per head)
#   CLS pooling       x1   new  (-0.0015 quality, 0.0158 disagreement)
#
# Left out on purpose: roberta-large (60 min for the eleventh member, whose
# marginal value is near zero at the plateau) and max pooling (same quality
# cost and diversity as the attention head already in the mix).
#
# Rules: Seoul Training text only, public pretrained start, decisions at 0.5.
# Domain-adapted checkpoints record klue/roberta-base as their architecture
# so they load without the local directory.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/final_diverse_log.md'
$TRAIN = 'src/mission3/local/train_frozen8.py'
$FULLDAPT = 'results/mission3/local/domain_adapted_full'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'

function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY $TRAIN @extra --epochs 3 --batch_size 16 --accum 2 --full_train --save_model --log $LOG 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}

Go "D3 mean lr3e-5 s11" @('--model','klue/roberta-base','--lr','3e-5','--normalize','raw','--seed','11','--tag_suffix','_D3')
Go "D4 attn lr3e-5 s13" @('--model','klue/roberta-base','--head','label_attention','--lr','3e-5','--normalize','raw','--seed','13','--tag_suffix','_D4')
Go "D7 cls lr2e-5 s19"  @('--model','klue/roberta-base','--head','cls','--lr','2e-5','--normalize','raw','--seed','19','--tag_suffix','_D7')
foreach ($s in 15, 16) {
    Go "D5 norm s$s" @('--model','klue/roberta-base','--lr','2e-5','--normalize','synonym+filler','--seed',"$s",'--tag_suffix','_D5')
}
"[DIVERSE_PART1] $(Get-Date -Format 'HH:mm:ss')"

"===== full-data MLM start $(Get-Date -Format 'HH:mm:ss') ====="
& $PY src/mission3/local/domain_pretrain.py --epochs 3 --include_holdout --out $FULLDAPT 2>&1 | Where-Object { $_ -notmatch $NOISE }
"===== full-data MLM end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
if (Test-Path "$FULLDAPT/config.json") {
    Go "D6 dapt mean s17" @('--model',$FULLDAPT,'--arch_name','klue/roberta-base','--lr','2e-5','--normalize','raw','--seed','17','--tag_suffix','_D6')
    Go "D6 dapt attn s18" @('--model',$FULLDAPT,'--arch_name','klue/roberta-base','--head','label_attention','--lr','2e-5','--normalize','raw','--seed','18','--tag_suffix','_D6')
} else {
    "[WARN] full-data MLM produced nothing; shipping six families instead of seven"
}
"[DIVERSE_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
