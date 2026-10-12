# Ship roberta-large as a twelfth/thirteenth/fourteenth family member.
#
# Why: it is the only candidate in ~100 experiments that pays. Added as a
# family of three it is +0.00192 on the holdout at a 98% split-half win rate,
# while adding three DUPLICATE existing members is -0.00119 at 0%. A nested
# check -- decide on half A, measure on half B -- gives +0.00190 +/- 0.00007
# with 97% positive. So this is not a member-count effect.
#
# Why epochs 2, not 3: all three holdout runs peaked at epoch 2 and the
# trainer saves FINAL-epoch weights, not best-epoch. A 3-epoch run would
# therefore ship the declining epoch-3 model. But a 2-epoch run is not the
# same model as the epoch-2 snapshot of a 3-epoch run -- the LR schedule
# decays over a shorter horizon -- so the first two runs here re-measure the
# ensemble contribution under the exact schedule that will be shipped,
# BEFORE the full-data weights are trusted.
#
# Settings read back from history_roberta-large_raw_s1_LGC.json, not from
# memory: lr 1.5e-5, batch 4 x accum 8, grad checkpointing.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$L = @('--model','klue/roberta-large','--lr','1.5e-5','--epochs','2','--normalize','raw',
       '--batch_size','4','--accum','8','--grad_checkpoint','--log',$LOG)
function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen9.py @L @extra 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
# Stage 1 -- holdout, 2-epoch schedule. Verifies the gain survives the change.
Go "e2 holdout mean s1"   @('--seed','1','--tag_suffix','_E2')
Go "e2 holdout attn s42"  @('--seed','42','--head','label_attention','--tag_suffix','_E2')
"[LGF_STAGE1] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
# Stage 2 -- full 29,200 calls, shippable weights. Discarded if stage 1 fails.
Go "full mean s1"   @('--seed','1','--full_train','--save_model','--tag_suffix','_LGF')
Go "full mean s7"   @('--seed','7','--full_train','--save_model','--tag_suffix','_LGF')
Go "full attn s42"  @('--seed','42','--head','label_attention','--full_train','--save_model','--tag_suffix','_LGF')
"[LGF_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
