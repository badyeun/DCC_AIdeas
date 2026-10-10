# Overnight run, 2026-10-03. One file, no chains: chained scripts were the
# most fragile part of this campaign (a bad path, an encoding, a sentinel
# that never fired). Everything is sequential here, with an ASCII sentinel
# after each block so progress is visible.
#
# All of it obeys the brief: Seoul Training text only (utterances[].text),
# open-source pretrained checkpoints, no external data, and every decision
# still taken at 0.5 for all nine symptoms. Nothing here touches the cut.
#
# Order is by expected value, best first, so an interruption costs least.
#   11  DAPT + attention head  - the one DAPT arm that gained (+0.0024, 1 seed)
#   10  roberta-large LR sweep - large ran twice ever, one seed, one rate,
#                                never ensembled; its rate was inherited from
#                                the base sweep and large usually wants less
#   12  untouched cheap knobs  - scheduler, warmup, dropout, max_length
#   13  more base seeds        - pushes ensemble size past 12
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$ADAPTED = 'results/mission3/local/domain_adapted'
$TRAIN = 'src/mission3/local/train_frozen6.py'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'

function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY $TRAIN @extra --epochs 3 --normalize raw --log $LOG 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}

# --- 11: DAPT with the attention head -----------------------------------
$D = @('--model',$ADAPTED,'--head','label_attention','--lr','2e-5','--batch_size','16','--accum','2','--tag_suffix','_DAPT_LS')
foreach ($s in 1, 7) { Go "dapt attn s$s" ($D + @('--seed',"$s")) }
"[P11_DONE] $(Get-Date -Format 'HH:mm:ss')"

# --- 10: roberta-large, its own learning rate ---------------------------
# batch 8 x accum 4 with checkpointing: large at 512 tokens does not fit at 16.
$L = @('--model','klue/roberta-large','--batch_size','8','--accum','4','--grad_checkpoint','--tag_suffix','_LS')
Go "large lr1e-5 s42"   ($L + @('--lr','1e-5','--seed','42'))
Go "large lr1.5e-5 s42" ($L + @('--lr','1.5e-5','--seed','42'))
Go "large lr1e-5 s1"    ($L + @('--lr','1e-5','--seed','1'))
Go "large lr1e-5 s7"    ($L + @('--lr','1e-5','--seed','7'))
"[P10_DONE] $(Get-Date -Format 'HH:mm:ss')"

# --- 12: knobs never swept ----------------------------------------------
$B = @('--model','klue/roberta-base','--lr','2e-5','--batch_size','16','--accum','2','--seed','42','--tag_suffix','_LS')
Go "cosine s42"     ($B + @('--scheduler','cosine'))
Go "warmup0.2 s42"  ($B + @('--warmup','0.2'))
Go "dropout0.2 s42" ($B + @('--dropout','0.2'))
Go "maxlen256 s42"  ($B + @('--max_length','256'))
"[P12_DONE] $(Get-Date -Format 'HH:mm:ss')"

# --- 13: more seeds of the settled recipe, for ensemble size -------------
foreach ($s in 2026, 7777, 31) {
    Go "base mean s$s" @('--model','klue/roberta-base','--lr','2e-5','--batch_size','16','--accum','2','--seed',"$s",'--tag_suffix','_LS')
    Go "base attn s$s" @('--model','klue/roberta-base','--head','label_attention','--lr','2e-5','--batch_size','16','--accum','2','--seed',"$s",'--tag_suffix','_lr2e5_LS')
}
"[P13_DONE] $(Get-Date -Format 'HH:mm:ss')"
"[OVERNIGHT_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
