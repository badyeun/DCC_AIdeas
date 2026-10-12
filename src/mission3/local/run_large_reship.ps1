# Re-ship the three klue/roberta-large members at the epoch that was validated.
#
# The first attempt trained them with --epochs 2, reasoning that the trainer
# saves final-epoch weights and large peaks at epoch 2. That reasoning was
# right about the save and wrong about the model: a 2-epoch run decays the
# learning rate to zero over a shorter horizon, so it is a different model.
# Re-measured on the holdout it cost -0.00172 of the ensemble gain
# (+0.00367 -> +0.00195), roughly half of it.
#
# --epochs 3 keeps the validated schedule; --save_at_epoch 2 ships the epoch
# that schedule peaks at; --stop_at_save skips the third epoch, which is only
# ever discarded. The epoch number comes from the holdout runs and is fixed
# before this queue starts, so --full_train still chooses nothing.
#
# Stage 0 proves the new flag does what it claims on a one-minute model
# before three hours are spent on it. The flag was added today and could not
# be tested on CPU -- the trainer assumes CUDA -- so it is tested here, on
# the GPU, with the queue refusing to continue if it fails.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated|newly initialized|should probably'

"===== stage 0 smoke test $(Get-Date -Format 'HH:mm:ss') ====="
& $PY src/mission3/local/train_frozen10.py --model klue/roberta-base --epochs 3 `
    --save_at_epoch 2 --stop_at_save --train_frac 0.01 --max_length 64 `
    --batch_size 8 --eval_batch_size 128 --save_model --tag_suffix '_SMOKE' `
    --log $LOG 2>&1 | Where-Object { $_ -notmatch $NOISE }
& $PY src/mission3/local/check_save_at_epoch.py
if ($LASTEXITCODE -ne 0) {
    "[LGE2_ABORT] --save_at_epoch 검증 실패. 재학습을 시작하지 않습니다."
    exit 1
}
# Glob, do not predict the name -- the tag is assembled from six fields and
# the guessed name left a 1%-data maxlen-64 checkpoint sitting in the
# submission folder, where an assembly could have picked it up.
Get-ChildItem 'results/mission3/local/submission_models/*_SMOKE.ckpt' -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem 'results/mission3/local/probs_*_SMOKE.npy','results/mission3/local/history_*_SMOKE.json' -ErrorAction SilentlyContinue | Remove-Item -Force

$L = @('--model','klue/roberta-large','--lr','1.5e-5','--epochs','3','--normalize','raw',
       '--batch_size','4','--accum','8','--grad_checkpoint','--pos_weight','sqrt5',
       '--save_at_epoch','2','--stop_at_save','--full_train','--save_model',
       '--tag_suffix','_LGE2','--log',$LOG)
function Go($label, $extra) {
    "===== $label start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen10.py @L @extra 2>&1 |
        Where-Object { $_ -notmatch $NOISE }
    "===== $label end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
Go "full mean s1 (e2 of 3)"  @('--seed','1')
Go "full mean s7 (e2 of 3)"  @('--seed','7')
Go "full attn s42 (e2 of 3)" @('--seed','42','--head','label_attention')
"[LGE2_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
