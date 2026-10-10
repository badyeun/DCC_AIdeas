# Phase eight: finish the two loose ends.
#
# 1. Label smoothing toward the base rate was meant only as the control for
#    distillation -- zero teacher knowledge. It beat the distillation it was
#    controlling for (F1 0.64143 vs 0.63248, ceiling 0.64636 vs 0.64667), which
#    says the ranking gain came from softening the targets, not from what the
#    teacher knew. One seed is not enough to act on, so two more.
# 2. kf-deberta and KR-ELECTRA: the session ended mid-run. Out of scope for the
#    label-side work, but round 1 only ever asked whether MIXING backbones
#    helps, never whether a better single one exists.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:/DevBox/projects/DCC_AIdeas
$LOG = 'results/mission3/local/labelside_log.md'
$CONST = 'results/mission3/local/soft_targets_constant.npy'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$BASE = @('--lr','2e-5','--epochs','3','--normalize','raw','--batch_size','16',
          '--accum','2','--log',$LOG,'--tag_suffix','_LS')
function Go($label, $extra) {
    "===== $label  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/train_frozen5.py @BASE @extra 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}
Go "라벨스무딩 const0.3 s1"  @('--model','klue/roberta-base','--seed','1','--soft_targets',$CONST,'--soft_alpha','0.3')
Go "라벨스무딩 const0.3 s7"  @('--model','klue/roberta-base','--seed','7','--soft_targets',$CONST,'--soft_alpha','0.3')
Go "kf-deberta s42" @('--model','kakaobank/kf-deberta-base','--seed','42')
Go "KR-ELECTRA s42" @('--model','snunlp/KR-ELECTRA-discriminator','--seed','42')
"[PHASE8_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
