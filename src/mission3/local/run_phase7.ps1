# Phase seven: two Korean backbones round 1 never tried.
#
# This is NOT label-side work -- it is queued last so it displaces nothing.
# The reason to run it at all: round 1 concluded "backbone diversity hurts"
# from four models, but that experiment asked whether MIXING backbones helps.
# It never asked whether a single BETTER backbone exists. kf-deberta-base is
# a DeBERTa-v2 trained on Korean financial and general text, and
# KR-ELECTRA-discriminator is a Korean ELECTRA -- both score above
# klue/roberta-base on Korean classification benchmarks.
#
# Judged the same way as everything else: paired seeds, F1@0.5.
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location C:\DevBox\projects\DCC_AIdeas
$LOG = 'results\mission3\local\labelside_log.md'
$NOISE = 'Loading weights|it/s\]|LOAD REPORT|UNEXPECTED|MISSING|^Key |^-+\+|^Notes:|^- |unauthenticated'
$BASE = @('--lr','2e-5','--epochs','3','--normalize','raw','--batch_size','16',
          '--accum','2','--log',$LOG,'--tag_suffix','_LS')
function Go($label, $extra) {
    "===== $label  시작 $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src\mission3\local\train_frozen4.py @BASE @extra 2>&1 | Where-Object { $_ -notmatch $NOISE }
    "===== $label  종료 $(Get-Date -Format 'HH:mm:ss')  exit=$LASTEXITCODE ====="
}
Go "kf-deberta s42"  @('--model','kakaobank/kf-deberta-base','--seed','42')
Go "KR-ELECTRA s42"  @('--model','snunlp/KR-ELECTRA-discriminator','--seed','42')
Go "kf-deberta s1"   @('--model','kakaobank/kf-deberta-base','--seed','1')
"[PHASE7_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
