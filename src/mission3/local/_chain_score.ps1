# After the last exploration run, measure what the submission actually does.
#
# Two numbers are still missing and both need the GPU to itself:
#   * inference wall-clock on the real 3,640 Validation calls. The "14 minutes"
#     figure reported earlier was computed, not measured, and was retracted;
#     the 16-member estimate (2.6x base) is likewise arithmetic.
#   * Macro F1 on the official Validation set, once, as the submission
#     decision. Every composition choice was fixed on the Training holdout.
#
# Both folders are measured so the fp16 copy is confirmed equal in practice,
# not just at the probability level (0 label flips in 5,760 decisions).
Set-Location C:/DevBox/projects/DCC_AIdeas
$dir = 'C:/DevBox/projects/DCC_AIdeas/results/mission3/local'
while (-not (Select-String -Path "$dir/_final2.log" -Pattern 'LGW2_DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
$PY = 'C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe'
$env:PYTHONPATH = 'C:\DevBox\pylibs'
$env:PYTHONIOENCODING = 'utf-8'
# 기존 11개를 먼저 돌립니다. 0.65658 이 재현되지 않으면 제 측정 경로가
# 잘못된 것이므로, 새 숫자를 믿기 전에 그것부터 알아야 합니다.
foreach ($d in 'submission_final', 'submission_final16', 'submission_final16_fp16') {
    "===== score $d start $(Get-Date -Format 'HH:mm:ss') ====="
    & $PY src/mission3/local/score_validation.py --dir "results/mission3/local/$d" 2>&1
    "===== score $d end $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ====="
}
"[SCORE_DONE] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
