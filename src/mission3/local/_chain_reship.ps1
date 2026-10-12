# Chain: wait for the exploration wave, then re-ship the three large members
# at the validated epoch. The smoke gate inside run_large_reship.ps1 failed
# its first attempt only because the checker predicted the checkpoint's
# filename instead of looking for it; the feature itself saved epoch 2 of a
# 3-epoch schedule correctly, and the checker now globs.
Set-Location C:/DevBox/projects/DCC_AIdeas
$log = 'C:/DevBox/projects/DCC_AIdeas/results/mission3/local/_lge2.log'
while (-not (Select-String -Path $log -Pattern 'LGW_DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_large_reship.ps1
