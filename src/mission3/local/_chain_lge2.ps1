# Chain: wait for the tapt full run, re-ship the three large members at the
# validated epoch, and only then explore more large recipes.
Set-Location C:/DevBox/projects/DCC_AIdeas
$dir = 'C:/DevBox/projects/DCC_AIdeas/results/mission3/local'
while (-not (Select-String -Path "$dir/_lgx.log" -Pattern 'LGX_DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_large_reship.ps1
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_large_wave.ps1
