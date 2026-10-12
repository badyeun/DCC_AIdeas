# Chain: wait for the candidate-backbone wave, then start the within-backbone wave.
Set-Location C:/DevBox/projects/DCC_AIdeas
$log = 'C:/DevBox/projects/DCC_AIdeas/results/mission3/local/_cand.log'
while (-not (Select-String -Path $log -Pattern 'CAND_DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_wave2.ps1
