# Chain: wait for phase 8, then start phase 9 (domain-adaptive pretraining).
Set-Location C:/DevBox/projects/DCC_AIdeas
$log = 'C:/DevBox/projects/DCC_AIdeas/results/mission3/local/_p8_console.log'
while (-not (Select-String -Path $log -Pattern 'PHASE8_DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_phase9.ps1
