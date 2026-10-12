# Chain: wait for the two full-data large queues, then more large recipes.
Set-Location C:/DevBox/projects/DCC_AIdeas
$log = 'C:/DevBox/projects/DCC_AIdeas/results/mission3/local/_lgx.log'
while (-not (Select-String -Path $log -Pattern 'LGX_DONE' -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
& powershell -NoProfile -ExecutionPolicy Bypass -File C:/DevBox/projects/DCC_AIdeas/src/mission3/local/run_large_wave.ps1
